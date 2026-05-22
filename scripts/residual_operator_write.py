"""Fit a persistent residual-stream operator from context traces.

This is a diagnostic between direct residual replay and MLP down-projection
TSOC writes. Direct replay patches each eval prompt with its own teacher delta;
the MLP write has to infer that delta through down-projection features. This
script asks a narrower question: if we fit a linear operator directly on the
block residual stream, can it generalize from write probes to held-out prompts?
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterator

import torch
from torch.nn import functional as F

from caic.evaluation import evaluate_yes_no, format_question_prompt
from caic.experiment import load_domain_rows
from caic.modeling import capture_block_io, get_decoder_layers, load_model_and_tokenizer
from caic.modeling import capture_layer_io
from caic.synthetic import (
    DomainSpec,
    general_guard_questions,
    make_candidate_probes,
    make_gauntlet_questions,
    make_minimal_pair_questions,
    make_near_collision_questions,
    make_null_document,
)
from caic.tsoc import (
    block_source_targets,
    mean_row_cosine,
    mean_row_l2,
    protected_ridge_update,
    principal_components,
    project_rows_away_from_basis,
    projection_energy_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--domains-jsonl", required=True)
    parser.add_argument("--papers", type=int, default=1)
    parser.add_argument(
        "--paper-offset",
        type=int,
        default=0,
        help="Skip this many domain rows before running; useful for targeted layer sweeps.",
    )
    parser.add_argument("--output", default="runs/residual_operator_write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[20])
    parser.add_argument("--trace-probes", type=int, default=32)
    parser.add_argument(
        "--contrastive-trace-pairs",
        action="store_true",
        help="Use one-edit opposite-label pairs as trace probes and contrast their teacher deltas.",
    )
    parser.add_argument(
        "--contrastive-auxiliary-pairs",
        type=int,
        default=0,
        help="Add this many one-edit opposite-label pair traces as contrastive auxiliary rows.",
    )
    parser.add_argument("--contrastive-auxiliary-scale", type=float, default=0.5)
    parser.add_argument("--contrastive-target-scale", type=float, default=0.5)
    parser.add_argument(
        "--surprise-weighting",
        choices=["none", "info_gain", "novelty", "target_norm", "combined"],
        default="none",
        help=(
            "Weight positive write rows by label-free surprise signals. "
            "combined uses context/null KL, target norm, and guard dissimilarity."
        ),
    )
    parser.add_argument("--surprise-top-k", type=int, default=128)
    parser.add_argument("--surprise-weight-floor", type=float, default=0.25)
    parser.add_argument("--surprise-weight-temperature", type=float, default=1.5)
    parser.add_argument(
        "--activation-energy-weighting",
        choices=["none", "block_action", "mlp_key", "combined"],
        default="none",
        help="Weight rows by single-forward activation energy from the context-conditioned trace.",
    )
    parser.add_argument("--activation-energy-weight-floor", type=float, default=0.25)
    parser.add_argument("--activation-energy-weight-temperature", type=float, default=2.0)
    parser.add_argument(
        "--object-default-basis-papers",
        type=int,
        default=0,
        help=(
            "Project target deltas away from top PCs of other same-task domains. "
            "This approximates object-conditioned expected state and writes only the deviation."
        ),
    )
    parser.add_argument("--object-default-basis-probes", type=int, default=8)
    parser.add_argument("--object-default-basis-components", type=int, default=4)
    parser.add_argument("--object-default-projection-strength", type=float, default=1.0)
    parser.add_argument("--trace-last-tokens", type=int, default=1)
    parser.add_argument("--eval-capture-last-tokens", type=int, default=1)
    parser.add_argument("--target-mode", choices=["teacher_delta", "teacher_source"], default="teacher_delta")
    parser.add_argument("--key-site", choices=["block_input", "block_output"], default="block_output")
    parser.add_argument(
        "--solve-mode",
        choices=[
            "vector_ridge",
            "margin_gradient",
            "teacher_kl_gradient",
            "teacher_kl_weighted_vector",
            "teacher_logit_jacobian",
            "vector_logit_calibrated",
            "vector_kl_orthogonalized",
        ],
        default="vector_ridge",
        help=(
            "margin_gradient fits labeled Yes/No scalar constraints; "
            "teacher_kl_gradient fits generic teacher-distribution KL constraints; "
            "teacher_kl_weighted_vector keeps vector targets but gates rows by generic KL relevance; "
            "teacher_logit_jacobian matches teacher next-token logit changes through local residual Jacobians; "
            "vector_logit_calibrated rescales a vector write by teacher-logit Jacobian fit; "
            "vector_kl_orthogonalized removes the generic KL-gradient update direction from a vector write."
        ),
    )
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=500.0)
    parser.add_argument("--negative-guards", type=int, default=8)
    parser.add_argument("--rival-negative-guards", type=int, default=8)
    parser.add_argument("--operator-gate", action="store_true")
    parser.add_argument("--operator-gate-threshold", type=float, default=0.95)
    parser.add_argument("--operator-gate-temperature", type=float, default=80.0)
    parser.add_argument("--operator-final-token-only", action="store_true")
    parser.add_argument("--eval-questions", type=int, default=30)
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument(
        "--teacher-kl-top-k",
        type=int,
        default=128,
        help="Use the teacher's top-k next-token distribution for teacher_kl_gradient; <=0 uses the full vocab.",
    )
    parser.add_argument("--teacher-kl-weight-threshold", type=float, default=0.0)
    parser.add_argument("--teacher-kl-weight-temperature", type=float, default=8.0)
    parser.add_argument("--teacher-kl-weight-floor", type=float, default=0.25)
    parser.add_argument("--teacher-logit-top-k", type=int, default=4)
    parser.add_argument("--teacher-logit-center", action="store_true", default=True)
    parser.add_argument("--no-teacher-logit-center", dest="teacher_logit_center", action="store_false")
    parser.add_argument("--teacher-logit-calibration-max-scale", type=float, default=1.0)
    parser.add_argument("--functional-orthogonalize-strength", type=float, default=1.0)
    parser.add_argument("--functional-orthogonalize-eta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def prompts_for_questions(tokenizer, questions: list, paper: str | None, use_chat_template: bool) -> list[str]:
    return [
        format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        for record in questions
    ]


def negative_questions_for_domain(domain: DomainSpec, args: argparse.Namespace, paper_idx: int) -> list:
    questions = general_guard_questions()[: args.negative_guards]
    if args.rival_negative_guards > 0:
        rivals = make_near_collision_questions(
            domain,
            args.rival_negative_guards * 2,
            seed=args.seed * 300_000 + paper_idx,
            include_rival_prompts=True,
        )
        rival_only = [record for record in rivals if record.category == "near_collision_rival"]
        questions.extend(rival_only[: args.rival_negative_guards])
    return questions


def block_rows_for_site(capture, key_site: str) -> torch.Tensor:
    if key_site == "block_input":
        return capture.inputs
    if key_site == "block_output":
        return capture.outputs
    raise ValueError(f"Unknown key site: {key_site}")


def target_rows_for_mode(target_mode: str, full_capture, null_capture) -> torch.Tensor:
    if target_mode == "teacher_delta":
        return full_capture.outputs.float() - null_capture.outputs.float()
    if target_mode == "teacher_source":
        return block_source_targets(
            full_capture.inputs,
            full_capture.outputs,
            null_capture.inputs,
            null_capture.outputs,
        )
    raise ValueError(f"Unknown target mode: {target_mode}")


def contrastive_targets(rows: torch.Tensor, scale: float) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"contrastive_targets expects rows [n, dim], got {tuple(rows.shape)}")
    if rows.shape[0] % 2 != 0:
        raise ValueError("contrastive trace rows must be an even number of pair rows.")
    paired = rows.float().reshape(rows.shape[0] // 2, 2, rows.shape[1])
    diff = paired[:, 0, :] - paired[:, 1, :]
    out = torch.empty_like(paired)
    out[:, 0, :] = scale * diff
    out[:, 1, :] = -scale * diff
    return out.reshape_as(rows).contiguous()


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer returned empty Yes/No completions.")
    return yes_ids[0], no_ids[0]


def hidden_from_output(module_output: Any) -> torch.Tensor:
    if isinstance(module_output, torch.Tensor):
        return module_output
    if isinstance(module_output, (tuple, list)) and module_output and isinstance(module_output[0], torch.Tensor):
        return module_output[0]
    raise TypeError(f"Could not resolve hidden-state tensor from {type(module_output)!r}.")


def replace_hidden(module_output: Any, patched_hidden: torch.Tensor) -> Any:
    if isinstance(module_output, torch.Tensor):
        return patched_hidden
    if isinstance(module_output, tuple):
        return (patched_hidden, *module_output[1:])
    if isinstance(module_output, list):
        out = list(module_output)
        out[0] = patched_hidden
        return out
    raise TypeError(f"Could not replace hidden-state tensor in {type(module_output)!r}.")


def capture_margin_gradients(
    model,
    tokenizer,
    prompts: list[str],
    questions: list,
    layer_idx: int,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Gradient of the correct Yes/No logit margin wrt final block output."""

    yes_id, no_id = yes_no_token_ids(tokenizer)
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    layer = layers[resolved]
    rows: list[torch.Tensor] = []

    for prompt, record in zip(prompts, questions):
        stored: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            hidden = hidden_from_output(output)
            patched = hidden.detach().clone().requires_grad_(True)
            stored["hidden"] = patched
            return replace_hidden(output, patched)

        tokens = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        model.zero_grad(set_to_none=True)
        handle = layer.register_forward_hook(hook)
        try:
            outputs = model(**tokens, use_cache=False)
            logits = outputs.logits[:, -1, :]
            if record.answer:
                margin = logits[0, yes_id] - logits[0, no_id]
            else:
                margin = logits[0, no_id] - logits[0, yes_id]
            margin.backward()
            grad = stored["hidden"].grad
            if grad is None:
                raise RuntimeError("No gradient captured for residual hidden state.")
            rows.append(grad[0, -1, :].detach().float().cpu())
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    return torch.stack(rows, dim=0)


def teacher_distribution_from_logits(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    logits = logits.detach().float()
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return torch.softmax(logits, dim=-1)
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    return indices, probs


def prompt_topk_kl_scores(
    model,
    tokenizer,
    teacher_prompts: list[str],
    null_prompts: list[str],
    device: torch.device,
    max_length: int,
    top_k: int,
) -> torch.Tensor:
    """Approximate KL(p_context || p_null) at the next-token distribution.

    This is label-free: it measures how much the support context changes the
    model's local prediction relative to a length/style null context.
    """

    if len(teacher_prompts) != len(null_prompts):
        raise ValueError("teacher_prompts and null_prompts must have matching length.")
    scores: list[torch.Tensor] = []
    for teacher_prompt, null_prompt in zip(teacher_prompts, null_prompts):
        teacher_tokens = tokenizer(
            [teacher_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        null_tokens = tokenizer(
            [null_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        teacher_tokens = {name: value.to(device) for name, value in teacher_tokens.items()}
        null_tokens = {name: value.to(device) for name, value in null_tokens.items()}
        with torch.no_grad():
            teacher_logits = model(**teacher_tokens, use_cache=False).logits[:, -1, :].float()
            null_logits = model(**null_tokens, use_cache=False).logits[:, -1, :].float()
            if top_k > 0 and top_k < teacher_logits.shape[-1]:
                values, indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                teacher_probs = torch.softmax(values, dim=-1)
                null_selected = torch.gather(null_logits, dim=-1, index=indices)
                null_log_probs = torch.log_softmax(null_selected, dim=-1)
                teacher_log_probs = torch.log_softmax(values, dim=-1)
                kl = torch.sum(teacher_probs * (teacher_log_probs - null_log_probs), dim=-1)
            else:
                teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
                teacher_probs = torch.exp(teacher_log_probs)
                null_log_probs = torch.log_softmax(null_logits, dim=-1)
                kl = torch.sum(teacher_probs * (teacher_log_probs - null_log_probs), dim=-1)
            scores.append(kl.detach().cpu().float().squeeze(0))
    return torch.stack(scores, dim=0)


def repeat_prompt_scores(scores: torch.Tensor, capture_last_tokens: int) -> torch.Tensor:
    if capture_last_tokens <= 0:
        raise ValueError("capture_last_tokens must be positive.")
    return scores.detach().float().repeat_interleave(capture_last_tokens)


def standardized(values: torch.Tensor) -> torch.Tensor:
    values_f = values.detach().float().flatten()
    if values_f.numel() == 0:
        return values_f
    std = values_f.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return torch.zeros_like(values_f)
    return (values_f - values_f.mean()) / std


def max_cosine_to_rows(rows: torch.Tensor, prototypes: torch.Tensor | None) -> torch.Tensor:
    if prototypes is None or prototypes.numel() == 0:
        return torch.zeros(rows.shape[0], dtype=torch.float32)
    rows_n = F.normalize(rows.detach().float(), dim=1)
    prototypes_n = F.normalize(prototypes.detach().float(), dim=1)
    return (rows_n @ prototypes_n.T).amax(dim=1).cpu().float()


def surprise_row_weights(
    mode: str,
    trace_keys: torch.Tensor,
    targets: torch.Tensor,
    guard_keys: torch.Tensor | None,
    info_gain: torch.Tensor | None,
    *,
    floor: float,
    temperature: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Return positive-row weights for a protected ridge solve.

    The weights are normalized to mean 1 so `eta` remains roughly comparable
    across weighted and unweighted runs.
    """

    if mode == "none":
        return None, {}
    if floor < 0 or floor > 1:
        raise ValueError("--surprise-weight-floor must be in [0, 1].")
    rows = trace_keys.shape[0]
    components: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    if mode in {"info_gain", "combined"}:
        if info_gain is None:
            raise ValueError(f"{mode} surprise weighting requires info_gain scores.")
        if info_gain.numel() != rows:
            raise ValueError(f"info_gain rows {info_gain.numel()} != trace rows {rows}.")
        info = info_gain.detach().float().clamp_min(0)
        components.append(standardized(torch.log1p(info)))
        stats["surprise_info_gain_mean"] = float(info.mean().item())
        stats["surprise_info_gain_max"] = float(info.max().item())
    if mode in {"target_norm", "combined"}:
        target_norm = torch.linalg.vector_norm(targets.detach().float(), dim=1)
        components.append(standardized(torch.log1p(target_norm)))
        stats["surprise_target_norm_mean"] = float(target_norm.mean().item())
        stats["surprise_target_norm_max"] = float(target_norm.max().item())
    if mode in {"novelty", "combined"}:
        guard_cos = max_cosine_to_rows(trace_keys, guard_keys)
        novelty = 1.0 - guard_cos
        components.append(standardized(novelty))
        stats["surprise_guard_cos_mean"] = float(guard_cos.mean().item())
        stats["surprise_guard_cos_max"] = float(guard_cos.max().item())
    if not components:
        raise ValueError(f"Unknown surprise weighting mode: {mode}")
    score = torch.stack(components, dim=0).mean(dim=0)
    gate = floor + (1.0 - floor) * torch.sigmoid(temperature * score)
    weights = gate / gate.mean().clamp_min(1e-8)
    stats.update(
        {
            "surprise_weighting": mode,
            "surprise_weight_floor": float(floor),
            "surprise_weight_temperature": float(temperature),
            "surprise_weight_mean": float(weights.mean().item()),
            "surprise_weight_min": float(weights.min().item()),
            "surprise_weight_max": float(weights.max().item()),
            "surprise_score_mean": float(score.mean().item()),
            "surprise_score_min": float(score.min().item()),
            "surprise_score_max": float(score.max().item()),
        }
    )
    return weights.contiguous(), stats


def apply_positive_row_weights(
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights is None:
        return keys, targets
    if weights.ndim != 1 or weights.shape[0] != keys.shape[0]:
        raise ValueError(f"weights must be [{keys.shape[0]}], got {tuple(weights.shape)}")
    scale = weights.detach().float().clamp_min(1e-8).sqrt().unsqueeze(1)
    return keys.float() * scale, targets.float() * scale


def normalized_gate_from_score(
    score: torch.Tensor,
    *,
    floor: float,
    temperature: float,
) -> torch.Tensor:
    if floor < 0 or floor > 1:
        raise ValueError("weight floor must be in [0, 1].")
    gate = floor + (1.0 - floor) * torch.sigmoid(temperature * score.detach().float())
    return gate / gate.mean().clamp_min(1e-8)


def combine_row_weights(*weights: torch.Tensor | None) -> torch.Tensor | None:
    kept = [weight.detach().float() for weight in weights if weight is not None]
    if not kept:
        return None
    combined = kept[0]
    for weight in kept[1:]:
        if weight.shape != combined.shape:
            raise ValueError(f"Cannot combine row weights with shapes {tuple(combined.shape)} and {tuple(weight.shape)}")
        combined = combined * weight
    return combined / combined.mean().clamp_min(1e-8)


def activation_energy_row_weights(
    mode: str,
    full_capture,
    full_mlp_capture,
    *,
    floor: float,
    temperature: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Single-forward energy proxy for how atypically hard the trace is working.

    `block_action` uses the residual movement made by the block. `mlp_key` uses
    the norm of post-activation MLP features entering the down projection. Both
    are available from the context-conditioned forward trace.
    """

    if mode == "none":
        return None, {}
    components: list[torch.Tensor] = []
    stats: dict[str, float] = {"activation_energy_weighting": mode}
    if mode in {"block_action", "combined"}:
        action = torch.linalg.vector_norm(
            full_capture.outputs.float() - full_capture.inputs.float(),
            dim=1,
        )
        components.append(standardized(torch.log1p(action)))
        stats["activation_energy_block_action_mean"] = float(action.mean().item())
        stats["activation_energy_block_action_max"] = float(action.max().item())
    if mode in {"mlp_key", "combined"}:
        if full_mlp_capture is None:
            raise ValueError(f"{mode} activation energy requires MLP captures.")
        mlp_key = torch.linalg.vector_norm(full_mlp_capture.keys.float(), dim=1)
        mlp_key = mlp_key / (full_mlp_capture.keys.shape[1] ** 0.5)
        components.append(standardized(torch.log1p(mlp_key)))
        stats["activation_energy_mlp_key_mean"] = float(mlp_key.mean().item())
        stats["activation_energy_mlp_key_max"] = float(mlp_key.max().item())
    if not components:
        raise ValueError(f"Unknown activation energy mode: {mode}")
    score = torch.stack(components, dim=0).mean(dim=0)
    weights = normalized_gate_from_score(score, floor=floor, temperature=temperature)
    stats.update(
        {
            "activation_energy_weight_floor": float(floor),
            "activation_energy_weight_temperature": float(temperature),
            "activation_energy_weight_mean": float(weights.mean().item()),
            "activation_energy_weight_min": float(weights.min().item()),
            "activation_energy_weight_max": float(weights.max().item()),
            "activation_energy_score_mean": float(score.mean().item()),
            "activation_energy_score_min": float(score.min().item()),
            "activation_energy_score_max": float(score.max().item()),
        }
    )
    return weights.contiguous(), stats


def remove_basis_with_strength(rows: torch.Tensor, basis: torch.Tensor, strength: float) -> torch.Tensor:
    if basis.numel() == 0 or strength <= 0:
        return rows
    residual = project_rows_away_from_basis(rows, basis)
    return rows.float() + strength * (residual - rows.float())


def choose_default_domains(domains: list[DomainSpec], paper_idx: int, count: int) -> list[DomainSpec]:
    if count <= 0:
        return []
    candidates = [domain for idx, domain in enumerate(domains) if idx != paper_idx]
    return candidates[:count]


def capture_teacher_kl_gradients(
    model,
    tokenizer,
    student_prompts: list[str],
    teacher_prompts: list[str],
    layer_idx: int,
    device: torch.device,
    max_length: int,
    top_k: int,
) -> torch.Tensor:
    """Gradient of teacher-distribution KL wrt final block output.

    This is deliberately label-free. The "teacher" is just the same base model
    with the document/context present. The gradient marks residual directions
    that would make the no-context student distribution look more like that
    context-conditioned teacher.
    """

    if len(student_prompts) != len(teacher_prompts):
        raise ValueError("student_prompts and teacher_prompts must have matching length.")
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    layer = layers[resolved]
    rows: list[torch.Tensor] = []

    for student_prompt, teacher_prompt in zip(student_prompts, teacher_prompts):
        teacher_tokens = tokenizer(
            [teacher_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        teacher_tokens = {name: value.to(device) for name, value in teacher_tokens.items()}
        with torch.no_grad():
            teacher_logits = model(**teacher_tokens, use_cache=False).logits[:, -1, :]
            teacher_distribution = teacher_distribution_from_logits(teacher_logits, top_k)

        stored: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            hidden = hidden_from_output(output)
            patched = hidden.detach().clone().requires_grad_(True)
            stored["hidden"] = patched
            return replace_hidden(output, patched)

        student_tokens = tokenizer(
            [student_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        student_tokens = {name: value.to(device) for name, value in student_tokens.items()}
        model.zero_grad(set_to_none=True)
        handle = layer.register_forward_hook(hook)
        try:
            outputs = model(**student_tokens, use_cache=False)
            student_logits = outputs.logits[:, -1, :].float()
            if isinstance(teacher_distribution, tuple):
                indices, teacher_probs = teacher_distribution
                student_selected = torch.gather(student_logits, dim=-1, index=indices)
                student_log_probs = torch.log_softmax(student_selected, dim=-1)
                loss = torch.nn.functional.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                )
            else:
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                loss = torch.nn.functional.kl_div(
                    student_log_probs,
                    teacher_distribution,
                    reduction="batchmean",
                )
            loss.backward()
            grad = stored["hidden"].grad
            if grad is None:
                raise RuntimeError("No gradient captured for residual hidden state.")
            rows.append(grad[0, -1, :].detach().float().cpu())
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    return torch.stack(rows, dim=0)


def capture_teacher_logit_jacobian_constraints(
    model,
    tokenizer,
    student_prompts: list[str],
    teacher_prompts: list[str],
    layer_idx: int,
    device: torch.device,
    max_length: int,
    top_k: int,
    center: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return gradients and desired teacher logit changes.

    For each prompt, choose the teacher's top-k next-token logits. The desired
    scalar constraints are the teacher-minus-student logit changes for those
    tokens. This is label-free: the context teacher supplies the functional
    target directly.
    """

    if len(student_prompts) != len(teacher_prompts):
        raise ValueError("student_prompts and teacher_prompts must have matching length.")
    if top_k <= 0:
        raise ValueError("teacher_logit_jacobian requires --teacher-logit-top-k > 0.")
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    layer = layers[resolved]
    gradient_rows: list[torch.Tensor] = []
    desired_rows: list[torch.Tensor] = []

    for student_prompt, teacher_prompt in zip(student_prompts, teacher_prompts):
        teacher_tokens = tokenizer(
            [teacher_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        teacher_tokens = {name: value.to(device) for name, value in teacher_tokens.items()}
        with torch.no_grad():
            teacher_logits = model(**teacher_tokens, use_cache=False).logits[:, -1, :].float()
            _values, indices = torch.topk(teacher_logits, k=min(top_k, teacher_logits.shape[-1]), dim=-1)

        stored: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            hidden = hidden_from_output(output)
            patched = hidden.detach().clone().requires_grad_(True)
            stored["hidden"] = patched
            return replace_hidden(output, patched)

        student_tokens = tokenizer(
            [student_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        student_tokens = {name: value.to(device) for name, value in student_tokens.items()}
        model.zero_grad(set_to_none=True)
        handle = layer.register_forward_hook(hook)
        try:
            outputs = model(**student_tokens, use_cache=False)
            student_logits = outputs.logits[:, -1, :].float()
            teacher_selected = torch.gather(teacher_logits, dim=-1, index=indices).squeeze(0)
            student_selected = torch.gather(student_logits, dim=-1, index=indices).squeeze(0)
            desired = teacher_selected - student_selected.detach()
            if center:
                desired = desired - desired.mean()
            hidden = stored["hidden"]
            for idx, token_idx in enumerate(indices.squeeze(0).tolist()):
                grad = torch.autograd.grad(
                    student_logits[0, token_idx],
                    hidden,
                    retain_graph=idx < indices.shape[1] - 1,
                )[0]
                gradient_rows.append(grad[0, -1, :].detach().float().cpu())
                desired_rows.append(desired[idx].detach().float().cpu())
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    return torch.stack(gradient_rows, dim=0), torch.stack(desired_rows, dim=0), min(top_k, teacher_logits.shape[-1])


def margin_gradient_update(
    keys: torch.Tensor,
    targets: torch.Tensor,
    gradients: torch.Tensor,
    negative_keys: torch.Tensor | None,
    negative_gradients: torch.Tensor | None,
    ridge: float,
    negative_weight: float,
    eta: float,
    max_update_norm: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if keys.shape != targets.shape or keys.shape != gradients.shape:
        raise ValueError(
            "margin_gradient solve expects keys, targets, and gradients "
            f"to have matching shape, got {tuple(keys.shape)}, {tuple(targets.shape)}, {tuple(gradients.shape)}"
        )
    k_pos = keys.detach().float()
    g_pos = gradients.detach().float()
    desired_pos = eta * torch.sum(g_pos * targets.detach().float(), dim=1)
    all_keys = [k_pos]
    all_grads = [g_pos]
    all_desired = [desired_pos]
    negative_rows = 0
    if (
        negative_keys is not None
        and negative_gradients is not None
        and negative_keys.numel() > 0
        and negative_gradients.numel() > 0
    ):
        if negative_keys.shape != negative_gradients.shape:
            raise ValueError("negative_keys and negative_gradients must have matching shapes.")
        scale = negative_weight**0.5
        all_keys.append(negative_keys.detach().float() * scale)
        all_grads.append(negative_gradients.detach().float())
        all_desired.append(torch.zeros(negative_keys.shape[0], dtype=torch.float32))
        negative_rows = negative_keys.shape[0]
    k_all = torch.cat(all_keys, dim=0)
    g_all = torch.cat(all_grads, dim=0)
    desired = torch.cat(all_desired, dim=0)
    gram = (k_all @ k_all.T) * (g_all @ g_all.T)
    system = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    alpha = torch.linalg.pinv(system) @ desired
    update = torch.einsum("n,no,ni->oi", alpha, g_all, k_all)
    if max_update_norm is not None:
        norm = torch.linalg.vector_norm(update)
        if float(norm.item()) > max_update_norm:
            update = update * (max_update_norm / (float(norm.item()) + 1e-12))
    predicted = torch.sum(g_pos * (k_pos @ update.T), dim=1)
    fit_rmse = torch.sqrt(torch.mean((predicted - desired_pos).square())).item()
    if negative_rows:
        k_neg = all_keys[1]
        g_neg = all_grads[1]
        neg_pred = torch.sum(g_neg * (k_neg @ update.T), dim=1)
        negative_rmse = torch.sqrt(torch.mean(neg_pred.square())).item()
    else:
        negative_rmse = 0.0
    stats = {
        "positive_rows": int(k_pos.shape[0]),
        "negative_rows": int(negative_rows),
        "ridge": float(ridge),
        "negative_weight": float(negative_weight),
        "eta": float(eta),
        "update_fro": float(torch.linalg.vector_norm(update).item()),
        "target_fro": float(torch.linalg.vector_norm(targets.float()).item()),
        "fit_rmse": float(fit_rmse),
        "negative_rmse": float(negative_rmse),
        "mean_desired_margin_delta": float(desired_pos.mean().item()),
        "mean_abs_desired_margin_delta": float(desired_pos.abs().mean().item()),
    }
    return update.contiguous(), stats


def scalar_gradient_update(
    keys: torch.Tensor,
    gradients: torch.Tensor,
    desired: torch.Tensor,
    negative_keys: torch.Tensor | None,
    negative_gradients: torch.Tensor | None,
    ridge: float,
    negative_weight: float,
    eta: float,
    max_update_norm: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if keys.shape != gradients.shape:
        raise ValueError(f"keys and gradients must match, got {tuple(keys.shape)} and {tuple(gradients.shape)}")
    if desired.ndim != 1 or desired.shape[0] != keys.shape[0]:
        raise ValueError(f"desired must be [{keys.shape[0]}], got {tuple(desired.shape)}")
    k_pos = keys.detach().float()
    g_pos = gradients.detach().float()
    desired_pos = eta * desired.detach().float()
    all_keys = [k_pos]
    all_grads = [g_pos]
    all_desired = [desired_pos]
    negative_rows = 0
    if (
        negative_keys is not None
        and negative_gradients is not None
        and negative_keys.numel() > 0
        and negative_gradients.numel() > 0
    ):
        if negative_keys.shape != negative_gradients.shape:
            raise ValueError("negative_keys and negative_gradients must have matching shapes.")
        scale = negative_weight**0.5
        all_keys.append(negative_keys.detach().float() * scale)
        all_grads.append(negative_gradients.detach().float())
        all_desired.append(torch.zeros(negative_keys.shape[0], dtype=torch.float32))
        negative_rows = negative_keys.shape[0]
    k_all = torch.cat(all_keys, dim=0)
    g_all = torch.cat(all_grads, dim=0)
    desired_all = torch.cat(all_desired, dim=0)
    gram = (k_all @ k_all.T) * (g_all @ g_all.T)
    system = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    alpha = torch.linalg.pinv(system) @ desired_all
    update = torch.einsum("n,no,ni->oi", alpha, g_all, k_all)
    if max_update_norm is not None:
        norm = torch.linalg.vector_norm(update)
        if float(norm.item()) > max_update_norm:
            update = update * (max_update_norm / (float(norm.item()) + 1e-12))
    predicted = torch.sum(g_pos * (k_pos @ update.T), dim=1)
    fit_rmse = torch.sqrt(torch.mean((predicted - desired_pos).square())).item()
    if negative_rows:
        k_neg = all_keys[1]
        g_neg = all_grads[1]
        neg_pred = torch.sum(g_neg * (k_neg @ update.T), dim=1)
        negative_rmse = torch.sqrt(torch.mean(neg_pred.square())).item()
    else:
        negative_rmse = 0.0
    return update.contiguous(), {
        "positive_rows": int(k_pos.shape[0]),
        "negative_rows": int(negative_rows),
        "ridge": float(ridge),
        "negative_weight": float(negative_weight),
        "eta": float(eta),
        "update_fro": float(torch.linalg.vector_norm(update).item()),
        "target_fro": float(torch.linalg.vector_norm(desired_pos).item()),
        "fit_rmse": float(fit_rmse),
        "negative_rmse": float(negative_rmse),
        "mean_desired_scalar": float(desired_pos.mean().item()),
        "mean_abs_desired_scalar": float(desired_pos.abs().mean().item()),
    }


def calibrate_update_by_scalar_constraints(
    update: torch.Tensor,
    keys: torch.Tensor,
    gradients: torch.Tensor,
    desired: torch.Tensor,
    negative_keys: torch.Tensor | None,
    negative_gradients: torch.Tensor | None,
    *,
    negative_weight: float,
    max_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if keys.shape != gradients.shape:
        raise ValueError(f"keys and gradients must match, got {tuple(keys.shape)} and {tuple(gradients.shape)}")
    effect = keys.detach().float() @ update.detach().float().T
    predicted = torch.sum(gradients.detach().float() * effect, dim=1)
    desired_f = desired.detach().float()
    pred_all = [predicted]
    desired_all = [desired_f]
    if (
        negative_keys is not None
        and negative_gradients is not None
        and negative_keys.numel() > 0
        and negative_gradients.numel() > 0
    ):
        neg_effect = negative_keys.detach().float() @ update.detach().float().T
        neg_predicted = torch.sum(negative_gradients.detach().float() * neg_effect, dim=1)
        scale = negative_weight**0.5
        pred_all.append(scale * neg_predicted)
        desired_all.append(torch.zeros_like(neg_predicted))
    pred = torch.cat(pred_all, dim=0)
    target = torch.cat(desired_all, dim=0)
    denom = torch.dot(pred, pred).clamp_min(1e-12)
    raw_scale = torch.dot(pred, target) / denom
    scale_value = float(raw_scale.clamp(0.0, max_scale).item())
    calibrated = update * scale_value
    calibrated_pred = scale_value * predicted
    fit_rmse = torch.sqrt(torch.mean((calibrated_pred - desired_f).square())).item()
    return calibrated.contiguous(), {
        "logit_calibration_raw_scale": float(raw_scale.item()),
        "logit_calibration_scale": scale_value,
        "logit_calibration_max_scale": float(max_scale),
        "logit_calibration_predicted_mean": float(predicted.mean().item()),
        "logit_calibration_predicted_abs_mean": float(predicted.abs().mean().item()),
        "logit_calibration_desired_mean": float(desired_f.mean().item()),
        "logit_calibration_desired_abs_mean": float(desired_f.abs().mean().item()),
        "logit_calibration_fit_rmse": float(fit_rmse),
        "logit_calibration_uncalibrated_update_fro": float(torch.linalg.vector_norm(update).item()),
        "logit_calibration_update_fro": float(torch.linalg.vector_norm(calibrated).item()),
    }


def project_update_away_from(
    update: torch.Tensor,
    nuisance: torch.Tensor,
    *,
    strength: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    update_flat = update.detach().float().flatten()
    nuisance_flat = nuisance.detach().float().flatten()
    nuisance_norm = torch.linalg.vector_norm(nuisance_flat).clamp_min(1e-8)
    nuisance_unit = nuisance_flat / nuisance_norm
    projection = torch.dot(update_flat, nuisance_unit)
    adjusted_flat = update_flat - strength * projection * nuisance_unit
    adjusted = adjusted_flat.reshape_as(update).contiguous()
    update_norm = torch.linalg.vector_norm(update_flat).clamp_min(1e-8)
    adjusted_norm = torch.linalg.vector_norm(adjusted_flat)
    return adjusted, {
        "functional_orthogonalize_strength": float(strength),
        "functional_orthogonalize_update_fro": float(update_norm.item()),
        "functional_orthogonalize_nuisance_fro": float(nuisance_norm.item()),
        "functional_orthogonalize_projection_fro": float(abs(projection.item())),
        "functional_orthogonalize_cosine": float((projection / update_norm).item()),
        "functional_orthogonalize_adjusted_fro": float(adjusted_norm.item()),
    }


def teacher_kl_weighted_vector_update(
    keys: torch.Tensor,
    targets: torch.Tensor,
    gradients: torch.Tensor,
    negative_keys: torch.Tensor | None,
    ridge: float,
    negative_weight: float,
    eta: float,
    max_update_norm: float | None,
    threshold: float,
    temperature: float,
    floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if keys.shape != targets.shape or keys.shape != gradients.shape:
        raise ValueError(
            "teacher_kl_weighted_vector expects keys, targets, and gradients "
            f"to have matching shape, got {tuple(keys.shape)}, {tuple(targets.shape)}, {tuple(gradients.shape)}"
        )
    target = targets.detach().float()
    grad = gradients.detach().float()
    target_norm = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-8)
    grad_norm = torch.linalg.vector_norm(grad, dim=1).clamp_min(1e-8)
    kl_reducing_cosine = -torch.sum(target * grad, dim=1) / (target_norm * grad_norm)
    gate = floor + (1.0 - floor) * torch.sigmoid((kl_reducing_cosine - threshold) * temperature)
    scale = torch.sqrt(gate).unsqueeze(1)
    update, stats_obj = protected_ridge_update(
        keys.detach().float() * scale,
        target * scale,
        negative_keys=negative_keys,
        ridge=ridge,
        negative_weight=negative_weight,
        eta=eta,
        max_update_norm=max_update_norm,
    )
    stats = stats_obj.__dict__
    stats.update(
        {
            "teacher_kl_weight_threshold": float(threshold),
            "teacher_kl_weight_temperature": float(temperature),
            "teacher_kl_weight_floor": float(floor),
            "teacher_kl_weight_mean_gate": float(gate.mean().item()),
            "teacher_kl_weight_min_gate": float(gate.min().item()),
            "teacher_kl_weight_max_gate": float(gate.max().item()),
            "teacher_kl_weight_mean_reducing_cosine": float(kl_reducing_cosine.mean().item()),
            "teacher_kl_weight_min_reducing_cosine": float(kl_reducing_cosine.min().item()),
            "teacher_kl_weight_max_reducing_cosine": float(kl_reducing_cosine.max().item()),
        }
    )
    return update, stats


@contextmanager
def residual_operator_hook(
    model,
    layer_idx: int,
    update: torch.Tensor,
    key_site: str,
    gate_keys: torch.Tensor | None,
    gate_threshold: float,
    gate_temperature: float,
    final_token_only: bool,
) -> Iterator[None]:
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    layer = layers[resolved]
    update_cpu = update.detach().float().cpu()
    gate_keys_cpu = gate_keys.detach().float().cpu() if gate_keys is not None else None

    def hook(_module, inputs, output):
        hidden_out = hidden_from_output(output)
        if key_site == "block_input":
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("Expected decoder block input hidden states.")
            key = inputs[0]
        elif key_site == "block_output":
            key = hidden_out
        else:
            raise ValueError(f"Unknown key site: {key_site}")
        update_device = update_cpu.to(device=key.device, dtype=torch.float32)
        key_f = key.float()
        effect = F.linear(key_f, update_device)
        if gate_keys_cpu is not None and gate_keys_cpu.numel() > 0:
            keys = gate_keys_cpu.to(device=key.device, dtype=torch.float32)
            similarity = (
                F.normalize(key_f, dim=-1)
                @ F.normalize(keys, dim=-1).T
            ).amax(dim=-1, keepdim=True)
            gate = torch.sigmoid((similarity - gate_threshold) * gate_temperature)
            effect = effect * gate
        if final_token_only and effect.ndim == 3 and effect.shape[1] > 1:
            mask = torch.zeros_like(effect[..., :1])
            mask[:, -1:, :] = 1.0
            effect = effect * mask
        patched = hidden_out + effect.to(dtype=hidden_out.dtype)
        return replace_hidden(output, patched)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def add_eval_metrics(row: dict, prefix: str, result) -> None:
    row.update(result.to_dict(prefix))


def mean_max_cosine(rows: torch.Tensor, prototypes: torch.Tensor) -> float:
    if rows.numel() == 0 or prototypes.numel() == 0:
        return 0.0
    rows_n = F.normalize(rows.float(), dim=1)
    proto_n = F.normalize(prototypes.float(), dim=1)
    return float((rows_n @ proto_n.T).amax(dim=1).mean().item())


def repeated_answer_labels(questions: list, rows_per_question: int) -> torch.Tensor:
    labels = [1.0 if record.answer else -1.0 for record in questions]
    return torch.repeat_interleave(torch.tensor(labels, dtype=torch.float32), rows_per_question)


def add_labeled_alignment_metrics(row: dict, effect: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> None:
    labels = labels.to(dtype=torch.float32)
    for name, mask in (
        ("positive", labels > 0),
        ("negative", labels < 0),
    ):
        if int(mask.sum().item()) == 0:
            continue
        effect_rows = effect[mask]
        target_rows = target[mask]
        row[f"{name}_effect_target_cosine"] = mean_row_cosine(effect_rows, target_rows)
        row[f"{name}_effect_target_l2"] = mean_row_l2(effect_rows, target_rows)
        row[f"{name}_effect_target_norm_ratio"] = float(
            torch.linalg.vector_norm(effect_rows, dim=1).mean().item()
            / (torch.linalg.vector_norm(target_rows, dim=1).mean().item() + 1e-12)
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    updates_path = output_dir / "updates.jsonl"
    trigger_path = output_dir / "trigger_overlap.jsonl"
    for path in (metrics_path, updates_path, trigger_path):
        if path.exists():
            path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    if args.solve_mode in {
        "margin_gradient",
        "teacher_kl_gradient",
        "teacher_kl_weighted_vector",
        "teacher_logit_jacobian",
        "vector_logit_calibrated",
        "vector_kl_orthogonalized",
    }:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if args.paper_offset < 0:
        raise ValueError("--paper-offset must be non-negative.")
    domains_all, eval_sets_all = load_domain_rows(Path(args.domains_jsonl), args.paper_offset + args.papers)
    domains = domains_all[args.paper_offset :]
    eval_sets = eval_sets_all[args.paper_offset :]

    for local_paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        paper_idx = args.paper_offset + local_paper_idx
        started = time.time()
        if args.contrastive_trace_pairs:
            probes = make_minimal_pair_questions(
                domain,
                pair_count=max(1, args.trace_probes // 2),
                seed=args.seed * 500_000 + paper_idx,
            )
            if len(probes) < 2:
                raise ValueError("Could not generate enough contrastive trace probes.")
        else:
            probes = make_candidate_probes(domain, args.trace_probes, seed=args.seed * 500_000 + paper_idx)
        paper = domain.render_paper()
        null_doc = make_null_document(
            seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id)),
            approx_words=len(paper.split()),
        )
        full_prompts = prompts_for_questions(tokenizer, probes, paper, args.chat_template)
        null_prompts = prompts_for_questions(tokenizer, probes, null_doc, args.chat_template)
        key_prompts = prompts_for_questions(tokenizer, probes, None, args.chat_template)
        trace_info_gain = None
        if args.surprise_weighting in {"info_gain", "combined"}:
            trace_info_gain = repeat_prompt_scores(
                prompt_topk_kl_scores(
                    model,
                    tokenizer,
                    full_prompts,
                    null_prompts,
                    device,
                    args.max_length,
                    args.surprise_top_k,
                ),
                args.trace_last_tokens,
            )
        full_blocks = capture_block_io(
            model, tokenizer, full_prompts, args.layers, device, args.batch_size, args.max_length, args.trace_last_tokens
        )
        full_mlp_blocks = None
        if args.activation_energy_weighting in {"mlp_key", "combined"}:
            full_mlp_blocks = capture_layer_io(
                model,
                tokenizer,
                full_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
        null_blocks = capture_block_io(
            model, tokenizer, null_prompts, args.layers, device, args.batch_size, args.max_length, args.trace_last_tokens
        )
        key_blocks = capture_block_io(
            model, tokenizer, key_prompts, args.layers, device, args.batch_size, args.max_length, args.trace_last_tokens
        )
        auxiliary_full_blocks = None
        auxiliary_null_blocks = None
        auxiliary_key_blocks = None
        auxiliary_full_mlp_blocks = None
        auxiliary_info_gain = None
        auxiliary_prompts_count = 0
        if args.contrastive_auxiliary_pairs > 0:
            auxiliary_probes = make_minimal_pair_questions(
                domain,
                pair_count=args.contrastive_auxiliary_pairs,
                seed=args.seed * 700_000 + paper_idx + 17,
            )
            auxiliary_prompts_count = len(auxiliary_probes)
            auxiliary_full_prompts = prompts_for_questions(tokenizer, auxiliary_probes, paper, args.chat_template)
            auxiliary_null_prompts = prompts_for_questions(tokenizer, auxiliary_probes, null_doc, args.chat_template)
            auxiliary_key_prompts = prompts_for_questions(tokenizer, auxiliary_probes, None, args.chat_template)
            if args.surprise_weighting in {"info_gain", "combined"}:
                auxiliary_info_gain = repeat_prompt_scores(
                    prompt_topk_kl_scores(
                        model,
                        tokenizer,
                        auxiliary_full_prompts,
                        auxiliary_null_prompts,
                        device,
                        args.max_length,
                        args.surprise_top_k,
                    ),
                    args.trace_last_tokens,
                )
            auxiliary_full_blocks = capture_block_io(
                model,
                tokenizer,
                auxiliary_full_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
            if args.activation_energy_weighting in {"mlp_key", "combined"}:
                auxiliary_full_mlp_blocks = capture_layer_io(
                    model,
                    tokenizer,
                    auxiliary_full_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    args.trace_last_tokens,
                )
            auxiliary_null_blocks = capture_block_io(
                model,
                tokenizer,
                auxiliary_null_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
            auxiliary_key_blocks = capture_block_io(
                model,
                tokenizer,
                auxiliary_key_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
        guard_questions = negative_questions_for_domain(domain, args, paper_idx)
        guard_prompts = prompts_for_questions(tokenizer, guard_questions, None, args.chat_template)
        guard_blocks = capture_block_io(
            model,
            tokenizer,
            guard_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            args.eval_capture_last_tokens,
        )
        default_basis_rows_by_layer: dict[int, torch.Tensor] = {}
        default_basis_domains = choose_default_domains(
            domains,
            local_paper_idx,
            args.object_default_basis_papers,
        )
        if default_basis_domains:
            default_rows: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in args.layers}
            for default_idx, default_domain in enumerate(default_basis_domains):
                default_probes = make_candidate_probes(
                    default_domain,
                    args.object_default_basis_probes,
                    seed=args.seed * 900_000 + paper_idx * 1_000 + default_idx,
                )
                default_paper = default_domain.render_paper()
                default_null = make_null_document(
                    seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(default_domain.domain_id)),
                    approx_words=len(default_paper.split()),
                )
                default_full_prompts = prompts_for_questions(
                    tokenizer,
                    default_probes,
                    default_paper,
                    args.chat_template,
                )
                default_null_prompts = prompts_for_questions(
                    tokenizer,
                    default_probes,
                    default_null,
                    args.chat_template,
                )
                default_full_blocks = capture_block_io(
                    model,
                    tokenizer,
                    default_full_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    args.trace_last_tokens,
                )
                default_null_blocks = capture_block_io(
                    model,
                    tokenizer,
                    default_null_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    args.trace_last_tokens,
                )
                for layer_idx in args.layers:
                    default_rows[layer_idx].append(
                        target_rows_for_mode(
                            args.target_mode,
                            default_full_blocks[layer_idx],
                            default_null_blocks[layer_idx],
                        )
                    )
            for layer_idx, layer_rows in default_rows.items():
                if layer_rows:
                    basis_source = torch.cat(layer_rows, dim=0)
                    default_basis_rows_by_layer[layer_idx] = principal_components(
                        basis_source,
                        args.object_default_basis_components,
                    )

        updates: dict[int, torch.Tensor] = {}
        trace_keys_by_layer: dict[int, torch.Tensor] = {}
        targets_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx in args.layers:
            trace_keys = block_rows_for_site(key_blocks[layer_idx], args.key_site)
            guard_keys = block_rows_for_site(guard_blocks[layer_idx], args.key_site)
            targets = target_rows_for_mode(args.target_mode, full_blocks[layer_idx], null_blocks[layer_idx])
            layer_info_gain = trace_info_gain
            if args.contrastive_trace_pairs:
                targets = contrastive_targets(targets, scale=args.contrastive_target_scale)
            if auxiliary_key_blocks is not None and auxiliary_full_blocks is not None and auxiliary_null_blocks is not None:
                auxiliary_keys = block_rows_for_site(auxiliary_key_blocks[layer_idx], args.key_site)
                auxiliary_targets = contrastive_targets(
                    target_rows_for_mode(
                        args.target_mode,
                        auxiliary_full_blocks[layer_idx],
                        auxiliary_null_blocks[layer_idx],
                    ),
                    scale=args.contrastive_auxiliary_scale,
                )
                trace_keys = torch.cat([trace_keys, auxiliary_keys], dim=0)
                targets = torch.cat([targets, auxiliary_targets], dim=0)
                if layer_info_gain is not None and auxiliary_info_gain is not None:
                    layer_info_gain = torch.cat([layer_info_gain, auxiliary_info_gain], dim=0)
            full_mlp_layer = full_mlp_blocks[layer_idx] if full_mlp_blocks is not None else None
            energy_full_capture = full_blocks[layer_idx]
            if (
                auxiliary_full_blocks is not None
                and auxiliary_key_blocks is not None
                and auxiliary_null_blocks is not None
            ):
                class _CombinedCapture:
                    pass

                combined_capture = _CombinedCapture()
                combined_capture.inputs = torch.cat(
                    [full_blocks[layer_idx].inputs, auxiliary_full_blocks[layer_idx].inputs],
                    dim=0,
                )
                combined_capture.outputs = torch.cat(
                    [full_blocks[layer_idx].outputs, auxiliary_full_blocks[layer_idx].outputs],
                    dim=0,
                )
                energy_full_capture = combined_capture
                if full_mlp_blocks is not None and auxiliary_full_mlp_blocks is not None:
                    combined_mlp = _CombinedCapture()
                    combined_mlp.keys = torch.cat(
                        [full_mlp_blocks[layer_idx].keys, auxiliary_full_mlp_blocks[layer_idx].keys],
                        dim=0,
                    )
                    full_mlp_layer = combined_mlp
            default_basis = default_basis_rows_by_layer.get(layer_idx)
            object_default_stats = {
                "object_default_basis_papers": int(len(default_basis_domains)),
                "object_default_basis_components": int(default_basis.shape[0]) if default_basis is not None else 0,
                "object_default_projection_strength": float(args.object_default_projection_strength),
            }
            if default_basis is not None and default_basis.numel() > 0 and args.object_default_projection_strength > 0:
                object_default_stats["object_default_target_energy_removed"] = projection_energy_fraction(
                    targets,
                    default_basis,
                )
                targets = remove_basis_with_strength(
                    targets,
                    default_basis,
                    args.object_default_projection_strength,
                )
                object_default_stats["object_default_target_fro_after"] = float(
                    torch.linalg.vector_norm(targets.float()).item()
                )
            surprise_weights, surprise_stats = surprise_row_weights(
                args.surprise_weighting,
                trace_keys,
                targets,
                guard_keys,
                layer_info_gain,
                floor=args.surprise_weight_floor,
                temperature=args.surprise_weight_temperature,
            )
            energy_weights, energy_stats = activation_energy_row_weights(
                args.activation_energy_weighting,
                energy_full_capture,
                full_mlp_layer,
                floor=args.activation_energy_weight_floor,
                temperature=args.activation_energy_weight_temperature,
            )
            combined_weights = combine_row_weights(surprise_weights, energy_weights)
            solve_trace_keys, solve_targets = apply_positive_row_weights(trace_keys, targets, combined_weights)
            if args.solve_mode == "vector_ridge":
                update, stats_obj = protected_ridge_update(
                    solve_trace_keys,
                    solve_targets,
                    negative_keys=guard_keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
                stats = stats_obj.__dict__
                stats.update(object_default_stats)
                stats.update(surprise_stats)
                stats.update(energy_stats)
            elif args.solve_mode == "vector_logit_calibrated":
                update, stats_obj = protected_ridge_update(
                    solve_trace_keys,
                    solve_targets,
                    negative_keys=guard_keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
                stats = stats_obj.__dict__
                stats.update(object_default_stats)
                stats.update(surprise_stats)
                stats.update(energy_stats)
                gradients, desired, constraints_per_prompt = capture_teacher_logit_jacobian_constraints(
                    model,
                    tokenizer,
                    key_prompts,
                    full_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_logit_top_k,
                    args.teacher_logit_center,
                )
                guard_teacher_prompts = prompts_for_questions(
                    tokenizer,
                    guard_questions,
                    null_doc,
                    args.chat_template,
                )
                guard_gradients, _guard_desired, guard_constraints_per_prompt = (
                    capture_teacher_logit_jacobian_constraints(
                        model,
                        tokenizer,
                        guard_prompts,
                        guard_teacher_prompts,
                        layer_idx,
                        device,
                        args.max_length,
                        args.teacher_logit_top_k,
                        args.teacher_logit_center,
                    )
                )
                update, calibration_stats = calibrate_update_by_scalar_constraints(
                    update,
                    trace_keys.repeat_interleave(constraints_per_prompt, dim=0),
                    gradients,
                    desired,
                    guard_keys.repeat_interleave(guard_constraints_per_prompt, dim=0),
                    guard_gradients,
                    negative_weight=args.negative_weight,
                    max_scale=args.teacher_logit_calibration_max_scale,
                )
                stats.update(calibration_stats)
                stats["update_fro"] = float(torch.linalg.vector_norm(update).item())
                stats["teacher_logit_top_k"] = int(args.teacher_logit_top_k)
                stats["teacher_logit_center"] = bool(args.teacher_logit_center)
                stats["trace_prompt_rows"] = int(trace_keys.shape[0])
                stats["guard_prompt_rows"] = int(guard_keys.shape[0])
            elif args.solve_mode == "vector_kl_orthogonalized":
                update, stats_obj = protected_ridge_update(
                    solve_trace_keys,
                    solve_targets,
                    negative_keys=guard_keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
                stats = stats_obj.__dict__
                stats.update(object_default_stats)
                stats.update(surprise_stats)
                stats.update(energy_stats)
                gradients = capture_teacher_kl_gradients(
                    model,
                    tokenizer,
                    key_prompts,
                    full_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_kl_top_k,
                )
                guard_teacher_prompts = prompts_for_questions(
                    tokenizer,
                    guard_questions,
                    null_doc,
                    args.chat_template,
                )
                guard_gradients = capture_teacher_kl_gradients(
                    model,
                    tokenizer,
                    guard_prompts,
                    guard_teacher_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_kl_top_k,
                )
                nuisance_update, nuisance_stats = margin_gradient_update(
                    trace_keys,
                    targets,
                    gradients,
                    guard_keys,
                    guard_gradients,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.functional_orthogonalize_eta,
                    max_update_norm=args.max_update_norm,
                )
                update, orthogonalize_stats = project_update_away_from(
                    update,
                    nuisance_update,
                    strength=args.functional_orthogonalize_strength,
                )
                stats.update({f"nuisance_{key}": value for key, value in nuisance_stats.items()})
                stats.update(orthogonalize_stats)
                stats["update_fro"] = float(torch.linalg.vector_norm(update).item())
                stats["teacher_kl_top_k"] = int(args.teacher_kl_top_k)
            elif args.solve_mode == "margin_gradient":
                gradients = capture_margin_gradients(
                    model,
                    tokenizer,
                    key_prompts,
                    probes,
                    layer_idx,
                    device,
                    args.max_length,
                )
                guard_gradients = capture_margin_gradients(
                    model,
                    tokenizer,
                    guard_prompts,
                    guard_questions,
                    layer_idx,
                    device,
                    args.max_length,
                )
                if args.trace_last_tokens != 1:
                    raise ValueError("margin_gradient solve currently expects --trace-last-tokens 1.")
                if args.eval_capture_last_tokens != 1:
                    raise ValueError("margin_gradient solve currently expects --eval-capture-last-tokens 1.")
                update, stats = margin_gradient_update(
                    trace_keys,
                    targets,
                    gradients,
                    guard_keys,
                    guard_gradients,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
            elif args.solve_mode == "teacher_kl_gradient":
                gradients = capture_teacher_kl_gradients(
                    model,
                    tokenizer,
                    key_prompts,
                    full_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_kl_top_k,
                )
                guard_teacher_prompts = prompts_for_questions(
                    tokenizer,
                    guard_questions,
                    null_doc,
                    args.chat_template,
                )
                guard_gradients = capture_teacher_kl_gradients(
                    model,
                    tokenizer,
                    guard_prompts,
                    guard_teacher_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_kl_top_k,
                )
                if args.trace_last_tokens != 1:
                    raise ValueError("teacher_kl_gradient solve currently expects --trace-last-tokens 1.")
                if args.eval_capture_last_tokens != 1:
                    raise ValueError("teacher_kl_gradient solve currently expects --eval-capture-last-tokens 1.")
                update, stats = margin_gradient_update(
                    trace_keys,
                    targets,
                    gradients,
                    guard_keys,
                    guard_gradients,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
                stats["teacher_kl_top_k"] = int(args.teacher_kl_top_k)
            elif args.solve_mode == "teacher_kl_weighted_vector":
                gradients = capture_teacher_kl_gradients(
                    model,
                    tokenizer,
                    key_prompts,
                    full_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_kl_top_k,
                )
                if args.trace_last_tokens != 1:
                    raise ValueError("teacher_kl_weighted_vector solve currently expects --trace-last-tokens 1.")
                if args.eval_capture_last_tokens != 1:
                    raise ValueError("teacher_kl_weighted_vector solve currently expects --eval-capture-last-tokens 1.")
                update, stats = teacher_kl_weighted_vector_update(
                    trace_keys,
                    targets,
                    gradients,
                    guard_keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                    threshold=args.teacher_kl_weight_threshold,
                    temperature=args.teacher_kl_weight_temperature,
                    floor=args.teacher_kl_weight_floor,
                )
                stats["teacher_kl_top_k"] = int(args.teacher_kl_top_k)
            elif args.solve_mode == "teacher_logit_jacobian":
                gradients, desired, constraints_per_prompt = capture_teacher_logit_jacobian_constraints(
                    model,
                    tokenizer,
                    key_prompts,
                    full_prompts,
                    layer_idx,
                    device,
                    args.max_length,
                    args.teacher_logit_top_k,
                    args.teacher_logit_center,
                )
                guard_teacher_prompts = prompts_for_questions(
                    tokenizer,
                    guard_questions,
                    null_doc,
                    args.chat_template,
                )
                guard_gradients, _guard_desired, guard_constraints_per_prompt = (
                    capture_teacher_logit_jacobian_constraints(
                        model,
                        tokenizer,
                        guard_prompts,
                        guard_teacher_prompts,
                        layer_idx,
                        device,
                        args.max_length,
                        args.teacher_logit_top_k,
                        args.teacher_logit_center,
                    )
                )
                if args.trace_last_tokens != 1:
                    raise ValueError("teacher_logit_jacobian solve currently expects --trace-last-tokens 1.")
                if args.eval_capture_last_tokens != 1:
                    raise ValueError("teacher_logit_jacobian solve currently expects --eval-capture-last-tokens 1.")
                update, stats = scalar_gradient_update(
                    trace_keys.repeat_interleave(constraints_per_prompt, dim=0),
                    gradients,
                    desired,
                    guard_keys.repeat_interleave(guard_constraints_per_prompt, dim=0),
                    guard_gradients,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
                stats["teacher_logit_top_k"] = int(args.teacher_logit_top_k)
                stats["teacher_logit_center"] = bool(args.teacher_logit_center)
                stats["trace_prompt_rows"] = int(trace_keys.shape[0])
                stats["guard_prompt_rows"] = int(guard_keys.shape[0])
            else:
                raise ValueError(f"Unknown solve mode: {args.solve_mode}")
            updates[layer_idx] = update
            trace_keys_by_layer[layer_idx] = trace_keys
            targets_by_layer[layer_idx] = targets
            row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "layer": layer_idx,
                "target_mode": args.target_mode,
                "key_site": args.key_site,
                "solve_mode": args.solve_mode,
                "trace_rows": int(trace_keys.shape[0]),
                "guard_rows": int(guard_keys.shape[0]),
                "operator_gate": args.operator_gate,
                "operator_final_token_only": args.operator_final_token_only,
                "contrastive_trace_pairs": args.contrastive_trace_pairs,
                "contrastive_target_scale": args.contrastive_target_scale,
                "contrastive_auxiliary_pairs": args.contrastive_auxiliary_pairs,
                "contrastive_auxiliary_prompts": auxiliary_prompts_count,
                "contrastive_auxiliary_scale": args.contrastive_auxiliary_scale,
                "seconds": time.time() - started,
            }
            row.update(stats)
            append_jsonl(updates_path, row)

        gauntlets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": heldout_questions, **gauntlets}

        for group, questions in eval_groups.items():
            baseline = evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            context = evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=paper,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            hook_stack = []
            try:
                for layer_idx in args.layers:
                    hook_stack.append(
                        residual_operator_hook(
                            model,
                            layer_idx,
                            updates[layer_idx],
                            args.key_site,
                            trace_keys_by_layer[layer_idx] if args.operator_gate else None,
                            args.operator_gate_threshold,
                            args.operator_gate_temperature,
                            args.operator_final_token_only,
                        )
                    )
                    hook_stack[-1].__enter__()
                edited = evaluate_yes_no(
                    model,
                    tokenizer,
                    questions,
                    device,
                    paper=None,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
            finally:
                while hook_stack:
                    hook_stack.pop().__exit__(None, None, None)

            row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "group": group,
                "question_count": len(questions),
                "target_mode": args.target_mode,
                "key_site": args.key_site,
                "solve_mode": args.solve_mode,
                "operator_gate": args.operator_gate,
                "operator_final_token_only": args.operator_final_token_only,
                "seconds": time.time() - started,
            }
            add_eval_metrics(row, "baseline", baseline)
            add_eval_metrics(row, "context", context)
            add_eval_metrics(row, "edited", edited)
            row["accuracy_delta"] = edited.accuracy - baseline.accuracy
            row["internalization_ratio"] = (
                (edited.accuracy - baseline.accuracy)
                / (context.accuracy - baseline.accuracy + 1e-12)
            )
            append_jsonl(metrics_path, row)

            eval_prompts = prompts_for_questions(tokenizer, questions, None, args.chat_template)
            full_eval = capture_block_io(
                model,
                tokenizer,
                prompts_for_questions(tokenizer, questions, paper, args.chat_template),
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.eval_capture_last_tokens,
            )
            null_eval = capture_block_io(
                model,
                tokenizer,
                prompts_for_questions(tokenizer, questions, null_doc, args.chat_template),
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.eval_capture_last_tokens,
            )
            key_eval = capture_block_io(
                model,
                tokenizer,
                eval_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                args.eval_capture_last_tokens,
            )
            for layer_idx in args.layers:
                eval_keys = block_rows_for_site(key_eval[layer_idx], args.key_site)
                target = target_rows_for_mode(args.target_mode, full_eval[layer_idx], null_eval[layer_idx])
                effect = eval_keys.float() @ updates[layer_idx].float().T
                trigger_row = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "group": group,
                    "layer": layer_idx,
                    "target_mode": args.target_mode,
                    "key_site": args.key_site,
                    "solve_mode": args.solve_mode,
                    "trace_rows": int(trace_keys_by_layer[layer_idx].shape[0]),
                    "eval_rows": int(eval_keys.shape[0]),
                    "mean_max_cosine_to_trace_keys": mean_max_cosine(eval_keys, trace_keys_by_layer[layer_idx]),
                    "effect_target_cosine": mean_row_cosine(effect, target),
                    "effect_target_l2": mean_row_l2(effect, target),
                    "effect_target_norm_ratio": float(
                        torch.linalg.vector_norm(effect, dim=1).mean().item()
                        / (torch.linalg.vector_norm(target, dim=1).mean().item() + 1e-12)
                    ),
                    "seconds": time.time() - started,
                }
                labels = repeated_answer_labels(questions, args.eval_capture_last_tokens)
                if labels.shape[0] == effect.shape[0]:
                    add_labeled_alignment_metrics(trigger_row, effect, target, labels)
                append_jsonl(trigger_path, trigger_row)

    print(f"Wrote residual operator metrics to {metrics_path}")
    print(f"Wrote residual operator update stats to {updates_path}")
    print(f"Wrote residual operator trigger metrics to {trigger_path}")


if __name__ == "__main__":
    main()
