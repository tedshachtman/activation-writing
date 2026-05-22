"""CLI runner for Causal Activation-Imprint Consolidation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .synthetic import (
    DomainSpec,
    domain_from_dict,
    format_prompt,
    generate_domain,
    generate_domains,
    general_guard_questions,
    make_candidate_probes,
    make_eval_questions,
    make_gauntlet_questions,
    make_inverse_questions,
    make_minimal_pair_questions,
    make_null_document,
    make_near_collision_questions,
    negative_guard_prompts,
    question_from_dict,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for dependency-free dry runs
    def tqdm(iterable, **_kwargs):
        return iterable


def load_runtime_modules() -> None:
    """Import torch-dependent modules only when a real model run starts."""

    global torch
    global TrainConfig, prepare_qa_lora_model, train_naive_text_baseline, train_qa_lora_baseline
    global categorical_kl, causal_patch_weights, evaluate_yes_no, evaluate_yes_no_details
    global evaluate_yes_no_with_bias, fit_scalar_yes_bias, format_model_prompt, format_question_prompt
    global internalization_ratio, yes_no_distributions
    global PlasticityState, RLSConfig, select_d_optimal
    global ForwardCounter, capture_layer_io, clear_active_slot_weights, get_decoder_layers
    global install_additive_memory, load_model_and_tokenizer, memory_norms

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "CAIC model runs require PyTorch and the project dependencies. "
            "Install them with: pip install -e \".[dev]\""
        ) from exc

    from .baselines import (
        TrainConfig,
        prepare_qa_lora_model,
        train_naive_text_baseline,
        train_qa_lora_baseline,
    )
    from .evaluation import (
        categorical_kl,
        causal_patch_weights,
        evaluate_yes_no,
        evaluate_yes_no_details,
        evaluate_yes_no_with_bias,
        fit_scalar_yes_bias,
        format_model_prompt,
        format_question_prompt,
        internalization_ratio,
        yes_no_distributions,
    )
    from .memory import PlasticityState, RLSConfig, select_d_optimal
    from .modeling import (
        ForwardCounter,
        capture_layer_io,
        clear_active_slot_weights,
        get_decoder_layers,
        install_additive_memory,
        load_model_and_tokenizer,
        memory_norms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CAIC context-to-weight experiment.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--papers", type=int, default=1)
    parser.add_argument("--domains-jsonl", default=None)
    parser.add_argument("--domain-difficulty", choices=["easy", "medium", "standard"], default="easy")
    parser.add_argument("--candidate-probes", type=int, default=64)
    parser.add_argument("--candidate-inverse-probes", type=int, default=0)
    parser.add_argument("--candidate-minimal-pair-probes", type=int, default=0)
    parser.add_argument("--candidate-near-collision-probes", type=int, default=0)
    parser.add_argument("--write-probes", type=int, default=20)
    parser.add_argument("--eval-questions", type=int, default=30)
    parser.add_argument("--layers", nargs="+", type=int, default=[8])
    parser.add_argument("--output", default="runs/caic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--capture-last-tokens", type=int, default=1)
    parser.add_argument("--write-token-selection", choices=["suffix", "content", "final"], default="suffix")
    parser.add_argument("--gate-token-selection", choices=["same", "suffix", "content", "final"], default="same")
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument("--max-update-norm", type=float, default=5.0)
    parser.add_argument("--rls-device", default="cpu")
    parser.add_argument("--negative-guards", type=int, default=8)
    parser.add_argument("--guard-weight", type=float, default=0.25)
    parser.add_argument("--sentinel-prompts", type=int, default=8)
    parser.add_argument("--sentinel-kl-threshold", type=float, default=0.15)
    parser.add_argument("--sentinel-retries", type=int, default=4)
    parser.add_argument("--causal-filter", dest="causal_filter", action="store_true", default=True)
    parser.add_argument("--no-causal-filter", dest="causal_filter", action="store_false")
    parser.add_argument("--background-prompts", type=int, default=8)
    parser.add_argument("--baselines", nargs="*", default=[], choices=["naive_text", "qa_lora"])
    parser.add_argument("--baseline-steps", type=int, default=20)
    parser.add_argument("--baseline-lr", type=float, default=5e-5)
    parser.add_argument("--qa-lora-r", type=int, default=16)
    parser.add_argument("--teacher-gate", dest="teacher_gate", action="store_true", default=True)
    parser.add_argument("--no-teacher-gate", dest="teacher_gate", action="store_false")
    parser.add_argument("--teacher-search-budget", type=int, default=50)
    parser.add_argument("--teacher-min-accuracy", type=float, default=0.65)
    parser.add_argument("--teacher-min-delta", type=float, default=0.10)
    parser.add_argument("--memory-gate", dest="memory_gate", action="store_true", default=True)
    parser.add_argument("--no-memory-gate", dest="memory_gate", action="store_false")
    parser.add_argument("--memory-gate-threshold", type=float, default=0.95)
    parser.add_argument("--memory-gate-temperature", type=float, default=80.0)
    parser.add_argument("--slot-memory", action="store_true")
    parser.add_argument("--independent-slot-plasticity", action="store_true")
    parser.add_argument("--activation-slot-routing", action="store_true")
    parser.add_argument("--domain-latch-diagnostics", action="store_true")
    parser.add_argument("--domain-latch-layer", type=int, default=8)
    parser.add_argument("--domain-latch-probes", type=int, default=8)
    parser.add_argument("--domain-latch-threshold", type=float, default=0.0)
    parser.add_argument("--domain-latch-margin", type=float, default=0.02)
    parser.add_argument("--safe-write-search", dest="safe_write_search", action="store_true", default=True)
    parser.add_argument("--no-safe-write-search", dest="safe_write_search", action="store_false")
    parser.add_argument("--validation-probes", type=int, default=16)
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--validation-min-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--validation-min-fold-delta", type=float, default=-0.001)
    parser.add_argument("--validation-min-positive-delta", type=float, default=-1.0)
    parser.add_argument("--validation-min-negative-delta", type=float, default=-1.0)
    parser.add_argument("--search-etas", nargs="+", type=float, default=[3.0, 10.0])
    parser.add_argument("--search-max-update-norms", nargs="+", type=float, default=[200.0, 500.0])
    parser.add_argument("--search-gate-thresholds", nargs="+", type=float, default=[0.95])
    parser.add_argument("--positive-label-weight", type=float, default=1.0)
    parser.add_argument("--negative-label-weight", type=float, default=1.0)
    parser.add_argument("--balanced-write-selection", action="store_true")
    parser.add_argument("--write-selection-positive-fraction", type=float, default=0.5)
    parser.add_argument(
        "--target-mode",
        choices=["doc_null", "answer_direction", "validity_probe"],
        default="doc_null",
    )
    parser.add_argument("--answer-target-scale", type=float, default=1.0)
    parser.add_argument("--validity-target-margin", type=float, default=2.0)
    parser.add_argument("--project-answer-direction", action="store_true")
    parser.add_argument("--gauntlet", action="store_true")
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--gauntlet-validation", action="store_true")
    parser.add_argument("--gauntlet-validation-questions", type=int, default=20)
    parser.add_argument("--gauntlet-min-bucket-delta", type=float, default=-1.0)
    parser.add_argument("--bias-rival-baseline", action="store_true")
    parser.add_argument("--allow-noop-write", action="store_true")
    parser.add_argument("--diagnostics", dest="diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="diagnostics", action="store_false")
    parser.add_argument("--dry-run-synthetic", action="store_true")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_domain_rows(path: Path, count: int) -> tuple[list[DomainSpec], list[list]]:
    domains: list[DomainSpec] = []
    eval_sets: list[list] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            domains.append(domain_from_dict(row["domain"]))
            eval_sets.append([question_from_dict(item) for item in row.get("eval_questions", [])])
            if len(domains) >= count:
                break
    if len(domains) < count:
        raise RuntimeError(f"Loaded {len(domains)}/{count} domains from {path}.")
    return domains, eval_sets


def build_candidate_pool(domain: DomainSpec, args: argparse.Namespace, paper_idx: int) -> list:
    seed_base = args.seed * 100_000 + paper_idx
    candidates = make_candidate_probes(domain, args.candidate_probes, seed=seed_base)
    if args.candidate_inverse_probes > 0:
        candidates.extend(
            make_inverse_questions(
                domain,
                args.candidate_inverse_probes,
                seed=seed_base + 20_000,
            )
        )
    if args.candidate_minimal_pair_probes > 0:
        pair_count = (args.candidate_minimal_pair_probes + 1) // 2
        candidates.extend(
            make_minimal_pair_questions(
                domain,
                pair_count,
                seed=seed_base + 30_000,
            )[: args.candidate_minimal_pair_probes]
        )
    if args.candidate_near_collision_probes > 0:
        candidates.extend(
            make_near_collision_questions(
                domain,
                args.candidate_near_collision_probes,
                seed=seed_base + 40_000,
            )
        )
    return candidates


def question_category_counts(questions: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in questions:
        counts[record.category] = counts.get(record.category, 0) + 1
    return counts


def select_write_questions(
    question_keys: torch.Tensor,
    questions: list,
    k: int,
    weights: torch.Tensor,
    ridge: float,
    balanced: bool,
    positive_fraction: float,
) -> list[int]:
    try:
        selector = select_d_optimal
    except NameError:  # available during tests/dry imports before load_runtime_modules()
        from .memory import select_d_optimal as selector

    if not balanced:
        return selector(question_keys, k, weights=weights, ridge=ridge)

    positive_rows = [idx for idx, record in enumerate(questions) if record.answer]
    negative_rows = [idx for idx, record in enumerate(questions) if not record.answer]
    if not positive_rows or not negative_rows:
        return selector(question_keys, k, weights=weights, ridge=ridge)

    positive_fraction = min(1.0, max(0.0, positive_fraction))
    target_positive = int(round(k * positive_fraction))
    target_positive = min(len(positive_rows), max(1, target_positive))
    target_negative = min(len(negative_rows), max(1, k - target_positive))

    while target_positive + target_negative < min(k, len(positive_rows) + len(negative_rows)):
        if len(positive_rows) - target_positive >= len(negative_rows) - target_negative:
            target_positive += int(target_positive < len(positive_rows))
        else:
            target_negative += int(target_negative < len(negative_rows))
        if target_positive >= len(positive_rows) and target_negative >= len(negative_rows):
            break

    selected: list[int] = []
    if target_positive > 0:
        local = selector(
            question_keys[positive_rows],
            target_positive,
            weights=weights[positive_rows],
            ridge=ridge,
        )
        selected.extend(positive_rows[idx] for idx in local)
    if target_negative > 0:
        local = selector(
            question_keys[negative_rows],
            target_negative,
            weights=weights[negative_rows],
            ridge=ridge,
        )
        selected.extend(negative_rows[idx] for idx in local)
    return selected[:k]


def capture_three_passes(
    model: Any,
    tokenizer: Any,
    domain: DomainSpec,
    questions: list,
    layer_indices: list[int],
    device: torch.device,
    batch_size: int,
    max_length: int,
    use_chat_template: bool,
    capture_last_tokens: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    paper = domain.render_paper()
    null_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id))
    null_doc = make_null_document(seed=null_seed, approx_words=len(paper.split()))
    doc_prompts = [
        format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        for record in questions
    ]
    null_prompts = [
        format_question_prompt(tokenizer, record.question, paper=null_doc, use_chat_template=use_chat_template)
        for record in questions
    ]
    student_prompts = [
        format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=use_chat_template)
        for record in questions
    ]

    doc = capture_layer_io(
        model,
        tokenizer,
        doc_prompts,
        layer_indices,
        device,
        batch_size,
        max_length,
        capture_last_tokens=capture_last_tokens,
    )
    null = capture_layer_io(
        model,
        tokenizer,
        null_prompts,
        layer_indices,
        device,
        batch_size,
        max_length,
        capture_last_tokens=capture_last_tokens,
    )
    student = capture_layer_io(
        model,
        tokenizer,
        student_prompts,
        layer_indices,
        device,
        batch_size,
        max_length,
        capture_last_tokens=capture_last_tokens,
    )

    keys = {idx: student[idx].keys for idx in layer_indices}
    student_outputs = {idx: student[idx].outputs for idx in layer_indices}
    content_deltas = {idx: doc[idx].outputs - null[idx].outputs for idx in layer_indices}
    return keys, student_outputs, content_deltas


def final_token_rows(question_count: int, capture_last_tokens: int) -> list[int]:
    return [idx * capture_last_tokens + capture_last_tokens - 1 for idx in range(question_count)]


def suffix_token_rows(selected_questions: list[int], capture_last_tokens: int) -> list[int]:
    rows: list[int] = []
    for question_idx in selected_questions:
        start = question_idx * capture_last_tokens
        rows.extend(range(start, start + capture_last_tokens))
    return rows


def content_token_rows(
    tokenizer: Any,
    domain: DomainSpec,
    questions: list,
    capture_last_tokens: int,
    max_length: int,
    use_chat_template: bool,
) -> list[list[int]]:
    terms = [domain.title, *domain.operators, *domain.marks]
    rows_by_question: list[list[int]] = []
    for question_idx, record in enumerate(questions):
        prompt = format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=use_chat_template)
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = encoded["offset_mapping"]
        token_count = len(offsets)
        suffix_start_token = max(0, token_count - capture_last_tokens)
        suffix_start_row = question_idx * capture_last_tokens + max(0, capture_last_tokens - token_count)

        spans: list[tuple[int, int]] = []
        lowered = prompt.lower()
        for term in terms:
            term_lower = term.lower()
            start = 0
            while True:
                hit = lowered.find(term_lower, start)
                if hit < 0:
                    break
                spans.append((hit, hit + len(term_lower)))
                start = hit + len(term_lower)

        rows: list[int] = []
        for token_idx in range(suffix_start_token, token_count):
            token_start, token_end = offsets[token_idx]
            if token_end <= token_start:
                continue
            if any(token_start < span_end and token_end > span_start for span_start, span_end in spans):
                rows.append(suffix_start_row + (token_idx - suffix_start_token))
        if not rows:
            rows.append(question_idx * capture_last_tokens + capture_last_tokens - 1)
        rows_by_question.append(rows)
    return rows_by_question


def selected_rows(
    selected_questions: list[int],
    capture_last_tokens: int,
    mode: str,
    content_rows_by_question: list[list[int]],
) -> list[int]:
    if mode == "suffix":
        return suffix_token_rows(selected_questions, capture_last_tokens)
    if mode == "final":
        return [idx * capture_last_tokens + capture_last_tokens - 1 for idx in selected_questions]
    if mode == "content":
        rows: list[int] = []
        for question_idx in selected_questions:
            rows.extend(content_rows_by_question[question_idx])
        return sorted(set(rows))
    raise ValueError(f"Unknown token selection mode: {mode}")


def build_write_matrices(
    keys: dict[int, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    weights: dict[int, torch.Tensor],
    selected_questions: list[int],
    positive_rows: list[int],
    question_weight_multipliers: list[float],
    guard_keys: dict[int, torch.Tensor] | None,
    guard_weight: float,
    capture_last_tokens: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    a_by_layer: dict[int, torch.Tensor] = {}
    d_by_layer: dict[int, torch.Tensor] = {}
    w_by_layer: dict[int, torch.Tensor] = {}
    row_weights = []
    selected_set = set(selected_questions)
    for row_idx in positive_rows:
        question_idx = row_idx // capture_last_tokens
        if question_idx not in selected_set:
            raise ValueError(f"Positive row {row_idx} maps to unselected question {question_idx}.")
        row_weights.append((question_idx, row_idx))
    for layer_idx in keys:
        pos_keys = keys[layer_idx][positive_rows].T.contiguous()
        pos_deltas = deltas[layer_idx][positive_rows].T.contiguous()
        pos_weights = torch.tensor(
            [
                float(weights[layer_idx][question_idx]) * question_weight_multipliers[question_idx]
                for question_idx, _row_idx in row_weights
            ],
            dtype=torch.float32,
        )
        if guard_keys is not None and guard_keys[layer_idx].numel() > 0:
            neg_keys = guard_keys[layer_idx].T.contiguous()
            neg_deltas = torch.zeros(pos_deltas.shape[0], neg_keys.shape[1])
            neg_weights = torch.full((neg_keys.shape[1],), guard_weight)
            a_by_layer[layer_idx] = torch.cat([pos_keys, neg_keys], dim=1)
            d_by_layer[layer_idx] = torch.cat([pos_deltas, neg_deltas], dim=1)
            w_by_layer[layer_idx] = torch.cat([pos_weights, neg_weights], dim=0)
        else:
            a_by_layer[layer_idx] = pos_keys
            d_by_layer[layer_idx] = pos_deltas
            w_by_layer[layer_idx] = pos_weights
    return a_by_layer, d_by_layer, w_by_layer


def project_rows_away_from_direction(rows: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    import torch

    if rows.numel() == 0:
        return rows
    unit = torch.nn.functional.normalize(direction.detach().float().flatten(), dim=0)
    return rows - (rows.float() @ unit).unsqueeze(1) * unit.unsqueeze(0)


def answer_unembedding_direction(model: Any, tokenizer: Any) -> torch.Tensor:
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise ValueError("Tokenizer returned empty Yes/No completions.")
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise AttributeError("Model has no lm_head.weight for answer-direction projection.")
    weight = lm_head.weight.detach().float().cpu()
    return weight[yes_ids[0]] - weight[no_ids[0]]


def project_deltas_away_from_answer_direction(
    deltas: dict[int, torch.Tensor],
    model: Any,
    tokenizer: Any,
) -> dict[int, torch.Tensor]:
    direction = answer_unembedding_direction(model, tokenizer)
    return {
        layer_idx: project_rows_away_from_direction(layer_deltas, direction)
        for layer_idx, layer_deltas in deltas.items()
    }


def answer_direction_targets_like(
    deltas: dict[int, torch.Tensor],
    questions: list,
    capture_last_tokens: int,
    model: Any,
    tokenizer: Any,
    scale: float,
) -> dict[int, torch.Tensor]:
    import torch

    if not questions:
        return {layer_idx: torch.zeros_like(layer_deltas) for layer_idx, layer_deltas in deltas.items()}
    direction = answer_unembedding_direction(model, tokenizer)
    unit = torch.nn.functional.normalize(direction.detach().float().flatten(), dim=0)
    rows: list[torch.Tensor] = []
    for record in questions:
        sign = 1.0 if record.answer else -1.0
        rows.extend([sign * scale * unit for _ in range(capture_last_tokens)])
    target = torch.stack(rows, dim=0)
    targets: dict[int, torch.Tensor] = {}
    for layer_idx, layer_deltas in deltas.items():
        if layer_deltas.shape != target.shape:
            raise ValueError(
                f"Answer-direction target shape {tuple(target.shape)} does not match "
                f"layer {layer_idx} deltas {tuple(layer_deltas.shape)}."
            )
        targets[layer_idx] = target.to(dtype=layer_deltas.dtype)
    return targets


def fit_linear_probe_direction(
    features: torch.Tensor,
    labels: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, float]:
    import torch

    x = features.detach().float().cpu()
    y = labels.detach().float().cpu()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got {tuple(x.shape)}")
    if y.ndim != 1 or y.numel() != x.shape[0]:
        raise ValueError(f"Expected labels [{x.shape[0]}], got {tuple(y.shape)}")
    x_aug = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    system = x_aug @ x_aug.T + ridge * torch.eye(x_aug.shape[0])
    alpha = torch.linalg.pinv(system) @ y
    weights = x_aug.T @ alpha
    return weights[:-1].contiguous(), float(weights[-1].item())


def validity_probe_targets_like(
    deltas: dict[int, torch.Tensor],
    student_outputs: dict[int, torch.Tensor],
    questions: list,
    domain: DomainSpec,
    capture_last_tokens: int,
    ridge: float,
    margin: float,
) -> dict[int, torch.Tensor]:
    import torch

    labels = []
    for record in questions:
        valid, _failures = domain.validate(record.chain)
        labels.append(1.0 if valid else -1.0)
    question_labels = torch.tensor(labels, dtype=torch.float32)
    row_labels = torch.repeat_interleave(question_labels, capture_last_tokens)
    final_rows = final_token_rows(len(questions), capture_last_tokens)

    targets: dict[int, torch.Tensor] = {}
    for layer_idx, layer_deltas in deltas.items():
        outputs = student_outputs[layer_idx].detach().float().cpu()
        probe, bias = fit_linear_probe_direction(outputs[final_rows], question_labels, ridge)
        norm_sq = float(torch.dot(probe, probe).item())
        if norm_sq <= 1e-12:
            targets[layer_idx] = torch.zeros_like(layer_deltas)
            continue
        scores = outputs @ probe + bias
        signed_scores = scores * row_labels
        gaps = torch.clamp(margin - signed_scores, min=0.0)
        target = (gaps * row_labels / (norm_sq + 1e-12)).unsqueeze(1) * probe.unsqueeze(0)
        if target.shape != layer_deltas.shape:
            raise ValueError(
                f"Validity-probe target shape {tuple(target.shape)} does not match "
                f"layer {layer_idx} deltas {tuple(layer_deltas.shape)}."
            )
        targets[layer_idx] = target.to(dtype=layer_deltas.dtype)
    return targets


def propose_updates(
    states: dict[int, PlasticityState],
    wrappers: dict,
    a_by_layer: dict[int, torch.Tensor],
    d_by_layer: dict[int, torch.Tensor],
    w_by_layer: dict[int, torch.Tensor],
    eta: float,
    max_update_norm: float | None = None,
    slot_id: int | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    deltas: dict[int, torch.Tensor] = {}
    next_ps: dict[int, torch.Tensor] = {}
    for layer_idx, state in states.items():
        old_eta = state.config.eta
        old_max_update_norm = state.config.max_update_norm
        state.config.eta = eta
        if max_update_norm is not None:
            state.config.max_update_norm = max_update_norm
        delta, next_p = state.propose(
            wrappers[layer_idx].memory_for_slot(slot_id).detach(),
            a_by_layer[layer_idx],
            d_by_layer[layer_idx],
            w_by_layer[layer_idx],
        )
        state.config.eta = old_eta
        state.config.max_update_norm = old_max_update_norm
        deltas[layer_idx] = delta
        next_ps[layer_idx] = next_p
    return deltas, next_ps


def apply_memory_deltas(
    wrappers: dict,
    deltas: dict[int, torch.Tensor],
    sign: float = 1.0,
    slot_id: int | None = None,
) -> None:
    with torch.no_grad():
        for layer_idx, delta in deltas.items():
            wrappers[layer_idx].add_memory_(sign * delta, slot_id=slot_id)


def snapshot_wrappers(wrappers: dict) -> dict[int, dict[str, Any]]:
    return {
        layer_idx: {
            "memory": wrapper.memory.detach().clone(),
            "gate_keys": wrapper.gate_keys.detach().clone(),
            "gate_threshold": wrapper.gate_threshold,
            "gate_temperature": wrapper.gate_temperature,
            "object_gate_keys": wrapper.object_gate_keys.detach().clone(),
            "object_gate_threshold": wrapper.object_gate_threshold,
            "object_gate_temperature": wrapper.object_gate_temperature,
            "object_gate_floor": wrapper.object_gate_floor,
            "object_density_gates": list(wrapper.object_density_gates),
            "slot_memories": [slot.detach().clone() for slot in wrapper.slot_memories],
            "slot_gate_keys": [keys.detach().clone() for keys in wrapper.slot_gate_keys],
            "slot_terms": list(wrapper.slot_terms),
        }
        for layer_idx, wrapper in wrappers.items()
    }


def restore_wrappers(wrappers: dict, snapshot: dict[int, dict[str, Any]]) -> None:
    with torch.no_grad():
        for layer_idx, state in snapshot.items():
            wrapper = wrappers[layer_idx]
            wrapper.copy_memory_(state["memory"])
            wrapper.gate_keys = state["gate_keys"].to(
                device=wrapper.memory.device,
                dtype=wrapper.memory.dtype,
            ).contiguous()
            wrapper.gate_threshold = state["gate_threshold"]
            wrapper.gate_temperature = state["gate_temperature"]
            wrapper.object_gate_keys = state.get(
                "object_gate_keys",
                torch.empty(0, wrapper.in_features),
            ).to(
                device=wrapper.memory.device,
                dtype=wrapper.memory.dtype,
            ).contiguous()
            wrapper.object_gate_threshold = state.get("object_gate_threshold", 0.90)
            wrapper.object_gate_temperature = state.get("object_gate_temperature", 40.0)
            wrapper.object_gate_floor = state.get("object_gate_floor", 0.0)
            wrapper.object_density_gates = list(state.get("object_density_gates", []))
            wrapper.slot_memories = [
                slot.to(device=wrapper.memory.device, dtype=wrapper.memory.dtype).contiguous()
                for slot in state.get("slot_memories", [])
            ]
            wrapper.slot_gate_keys = [
                keys.to(device=wrapper.memory.device, dtype=wrapper.memory.dtype).contiguous()
                for keys in state.get("slot_gate_keys", [])
            ]
            wrapper.slot_terms = list(state.get("slot_terms", []))
            wrapper.set_active_slot_weights_(None)


def clear_slot_memory(wrappers: dict, slot_id: int) -> None:
    with torch.no_grad():
        for wrapper in wrappers.values():
            if slot_id >= len(wrapper.slot_memories):
                continue
            wrapper.slot_memories[slot_id].zero_()
            wrapper.slot_gate_keys[slot_id] = torch.empty(
                0,
                wrapper.in_features,
                device=wrapper.memory.device,
                dtype=wrapper.memory.dtype,
            )


def configure_current_gate(
    wrappers: dict,
    keys: dict[int, torch.Tensor],
    selected_rows: list[int],
    snapshot: dict[int, dict[str, Any]],
    threshold: float,
    temperature: float,
    slot_id: int | None = None,
) -> None:
    for layer_idx, wrapper in wrappers.items():
        if slot_id is not None:
            wrapper.set_slot_gate_keys_(
                slot_id,
                keys[layer_idx][selected_rows],
                threshold=threshold,
                temperature=temperature,
            )
            continue
        old_keys = snapshot[layer_idx]["gate_keys"]
        new_keys = keys[layer_idx][selected_rows]
        if old_keys.numel() > 0:
            combined = torch.cat([old_keys.cpu(), new_keys.cpu()], dim=0)
        else:
            combined = new_keys
        wrapper.set_gate_keys_(combined, threshold=threshold, temperature=temperature, append=False)


def apply_with_sentinel_retry(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    wrappers: dict,
    states: dict[int, PlasticityState],
    a_by_layer: dict[int, torch.Tensor],
    d_by_layer: dict[int, torch.Tensor],
    w_by_layer: dict[int, torch.Tensor],
    sentinel_prompts: list[str],
    eta: float,
    threshold: float,
    retries: int,
    max_length: int,
    use_chat_template: bool,
    slot_id: int | None = None,
) -> tuple[float, float]:
    if threshold <= 0 or not sentinel_prompts:
        deltas, next_ps = propose_updates(states, wrappers, a_by_layer, d_by_layer, w_by_layer, eta, slot_id=slot_id)
        apply_memory_deltas(wrappers, deltas, slot_id=slot_id)
        for layer_idx, next_p in next_ps.items():
            states[layer_idx].commit(next_p)
        return eta, 0.0

    before = yes_no_distributions(
        model,
        tokenizer,
        sentinel_prompts,
        device,
        max_length=max_length,
        use_chat_template=use_chat_template,
    )
    attempt_eta = eta
    last_kl = 0.0
    for _attempt in range(retries + 1):
        deltas, next_ps = propose_updates(
            states,
            wrappers,
            a_by_layer,
            d_by_layer,
            w_by_layer,
            attempt_eta,
            slot_id=slot_id,
        )
        apply_memory_deltas(wrappers, deltas, slot_id=slot_id)
        after = yes_no_distributions(
            model,
            tokenizer,
            sentinel_prompts,
            device,
            max_length=max_length,
            use_chat_template=use_chat_template,
        )
        last_kl = categorical_kl(before, after)
        if last_kl <= threshold or _attempt == retries:
            for layer_idx, next_p in next_ps.items():
                states[layer_idx].commit(next_p)
            return attempt_eta, last_kl
        apply_memory_deltas(wrappers, deltas, sign=-1.0, slot_id=slot_id)
        attempt_eta *= 0.5
    return attempt_eta, last_kl


class FoldEval:
    def __init__(self, results: list[Any]):
        self.results = results
        self.fold_accuracies = [result.accuracy for result in results]
        self.fold_mean_margins = [result.mean_margin for result in results]
        self.fold_positive_accuracies = [result.positive_accuracy for result in results]
        self.fold_negative_accuracies = [result.negative_accuracy for result in results]
        self.n = sum(result.n for result in results)
        self.correct = sum(result.correct for result in results)
        self.accuracy = self.correct / self.n if self.n else 0.0
        self.mean_margin = (
            sum(result.mean_margin * result.n for result in results) / self.n
            if self.n
            else 0.0
        )
        self.positive_n = sum(result.positive_n for result in results)
        self.positive_correct = sum(result.positive_correct for result in results)
        self.positive_accuracy = self.positive_correct / self.positive_n if self.positive_n else 0.0
        self.positive_mean_margin = (
            sum(result.positive_mean_margin * result.positive_n for result in results) / self.positive_n
            if self.positive_n
            else 0.0
        )
        self.negative_n = sum(result.negative_n for result in results)
        self.negative_correct = sum(result.negative_correct for result in results)
        self.negative_accuracy = self.negative_correct / self.negative_n if self.negative_n else 0.0
        self.negative_mean_margin = (
            sum(result.negative_mean_margin * result.negative_n for result in results) / self.negative_n
            if self.negative_n
            else 0.0
        )


def evaluate_yes_no_folds(
    model: Any,
    tokenizer: Any,
    question_sets: list[list],
    device: Any,
    paper: str | None,
    max_length: int,
    use_chat_template: bool,
) -> FoldEval:
    return FoldEval(
        [
            evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=paper,
                max_length=max_length,
                use_chat_template=use_chat_template,
            )
            for questions in question_sets
        ]
    )


def safe_write_search(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    wrappers: dict,
    states: dict[int, PlasticityState],
    keys: dict[int, torch.Tensor],
    selected_rows: list[int],
    a_by_layer: dict[int, torch.Tensor],
    d_by_layer: dict[int, torch.Tensor],
    w_by_layer: dict[int, torch.Tensor],
    validation_question_sets: list[list],
    validation_gauntlet_sets: dict[str, list],
    guard_questions: list,
    sentinel_prompts: list[str],
    args: argparse.Namespace,
    search_path: Path,
    paper_idx: int,
    slot_id: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    snapshot = snapshot_wrappers(wrappers)
    validation_pre = evaluate_yes_no_folds(
        model,
        tokenizer,
        validation_question_sets,
        device,
        paper=None,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    validation_gauntlet_pre = {
        bucket: evaluate_yes_no(
            model,
            tokenizer,
            questions,
            device,
            paper=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        for bucket, questions in validation_gauntlet_sets.items()
    }
    contamination_pre = evaluate_yes_no(
        model,
        tokenizer,
        guard_questions,
        device,
        paper=None,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    sentinel_before = yes_no_distributions(
        model,
        tokenizer,
        sentinel_prompts,
        device,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )

    best: dict[str, Any] | None = None
    best_deltas: dict[int, torch.Tensor] | None = None
    best_next_ps: dict[int, torch.Tensor] | None = None
    thresholds = args.search_gate_thresholds if args.memory_gate else [0.0]
    trial_count = 0
    accepted_count = 0

    for gate_threshold in thresholds:
        for eta in args.search_etas:
            for max_norm in args.search_max_update_norms:
                trial_count += 1
                restore_wrappers(wrappers, snapshot)
                if args.memory_gate:
                    configure_current_gate(
                        wrappers,
                        keys,
                        selected_rows,
                        snapshot,
                        gate_threshold,
                        args.memory_gate_temperature,
                        slot_id=slot_id,
                    )
                deltas, next_ps = propose_updates(
                    states,
                    wrappers,
                    a_by_layer,
                    d_by_layer,
                    w_by_layer,
                    eta,
                    max_update_norm=max_norm,
                    slot_id=slot_id,
                )
                apply_memory_deltas(wrappers, deltas, slot_id=slot_id)
                validation_post = evaluate_yes_no_folds(
                    model,
                    tokenizer,
                    validation_question_sets,
                    device,
                    paper=None,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                validation_gauntlet_post = {
                    bucket: evaluate_yes_no(
                        model,
                        tokenizer,
                        questions,
                        device,
                        paper=None,
                        max_length=args.max_length,
                        use_chat_template=args.chat_template,
                    )
                    for bucket, questions in validation_gauntlet_sets.items()
                }
                contamination_post = evaluate_yes_no(
                    model,
                    tokenizer,
                    guard_questions,
                    device,
                    paper=None,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                sentinel_after = yes_no_distributions(
                    model,
                    tokenizer,
                    sentinel_prompts,
                    device,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                sentinel_kl = categorical_kl(sentinel_before, sentinel_after)
                row = {
                    "paper_idx": paper_idx,
                    "gate_threshold": gate_threshold if args.memory_gate else None,
                    "eta": eta,
                    "max_update_norm": max_norm,
                    "validation_pre_accuracy": validation_pre.accuracy,
                    "validation_post_accuracy": validation_post.accuracy,
                    "validation_accuracy_delta": validation_post.accuracy - validation_pre.accuracy,
                    "validation_pre_mean_margin": validation_pre.mean_margin,
                    "validation_post_mean_margin": validation_post.mean_margin,
                    "validation_margin_delta": validation_post.mean_margin - validation_pre.mean_margin,
                    "validation_pre_positive_accuracy": validation_pre.positive_accuracy,
                    "validation_post_positive_accuracy": validation_post.positive_accuracy,
                    "validation_positive_accuracy_delta": (
                        validation_post.positive_accuracy - validation_pre.positive_accuracy
                    ),
                    "validation_pre_negative_accuracy": validation_pre.negative_accuracy,
                    "validation_post_negative_accuracy": validation_post.negative_accuracy,
                    "validation_negative_accuracy_delta": (
                        validation_post.negative_accuracy - validation_pre.negative_accuracy
                    ),
                    "validation_pre_positive_mean_margin": validation_pre.positive_mean_margin,
                    "validation_post_positive_mean_margin": validation_post.positive_mean_margin,
                    "validation_positive_margin_delta": (
                        validation_post.positive_mean_margin - validation_pre.positive_mean_margin
                    ),
                    "validation_pre_negative_mean_margin": validation_pre.negative_mean_margin,
                    "validation_post_negative_mean_margin": validation_post.negative_mean_margin,
                    "validation_negative_margin_delta": (
                        validation_post.negative_mean_margin - validation_pre.negative_mean_margin
                    ),
                    "validation_fold_pre_accuracies": validation_pre.fold_accuracies,
                    "validation_fold_post_accuracies": validation_post.fold_accuracies,
                    "validation_fold_accuracy_deltas": [
                        post - pre
                        for pre, post in zip(validation_pre.fold_accuracies, validation_post.fold_accuracies)
                    ],
                    "validation_fold_pre_positive_accuracies": validation_pre.fold_positive_accuracies,
                    "validation_fold_post_positive_accuracies": validation_post.fold_positive_accuracies,
                    "validation_fold_positive_accuracy_deltas": [
                        post - pre
                        for pre, post in zip(
                            validation_pre.fold_positive_accuracies,
                            validation_post.fold_positive_accuracies,
                        )
                    ],
                    "validation_fold_pre_negative_accuracies": validation_pre.fold_negative_accuracies,
                    "validation_fold_post_negative_accuracies": validation_post.fold_negative_accuracies,
                    "validation_fold_negative_accuracy_deltas": [
                        post - pre
                        for pre, post in zip(
                            validation_pre.fold_negative_accuracies,
                            validation_post.fold_negative_accuracies,
                        )
                    ],
                    "validation_min_accuracy_delta": min(
                        (
                            post - pre
                            for pre, post in zip(validation_pre.fold_accuracies, validation_post.fold_accuracies)
                        ),
                        default=0.0,
                    ),
                    "contamination_pre_accuracy": contamination_pre.accuracy,
                    "contamination_post_accuracy": contamination_post.accuracy,
                    "contamination_accuracy_delta": contamination_post.accuracy - contamination_pre.accuracy,
                    "contamination_pre_mean_margin": contamination_pre.mean_margin,
                    "contamination_post_mean_margin": contamination_post.mean_margin,
                    "sentinel_kl": sentinel_kl,
                    "accepted": False,
                }
                gauntlet_bucket_deltas = []
                for bucket, pre_result in validation_gauntlet_pre.items():
                    post_result = validation_gauntlet_post[bucket]
                    delta = post_result.accuracy - pre_result.accuracy
                    gauntlet_bucket_deltas.append(delta)
                    row[f"validation_gauntlet_{bucket}_pre_accuracy"] = pre_result.accuracy
                    row[f"validation_gauntlet_{bucket}_post_accuracy"] = post_result.accuracy
                    row[f"validation_gauntlet_{bucket}_accuracy_delta"] = delta
                    row[f"validation_gauntlet_{bucket}_pre_negative_accuracy"] = pre_result.negative_accuracy
                    row[f"validation_gauntlet_{bucket}_post_negative_accuracy"] = post_result.negative_accuracy
                    row[f"validation_gauntlet_{bucket}_negative_accuracy_delta"] = (
                        post_result.negative_accuracy - pre_result.negative_accuracy
                    )
                row["validation_gauntlet_min_bucket_delta"] = (
                    min(gauntlet_bucket_deltas) if gauntlet_bucket_deltas else 0.0
                )
                row["accepted"] = (
                    contamination_post.accuracy >= contamination_pre.accuracy
                    and sentinel_kl <= args.sentinel_kl_threshold
                    and validation_post.accuracy - validation_pre.accuracy >= args.validation_min_accuracy_delta
                    and validation_post.accuracy > validation_pre.accuracy
                    and row["validation_min_accuracy_delta"] >= args.validation_min_fold_delta
                    and row["validation_positive_accuracy_delta"] >= args.validation_min_positive_delta
                    and row["validation_negative_accuracy_delta"] >= args.validation_min_negative_delta
                    and row["validation_gauntlet_min_bucket_delta"] >= args.gauntlet_min_bucket_delta
                )
                append_jsonl(search_path, row)

                if row["accepted"]:
                    accepted_count += 1
                    score = (
                        row["validation_accuracy_delta"],
                        row["validation_negative_accuracy_delta"],
                        row["validation_post_negative_accuracy"],
                        row["validation_min_accuracy_delta"],
                        row["validation_post_accuracy"],
                        -row["sentinel_kl"],
                        row["contamination_post_mean_margin"],
                        row["validation_margin_delta"],
                        max_norm,
                    )
                    if best is None or score > best["score"]:
                        best = dict(row)
                        best["score"] = score
                        best_deltas = {layer_idx: delta.detach().clone() for layer_idx, delta in deltas.items()}
                        best_next_ps = {layer_idx: next_p.detach().clone() for layer_idx, next_p in next_ps.items()}

    restore_wrappers(wrappers, snapshot)
    if best is None:
        if args.allow_noop_write:
            if slot_id is not None:
                clear_slot_memory(wrappers, slot_id)
            return {
                "write_applied": False,
                "safe_search_accepted": False,
                "validation_pre_accuracy": validation_pre.accuracy,
                "validation_post_accuracy": validation_pre.accuracy,
                "validation_accuracy_delta": 0.0,
                "validation_pre_mean_margin": validation_pre.mean_margin,
                "validation_post_mean_margin": validation_pre.mean_margin,
                "validation_margin_delta": 0.0,
                "validation_pre_positive_accuracy": validation_pre.positive_accuracy,
                "validation_post_positive_accuracy": validation_pre.positive_accuracy,
                "validation_positive_accuracy_delta": 0.0,
                "validation_pre_negative_accuracy": validation_pre.negative_accuracy,
                "validation_post_negative_accuracy": validation_pre.negative_accuracy,
                "validation_negative_accuracy_delta": 0.0,
                "validation_pre_positive_mean_margin": validation_pre.positive_mean_margin,
                "validation_post_positive_mean_margin": validation_pre.positive_mean_margin,
                "validation_positive_margin_delta": 0.0,
                "validation_pre_negative_mean_margin": validation_pre.negative_mean_margin,
                "validation_post_negative_mean_margin": validation_pre.negative_mean_margin,
                "validation_negative_margin_delta": 0.0,
                "validation_min_accuracy_delta": 0.0,
                "validation_gauntlet_min_bucket_delta": 0.0,
                "contamination_pre_accuracy": contamination_pre.accuracy,
                "contamination_post_accuracy": contamination_pre.accuracy,
                "contamination_accuracy_delta": 0.0,
                "sentinel_kl": 0.0,
                "eta_used": 0.0,
                "max_update_norm_used": 0.0,
                "memory_gate_threshold_used": None,
                "safe_search_trials": trial_count,
                "safe_search_accepted_trials": accepted_count,
                "safe_search_seconds": time.time() - started,
            }
        raise RuntimeError(
            "Safe write search found no candidate that improved validation while preserving contamination guards. "
            f"Inspect {search_path}."
        )

    assert best_deltas is not None
    assert best_next_ps is not None
    if args.memory_gate:
        configure_current_gate(
            wrappers,
            keys,
            selected_rows,
            snapshot,
            best["gate_threshold"],
            args.memory_gate_temperature,
            slot_id=slot_id,
        )
    apply_memory_deltas(wrappers, best_deltas, slot_id=slot_id)
    for layer_idx, next_p in best_next_ps.items():
        states[layer_idx].commit(next_p)

    best.pop("score", None)
    return {
        "write_applied": True,
        "safe_search_accepted": True,
        "validation_pre_accuracy": best["validation_pre_accuracy"],
        "validation_post_accuracy": best["validation_post_accuracy"],
        "validation_accuracy_delta": best["validation_accuracy_delta"],
        "validation_pre_mean_margin": best["validation_pre_mean_margin"],
        "validation_post_mean_margin": best["validation_post_mean_margin"],
        "validation_margin_delta": best["validation_margin_delta"],
        "validation_pre_positive_accuracy": best["validation_pre_positive_accuracy"],
        "validation_post_positive_accuracy": best["validation_post_positive_accuracy"],
        "validation_positive_accuracy_delta": best["validation_positive_accuracy_delta"],
        "validation_pre_negative_accuracy": best["validation_pre_negative_accuracy"],
        "validation_post_negative_accuracy": best["validation_post_negative_accuracy"],
        "validation_negative_accuracy_delta": best["validation_negative_accuracy_delta"],
        "validation_pre_positive_mean_margin": best["validation_pre_positive_mean_margin"],
        "validation_post_positive_mean_margin": best["validation_post_positive_mean_margin"],
        "validation_positive_margin_delta": best["validation_positive_margin_delta"],
        "validation_pre_negative_mean_margin": best["validation_pre_negative_mean_margin"],
        "validation_post_negative_mean_margin": best["validation_post_negative_mean_margin"],
        "validation_negative_margin_delta": best["validation_negative_margin_delta"],
        "validation_min_accuracy_delta": best["validation_min_accuracy_delta"],
        "validation_gauntlet_min_bucket_delta": best.get("validation_gauntlet_min_bucket_delta", 0.0),
        "contamination_pre_accuracy": best["contamination_pre_accuracy"],
        "contamination_post_accuracy": best["contamination_post_accuracy"],
        "contamination_accuracy_delta": best["contamination_accuracy_delta"],
        "sentinel_kl": best["sentinel_kl"],
        "eta_used": best["eta"],
        "max_update_norm_used": best["max_update_norm"],
        "memory_gate_threshold_used": best["gate_threshold"],
        "safe_search_trials": trial_count,
        "safe_search_accepted_trials": accepted_count,
        "safe_search_seconds": time.time() - started,
    }


def init_background_plasticity(
    model: Any,
    tokenizer: Any,
    states: dict[int, PlasticityState],
    layer_indices: list[int],
    device: torch.device,
    prompt_count: int,
    seed: int,
    batch_size: int,
    max_length: int,
    use_chat_template: bool,
    capture_last_tokens: int,
) -> None:
    if prompt_count <= 0:
        return
    prompts = [
        format_model_prompt(tokenizer, prompt, use_chat_template=use_chat_template)
        for prompt in negative_guard_prompts(seed, prompt_count)
    ]
    captures = capture_layer_io(
        model,
        tokenizer,
        prompts,
        layer_indices,
        device,
        batch_size,
        max_length,
        capture_last_tokens=capture_last_tokens,
    )
    for layer_idx, state in states.items():
        state.init_from_keys_diagonal(captures[layer_idx].keys.T, mu=1.0)


def evaluate_retention(
    model: Any,
    tokenizer: Any,
    eval_sets: list[list],
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
) -> float:
    if not eval_sets:
        return 0.0
    accs = retention_accuracies(model, tokenizer, eval_sets, device, max_length, use_chat_template)
    return sum(accs) / len(accs)


def flatten_question_sets(question_sets: list[list]) -> list:
    return [record for questions in question_sets for record in questions]


def add_gauntlet_metrics(
    row: dict[str, Any],
    prefix: str,
    results_by_bucket: dict[str, Any],
) -> None:
    for bucket, result in results_by_bucket.items():
        row.update(result.to_dict(f"{prefix}_{bucket}"))


def retention_accuracies(
    model: Any,
    tokenizer: Any,
    eval_sets: list[list],
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
) -> list[float]:
    return [
        evaluate_yes_no(
            model,
            tokenizer,
            questions,
            device,
            paper=None,
            max_length=max_length,
            use_chat_template=use_chat_template,
        ).accuracy
        for questions in eval_sets
    ]


def gate_similarity_diagnostics(
    model: Any,
    tokenizer: Any,
    wrappers: dict,
    questions: list,
    layer_indices: list[int],
    device: Any,
    batch_size: int,
    max_length: int,
    use_chat_template: bool,
    capture_last_tokens: int,
    prefix: str,
) -> tuple[dict[str, float], dict[int, list[float]]]:
    if not questions:
        return {}, {}
    prompts = [
        format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=use_chat_template)
        for record in questions
    ]
    captures = capture_layer_io(
        model,
        tokenizer,
        prompts,
        layer_indices,
        device,
        batch_size,
        max_length,
        capture_last_tokens=capture_last_tokens,
    )
    summary: dict[str, float] = {}
    per_question: dict[int, list[float]] = {}
    for layer_idx in layer_indices:
        gate_keys = wrappers[layer_idx].gate_keys.detach().float().cpu()
        if gate_keys.numel() == 0:
            per_question[layer_idx] = [0.0 for _ in questions]
            summary[f"{prefix}_layer_{layer_idx}_gate_key_count"] = 0.0
            summary[f"{prefix}_layer_{layer_idx}_mean_max_similarity"] = 0.0
            summary[f"{prefix}_layer_{layer_idx}_token_hit_rate"] = 0.0
            summary[f"{prefix}_layer_{layer_idx}_question_hit_rate"] = 0.0
            continue

        keys = captures[layer_idx].keys.reshape(len(questions), capture_last_tokens, -1).float()
        key_norm = torch.nn.functional.normalize(gate_keys, dim=-1)
        probe_norm = torch.nn.functional.normalize(keys.reshape(-1, keys.shape[-1]), dim=-1)
        token_max = (probe_norm @ key_norm.T).amax(dim=-1).reshape(len(questions), capture_last_tokens)
        question_max = token_max.amax(dim=-1)
        threshold = wrappers[layer_idx].gate_threshold
        per_question[layer_idx] = [float(value) for value in question_max]
        summary[f"{prefix}_layer_{layer_idx}_gate_key_count"] = float(gate_keys.shape[0])
        summary[f"{prefix}_layer_{layer_idx}_mean_max_similarity"] = float(question_max.mean().item())
        summary[f"{prefix}_layer_{layer_idx}_token_hit_rate"] = float((token_max >= threshold).float().mean().item())
        summary[f"{prefix}_layer_{layer_idx}_question_hit_rate"] = float((question_max >= threshold).float().mean().item())
    return summary, per_question


def resolve_layer_idx(model: Any, layer_idx: int) -> int:
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    if resolved < 0 or resolved >= len(layers):
        raise IndexError(f"Layer index {layer_idx} resolved to {resolved}, but model has {len(layers)} layers.")
    return resolved


def domain_latch_terms(domain: DomainSpec) -> list[str]:
    return [domain.title, *domain.operators, *domain.marks]


def token_indices_for_terms(
    tokenizer: Any,
    prompt: str,
    terms: list[str],
    max_length: int,
) -> list[int]:
    if not terms:
        return []
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
    except (NotImplementedError, TypeError):
        return []

    offsets = encoded.get("offset_mapping", [])
    lowered = prompt.lower()
    spans: list[tuple[int, int]] = []
    for term in terms:
        term_lower = term.lower()
        start = 0
        while True:
            hit = lowered.find(term_lower, start)
            if hit < 0:
                break
            spans.append((hit, hit + len(term_lower)))
            start = hit + len(term_lower)
    if not spans:
        return []

    indices: list[int] = []
    for token_idx, offset in enumerate(offsets):
        token_start, token_end = offset
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in spans):
            indices.append(token_idx)
    return indices


def capture_content_latents(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    terms: list[str],
    layer_idx: int,
    device: torch.device,
    max_length: int,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool residual activations at content-token positions for domain routing."""

    resolved = resolve_layer_idx(model, layer_idx)
    latents: list[torch.Tensor] = []
    hits: list[bool] = []
    clear_active_slot_weights(model)
    batch_size = max(1, batch_size)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_indices = [
            token_indices_for_terms(tokenizer, prompt, terms, max_length)
            for prompt in batch
        ]
        tokens = tokenizer(
            batch,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attention_mask = tokens.get("attention_mask")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens, use_cache=False, output_hidden_states=True)
        hidden_batch = outputs.hidden_states[resolved + 1].detach().float().cpu()
        mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None
        for batch_idx, token_indices in enumerate(batch_indices):
            hidden = hidden_batch[batch_idx]
            if mask_cpu is not None:
                actual_len = int(mask_cpu[batch_idx].sum().item())
            else:
                actual_len = hidden.shape[0]
            if tokenizer.padding_side == "left":
                pad_offset = max(0, hidden.shape[0] - actual_len)
            else:
                pad_offset = 0
            valid_indices = [
                idx + pad_offset
                for idx in token_indices
                if 0 <= idx < actual_len and 0 <= idx + pad_offset < hidden.shape[0]
            ]
            if valid_indices:
                latents.append(hidden[valid_indices].mean(dim=0))
                hits.append(True)
                continue
            fallback_idx = min(hidden.shape[0] - 1, max(0, pad_offset + actual_len - 1))
            latents.append(hidden[fallback_idx])
            hits.append(False)
    if not latents:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.bool)
    return torch.stack(latents, dim=0), torch.tensor(hits, dtype=torch.bool)


def whiten_latents(
    latents: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if latents.numel() == 0:
        return latents
    return (latents.float() - mean.float()) / scale.float().clamp_min(1e-4)


class ActivationSlotRouter:
    """Two-factor slot router: activation domain latch plus suffix behavior gate."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        layer_idx: int,
        max_length: int,
        batch_size: int,
        threshold: float,
        margin: float,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_idx = layer_idx
        self.max_length = max_length
        self.batch_size = batch_size
        self.threshold = threshold
        self.margin = margin
        self.slot_terms: list[list[str]] = []
        self.raw_slot_latents: list[torch.Tensor] = []
        self.whiten_mean: torch.Tensor | None = None
        self.whiten_scale: torch.Tensor | None = None
        self.prototypes: torch.Tensor | None = None
        self._weight_cache: dict[str, torch.Tensor] = {}

    def add_slot(self, slot_id: int, domain: DomainSpec, prompts: list[str]) -> dict[str, float]:
        terms = domain_latch_terms(domain)
        latents, hits = capture_content_latents(
            self.model,
            self.tokenizer,
            prompts,
            terms,
            self.layer_idx,
            self.device,
            self.max_length,
            self.batch_size,
        )
        if latents.numel() == 0:
            raise ValueError("Cannot add activation slot without prototype prompts.")
        while len(self.raw_slot_latents) <= slot_id:
            self.raw_slot_latents.append(torch.empty(0, latents.shape[-1]))
            self.slot_terms.append([])
        usable = latents[hits] if bool(hits.any()) else latents
        self.raw_slot_latents[slot_id] = usable.detach().float().cpu()
        self.slot_terms[slot_id] = terms
        self._recompute()
        return {
            "activation_slot_prototype_prompts": float(latents.shape[0]),
            "activation_slot_term_hit_rate": float(hits.float().mean().item()) if hits.numel() else 0.0,
        }

    def all_terms(self) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for slot_terms in self.slot_terms:
            for term in slot_terms:
                lower = term.lower()
                if lower not in seen:
                    seen.add(lower)
                    terms.append(term)
        return terms

    def _recompute(self) -> None:
        populated = [latents for latents in self.raw_slot_latents if latents.numel() > 0]
        if not populated:
            self.whiten_mean = None
            self.whiten_scale = None
            self.prototypes = None
            return
        all_latents = torch.cat(populated, dim=0).float()
        self.whiten_mean = all_latents.mean(dim=0)
        scale = all_latents.std(dim=0, unbiased=False)
        self.whiten_scale = scale.clamp_min(1e-3)
        prototypes: list[torch.Tensor] = []
        for latents in self.raw_slot_latents:
            if latents.numel() == 0:
                prototypes.append(torch.zeros(all_latents.shape[-1]))
                continue
            whitened = whiten_latents(latents, self.whiten_mean, self.whiten_scale)
            prototypes.append(torch.nn.functional.normalize(whitened.mean(dim=0), dim=0))
        self.prototypes = torch.stack(prototypes, dim=0).float()
        self._weight_cache.clear()

    def classify(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.prototypes is None or self.whiten_mean is None or self.whiten_scale is None:
            return (
                torch.zeros(len(prompts), 0),
                torch.zeros(len(prompts), dtype=torch.bool),
                torch.zeros(len(prompts), dtype=torch.bool),
            )
        latents, hits = capture_content_latents(
            self.model,
            self.tokenizer,
            prompts,
            self.all_terms(),
            self.layer_idx,
            self.device,
            self.max_length,
            self.batch_size,
        )
        whitened = whiten_latents(latents, self.whiten_mean, self.whiten_scale)
        normed = torch.nn.functional.normalize(whitened, dim=1)
        similarities = normed @ self.prototypes.T
        return similarities, hits, torch.ones(len(prompts), dtype=torch.bool)

    def __call__(self, prompts: list[str]) -> torch.Tensor:
        if prompts and all(prompt in self._weight_cache for prompt in prompts):
            return torch.stack([self._weight_cache[prompt] for prompt in prompts], dim=0)

        missing_prompts: list[str] = []
        seen_missing: set[str] = set()
        for prompt in prompts:
            if prompt in self._weight_cache or prompt in seen_missing:
                continue
            seen_missing.add(prompt)
            missing_prompts.append(prompt)

        if missing_prompts:
            missing_weights = self._compute_weights(missing_prompts)
            for prompt, row in zip(missing_prompts, missing_weights):
                self._weight_cache[prompt] = row.detach().float().cpu()

        if not prompts:
            return torch.zeros(0, len(self.slot_terms), dtype=torch.float32)
        return torch.stack([self._weight_cache[prompt] for prompt in prompts], dim=0)

    def _compute_weights(self, prompts: list[str]) -> torch.Tensor:
        similarities, hits, _valid = self.classify(prompts)
        if similarities.numel() == 0:
            return torch.zeros(len(prompts), len(self.slot_terms), dtype=torch.float32)
        if similarities.shape[1] == 1:
            weights = torch.zeros(len(prompts), 1, dtype=torch.float32)
            weights[hits, 0] = 1.0
            return weights
        top_values, top_indices = similarities.topk(k=min(2, similarities.shape[1]), dim=1)
        best = top_values[:, 0]
        if top_values.shape[1] > 1:
            margin = top_values[:, 0] - top_values[:, 1]
        else:
            margin = torch.full_like(best, float("inf"))
        active = hits & (best >= self.threshold) & (margin >= self.margin)
        weights = torch.zeros(len(prompts), similarities.shape[1], dtype=torch.float32)
        for row_idx, is_active in enumerate(active.tolist()):
            if is_active:
                weights[row_idx, int(top_indices[row_idx, 0].item())] = 1.0
        return weights


def activation_domain_latch_diagnostics(
    model: Any,
    tokenizer: Any,
    domains: list[DomainSpec],
    paper_idx: int,
    args: argparse.Namespace,
    device: torch.device,
    path: Path,
) -> dict[str, float]:
    if not domains:
        return {}

    train_prompts: list[str] = []
    train_labels: list[int] = []
    eval_prompts: list[str] = []
    eval_labels: list[int] = []
    all_terms: list[str] = []
    for domain_idx, domain in enumerate(domains):
        all_terms.extend(domain_latch_terms(domain))
        train_questions = make_candidate_probes(
            domain,
            args.domain_latch_probes,
            seed=args.seed * 500_000 + domain_idx,
        )
        eval_questions = make_eval_questions(
            domain,
            args.domain_latch_probes,
            seed=args.seed * 500_000 + domain_idx + 10_000,
        )
        train_prompts.extend(
            format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=args.chat_template)
            for record in train_questions
        )
        train_labels.extend([domain_idx] * len(train_questions))
        eval_prompts.extend(
            format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=args.chat_template)
            for record in eval_questions
        )
        eval_labels.extend([domain_idx] * len(eval_questions))

    train_latents, train_hits = capture_content_latents(
        model,
        tokenizer,
        train_prompts,
        all_terms,
        args.domain_latch_layer,
        device,
        args.max_length,
        args.batch_size,
    )
    eval_latents, eval_hits = capture_content_latents(
        model,
        tokenizer,
        eval_prompts,
        all_terms,
        args.domain_latch_layer,
        device,
        args.max_length,
        args.batch_size,
    )
    if train_latents.numel() == 0 or eval_latents.numel() == 0:
        return {}

    mean = train_latents.mean(dim=0)
    scale = train_latents.std(dim=0, unbiased=False).clamp_min(1e-3)
    train_whitened = whiten_latents(train_latents, mean, scale)
    eval_whitened = whiten_latents(eval_latents, mean, scale)
    prototypes: list[torch.Tensor] = []
    for domain_idx in range(len(domains)):
        rows = [idx for idx, label in enumerate(train_labels) if label == domain_idx and bool(train_hits[idx])]
        if not rows:
            rows = [idx for idx, label in enumerate(train_labels) if label == domain_idx]
        proto = train_whitened[rows].mean(dim=0)
        prototypes.append(torch.nn.functional.normalize(proto, dim=0))
    prototype_matrix = torch.stack(prototypes, dim=0)
    similarities = torch.nn.functional.normalize(eval_whitened, dim=1) @ prototype_matrix.T
    top_values, top_indices = similarities.topk(k=min(2, similarities.shape[1]), dim=1)
    best = top_values[:, 0]
    second = top_values[:, 1] if top_values.shape[1] > 1 else torch.zeros_like(best)
    margins = best - second
    labels = torch.tensor(eval_labels, dtype=torch.long)
    predictions = top_indices[:, 0]
    correct = predictions == labels
    true_similarities = similarities[torch.arange(len(labels)), labels]
    if similarities.shape[1] == 1:
        active = eval_hits
    else:
        active = eval_hits & (best >= args.domain_latch_threshold) & (margins >= args.domain_latch_margin)

    summary = {
        "domain_latch_layer": float(args.domain_latch_layer),
        "domain_latch_domains": float(len(domains)),
        "domain_latch_train_prompts": float(len(train_prompts)),
        "domain_latch_eval_prompts": float(len(eval_prompts)),
        "domain_latch_train_term_hit_rate": float(train_hits.float().mean().item()),
        "domain_latch_eval_term_hit_rate": float(eval_hits.float().mean().item()),
        "domain_latch_accuracy": float(correct.float().mean().item()),
        "domain_latch_active_accuracy": float(correct[active].float().mean().item()) if bool(active.any()) else 0.0,
        "domain_latch_active_rate": float(active.float().mean().item()),
        "domain_latch_mean_true_similarity": float(true_similarities.mean().item()),
        "domain_latch_mean_top1_similarity": float(best.mean().item()),
        "domain_latch_mean_margin": float(margins.mean().item()),
        "domain_latch_min_margin": float(margins.min().item()),
    }
    append_jsonl(path, {"paper_idx": paper_idx, **summary})
    return summary


def append_question_diagnostics(
    path: Path,
    paper_idx: int,
    domain: DomainSpec,
    before: list,
    after: list,
    gate_by_layer: dict[int, list[float]],
) -> None:
    for question_idx, (pre, post) in enumerate(zip(before, after)):
        row = {
            "paper_idx": paper_idx,
            "domain_id": domain.domain_id,
            "title": domain.title,
            "question_idx": question_idx,
            "question": post.question,
            "answer": post.answer,
            "pre_prediction": pre.prediction,
            "post_prediction": post.prediction,
            "pre_correct": pre.correct,
            "post_correct": post.correct,
            "pre_margin": pre.margin,
            "post_margin": post.margin,
            "margin_delta": post.margin - pre.margin,
            "flipped": pre.prediction != post.prediction,
            "flip_became_correct": (not pre.correct) and post.correct,
            "flip_became_wrong": pre.correct and (not post.correct),
            "category": post.category,
        }
        for layer_idx, values in gate_by_layer.items():
            if question_idx < len(values):
                row[f"layer_{layer_idx}_gate_max_similarity"] = values[question_idx]
        append_jsonl(path, row)


def plot_metrics(metrics_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    if not metrics_path.exists():
        return
    df = pd.read_json(metrics_path, lines=True)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["paper_idx"], df["aic_latest_accuracy"], marker="o", label="CAIC latest")
    ax.plot(df["paper_idx"], df["context_latest_accuracy"], marker="o", label="Context upper")
    ax.plot(df["paper_idx"], df["pre_no_doc_latest_accuracy"], marker="o", label="No-doc pre")
    if "retention_mean_accuracy" in df:
        ax.plot(df["paper_idx"], df["retention_mean_accuracy"], marker="o", label="Retention mean")
    ax.set_xlabel("Paper index")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_csv(metrics_path: Path, output_path: Path) -> None:
    try:
        import pandas as pd

        pd.read_json(metrics_path, lines=True).to_csv(output_path, index=False)
        return
    except ImportError:
        pass

    import csv

    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_optional_baselines(args: argparse.Namespace) -> dict[str, tuple[Any, Any, torch.device]]:
    baselines: dict[str, tuple[Any, Any, torch.device]] = {}
    for name in args.baselines:
        model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
        if name == "qa_lora":
            model = prepare_qa_lora_model(model, r=args.qa_lora_r)
            model.to(device)
        baselines[name] = (model, tokenizer, device)
    return baselines


def select_teacher_validated_domains(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    gate_path: Path | None = None,
) -> tuple[list[DomainSpec], list[list], list[dict[str, Any]]]:
    selected_domains: list[DomainSpec] = []
    selected_eval_sets: list[list] = []
    gate_rows: list[dict[str, Any]] = []
    if gate_path is not None and gate_path.exists():
        gate_path.unlink()

    for candidate_idx in range(args.teacher_search_budget):
        domain = generate_domain(args.seed, candidate_idx, difficulty=args.domain_difficulty)
        eval_questions = make_eval_questions(
            domain,
            args.eval_questions,
            seed=args.seed * 10_000 + candidate_idx,
        )
        no_doc = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            paper=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        context = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            paper=domain.render_paper(),
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        row = {
            "candidate_idx": candidate_idx,
            "domain_id": domain.domain_id,
            "title": domain.title,
            "accepted": False,
            "no_doc_accuracy": no_doc.accuracy,
            "context_accuracy": context.accuracy,
            "context_delta": context.accuracy - no_doc.accuracy,
            "context_margin": context.mean_margin,
        }
        eps = 1e-8
        accepted = (
            context.accuracy + eps >= args.teacher_min_accuracy
            and context.accuracy - no_doc.accuracy + eps >= args.teacher_min_delta
        )
        row["accepted"] = accepted
        gate_rows.append(row)
        if gate_path is not None:
            append_jsonl(gate_path, row)
        if accepted:
            selected_domains.append(domain)
            selected_eval_sets.append(eval_questions)
            if len(selected_domains) >= args.papers:
                break

    if len(selected_domains) < args.papers:
        best = sorted(
            gate_rows,
            key=lambda item: (item["context_delta"], item["context_accuracy"]),
            reverse=True,
        )[:5]
        best_text = json.dumps(best, indent=2)
        raise RuntimeError(
            f"Teacher gate accepted {len(selected_domains)}/{args.papers} domains "
            f"within budget {args.teacher_search_budget}. Best candidates:\n{best_text}"
        )

    return selected_domains, selected_eval_sets, gate_rows


def run() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    if args.dry_run_synthetic:
        if args.domains_jsonl:
            domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)
        else:
            domains = generate_domains(args.papers, seed=args.seed, difficulty=args.domain_difficulty)
            eval_sets = [
                make_eval_questions(domain, args.eval_questions, seed=args.seed * 10_000 + idx)
                for idx, domain in enumerate(domains)
            ]
    else:
        load_runtime_modules()
        model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
        forward_counter = ForwardCounter(model).install()
        if args.domains_jsonl:
            domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)
        elif args.teacher_gate:
            domains, eval_sets, gate_rows = select_teacher_validated_domains(
                model,
                tokenizer,
                device,
                args,
                gate_path=output_dir / "teacher_gate.jsonl",
            )
        else:
            domains = generate_domains(args.papers, seed=args.seed, difficulty=args.domain_difficulty)
            eval_sets = [
                make_eval_questions(domain, args.eval_questions, seed=args.seed * 10_000 + idx)
                for idx, domain in enumerate(domains)
            ]
    if not args.dry_run_synthetic:
        teacher_gate_compute = forward_counter.snapshot()

    domain_rows = []
    for domain, eval_questions in zip(domains, eval_sets):
        domain_rows.append(
            {
                "domain": json.loads(domain.to_json()),
                "paper": domain.render_paper(),
                "eval_questions": [record.to_dict() for record in eval_questions],
            }
        )
    write_jsonl(output_dir / "domains.jsonl", domain_rows)

    if args.dry_run_synthetic:
        print(f"Wrote {len(domain_rows)} synthetic domains to {output_dir / 'domains.jsonl'}")
        return

    wrappers = install_additive_memory(model, args.layers, memory_dtype=torch.float32)
    layer_indices = sorted(wrappers.keys())
    activation_router = None
    if args.activation_slot_routing:
        if not args.slot_memory:
            raise RuntimeError("--activation-slot-routing requires --slot-memory.")
        activation_router = ActivationSlotRouter(
            model,
            tokenizer,
            device,
            args.domain_latch_layer,
            args.max_length,
            args.batch_size,
            args.domain_latch_threshold,
            args.domain_latch_margin,
        )
        setattr(model, "_caic_activation_slot_router", activation_router)
    rls_config = RLSConfig(
        ridge=args.ridge,
        eta=args.eta,
        max_update_norm=args.max_update_norm,
        device=args.rls_device,
        dtype=torch.float32,
    )
    states = {
        layer_idx: PlasticityState(wrappers[layer_idx].in_features, rls_config)
        for layer_idx in layer_indices
    }
    init_background_plasticity(
        model,
        tokenizer,
        states,
        layer_indices,
        device,
        args.background_prompts,
        args.seed,
        args.batch_size,
        args.max_length,
        args.chat_template,
        args.capture_last_tokens,
    )
    background_plasticity = {layer_idx: state.p.detach().clone() for layer_idx, state in states.items()}

    baseline_models = load_optional_baselines(args)
    baseline_train_config = TrainConfig(
        steps=args.baseline_steps,
        lr=args.baseline_lr,
        max_length=args.max_length,
        seed=args.seed,
    )

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    search_path = output_dir / "write_search.jsonl"
    if search_path.exists():
        search_path.unlink()
    diagnostics_path = output_dir / "diagnostics.jsonl"
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    domain_latch_path = output_dir / "domain_latch.jsonl"
    if domain_latch_path.exists():
        domain_latch_path.unlink()

    guard_questions = general_guard_questions()

    learned_eval_sets: list[list] = []
    for paper_idx, (domain, eval_questions) in enumerate(tqdm(list(zip(domains, eval_sets)), desc="papers")):
        started = time.time()
        paper_compute_start = forward_counter.snapshot()
        paper = domain.render_paper()
        pre_no_doc = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            paper=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        pre_details = (
            evaluate_yes_no_details(
                model,
                tokenizer,
                eval_questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            if args.diagnostics
            else []
        )
        context = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            paper=paper,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        gauntlet_sets = (
            make_gauntlet_questions(
                domain,
                args.gauntlet_questions,
                seed=args.seed * 200_000 + paper_idx,
                include_near_collision=args.near_collision_gauntlet,
            )
            if args.gauntlet
            else {}
        )
        pre_gauntlet = {
            bucket: evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for bucket, questions in gauntlet_sets.items()
        }
        context_gauntlet = {
            bucket: evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=paper,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for bucket, questions in gauntlet_sets.items()
        }

        candidates = build_candidate_pool(domain, args, paper_idx)
        validation_probe_sets = [
            make_candidate_probes(
                domain,
                args.validation_probes,
                seed=args.seed * 100_000 + paper_idx + 555_000 + fold_idx * 10_000,
            )
            for fold_idx in range(args.validation_folds)
        ]
        validation_gauntlet_sets = (
            make_gauntlet_questions(
                domain,
                args.gauntlet_validation_questions,
                seed=args.seed * 300_000 + paper_idx,
                include_near_collision=args.near_collision_gauntlet,
            )
            if args.gauntlet_validation
            else {}
        )
        bias_rival_metrics: dict[str, Any] = {}
        if args.bias_rival_baseline:
            validation_for_bias = flatten_question_sets(validation_probe_sets)
            scalar_yes_bias, scalar_validation = fit_scalar_yes_bias(
                model,
                tokenizer,
                validation_for_bias,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            scalar_latest = evaluate_yes_no_with_bias(
                model,
                tokenizer,
                eval_questions,
                device,
                scalar_yes_bias,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            bias_rival_metrics["scalar_yes_bias"] = scalar_yes_bias
            bias_rival_metrics.update(scalar_validation.to_dict("scalar_bias_validation"))
            bias_rival_metrics.update(scalar_latest.to_dict("scalar_bias_latest"))
            for bucket, questions in gauntlet_sets.items():
                result = evaluate_yes_no_with_bias(
                    model,
                    tokenizer,
                    questions,
                    device,
                    scalar_yes_bias,
                    paper=None,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                bias_rival_metrics.update(result.to_dict(f"scalar_bias_gauntlet_{bucket}"))
        capture_started = time.time()
        keys, student_outputs, content_deltas = capture_three_passes(
            model,
            tokenizer,
            domain,
            candidates,
            layer_indices,
            device,
            args.batch_size,
            args.max_length,
            args.chat_template,
            args.capture_last_tokens,
        )
        if args.project_answer_direction:
            content_deltas = project_deltas_away_from_answer_direction(content_deltas, model, tokenizer)
        if args.target_mode == "answer_direction":
            content_deltas = answer_direction_targets_like(
                content_deltas,
                candidates,
                args.capture_last_tokens,
                model,
                tokenizer,
                args.answer_target_scale,
            )
        if args.target_mode == "validity_probe":
            content_deltas = validity_probe_targets_like(
                content_deltas,
                student_outputs,
                candidates,
                domain,
                args.capture_last_tokens,
                args.ridge,
                args.validity_target_margin,
            )
        capture_seconds = time.time() - capture_started
        candidate_final_rows = final_token_rows(len(candidates), args.capture_last_tokens)
        content_rows_by_question = content_token_rows(
            tokenizer,
            domain,
            candidates,
            args.capture_last_tokens,
            args.max_length,
            args.chat_template,
        )
        causal_seconds = 0.0
        if args.causal_filter:
            causal_started = time.time()
            final_student_outputs = {
                idx: student_outputs[idx][candidate_final_rows]
                for idx in layer_indices
            }
            final_content_deltas = {
                idx: content_deltas[idx][candidate_final_rows]
                for idx in layer_indices
            }
            weights = causal_patch_weights(
                model,
                tokenizer,
                candidates,
                layer_indices,
                final_student_outputs,
                final_content_deltas,
                device,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            causal_seconds = time.time() - causal_started
        else:
            weights = {idx: torch.ones(len(candidates), dtype=torch.float32) for idx in layer_indices}

        first_layer = layer_indices[0]
        aggregate_weights = torch.stack([weights[idx] for idx in layer_indices], dim=0).mean(dim=0)
        question_weight_multipliers = [
            args.positive_label_weight if record.answer else args.negative_label_weight
            for record in candidates
        ]
        selection_weights = aggregate_weights * torch.tensor(question_weight_multipliers, dtype=torch.float32)
        selected = select_write_questions(
            keys[first_layer][candidate_final_rows],
            candidates,
            args.write_probes,
            weights=selection_weights,
            ridge=1.0,
            balanced=args.balanced_write_selection,
            positive_fraction=args.write_selection_positive_fraction,
        )
        selected_positive_count = sum(1 for idx in selected if candidates[idx].answer)
        selected_negative_count = len(selected) - selected_positive_count
        candidate_category_counts = question_category_counts(candidates)
        selected_category_counts = question_category_counts([candidates[idx] for idx in selected])
        write_rows = selected_rows(
            selected,
            args.capture_last_tokens,
            args.write_token_selection,
            content_rows_by_question,
        )
        gate_mode = args.write_token_selection if args.gate_token_selection == "same" else args.gate_token_selection
        gate_rows = selected_rows(
            selected,
            args.capture_last_tokens,
            gate_mode,
            content_rows_by_question,
        )
        current_slot_id: int | None = None
        activation_slot_metrics: dict[str, float] = {}
        if args.slot_memory:
            slot_ids = [wrapper.add_slot_([domain.title]) for wrapper in wrappers.values()]
            if len(set(slot_ids)) != 1:
                raise RuntimeError(f"Slot ids diverged across layers: {slot_ids}")
            current_slot_id = slot_ids[0]
            if activation_router is not None:
                prototype_prompts = [
                    format_question_prompt(
                        tokenizer,
                        record.question,
                        paper=None,
                        use_chat_template=args.chat_template,
                    )
                    for record in candidates[: args.domain_latch_probes]
                ]
                activation_slot_metrics = activation_router.add_slot(current_slot_id, domain, prototype_prompts)
            if args.independent_slot_plasticity:
                for layer_idx, state in states.items():
                    state.commit(background_plasticity[layer_idx])
        if args.memory_gate and not args.safe_write_search:
            for layer_idx, wrapper in wrappers.items():
                if current_slot_id is None:
                    wrapper.set_gate_keys_(
                        keys[layer_idx][gate_rows],
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                        append=True,
                    )
                else:
                    wrapper.set_slot_gate_keys_(
                        current_slot_id,
                        keys[layer_idx][gate_rows],
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                    )

        guard_keys = None
        if args.negative_guards > 0:
            guard_prompts = negative_guard_prompts(args.seed + paper_idx, args.negative_guards)
            guard_prompts = [
                format_model_prompt(tokenizer, prompt, use_chat_template=args.chat_template)
                for prompt in guard_prompts
            ]
            guard_capture = capture_layer_io(
                model,
                tokenizer,
                guard_prompts,
                layer_indices,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.capture_last_tokens,
            )
            guard_keys = {idx: guard_capture[idx].keys for idx in layer_indices}

        a_by_layer, d_by_layer, w_by_layer = build_write_matrices(
            keys,
            content_deltas,
            weights,
            selected,
            write_rows,
            question_weight_multipliers,
            guard_keys,
            args.guard_weight,
            args.capture_last_tokens,
        )

        sentinel_prompts = negative_guard_prompts(args.seed + 99_999 + paper_idx, args.sentinel_prompts)
        write_started = time.time()
        if args.safe_write_search:
            write_result = safe_write_search(
                model,
                tokenizer,
                device,
                wrappers,
                states,
                keys,
                gate_rows,
                a_by_layer,
                d_by_layer,
                w_by_layer,
                validation_probe_sets,
                validation_gauntlet_sets,
                guard_questions,
                sentinel_prompts,
                args,
                search_path,
                paper_idx,
                slot_id=current_slot_id,
            )
        else:
            contamination_pre = evaluate_yes_no(
                model,
                tokenizer,
                guard_questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            eta_used, sentinel_kl = apply_with_sentinel_retry(
                model,
                tokenizer,
                device,
                wrappers,
                states,
                a_by_layer,
                d_by_layer,
                w_by_layer,
                sentinel_prompts,
                args.eta,
                args.sentinel_kl_threshold,
                args.sentinel_retries,
                args.max_length,
                args.chat_template,
                slot_id=current_slot_id,
            )
            write_result = {
                "write_applied": True,
                "safe_search_accepted": False,
                "validation_pre_accuracy": 0.0,
                "validation_post_accuracy": 0.0,
                "validation_accuracy_delta": 0.0,
                "validation_pre_mean_margin": 0.0,
                "validation_post_mean_margin": 0.0,
                "validation_margin_delta": 0.0,
                "validation_pre_positive_accuracy": 0.0,
                "validation_post_positive_accuracy": 0.0,
                "validation_positive_accuracy_delta": 0.0,
                "validation_pre_negative_accuracy": 0.0,
                "validation_post_negative_accuracy": 0.0,
                "validation_negative_accuracy_delta": 0.0,
                "validation_pre_positive_mean_margin": 0.0,
                "validation_post_positive_mean_margin": 0.0,
                "validation_positive_margin_delta": 0.0,
                "validation_pre_negative_mean_margin": 0.0,
                "validation_post_negative_mean_margin": 0.0,
                "validation_negative_margin_delta": 0.0,
                "validation_min_accuracy_delta": 0.0,
                "validation_gauntlet_min_bucket_delta": 0.0,
                "contamination_pre_accuracy": contamination_pre.accuracy,
                "sentinel_kl": sentinel_kl,
                "eta_used": eta_used,
                "max_update_norm_used": args.max_update_norm,
                "memory_gate_threshold_used": args.memory_gate_threshold if args.memory_gate else None,
                "safe_search_trials": 0,
                "safe_search_accepted_trials": 0,
                "safe_search_seconds": 0.0,
            }
        write_seconds = time.time() - write_started

        learned_eval_sets.append(eval_questions)
        aic_latest = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            paper=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        aic_gauntlet = {
            bucket: evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for bucket, questions in gauntlet_sets.items()
        }
        post_details = (
            evaluate_yes_no_details(
                model,
                tokenizer,
                eval_questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            if args.diagnostics
            else []
        )
        contamination_post = evaluate_yes_no(
            model,
            tokenizer,
            guard_questions,
            device,
            paper=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        retention_by_paper = retention_accuracies(
            model,
            tokenizer,
            learned_eval_sets,
            device,
            args.max_length,
            args.chat_template,
        )
        retention = sum(retention_by_paper) / len(retention_by_paper) if retention_by_paper else 0.0
        diagnostics_summary: dict[str, float] = {}
        if args.diagnostics:
            eval_gate_summary, eval_gate_per_question = gate_similarity_diagnostics(
                model,
                tokenizer,
                wrappers,
                eval_questions,
                layer_indices,
                device,
                args.batch_size,
                args.max_length,
                args.chat_template,
                args.capture_last_tokens,
                prefix="eval",
            )
            validation_gate_summary, _ = gate_similarity_diagnostics(
                model,
                tokenizer,
                wrappers,
                validation_probe_sets[0],
                layer_indices,
                device,
                args.batch_size,
                args.max_length,
                args.chat_template,
                args.capture_last_tokens,
                prefix="validation",
            )
            diagnostics_summary.update(eval_gate_summary)
            diagnostics_summary.update(validation_gate_summary)
            append_question_diagnostics(
                diagnostics_path,
                paper_idx,
                domain,
                pre_details,
                post_details,
                eval_gate_per_question,
            )
        if args.domain_latch_diagnostics:
            diagnostics_summary.update(
                activation_domain_latch_diagnostics(
                    model,
                    tokenizer,
                    domains[: paper_idx + 1],
                    paper_idx,
                    args,
                    device,
                    domain_latch_path,
                )
            )
        paper_compute = forward_counter.delta_since(paper_compute_start)

        row: dict[str, Any] = {
            "paper_idx": paper_idx,
            "domain_id": domain.domain_id,
            "title": domain.title,
            "selected_probe_count": len(selected),
            "selected_write_key_count": len(write_rows),
            "selected_gate_key_count": len(gate_rows),
            "selected_positive_probe_count": selected_positive_count,
            "selected_negative_probe_count": selected_negative_count,
            "candidate_probe_count": len(candidates),
            "base_candidate_probe_count": args.candidate_probes,
            "candidate_inverse_probe_count": args.candidate_inverse_probes,
            "candidate_minimal_pair_probe_count": args.candidate_minimal_pair_probes,
            "candidate_near_collision_probe_count": args.candidate_near_collision_probes,
            "candidate_category_counts": candidate_category_counts,
            "selected_category_counts": selected_category_counts,
            "capture_last_tokens": args.capture_last_tokens,
            "write_token_selection": args.write_token_selection,
            "gate_token_selection": gate_mode,
            "slot_memory": args.slot_memory,
            "activation_slot_routing": args.activation_slot_routing,
            "current_slot_id": current_slot_id if current_slot_id is not None else -1,
            "independent_slot_plasticity": args.independent_slot_plasticity,
            "positive_label_weight": args.positive_label_weight,
            "negative_label_weight": args.negative_label_weight,
            "balanced_write_selection": args.balanced_write_selection,
            "write_selection_positive_fraction": args.write_selection_positive_fraction,
            "target_mode": args.target_mode,
            "answer_target_scale": args.answer_target_scale,
            "validity_target_margin": args.validity_target_margin,
            "project_answer_direction": args.project_answer_direction,
            "validation_probe_count": sum(len(items) for items in validation_probe_sets),
            "validation_fold_count": len(validation_probe_sets),
            "near_collision_gauntlet": args.near_collision_gauntlet,
            "validation_gauntlet_probe_count": sum(len(items) for items in validation_gauntlet_sets.values()),
            "validation_gauntlet_min_bucket_delta": write_result["validation_gauntlet_min_bucket_delta"],
            "mean_causal_weight": float(aggregate_weights.mean().cpu()),
            "eta_used": write_result["eta_used"],
            "max_update_norm_used": write_result["max_update_norm_used"],
            "memory_gate_threshold_used": write_result["memory_gate_threshold_used"],
            "sentinel_kl": write_result["sentinel_kl"],
            "write_applied": write_result["write_applied"],
            "safe_search_accepted": write_result["safe_search_accepted"],
            "validation_pre_accuracy": write_result["validation_pre_accuracy"],
            "validation_post_accuracy": write_result["validation_post_accuracy"],
            "validation_accuracy_delta": write_result["validation_accuracy_delta"],
            "validation_pre_mean_margin": write_result["validation_pre_mean_margin"],
            "validation_post_mean_margin": write_result["validation_post_mean_margin"],
            "validation_margin_delta": write_result["validation_margin_delta"],
            "validation_pre_positive_accuracy": write_result["validation_pre_positive_accuracy"],
            "validation_post_positive_accuracy": write_result["validation_post_positive_accuracy"],
            "validation_positive_accuracy_delta": write_result["validation_positive_accuracy_delta"],
            "validation_pre_negative_accuracy": write_result["validation_pre_negative_accuracy"],
            "validation_post_negative_accuracy": write_result["validation_post_negative_accuracy"],
            "validation_negative_accuracy_delta": write_result["validation_negative_accuracy_delta"],
            "validation_pre_positive_mean_margin": write_result["validation_pre_positive_mean_margin"],
            "validation_post_positive_mean_margin": write_result["validation_post_positive_mean_margin"],
            "validation_positive_margin_delta": write_result["validation_positive_margin_delta"],
            "validation_pre_negative_mean_margin": write_result["validation_pre_negative_mean_margin"],
            "validation_post_negative_mean_margin": write_result["validation_post_negative_mean_margin"],
            "validation_negative_margin_delta": write_result["validation_negative_margin_delta"],
            "validation_min_accuracy_delta": write_result["validation_min_accuracy_delta"],
            "capture_seconds": capture_seconds,
            "causal_filter_seconds": causal_seconds,
            "write_seconds": write_seconds,
            "safe_search_trials": write_result["safe_search_trials"],
            "safe_search_accepted_trials": write_result["safe_search_accepted_trials"],
            "safe_search_seconds": write_result["safe_search_seconds"],
            "paper_forward_calls": paper_compute["forward_calls"],
            "paper_forward_tokens": paper_compute["forward_tokens"],
            "total_forward_calls": forward_counter.calls,
            "total_forward_tokens": forward_counter.tokens,
            "teacher_gate_forward_calls": teacher_gate_compute[0],
            "teacher_gate_forward_tokens": teacher_gate_compute[1],
            "seconds": time.time() - started,
            "internalization_ratio": internalization_ratio(
                aic_latest.accuracy,
                pre_no_doc.accuracy,
                context.accuracy,
            ),
            "retention_mean_accuracy": retention,
            "retention_min_accuracy": min(retention_by_paper) if retention_by_paper else 0.0,
            "retention_accuracies": retention_by_paper,
        }
        row.update(pre_no_doc.to_dict("pre_no_doc_latest"))
        row.update(context.to_dict("context_latest"))
        row.update(aic_latest.to_dict("aic_latest"))
        add_gauntlet_metrics(row, "pre_no_doc_gauntlet", pre_gauntlet)
        add_gauntlet_metrics(row, "context_gauntlet", context_gauntlet)
        add_gauntlet_metrics(row, "aic_gauntlet", aic_gauntlet)
        row.update(bias_rival_metrics)
        row.update(activation_slot_metrics)
        row["contamination_pre_accuracy"] = write_result["contamination_pre_accuracy"]
        row.update(contamination_post.to_dict("contamination_post"))
        row["contamination_accuracy_delta"] = contamination_post.accuracy - write_result["contamination_pre_accuracy"]
        row.update(diagnostics_summary)
        row.update(memory_norms(wrappers))

        for baseline_name, (baseline_model, baseline_tokenizer, baseline_device) in baseline_models.items():
            if baseline_name == "naive_text":
                train_naive_text_baseline(
                    baseline_model,
                    baseline_tokenizer,
                    domain,
                    baseline_device,
                    baseline_train_config,
                )
            elif baseline_name == "qa_lora":
                train_qa_lora_baseline(
                    baseline_model,
                    baseline_tokenizer,
                    domain,
                    candidates[: args.write_probes],
                    baseline_device,
                    baseline_train_config,
                )
            baseline_latest = evaluate_yes_no(
                baseline_model,
                baseline_tokenizer,
                eval_questions,
                baseline_device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            row.update(baseline_latest.to_dict(f"{baseline_name}_latest"))

        append_jsonl(metrics_path, row)
        write_summary_csv(metrics_path, output_dir / "summary.csv")
        plot_metrics(metrics_path, output_dir / "accuracy.png")

    print(f"Run complete. Metrics: {metrics_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
