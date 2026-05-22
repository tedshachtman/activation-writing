"""Sweep direct activation patches for candidate CAIC targets.

This is a controllability diagnostic. It patches proposed target deltas into
MLP down-projection outputs during evaluation, without writing weights. If a
target cannot move answers by direct activation patching, a closed-form memory
write that approximates the same target is unlikely to work.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Iterator

import torch

from caic.evaluation import answer_margin, format_question_prompt, yes_no_logprobs
from caic.experiment import (
    answer_unembedding_direction,
    build_candidate_pool,
    final_token_rows,
    fit_linear_probe_direction,
    load_domain_rows,
)
from caic.modeling import capture_layer_io, get_decoder_layers, get_mlp_down_module, load_model_and_tokenizer
from caic.synthetic import make_gauntlet_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--domains-jsonl", required=True)
    parser.add_argument("--papers", type=int, default=1)
    parser.add_argument("--output", default="runs/target_patch_sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 22])
    parser.add_argument("--capture-last-tokens", type=int, default=12)
    parser.add_argument("--patch-modes", nargs="+", choices=["final", "suffix"], default=["final", "suffix"])
    parser.add_argument(
        "--target-modes",
        nargs="+",
        choices=["answer_direction", "validity_probe"],
        default=["answer_direction", "validity_probe"],
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--validity-target-margin", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--candidate-probes", type=int, default=32)
    parser.add_argument("--candidate-inverse-probes", type=int, default=32)
    parser.add_argument("--candidate-minimal-pair-probes", type=int, default=16)
    parser.add_argument("--candidate-near-collision-probes", type=int, default=16)
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--max-eval-per-group", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


@contextmanager
def patched_suffix_output(
    model,
    layer_idx: int,
    replacement: torch.Tensor,
    device: torch.device,
) -> Iterator[None]:
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    module = get_mlp_down_module(layers[resolved])

    def hook(_module, _inputs, output):
        patched = output.clone()
        repl = replacement.to(device=device, dtype=patched.dtype)
        if repl.ndim == 2:
            repl = repl.unsqueeze(0)
        token_count = min(repl.shape[1], patched.shape[1])
        patched[:, -token_count:, :] = repl[:, -token_count:, :]
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def prompts_for_questions(tokenizer, questions: list, use_chat_template: bool) -> list[str]:
    return [
        format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=use_chat_template)
        for record in questions
    ]


def validity_labels(domain, questions: list) -> torch.Tensor:
    labels = []
    for record in questions:
        valid, _failures = domain.validate(record.chain)
        labels.append(1.0 if valid else -1.0)
    return torch.tensor(labels, dtype=torch.float32)


def answer_labels(questions: list) -> torch.Tensor:
    return torch.tensor([1.0 if record.answer else -1.0 for record in questions], dtype=torch.float32)


def answer_direction_targets(
    model,
    tokenizer,
    questions: list,
    capture_last_tokens: int,
    out_dim: int,
) -> torch.Tensor:
    direction = answer_unembedding_direction(model, tokenizer).float()
    if direction.numel() != out_dim:
        raise ValueError(f"Answer direction dim {direction.numel()} does not match output dim {out_dim}.")
    unit = torch.nn.functional.normalize(direction, dim=0)
    labels = answer_labels(questions)
    rows = [label * unit for label in labels for _ in range(capture_last_tokens)]
    return torch.stack(rows, dim=0)


def validity_probe_targets(
    train_outputs: torch.Tensor,
    train_questions: list,
    eval_outputs: torch.Tensor,
    eval_questions: list,
    domain,
    capture_last_tokens: int,
    ridge: float,
    margin: float,
) -> torch.Tensor:
    train_final_rows = final_token_rows(len(train_questions), capture_last_tokens)
    train_labels = validity_labels(domain, train_questions)
    probe, bias = fit_linear_probe_direction(train_outputs[train_final_rows], train_labels, ridge)
    norm_sq = float(torch.dot(probe, probe).item())
    if norm_sq <= 1e-12:
        return torch.zeros_like(eval_outputs)
    eval_question_labels = validity_labels(domain, eval_questions)
    row_labels = torch.repeat_interleave(eval_question_labels, capture_last_tokens)
    scores = eval_outputs.float() @ probe + bias
    signed_scores = scores * row_labels
    gaps = torch.clamp(margin - signed_scores, min=0.0)
    return (gaps * row_labels / (norm_sq + 1e-12)).unsqueeze(1) * probe.unsqueeze(0)


def rows_for_patch(
    outputs: torch.Tensor,
    deltas: torch.Tensor,
    question_idx: int,
    capture_last_tokens: int,
    patch_mode: str,
    scale: float,
) -> torch.Tensor:
    start = question_idx * capture_last_tokens
    end = start + capture_last_tokens
    patched = outputs[start:end].clone()
    delta = deltas[start:end]
    if patch_mode == "final":
        patched[-1] = patched[-1] + scale * delta[-1]
    elif patch_mode == "suffix":
        patched = patched + scale * delta
    else:
        raise ValueError(f"Unknown patch mode: {patch_mode}")
    return patched


def eval_questions_with_optional_patch(
    model,
    tokenizer,
    questions: list,
    prompts: list[str],
    device: torch.device,
    max_length: int,
    layer_idx: int | None = None,
    replacements: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    correct = 0
    pos_correct = 0
    neg_correct = 0
    margins = []
    pos_margins = []
    neg_margins = []
    for idx, (record, prompt) in enumerate(zip(questions, prompts)):
        if layer_idx is None:
            yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        else:
            assert replacements is not None
            with patched_suffix_output(model, layer_idx, replacements[idx], device):
                yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        pred = yes_lp >= no_lp
        is_correct = pred == record.answer
        correct += int(is_correct)
        margin = answer_margin(yes_lp, no_lp, record.answer)
        margins.append(margin)
        if record.answer:
            pos_correct += int(is_correct)
            pos_margins.append(margin)
        else:
            neg_correct += int(is_correct)
            neg_margins.append(margin)
    n = len(questions)
    pos_n = len(pos_margins)
    neg_n = len(neg_margins)
    return {
        "accuracy": correct / n if n else 0.0,
        "positive_accuracy": pos_correct / pos_n if pos_n else 0.0,
        "negative_accuracy": neg_correct / neg_n if neg_n else 0.0,
        "mean_margin": sum(margins) / n if n else 0.0,
        "positive_mean_margin": sum(pos_margins) / pos_n if pos_n else 0.0,
        "negative_mean_margin": sum(neg_margins) / neg_n if neg_n else 0.0,
    }


def add_prefixed(row: dict, prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    results_path = output_dir / "target_patch_sweep.jsonl"
    if results_path.exists():
        results_path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)

    for paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        train_questions = build_candidate_pool(domain, args, paper_idx)
        gauntlet_sets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": heldout_questions, **gauntlet_sets}
        eval_groups = {
            name: questions[: args.max_eval_per_group]
            for name, questions in eval_groups.items()
            if questions
        }

        train_prompts = prompts_for_questions(tokenizer, train_questions, args.chat_template)
        train_capture = capture_layer_io(
            model,
            tokenizer,
            train_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.capture_last_tokens,
        )

        for group_name, questions in eval_groups.items():
            prompts = prompts_for_questions(tokenizer, questions, args.chat_template)
            baseline = eval_questions_with_optional_patch(
                model,
                tokenizer,
                questions,
                prompts,
                device,
                args.max_length,
            )
            eval_capture = capture_layer_io(
                model,
                tokenizer,
                prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.capture_last_tokens,
            )

            for layer_idx in args.layers:
                eval_outputs = eval_capture[layer_idx].outputs.float()
                train_outputs = train_capture[layer_idx].outputs.float()
                targets_by_mode = {}
                if "answer_direction" in args.target_modes:
                    targets_by_mode["answer_direction"] = answer_direction_targets(
                        model,
                        tokenizer,
                        questions,
                        args.capture_last_tokens,
                        eval_outputs.shape[-1],
                    )
                if "validity_probe" in args.target_modes:
                    targets_by_mode["validity_probe"] = validity_probe_targets(
                        train_outputs,
                        train_questions,
                        eval_outputs,
                        questions,
                        domain,
                        args.capture_last_tokens,
                        args.ridge,
                        args.validity_target_margin,
                    )

                for target_mode, deltas in targets_by_mode.items():
                    for patch_mode in args.patch_modes:
                        for scale in args.scales:
                            replacements = [
                                rows_for_patch(
                                    eval_outputs,
                                    deltas,
                                    question_idx,
                                    args.capture_last_tokens,
                                    patch_mode,
                                    scale,
                                )
                                for question_idx in range(len(questions))
                            ]
                            patched = eval_questions_with_optional_patch(
                                model,
                                tokenizer,
                                questions,
                                prompts,
                                device,
                                args.max_length,
                                layer_idx=layer_idx,
                                replacements=replacements,
                            )
                            row = {
                                "paper_idx": paper_idx,
                                "domain_id": domain.domain_id,
                                "title": domain.title,
                                "group": group_name,
                                "layer": layer_idx,
                                "target_mode": target_mode,
                                "patch_mode": patch_mode,
                                "scale": scale,
                                "question_count": len(questions),
                                "seconds": time.time() - started,
                            }
                            add_prefixed(row, "baseline", baseline)
                            add_prefixed(row, "patched", patched)
                            row["accuracy_delta"] = patched["accuracy"] - baseline["accuracy"]
                            row["positive_accuracy_delta"] = (
                                patched["positive_accuracy"] - baseline["positive_accuracy"]
                            )
                            row["negative_accuracy_delta"] = (
                                patched["negative_accuracy"] - baseline["negative_accuracy"]
                            )
                            with results_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote target patch sweep metrics to {results_path}")


if __name__ == "__main__":
    main()
