"""Probe whether no-document write keys linearly encode synthetic labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from caic.evaluation import format_question_prompt
from caic.experiment import build_candidate_pool, load_domain_rows
from caic.modeling import capture_layer_io, load_model_and_tokenizer
from caic.synthetic import make_gauntlet_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--domains-jsonl", required=True)
    parser.add_argument("--papers", type=int, default=2)
    parser.add_argument("--output", default="runs/key_separability")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--capture-last-tokens", type=int, default=12)
    parser.add_argument("--candidate-probes", type=int, default=32)
    parser.add_argument("--candidate-inverse-probes", type=int, default=32)
    parser.add_argument("--candidate-minimal-pair-probes", type=int, default=16)
    parser.add_argument("--candidate-near-collision-probes", type=int, default=16)
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--label-mode", choices=["answer", "validity"], default="answer")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


def prompts_for_questions(tokenizer, questions: list, use_chat_template: bool) -> list[str]:
    return [
        format_question_prompt(tokenizer, record.question, paper=None, use_chat_template=use_chat_template)
        for record in questions
    ]


def labels_for_questions(domain, questions: list, label_mode: str) -> torch.Tensor:
    labels: list[float] = []
    for record in questions:
        if label_mode == "answer" or not record.chain:
            label = record.answer
        elif label_mode == "validity":
            label, _failures = domain.validate(record.chain)
        else:
            raise ValueError(f"Unknown label mode: {label_mode}")
        labels.append(1.0 if label else -1.0)
    return torch.tensor(labels, dtype=torch.float32)


def capture_features(
    model,
    tokenizer,
    questions: list,
    layers: list[int],
    device: torch.device,
    batch_size: int,
    max_length: int,
    capture_last_tokens: int,
    use_chat_template: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    prompts = prompts_for_questions(tokenizer, questions, use_chat_template)
    captures = capture_layer_io(
        model,
        tokenizer,
        prompts,
        layers,
        device,
        batch_size,
        max_length,
        capture_last_tokens=capture_last_tokens,
    )
    features: dict[int, dict[str, torch.Tensor]] = {}
    for layer, capture in captures.items():
        keys = capture.keys.reshape(len(questions), capture_last_tokens, -1).float()
        outputs = capture.outputs.reshape(len(questions), capture_last_tokens, -1).float()
        features[layer] = {
            "key_final": keys[:, -1, :],
            "key_suffix_mean": keys.mean(dim=1),
            "key_suffix_flat": keys.reshape(len(questions), -1),
            "output_final": outputs[:, -1, :],
            "output_suffix_mean": outputs.mean(dim=1),
            "output_suffix_flat": outputs.reshape(len(questions), -1),
        }
    return features


def normalize_train_eval(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-3)
    train_z = (train_x - mean) / scale
    eval_z = (eval_x - mean) / scale
    return train_z, eval_z


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)


def fit_dual_ridge(train_x: torch.Tensor, train_y: torch.Tensor, ridge: float) -> torch.Tensor:
    x = add_bias(train_x)
    system = x @ x.T + ridge * torch.eye(x.shape[0])
    alpha = torch.linalg.pinv(system) @ train_y
    return x.T @ alpha


def evaluate_probe(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor, ridge: float) -> dict[str, float]:
    train_z, eval_z = normalize_train_eval(train_x, eval_x)
    w = fit_dual_ridge(train_z, train_y, ridge)
    scores = add_bias(eval_z) @ w
    preds = torch.where(scores >= 0.0, torch.tensor(1.0), torch.tensor(-1.0))
    correct = preds == eval_y
    positive = eval_y > 0
    negative = eval_y < 0
    return {
        "accuracy": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "positive_accuracy": float(correct[positive].float().mean().item()) if bool(positive.any()) else 0.0,
        "negative_accuracy": float(correct[negative].float().mean().item()) if bool(negative.any()) else 0.0,
        "mean_signed_score": float((scores * eval_y).mean().item()) if scores.numel() else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    results_path = output_dir / "key_separability.jsonl"
    if results_path.exists():
        results_path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)
    layers = args.layers if args.layers is not None else [args.layer]

    for paper_idx, (domain, eval_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        train_questions = build_candidate_pool(domain, args, paper_idx)
        gauntlet_sets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": eval_questions, **gauntlet_sets}
        all_eval_questions = [record for questions in eval_groups.values() for record in questions]

        train_features = capture_features(
            model,
            tokenizer,
            train_questions,
            layers,
            device,
            args.batch_size,
            args.max_length,
            args.capture_last_tokens,
            args.chat_template,
        )
        eval_features = capture_features(
            model,
            tokenizer,
            all_eval_questions,
            layers,
            device,
            args.batch_size,
            args.max_length,
            args.capture_last_tokens,
            args.chat_template,
        )
        train_y = labels_for_questions(domain, train_questions, args.label_mode)

        offset = 0
        eval_slices: dict[str, tuple[int, int, torch.Tensor]] = {}
        for name, questions in eval_groups.items():
            next_offset = offset + len(questions)
            eval_slices[name] = (
                offset,
                next_offset,
                labels_for_questions(domain, questions, args.label_mode),
            )
            offset = next_offset

        for layer in layers:
            for mode, train_x in train_features[layer].items():
                eval_x_all = eval_features[layer][mode]
                train_result = evaluate_probe(train_x, train_y, train_x, train_y, args.ridge)
                row = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "layer": layer,
                    "capture_last_tokens": args.capture_last_tokens,
                    "feature_mode": mode,
                    "label_mode": args.label_mode,
                    "train_n": len(train_questions),
                    "seconds": time.time() - started,
                    "train_accuracy": train_result["accuracy"],
                    "train_positive_accuracy": train_result["positive_accuracy"],
                    "train_negative_accuracy": train_result["negative_accuracy"],
                    "train_mean_signed_score": train_result["mean_signed_score"],
                }
                for group_name, (start, end, eval_y) in eval_slices.items():
                    result = evaluate_probe(train_x, train_y, eval_x_all[start:end], eval_y, args.ridge)
                    for key, value in result.items():
                        row[f"{group_name}_{key}"] = value
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote key separability metrics to {results_path}")


if __name__ == "__main__":
    main()
