"""Standalone gradient baselines for existing CAIC domain runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Baselines require PyTorch. Install with: pip install -e \".[dev]\"") from exc

from .baselines import (
    TrainConfig,
    paper_text_examples,
    prepare_qa_lora_model,
    qa_examples,
    set_trainable,
    train_prompt_completion,
)
from .evaluation import evaluate_yes_no, format_question_prompt
from .modeling import ForwardCounter, load_model_and_tokenizer
from .synthetic import (
    domain_from_dict,
    general_guard_questions,
    make_candidate_probes,
    question_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baselines against domains from a CAIC run directory.")
    parser.add_argument("--run-dir", required=True, help="CAIC run directory containing domains.jsonl.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--baseline",
        choices=["qa_lora", "text_lora", "naive_text_full"],
        default="qa_lora",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--candidate-probes", type=int, default=48)
    parser.add_argument("--qa-lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_domains(run_dir: Path) -> tuple[list[Any], list[list[Any]]]:
    domains = []
    eval_sets = []
    with (run_dir / "domains.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            domains.append(domain_from_dict(row["domain"]))
            eval_sets.append([question_from_dict(item) for item in row["eval_questions"]])
    return domains, eval_sets


def mean_retention(
    model: Any,
    tokenizer: Any,
    eval_sets: list[list[Any]],
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
) -> float:
    if not eval_sets:
        return 0.0
    accs = [
        evaluate_yes_no(
            model,
            tokenizer,
            questions,
            device,
            max_length=max_length,
            use_chat_template=use_chat_template,
        ).accuracy
        for questions in eval_sets
    ]
    return sum(accs) / len(accs)


def prepare_model(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    if args.baseline in {"qa_lora", "text_lora"}:
        model = prepare_qa_lora_model(model, r=args.qa_lora_r)
        model.to(device)
    elif args.baseline == "naive_text_full":
        set_trainable(model, True)
    return model, tokenizer, device


def training_examples(args: argparse.Namespace, tokenizer: Any, domain: Any, paper_idx: int) -> list[tuple[str, str]]:
    if args.baseline == "qa_lora":
        candidates = make_candidate_probes(
            domain,
            args.candidate_probes,
            seed=args.seed * 100_000 + paper_idx,
        )
        if args.chat_template:
            return [
                (
                    format_question_prompt(
                        tokenizer,
                        record.question,
                        paper=None,
                        use_chat_template=True,
                    ),
                    f" {record.answer_text}",
                )
                for record in candidates
            ]
        return qa_examples(domain, candidates)
    return paper_text_examples(domain)


def run() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / f"{args.baseline}_baseline.jsonl"
    if output_path.exists():
        output_path.unlink()

    domains, eval_sets = load_domains(run_dir)
    model, tokenizer, device = prepare_model(args)
    counter = ForwardCounter(model).install()
    config = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    guard_questions = general_guard_questions()
    learned_eval_sets: list[list[Any]] = []

    for paper_idx, (domain, eval_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        compute_start = counter.snapshot()
        pre = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        examples = training_examples(args, tokenizer, domain, paper_idx)
        train_started = time.time()
        train_prompt_completion(model, tokenizer, examples, device, config)
        train_seconds = time.time() - train_started
        post = evaluate_yes_no(
            model,
            tokenizer,
            eval_questions,
            device,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        learned_eval_sets.append(eval_questions)
        retention = mean_retention(
            model,
            tokenizer,
            learned_eval_sets,
            device,
            args.max_length,
            args.chat_template,
        )
        contamination = evaluate_yes_no(
            model,
            tokenizer,
            guard_questions,
            device,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        compute = counter.delta_since(compute_start)
        row = {
            "baseline": args.baseline,
            "paper_idx": paper_idx,
            "domain_id": domain.domain_id,
            "title": domain.title,
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "candidate_probe_count": args.candidate_probes if args.baseline == "qa_lora" else 0,
            "train_example_count": len(examples),
            "train_seconds": train_seconds,
            "seconds": time.time() - started,
            "paper_forward_calls": compute["forward_calls"],
            "paper_forward_tokens": compute["forward_tokens"],
            "total_forward_calls": counter.calls,
            "total_forward_tokens": counter.tokens,
            "retention_mean_accuracy": retention,
        }
        row.update(pre.to_dict("pre_latest"))
        row.update(post.to_dict(f"{args.baseline}_latest"))
        row.update(contamination.to_dict("contamination"))
        append_jsonl(output_path, row)

    print(f"Baseline complete. Metrics: {output_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
