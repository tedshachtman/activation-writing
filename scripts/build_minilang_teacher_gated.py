"""Build mini-language benchmark rows with a strong context teacher.

This creates benchmark artifacts for the context-to-weights experiments where
the base model has no usable no-context knowledge but the context-conditioned
teacher solves the task. It deliberately avoids the yes/no benchmark's prior
problem: exact translation multiple choice can be truly 0/20 without context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.modeling import load_model_and_tokenizer
from scripts.minilang_write import (
    build_exhaustive_modified_questions,
    evaluate_mc,
    lesson_example_keys,
    question_key,
    render_lesson,
    trace_probe_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="runs/minilang_teacher_gated")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--accepted", type=int, default=1)
    parser.add_argument("--lessons", type=int, default=4)
    parser.add_argument("--lesson-examples", type=int, default=5)
    parser.add_argument("--eval-questions", type=int, default=20)
    parser.add_argument("--candidate-questions", type=int, default=0)
    parser.add_argument("--selection-mode", choices=["first", "teacher_filter"], default="first")
    parser.add_argument("--trace-probes", type=int, default=16)
    parser.add_argument("--allow-trace-overlap", action="store_true")
    parser.add_argument("--baseline-correct-max", type=int, default=0)
    parser.add_argument("--context-correct-min", type=int, default=20)
    parser.add_argument("--context-margin-min", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def trace_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        balanced_trace=False,
        ensemble_corpora=1,
        ensemble_include_anchor=False,
        ensemble_per_lesson=False,
        ensemble_seed_stride=100_000,
        ensemble_shared_probes=False,
        freeze_language_after=None,
        lessons=args.lessons,
        seed=seed,
        trace_probes=args.trace_probes,
    )


def build_filtered_questions(args: argparse.Namespace, seed: int, lesson_texts: list[str]):
    final_lesson_idx = args.lessons - 1
    blocked = set(lesson_example_keys(lesson_texts))
    if not args.allow_trace_overlap:
        blocked.update(trace_probe_keys(trace_args(args, seed)))
    candidates = build_exhaustive_modified_questions(
        seed + 91_000,
        final_lesson_idx,
        "heldout_translation",
    )
    random.Random(seed + 123_456).shuffle(candidates)
    if args.candidate_questions > 0:
        candidates = candidates[: args.candidate_questions]
    out = []
    seen = set()
    limit = args.eval_questions if args.selection_mode == "first" else None
    for question in candidates:
        key = question_key(question)
        if key in blocked or key in seen:
            continue
        seen.add(key)
        out.append(question)
        if limit is not None and len(out) >= limit:
            break
    return out


def metrics_from_details(details: list[dict]) -> dict:
    correct = sum(1 for detail in details if detail["correct"])
    margins = [detail["margin"] for detail in details]
    return {
        "accuracy": correct / len(details) if details else 0.0,
        "correct": correct,
        "n": len(details),
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "predictions": [detail["prediction"] for detail in details],
        "details": details,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    lessons_path = output_dir / "lessons.jsonl"
    questions_path = output_dir / "eval_questions.jsonl"
    details_path = output_dir / "eval_details.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    for path in (metrics_path, lessons_path, questions_path, details_path, accepted_path):
        if path.exists():
            path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    accepted = 0
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        lesson_texts = [
            render_lesson(idx, args.lesson_examples, seed)
            for idx in range(args.lessons)
        ]
        candidates = build_filtered_questions(args, seed, lesson_texts)
        if len(candidates) < args.eval_questions:
            append_jsonl(metrics_path, {"seed": seed, "accepted": False, "reason": "too_few_filtered_questions", "n": len(candidates)})
            continue
        full_context = "\n\n".join(lesson_texts)
        candidate_baseline = evaluate_mc(
            model,
            tokenizer,
            candidates,
            device,
            context=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        candidate_context = evaluate_mc(
            model,
            tokenizer,
            candidates,
            device,
            context=full_context,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        selected_indices = list(range(min(args.eval_questions, len(candidates))))
        eligible_count = 0
        if args.selection_mode == "teacher_filter":
            eligible = []
            for idx, (base_detail, ctx_detail) in enumerate(zip(candidate_baseline["details"], candidate_context["details"], strict=True)):
                if base_detail["correct"]:
                    continue
                if not ctx_detail["correct"]:
                    continue
                if ctx_detail["margin"] < args.context_margin_min:
                    continue
                eligible.append((ctx_detail["margin"], idx))
            eligible_count = len(eligible)
            if eligible_count < args.eval_questions:
                append_jsonl(
                    metrics_path,
                    {
                        "seed": seed,
                        "accepted": False,
                        "reason": "too_few_teacher_filtered_questions",
                        "candidate_count": len(candidates),
                        "eligible_count": eligible_count,
                        "candidate_baseline_correct": candidate_baseline["correct"],
                        "candidate_context_correct": candidate_context["correct"],
                    },
                )
                continue
            eligible.sort(reverse=True)
            selected_indices = [idx for _margin, idx in eligible[: args.eval_questions]]
        questions = [candidates[idx] for idx in selected_indices]
        baseline = metrics_from_details([candidate_baseline["details"][idx] for idx in selected_indices])
        context = metrics_from_details([candidate_context["details"][idx] for idx in selected_indices])
        ok = (
            baseline["correct"] <= args.baseline_correct_max
            and context["correct"] >= args.context_correct_min
            and context["mean_margin"] >= args.context_margin_min
        )
        row = {
            "seed": seed,
            "accepted": ok,
            "baseline_correct": baseline["correct"],
            "baseline_n": baseline["n"],
            "baseline_accuracy": baseline["accuracy"],
            "baseline_mean_margin": baseline["mean_margin"],
            "context_correct": context["correct"],
            "context_n": context["n"],
            "context_accuracy": context["accuracy"],
            "context_mean_margin": context["mean_margin"],
            "question_count": len(questions),
            "candidate_count": len(candidates),
            "eligible_count": eligible_count,
            "selection_mode": args.selection_mode,
        }
        append_jsonl(metrics_path, row)
        if not ok:
            continue
        for lesson_idx, text in enumerate(lesson_texts):
            append_jsonl(lessons_path, {"accepted_idx": accepted, "seed": seed, "lesson_idx": lesson_idx, "text": text})
        for question_idx, question in enumerate(questions):
            append_jsonl(questions_path, {
                "accepted_idx": accepted,
                "seed": seed,
                "question_idx": question_idx,
                "sentence": question.sentence,
                "answer": question.answer,
                "options": question.options,
                "answer_letter": question.answer_letter,
                "category": question.category,
            })
        for stage, metrics in (("baseline", baseline), ("context", context)):
            for idx, detail in enumerate(metrics["details"]):
                append_jsonl(details_path, {"accepted_idx": accepted, "seed": seed, "stage": stage, "idx": idx, **detail})
        append_jsonl(accepted_path, {"accepted_idx": accepted, **row})
        accepted += 1
        if accepted >= args.accepted:
            break
    print(f"Wrote {accepted} accepted mini-language benchmark(s) to {output_dir}")


if __name__ == "__main__":
    main()
