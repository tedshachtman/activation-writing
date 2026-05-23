"""Screen mini-language task/seed splits before running expensive writes.

This loads the model once, builds teacher-filtered eval sets for multiple
task-profile/seed pairs, and reports baseline versus full-context performance.
It is intended to find acquisition-informative local gates where raw writes can
be meaningfully compared against DICE variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.modeling import load_model_and_tokenizer
from scripts.minilang_continual_triangle import (
    build_task_questions,
    evaluate_task_mc,
    release_device_cache,
    render_task_lesson,
    task_profile,
)
from scripts.minilang_write import append_jsonl


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", default="runs/minilang_split_screen.jsonl")
    parser.add_argument("--tasks", default="0,1,2,3")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--lessons-per-task", type=int, default=6)
    parser.add_argument("--lesson-examples", type=int, default=8)
    parser.add_argument("--teacher-filter-candidates", type=int, default=20)
    parser.add_argument("--eval-questions", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    final_lesson_idx = args.lessons_per_task - 1

    started = time.time()
    print("[split-screen] loading model", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    print(f"[split-screen] loaded model on {device}", flush=True)

    for task_idx in tasks:
        profile = task_profile(task_idx)
        for seed in seeds:
            print(f"[split-screen] task={task_idx} language={profile.name} seed={seed}", flush=True)
            lessons = [
                render_task_lesson(profile, lesson_idx, args.lesson_examples, seed)
                for lesson_idx in range(args.lessons_per_task)
            ]
            context = "\n\n".join(lessons)
            candidates = build_task_questions(
                profile,
                args.teacher_filter_candidates,
                seed + 91_000,
                final_lesson_idx,
                "heldout_translation",
            )
            context_candidates = evaluate_task_mc(
                model,
                tokenizer,
                profile,
                candidates,
                device,
                context=context,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            release_device_cache(device)
            filtered = [
                question
                for question, detail in zip(candidates, context_candidates["details"], strict=True)
                if detail["correct"]
            ]
            eval_questions = filtered[: args.eval_questions]
            if not eval_questions:
                row = {
                    "task_idx": task_idx,
                    "language": profile.name,
                    "seed": seed,
                    "teacher_filter_candidates": len(candidates),
                    "teacher_filter_correct": len(filtered),
                    "teacher_filter_selected": 0,
                    "seconds": time.time() - started,
                }
                append_jsonl(output_path, row)
                print(json.dumps(row, sort_keys=True), flush=True)
                continue
            baseline = evaluate_task_mc(
                model,
                tokenizer,
                profile,
                eval_questions,
                device,
                context=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            release_device_cache(device)
            context_metrics = evaluate_task_mc(
                model,
                tokenizer,
                profile,
                eval_questions,
                device,
                context=context,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            release_device_cache(device)
            row = {
                "task_idx": task_idx,
                "language": profile.name,
                "seed": seed,
                "teacher_filter_candidates": len(candidates),
                "teacher_filter_correct": len(filtered),
                "teacher_filter_selected": len(eval_questions),
                "baseline_correct": baseline["correct"],
                "baseline_n": baseline["n"],
                "baseline_accuracy": baseline["accuracy"],
                "baseline_mean_margin": baseline["mean_margin"],
                "context_correct": context_metrics["correct"],
                "context_n": context_metrics["n"],
                "context_accuracy": context_metrics["accuracy"],
                "context_mean_margin": context_metrics["mean_margin"],
                "seconds": time.time() - started,
            }
            append_jsonl(output_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
