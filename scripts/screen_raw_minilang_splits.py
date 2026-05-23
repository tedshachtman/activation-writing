"""Run raw relational writes across task/seed splits and summarize results.

The context-only screener is useful but insufficient: a split can be solved
in-context while still not crossing any closed-book item after a raw write.
This helper launches the existing continual runner once per task/seed split,
then records the before-write, after-write, and sentinel-shift metrics in a
single JSONL summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", default="runs/raw_split_screen/summary.jsonl")
    parser.add_argument("--run-dir", default="runs/raw_split_screen")
    parser.add_argument("--tasks", default="0,1,2,3")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--lessons-per-task", type=int, default=6)
    parser.add_argument("--lesson-examples", type=int, default=8)
    parser.add_argument("--teacher-filter-candidates", type=int, default=20)
    parser.add_argument("--eval-questions", type=int, default=4)
    parser.add_argument("--layers", default="4,8,12,16,20,24,27")
    parser.add_argument("--sentinel-suite", default="core")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_for_run(task_idx: int, seed: int, run_dir: Path) -> dict:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    before = next((row for row in metrics if row.get("stage") == "before_write"), {})
    sentinel = next((row for row in metrics if row.get("stage") == "sentinel_after_step" and row.get("step") == 0), {})
    after = next((row for row in metrics if row.get("stage") == "after_step" and row.get("step") == 0), {})
    return {
        "task_idx": task_idx,
        "seed": seed,
        "run_dir": str(run_dir),
        "language": before.get("language", after.get("language")),
        "teacher_filter_correct": before.get("teacher_filter_correct"),
        "teacher_filter_selected": before.get("teacher_filter_selected"),
        "baseline_correct": before.get("baseline_correct"),
        "baseline_n": before.get("baseline_n"),
        "context_correct": before.get("context_correct"),
        "context_n": before.get("context_n"),
        "edited_correct": after.get("edited_correct"),
        "edited_n": after.get("edited_n"),
        "accuracy_delta": after.get("accuracy_delta"),
        "baseline_mean_margin": before.get("baseline_mean_margin"),
        "context_mean_margin": before.get("context_mean_margin"),
        "edited_mean_margin": after.get("edited_mean_margin"),
        "sentinel_correct_to_wrong": sentinel.get("sentinel_correct_to_wrong"),
        "sentinel_before_correct_mean_margin_drop": sentinel.get("sentinel_before_correct_mean_margin_drop"),
        "sentinel_max_margin_drop": sentinel.get("sentinel_max_margin_drop"),
    }


def main() -> None:
    args = parse_args()
    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    layers = [part.strip() for part in args.layers.split(",") if part.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    for task_idx in tasks:
        for seed in seeds:
            run_dir = run_root / f"raw_task{task_idx}_seed{seed}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "minilang_intrinsic_continual.py"),
                "--model",
                args.model,
                "--output",
                str(run_dir),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--seed",
                str(seed),
                "--tasks",
                "1",
                "--task-indices",
                str(task_idx),
                "--lessons-per-task",
                str(args.lessons_per_task),
                "--lesson-examples",
                str(args.lesson_examples),
                "--teacher-filter-eval",
                "--teacher-filter-candidates",
                str(args.teacher_filter_candidates),
                "--eval-questions",
                str(args.eval_questions),
                "--layers",
                *layers,
                "--intrinsic-surprise-target-mode",
                "relational_aggregate",
                "--intrinsic-surprise-token-mode",
                "final_aligned",
                "--intrinsic-surprise-relation-value-mode",
                "context",
                "--intrinsic-surprise-key-feature-top-k",
                "16",
                "--intrinsic-surprise-target-scale",
                "0.10",
                "--intrinsic-surprise-weight-mode",
                "exponential",
                "--intrinsic-surprise-exp-temperature",
                "2.0",
                "--intrinsic-surprise-exp-cap",
                "20",
                "--intrinsic-surprise-persistence-power",
                "1.0",
                "--intrinsic-surprise-output-penalty-rank",
                "256",
                "--intrinsic-surprise-output-penalty-weight",
                "10",
                "--intrinsic-surprise-input-penalty-features",
                "256",
                "--intrinsic-surprise-input-penalty-weight",
                "20",
                "--intrinsic-surprise-input-penalty-usage-power",
                "0.0",
                "--intrinsic-target-purifier",
                "none",
                "--sentinel-eval",
                "--sentinel-suite",
                args.sentinel_suite,
            ]
            print(f"[raw-split-screen] task={task_idx} seed={seed}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            row = row_for_run(task_idx, seed, run_dir)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
