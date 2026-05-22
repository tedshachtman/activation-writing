"""Audit saved mini-language runs for false-positive paths.

This script does not rerun the model. It analyzes saved `eval_details.jsonl`,
`eval_questions.jsonl`, `lessons.jsonl`, and `config.json` artifacts so the
headline result can be checked reproducibly even when local model inference is
too memory-heavy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from math import comb
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.minilang_write import (  # noqa: E402
    build_balanced_questions,
    build_questions,
    lesson_example_keys,
    question_key,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trace_keys_from_config(config: dict) -> set[tuple[str, str]]:
    class Args:
        pass

    args = Args()
    for key, value in config.items():
        setattr(args, key, value)
    build_trace_questions = build_balanced_questions if config.get("balanced_trace") else build_questions
    keys: set[tuple[str, str]] = set()
    for lesson_idx in range(config["lessons"]):
        trace_language_idx = (
            min(lesson_idx, config["freeze_language_after"])
            if config.get("freeze_language_after") is not None
            else lesson_idx
        )
        for question in build_trace_questions(
            config["trace_probes"],
            config["seed"] + lesson_idx * 10_000,
            lesson_idx,
            "trace_translation",
            language_idx=trace_language_idx,
        ):
            keys.add(question_key(question))
    return keys


def load_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    questions = read_jsonl(run_dir / "eval_questions.jsonl")
    details = read_jsonl(run_dir / "eval_details.jsonl")
    lessons = read_jsonl(run_dir / "lessons.jsonl")
    lesson_keys = lesson_example_keys([row["text"] for row in lessons])
    trace_keys = trace_keys_from_config(config)
    return {
        "run_dir": str(run_dir),
        "config": config,
        "questions": questions,
        "details": details,
        "lesson_keys": lesson_keys,
        "trace_keys": trace_keys,
    }


def rows_for_stage(run: dict, stage: str) -> list[dict]:
    return [row for row in run["details"] if row.get("stage") == stage]


def score(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    return {"correct": correct, "n": n, "accuracy": correct / n if n else 0.0}


def collapse_unique(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in rows:
        key = (row["sentence"], row["answer"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def exclude_overlaps(rows: list[dict], keys: set[tuple[str, str]]) -> list[dict]:
    return [row for row in rows if (row["sentence"], row["answer"]) not in keys]


def run_summary(run: dict) -> dict:
    questions = run["questions"]
    question_keys = [(row["sentence"], row["answer"]) for row in questions]
    duplicate_count = len(question_keys) - len(set(question_keys))
    lesson_overlap = sum(1 for key in set(question_keys) if key in run["lesson_keys"])
    trace_overlap = sum(1 for key in set(question_keys) if key in run["trace_keys"])
    stages = {}
    for stage in ("baseline", "context", "edited"):
        rows = rows_for_stage(run, stage)
        if not rows:
            continue
        unique_rows = collapse_unique(rows)
        no_lesson = exclude_overlaps(unique_rows, run["lesson_keys"])
        no_trace = exclude_overlaps(unique_rows, run["trace_keys"])
        clean = exclude_overlaps(no_lesson, run["trace_keys"])
        stages[stage] = {
            "raw": score(rows),
            "unique": score(unique_rows),
            "unique_no_lesson_overlap": score(no_lesson),
            "unique_no_trace_overlap": score(no_trace),
            "unique_no_lesson_or_trace_overlap": score(clean),
            "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
            "answer_letter_counts": dict(Counter(row["answer_letter"] for row in rows)),
        }
    return {
        "run_dir": run["run_dir"],
        "duplicate_question_count": duplicate_count,
        "unique_question_count": len(set(question_keys)),
        "lesson_overlap_unique_count": lesson_overlap,
        "trace_overlap_unique_count": trace_overlap,
        "stages": stages,
    }


def exact_sign_tail(successes: int, failures: int) -> float:
    n = successes + failures
    if n == 0:
        return 1.0
    return sum(comb(n, k) * 0.5**n for k in range(successes, n + 1))


def paired_comparison(run: dict, control: dict, stage: str = "edited") -> dict:
    left = rows_for_stage(run, stage)
    right = rows_for_stage(control, stage)
    if len(left) != len(right):
        raise ValueError(f"Cannot pair runs with different row counts: {len(left)} vs {len(right)}")
    left_only = sum(1 for a, b in zip(left, right) if a["correct"] and not b["correct"])
    right_only = sum(1 for a, b in zip(left, right) if b["correct"] and not a["correct"])

    def by_unique(rows: list[dict]) -> dict[tuple[str, str], bool]:
        out = {}
        for row in rows:
            out.setdefault((row["sentence"], row["answer"]), row["correct"])
        return out

    left_unique = by_unique(left)
    right_unique = by_unique(right)
    common = sorted(set(left_unique) & set(right_unique))
    unique_left_only = sum(1 for key in common if left_unique[key] and not right_unique[key])
    unique_right_only = sum(1 for key in common if right_unique[key] and not left_unique[key])
    return {
        "stage": stage,
        "item_left_only": left_only,
        "item_control_only": right_only,
        "item_one_sided_p": exact_sign_tail(left_only, right_only),
        "unique_left_only": unique_left_only,
        "unique_control_only": unique_right_only,
        "unique_one_sided_p": exact_sign_tail(unique_left_only, unique_right_only),
    }


def post_selection_summary(pattern: str) -> dict:
    values = []
    for metrics_path in glob.glob(pattern):
        path = Path(metrics_path)
        rows = read_jsonl(path)
        if not rows:
            continue
        last = rows[-1]
        if last.get("edited_n") != 16:
            continue
        values.append({"run_dir": str(path.parent), "edited_correct": last.get("edited_correct")})
    values = [row for row in values if row["edited_correct"] is not None]
    if not values:
        return {"pattern": pattern, "runs": 0}
    best = max(row["edited_correct"] for row in values)
    return {
        "pattern": pattern,
        "runs": len(values),
        "best_correct": best,
        "runs_at_or_above_11": sum(1 for row in values if row["edited_correct"] >= 11),
        "mean_correct": sum(row["edited_correct"] for row in values) / len(values),
        "top_runs": sorted(values, key=lambda row: row["edited_correct"], reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--control")
    parser.add_argument("--glob", default="runs/minilang_*/metrics.jsonl")
    parser.add_argument("--output", default="runs/minilang_saved_audit.json")
    args = parser.parse_args()

    run = load_run(Path(args.run))
    result = {
        "run": run_summary(run),
        "post_selection": post_selection_summary(args.glob),
    }
    if args.control:
        control = load_run(Path(args.control))
        result["control"] = run_summary(control)
        result["paired_vs_control"] = paired_comparison(run, control)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
