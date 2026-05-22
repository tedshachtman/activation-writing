"""Summarize intrinsic continual-learning runs.

The primary research scoreboard is task acquisition/retention under sequential
writes plus sentinel preservation. This script reads a run directory containing
``metrics.jsonl`` and emits the compact table we need for comparing methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_run_dir(path: Path) -> Path:
    if (path / "metrics.jsonl").exists():
        return path
    nested = path / path.name
    if (nested / "metrics.jsonl").exists():
        return nested
    raise FileNotFoundError(f"No metrics.jsonl found under {path}")


def _fmt_acc(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def summarize_run(path: Path) -> None:
    run_dir = _resolve_run_dir(path)
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    name = path.name
    print(f"\n## {name}")

    before = [row for row in rows if row.get("stage") == "before_write"]
    if before:
        print("task\tbaseline\tcontext\tfiltered")
        for row in before:
            print(
                f"{row['task_idx']}:{row['language']}\t"
                f"{_fmt_acc(row.get('baseline_accuracy'))}\t"
                f"{_fmt_acc(row.get('context_accuracy'))}\t"
                f"{row.get('teacher_filter_selected', '-')}"
            )

    after = [row for row in rows if row.get("stage") == "after_step"]
    if after:
        print("\nretention")
        print("step\ttask\tedited\tdelta\tretain_delta\tmargin\tretain_margin")
        for row in after:
            print(
                f"{row['step']}\t"
                f"{row['task_idx']}:{row['language']}\t"
                f"{_fmt_acc(row.get('edited_accuracy'))}\t"
                f"{_fmt_acc(row.get('accuracy_delta'))}\t"
                f"{_fmt_acc(row.get('retention_accuracy_delta_from_acquisition'))}\t"
                f"{_fmt_acc(row.get('edited_mean_margin'))}\t"
                f"{_fmt_acc(row.get('retention_margin_delta_from_acquisition'))}"
            )

    sent = [row for row in rows if row.get("stage") == "sentinel_after_step"]
    if sent:
        print("\nsentinel")
        print("step\tacc\tacc_delta\tc2w\tw2c\tmargin_delta\tbefore_correct_drop")
        for row in sent:
            print(
                f"{row['step']}\t"
                f"{_fmt_acc(row.get('sentinel_after_accuracy'))}\t"
                f"{_fmt_acc(row.get('sentinel_accuracy_delta'))}\t"
                f"{row.get('sentinel_correct_to_wrong', '-')}\t"
                f"{row.get('sentinel_wrong_to_correct', '-')}\t"
                f"{_fmt_acc(row.get('sentinel_margin_delta'))}\t"
                f"{_fmt_acc(row.get('sentinel_before_correct_mean_margin_drop'))}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        summarize_run(run)


if __name__ == "__main__":
    main()
