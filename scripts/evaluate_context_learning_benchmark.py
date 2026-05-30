"""Evaluate a model on a context-learning benchmark fixture.

This script performs no weight writes. It reports no-context baseline,
full-context teacher performance, and optional sentinel performance for a
benchmark produced by scripts/build_context_learning_benchmark.py.
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
from scripts.build_context_learning_benchmark import evaluate_generic_mc_no_cache, write_json
from scripts.minilang_continual_triangle import release_device_cache


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add_prefix(row: dict, prefix: str, metrics: dict) -> None:
    row[f"{prefix}_correct"] = metrics["correct"]
    row[f"{prefix}_n"] = metrics["n"]
    row[f"{prefix}_accuracy"] = metrics["accuracy"]
    row[f"{prefix}_mean_margin"] = metrics["mean_margin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="benchmarks/context_learning_v1_qwen35_0_8b")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    manifest = json.loads((benchmark_dir / "manifest.json").read_text())
    output_path = Path(args.output) if args.output else benchmark_dir / "eval_metrics.json"

    started = time.time()
    print(f"[benchmark-eval] loading {args.model}", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    print(f"[benchmark-eval] loaded on {device}", flush=True)

    task_rows = []
    total_baseline_correct = 0
    total_context_correct = 0
    total_n = 0
    for task in manifest["tasks"]:
        task_dir = benchmark_dir / "tasks" / task["task_id"]
        context = (task_dir / "context.txt").read_text().strip()
        items = read_jsonl(task_dir / "eval.jsonl")
        baseline = evaluate_generic_mc_no_cache(
            model,
            tokenizer,
            items,
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)
        context_items = []
        for item in items:
            row = dict(item)
            row["prompt"] = context + "\n\n" + row["prompt"]
            context_items.append(row)
        context_metrics = evaluate_generic_mc_no_cache(
            model,
            tokenizer,
            context_items,
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)
        row = {
            "task_id": task["task_id"],
            "task_family": task["task_family"],
            "seed": task["seed"],
        }
        add_prefix(row, "baseline", baseline)
        add_prefix(row, "context", context_metrics)
        task_rows.append(row)
        total_baseline_correct += baseline["correct"]
        total_context_correct += context_metrics["correct"]
        total_n += baseline["n"]
        print(json.dumps(row, sort_keys=True), flush=True)

    sentinel_metrics = None
    sentinel_file = manifest.get("sentinel_file")
    if sentinel_file:
        sentinels = read_jsonl(benchmark_dir / sentinel_file)
        sentinel_metrics = evaluate_generic_mc_no_cache(
            model,
            tokenizer,
            sentinels,
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "task_count": len(task_rows),
        "item_count": total_n,
        "baseline_correct": total_baseline_correct,
        "baseline_accuracy": total_baseline_correct / total_n if total_n else 0.0,
        "context_correct": total_context_correct,
        "context_accuracy": total_context_correct / total_n if total_n else 0.0,
        "tasks": task_rows,
        "sentinel": sentinel_metrics,
        "runtime_seconds": time.time() - started,
    }
    write_json(output_path, summary)
    print(json.dumps({"output": str(output_path), "baseline": total_baseline_correct, "context": total_context_correct, "n": total_n}, sort_keys=True))


if __name__ == "__main__":
    main()
