"""Run raw relational/context-value writes on a context-learning benchmark.

This is the first closed-book write baseline for the benchmark created by
scripts/build_context_learning_benchmark.py. It runs each task independently:

1. reset all additive MLP memories to zero;
2. write from that task's context only;
3. evaluate the task without context;
4. evaluate expanded sentinels.

The default write is deliberately the old acquisition-bearing raw carrier:
relational_aggregate + context values, no DICE/probe/sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.modeling import (
    clear_active_slot_weights,
    get_decoder_layers,
    install_additive_memory,
    load_model_and_tokenizer,
)
from scripts.build_context_learning_benchmark import evaluate_generic_mc_no_cache, write_json
from scripts.minilang_continual_triangle import release_device_cache
from scripts.minilang_write import add_sentinel_shift_metrics, parse_args as minilang_write_parse_args, run_intrinsic_surprise_writes


def default_write_args() -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = ["minilang_write_defaults"]
        return minilang_write_parse_args()
    finally:
        sys.argv = old_argv


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reset_memories(model, wrappers: dict[int, object]) -> None:
    for wrapper in wrappers.values():
        wrapper.copy_memory_(torch.zeros_like(wrapper.memory))
        wrapper.memory_scale = 1.0
        if hasattr(wrapper, "gate_keys"):
            wrapper.gate_keys = torch.empty_like(wrapper.gate_keys[:0])
        if hasattr(wrapper, "object_gate_keys"):
            wrapper.object_gate_keys = torch.empty_like(wrapper.object_gate_keys[:0])
        if hasattr(wrapper, "object_density_gates"):
            wrapper.object_density_gates = []
        if hasattr(wrapper, "slot_memories"):
            wrapper.slot_memories = []
            wrapper.slot_gate_keys = []
            wrapper.slot_terms = []
    clear_active_slot_weights(model)


def sentinel_shift(before: dict, after: dict) -> dict:
    row: dict = {}
    add_sentinel_shift_metrics(row, before, after, prefix="sentinel")
    return row


def configure_raw_args(
    args: argparse.Namespace,
    layers: list[int],
    script_args: argparse.Namespace,
) -> argparse.Namespace:
    args.layers = layers
    args.seed = script_args.seed
    args.max_length = script_args.max_length
    args.chat_template = script_args.chat_template
    args.intrinsic_surprise_write = True
    args.intrinsic_surprise_target_mode = "relational_aggregate"
    args.intrinsic_surprise_relation_value_mode = script_args.relation_value_mode
    args.intrinsic_surprise_context_target_mode = "full"
    args.intrinsic_surprise_token_mode = script_args.token_mode
    args.intrinsic_surprise_top_tokens = script_args.top_tokens
    args.intrinsic_surprise_key_feature_top_k = script_args.key_feature_top_k
    args.intrinsic_surprise_pair_top_k = script_args.pair_top_k
    args.intrinsic_surprise_relation_order = script_args.relation_order
    args.intrinsic_surprise_target_scale = script_args.target_scale
    args.intrinsic_surprise_weight_mode = script_args.weight_mode
    args.intrinsic_surprise_exp_temperature = script_args.exp_temperature
    args.intrinsic_surprise_exp_cap = script_args.exp_cap
    args.intrinsic_surprise_pair_quantile = script_args.pair_quantile
    args.intrinsic_surprise_row_quantile = script_args.row_quantile
    args.intrinsic_target_purifier = "none"
    args.intrinsic_surprise_output_penalty_rank = script_args.output_penalty_rank
    args.intrinsic_surprise_output_penalty_weight = script_args.output_penalty_weight
    args.intrinsic_surprise_input_penalty_features = script_args.input_penalty_features
    args.intrinsic_surprise_input_penalty_weight = script_args.input_penalty_weight
    args.intrinsic_surprise_input_penalty_usage_power = script_args.input_penalty_usage_power
    args.intrinsic_surprise_input_penalty_mode = script_args.input_penalty_mode
    args.ridge = script_args.ridge
    args.eta = script_args.eta
    args.max_update_norm = script_args.max_update_norm
    args.write_only_final = False
    args.dice_defer_apply = False
    args.cache_current_captures = False
    args.intrinsic_surprise_lesson_format = "raw"
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="benchmarks/context_learning_v1_qwen35_0_8b")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output-dir", default="runs/context_benchmark_raw_relational_qwen35_0_8b_s010")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-scale", type=float, default=0.10)
    parser.add_argument("--token-mode", choices=["last", "top", "all", "final_aligned"], default="all")
    parser.add_argument("--top-tokens", type=int, default=16)
    parser.add_argument("--key-feature-top-k", type=int, default=16)
    parser.add_argument("--pair-top-k", type=int, default=16)
    parser.add_argument("--relation-order", type=int, choices=[2, 3], default=2)
    parser.add_argument("--relation-value-mode", choices=["residual", "full", "context"], default="context")
    parser.add_argument("--weight-mode", choices=["linear", "exponential"], default="linear")
    parser.add_argument("--exp-temperature", type=float, default=1.0)
    parser.add_argument("--exp-cap", type=float, default=100.0)
    parser.add_argument("--pair-quantile", type=float, default=0.0)
    parser.add_argument("--row-quantile", type=float, default=0.0)
    parser.add_argument("--output-penalty-rank", type=int, default=256)
    parser.add_argument("--output-penalty-weight", type=float, default=10.0)
    parser.add_argument("--input-penalty-features", type=int, default=256)
    parser.add_argument("--input-penalty-weight", type=float, default=20.0)
    parser.add_argument("--input-penalty-usage-power", type=float, default=0.0)
    parser.add_argument("--input-penalty-mode", choices=["onehot", "svd", "hybrid"], default="onehot")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((benchmark_dir / "manifest.json").read_text())

    started = time.time()
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    layers = list(range(len(get_decoder_layers(model))))
    wrappers = install_additive_memory(model, layers)
    write_args = configure_raw_args(default_write_args(), layers, args)

    sentinels = read_jsonl(benchmark_dir / manifest["sentinel_file"])
    sentinel_before = evaluate_generic_mc_no_cache(
        model,
        tokenizer,
        sentinels,
        device,
        args.max_length,
        args.chat_template,
    )
    release_device_cache(device)

    summaries = []
    details_by_task = {}
    for task in manifest["tasks"]:
        reset_memories(model, wrappers)
        task_id = task["task_id"]
        task_dir = benchmark_dir / "tasks" / task_id
        context = (task_dir / "context.txt").read_text().strip()
        items = read_jsonl(task_dir / "eval.jsonl")
        updates_path = output_dir / f"{task_id}_updates.jsonl"
        if updates_path.exists():
            updates_path.unlink()
        task_started = time.time()
        print(f"[raw-benchmark] write task={task_id}", flush=True)
        run_intrinsic_surprise_writes(
            model,
            tokenizer,
            wrappers,
            [context],
            write_args,
            device,
            updates_path,
            slot_id=None,
        )
        release_device_cache(device)
        edited = evaluate_generic_mc_no_cache(
            model,
            tokenizer,
            items,
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)
        sentinel_after = evaluate_generic_mc_no_cache(
            model,
            tokenizer,
            sentinels,
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)
        shift = sentinel_shift(sentinel_before, sentinel_after)
        row = {
            "task_id": task_id,
            "task_family": task["task_family"],
            "seed": task["seed"],
            "edited_correct": edited["correct"],
            "edited_n": edited["n"],
            "edited_accuracy": edited["accuracy"],
            "edited_mean_margin": edited["mean_margin"],
            "sentinel_after_correct": sentinel_after["correct"],
            "sentinel_after_n": sentinel_after["n"],
            "sentinel_after_accuracy": sentinel_after["accuracy"],
            **shift,
            "seconds": time.time() - task_started,
        }
        summaries.append(row)
        details_by_task[task_id] = edited["details"]
        print(json.dumps(row, sort_keys=True), flush=True)
        reset_memories(model, wrappers)

    total_correct = sum(row["edited_correct"] for row in summaries)
    total_n = sum(row["edited_n"] for row in summaries)
    result = {
        "benchmark_dir": str(benchmark_dir),
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "write_config": {
            "target_mode": "relational_aggregate",
            "relation_value_mode": args.relation_value_mode,
            "token_mode": args.token_mode,
            "target_scale": args.target_scale,
            "layers": layers,
            "key_feature_top_k": args.key_feature_top_k,
            "pair_top_k": args.pair_top_k,
            "output_penalty_rank": args.output_penalty_rank,
            "output_penalty_weight": args.output_penalty_weight,
            "input_penalty_features": args.input_penalty_features,
            "input_penalty_weight": args.input_penalty_weight,
            "weight_mode": args.weight_mode,
            "pair_quantile": args.pair_quantile,
            "row_quantile": args.row_quantile,
        },
        "sentinel_before": sentinel_before,
        "task_summaries": summaries,
        "edited_total_correct": total_correct,
        "edited_total_n": total_n,
        "edited_total_accuracy": total_correct / total_n if total_n else 0.0,
        "runtime_seconds": time.time() - started,
    }
    write_json(output_dir / "summary.json", result)
    with (output_dir / "task_details.jsonl").open("w") as handle:
        for task_id, details in details_by_task.items():
            for detail in details:
                handle.write(json.dumps({"task_id": task_id, **detail}, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_dir / "summary.json"), "edited": total_correct, "n": total_n}, sort_keys=True))


if __name__ == "__main__":
    main()
