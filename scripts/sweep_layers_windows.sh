#!/usr/bin/env bash
set -euo pipefail

layers=(18 20 22)
windows=(12 24 48)
out_root="${1:-runs/sweeps/layer_window_stage1}"

for layer in "${layers[@]}"; do
  for window in "${windows[@]}"; do
    out_dir="$out_root/layer${layer}_window${window}"
    caic-run \
      --model Qwen/Qwen3-1.7B \
      --papers 1 \
      --domain-difficulty easy \
      --teacher-search-budget 10 \
      --teacher-min-accuracy 0.65 \
      --teacher-min-delta 0.10 \
      --candidate-probes 48 \
      --write-probes 24 \
      --validation-folds 3 \
      --eval-questions 20 \
      --layers "$layer" \
      --negative-guards 8 \
      --guard-weight 0.05 \
      --causal-filter \
      --capture-last-tokens "$window" \
      --eta 10.0 \
      --max-update-norm 500.0 \
      --memory-gate \
      --memory-gate-threshold 0.95 \
      --allow-noop-write \
      --output "$out_dir"
  done
done

python - "$out_root" <<'PY'
from pathlib import Path
import csv
import json
import sys

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("*/metrics.jsonl")):
    with metrics_path.open("r", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    first["run_dir"] = str(metrics_path.parent)
    rows.append(first)

if rows:
    fieldnames = sorted({key for row in rows for key in row})
    with (root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(root / "sweep_summary.csv")
PY

