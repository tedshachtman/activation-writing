#!/usr/bin/env bash
set -euo pipefail

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
  --layers 20 \
  --negative-guards 8 \
  --guard-weight 0.05 \
  --causal-filter \
  --capture-last-tokens 24 \
  --eta 10.0 \
  --max-update-norm 500.0 \
  --memory-gate \
  --memory-gate-threshold 0.95 \
  --output runs/tokenwindow_stage1_qwen17
