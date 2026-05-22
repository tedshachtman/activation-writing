#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-runs/tokenwindow_stage1_qwen17}"

caic-baseline \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-1.7B \
  --baseline qa_lora \
  --steps 10 \
  --lr 5e-5 \
  --candidate-probes 48 \
  --qa-lora-r 16 \
  --output "$RUN_DIR/qa_lora_baseline.jsonl"

caic-baseline \
  --run-dir "$RUN_DIR" \
  --model Qwen/Qwen3-1.7B \
  --baseline text_lora \
  --steps 10 \
  --lr 5e-5 \
  --qa-lora-r 16 \
  --output "$RUN_DIR/text_lora_baseline.jsonl"

