# Causal Activation-Imprint Consolidation

This repo is a small PyTorch prototype for the refined CAIC idea: turn context-induced internal behavior into direct persistent writes on selected MLP down-projections.

The implementation prioritizes inspectability over speed. It uses:

- executable synthetic mini-papers, not hand-labeled prose;
- teacher/document, null-prefix, and query-only passes;
- query-only student MLP keys, so writes fire in the deployment regime;
- module-output deltas, not raw residual deltas;
- optional causal patch scores to weight only deltas that improve answer likelihood;
- recursive least squares memory updates with per-layer plasticity matrices;
- negative guard prompts with zero target deltas;
- metrics and plots after every paper;
- optional naive text fine-tuning and Q&A LoRA baselines.

## Quick Start

Install dependencies in a Python environment with PyTorch and Transformers:

```bash
pip install -e ".[dev]"
```

Run a one-paper gated experiment with Qwen3-1.7B:

```bash
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
  --capture-last-tokens 24 \
  --eta 10.0 \
  --max-update-norm 500.0 \
  --memory-gate \
  --memory-gate-threshold 0.95 \
  --output runs/tokenwindow_stage1_qwen17
```

Current best two-paper sequential smoke test:

```bash
caic-run \
  --model Qwen/Qwen3-1.7B \
  --papers 2 \
  --domain-difficulty easy \
  --teacher-search-budget 80 \
  --teacher-min-accuracy 0.80 \
  --teacher-min-delta 0.20 \
  --candidate-probes 64 \
  --write-probes 32 \
  --validation-probes 20 \
  --validation-folds 4 \
  --validation-min-accuracy-delta 0.02 \
  --validation-min-fold-delta 0.0 \
  --validation-min-negative-delta 0.05 \
  --eval-questions 20 \
  --layers 22 \
  --capture-last-tokens 12 \
  --write-token-selection suffix \
  --gate-token-selection same \
  --negative-label-weight 4.0 \
  --search-etas 10.0 20.0 40.0 \
  --search-max-update-norms 500.0 1000.0 2000.0 \
  --negative-guards 8 \
  --guard-weight 0.05 \
  --causal-filter \
  --memory-gate \
  --memory-gate-threshold 0.95 \
  --allow-noop-write \
  --output runs/negweight_relaxed_l22w12_2paper_qwen17
```

For a ten-paper sequential run:

```bash
caic-run \
  --model Qwen/Qwen3-1.7B \
  --papers 10 \
  --candidate-probes 100 \
  --write-probes 24 \
  --eval-questions 30 \
  --layers 10 14 \
  --negative-guards 12 \
  --causal-filter \
  --output runs/stage2
```

Optional baselines:

```bash
scripts/run_stage1_baselines.sh runs/tokenwindow_stage1_qwen17
```

Layer/window sweep:

```bash
scripts/sweep_layers_windows.sh runs/sweeps/layer_window_stage1
```

## Core Update

For a selected layer, CAIC captures:

- `A`: MLP down-projection inputs from the query-only student run;
- `D`: content deltas from `doc+question` MLP output minus `null+question` MLP output;
- optional weights from causal patching.

It then updates the additive memory matrix `M` with recursive least squares:

```text
E = D - M A
G = P A (I + A^T P A)^-1
M <- M + eta E G^T
P <- P - G A^T P
```

`P` is the inverse covariance / plasticity matrix. Directions that have already been claimed become less plastic.

## Outputs

Each run writes:

- `teacher_gate.jsonl`: no-document and document-in-context scores for searched domains;
- `domains.jsonl`: executable domain specs and rendered papers;
- `metrics.jsonl`: per-step acquisition, retention, contamination, and baseline metrics;
- `summary.csv`: tabular metrics;
- `accuracy.png`: accuracy curves.

Metrics include labeled contamination sentinels (`contamination_pre_*`, `contamination_post_*`) in addition to sentinel KL. Treat a run as suspect if contamination accuracy drops or sentinel KL is high.

The default evaluator uses prompt-only Yes/No answer-token log-probabilities rather than free-form generation, which makes the early benchmark deterministic and cheap.

`--causal-filter` is enabled by default because the refined method should write deltas that move answer likelihood, not every delta that happens to change. Use `--no-causal-filter` for faster ablations.

`--teacher-gate` is also enabled by default. It searches candidate synthetic domains and only starts CAIC if the context teacher clears `--teacher-min-accuracy` and `--teacher-min-delta`. Use `--domain-difficulty standard` for the original multi-rule benchmark; `medium` adds two-rule domains with an exception; `easy` is a Stage 0 sanity benchmark for small models.

`--memory-gate` is enabled by default. It gates the additive memory output by cosine similarity to the selected student write keys, which prevents large writes from becoming always-on answer biases.

`--safe-write-search` is enabled by default. It tries a small grid of write strengths, requires labeled contamination guards and sentinel KL to stay clean, and chooses among accepted writes using multiple independent validation-probe folds rather than the held-out evaluation set.

`--capture-last-tokens` controls how many suffix tokens from each probe contribute write constraints. The default of 1 is the minimal final-token ablation; the Stage 1 script uses 24 because final-token-only writes were too prone to template-specific validation gains.

`--positive-label-weight` and `--negative-label-weight` control both D-optimal probe selection and RLS write weights. The current easy-domain benchmark has a strong Yes bias, so the working two-paper run upweights negative examples and requires validation negative-label accuracy to improve before committing the write.

`--gauntlet` adds falsification buckets: ordinary heldout, one-edit minimal pairs, and inverse-polarity questions. `--near-collision-gauntlet` adds chains selected because a same-vocabulary rival rule system would flip the label; this catches methods that learn the prompt family but not the actual rule. `--bias-rival-baseline` fits a scalar Yes bias on validation probes and reports it beside CAIC, so ordinary gains can be checked against a cheap answer-prior repair baseline.

`--slot-memory` stores each paper's accepted update in a separate memory slot. By default slot routing is a simple lexical domain latch over the domain title, multiplied by the existing suffix behavior gate. `--activation-slot-routing` replaces the lexical latch with a pooled content-activation latch, while keeping the suffix behavior gate as the actuator. `--domain-latch-diagnostics` writes `domain_latch.jsonl` and adds latch accuracy, margin, and activation-rate metrics to `metrics.jsonl`. `--gauntlet-validation` can require safe-write candidates to preserve the gauntlet buckets before a slot write is committed.

See `RESEARCH_LOG.md` for the current positive result and failed ablations. The present best run is a small but clean Stage 1 result, not a solved continual-learning method.

The standalone baseline runner uses the exact `domains.jsonl` from a CAIC run:

```bash
caic-baseline \
  --run-dir runs/tokenwindow_stage1_qwen17 \
  --model Qwen/Qwen3-1.7B \
  --baseline qa_lora \
  --steps 100 \
  --output runs/tokenwindow_stage1_qwen17/qa_lora_100step_baseline.jsonl
```
