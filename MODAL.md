# Running Modal Jobs From Codex Cloud

This repo is set up so a fresh GitHub checkout can launch Modal GPU jobs without
copying local run artifacts. Modal mounts the checked-out repository into the
remote container and writes experiment outputs to the persistent `caic-runs`
Modal volume at `/modal-runs`.

## Required Secrets

Codex cloud needs Modal credentials in its environment:

```bash
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

Create them locally with Modal if needed:

```bash
modal token new
```

For gated Hugging Face models, also configure Modal/Hugging Face credentials as
needed. The default Qwen/Qwen3-1.7B benchmark uses a public model.

Do not commit `.modal/`, tokens, downloaded models, or run artifacts. The
repository ignore rules and Modal mount filters exclude these by default.

## One-Time Setup In Codex Cloud

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

Confirm the Modal CLI is authenticated:

```bash
modal profile current
```

Run a minimal remote smoke test:

```bash
modal run scripts/modal_smoke.py
```

Run the GPU/image smoke test:

```bash
modal run scripts/modal_caic_runner.py --smoke
```

## Run The Standard Two-Task Benchmark

Print the command first:

```bash
python scripts/continual_benchmark_grid.py --preset qrico_key16 --tag cloud_smoke
```

Launch the current safe baseline on Modal:

```bash
python scripts/continual_benchmark_grid.py \
  --modal \
  --run \
  --preset qrico_key16 \
  --tag cloud_qrico_key16
```

Launch the current SEAL-Q diagnostic:

```bash
python scripts/continual_benchmark_grid.py \
  --modal \
  --run \
  --preset seal_qrico \
  --tag cloud_seal_qrico
```

Launch multiple presets in one Codex cloud command:

```bash
python scripts/continual_benchmark_grid.py \
  --modal \
  --run \
  --preset qrico_key16 \
  --preset seal_qrico_no_apply \
  --preset seal_qrico \
  --tag cloud_compare
```

## Run Any Command On Modal

Use the generic runner when you want a one-off command:

```bash
modal run scripts/modal_caic_runner.py --command \
  "python scripts/minilang_intrinsic_continual.py --help"
```

Long-running benchmark commands should write under `/modal-runs/...` so artifacts
land in the persistent Modal volume.

## Download Results

After a run completes, download artifacts from the `caic-runs` volume:

```bash
modal volume ls caic-runs /
modal volume get caic-runs /cloud_qrico_key16_qrico_key16 runs/cloud_qrico_key16_qrico_key16
```

If the exact output directory is unclear, inspect the Modal logs or use the run
tag printed by `scripts/continual_benchmark_grid.py`.
