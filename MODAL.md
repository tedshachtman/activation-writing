# Running Modal Jobs From Codex Cloud

This repo is set up so a fresh GitHub checkout can launch Modal GPU jobs without
copying local run artifacts. Modal mounts the checked-out repository into the
remote container and writes experiment outputs to the persistent `caic-runs`
Modal volume at `/modal-runs`.

## Required Secrets

The GitHub repo has these Actions secrets set:

```bash
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

Those are enough for the manual GitHub Actions workflow below. If you are
running Modal directly from an interactive Codex cloud shell instead of through
Actions, that shell also needs the same two environment variables.

Create them locally with Modal if needed:

```bash
modal token new
```

For gated Hugging Face models, also configure Modal/Hugging Face credentials as
needed. The default Qwen/Qwen3-1.7B benchmark uses a public model.

Do not commit `.modal/`, tokens, downloaded models, or run artifacts. The
repository ignore rules and Modal mount filters exclude these by default.

## Run From GitHub Actions

Use the manual workflow when you want Codex cloud to launch Modal without
handling Modal secrets directly.

Smoke test:

```bash
gh workflow run modal-benchmark.yml \
  --repo tedshachtman/activation-writing \
  -f mode=smoke
```

Run the current safe baseline preset:

```bash
gh workflow run modal-benchmark.yml \
  --repo tedshachtman/activation-writing \
  -f mode=preset \
  -f preset=qrico_key16 \
  -f tag=cloud_qrico_key16
```

Run an arbitrary command through the generic Modal runner:

```bash
gh workflow run modal-benchmark.yml \
  --repo tedshachtman/activation-writing \
  -f mode=command \
  -f command='python scripts/minilang_intrinsic_continual.py --help'
```

Watch the latest workflow run:

```bash
gh run list --repo tedshachtman/activation-writing --workflow modal-benchmark.yml --limit 1
gh run watch --repo tedshachtman/activation-writing
```

## Run From Codex Cloud Without `gh`

Codex cloud containers may not have a git remote or the `gh` CLI. Use the
standard-library launcher instead:

```bash
python scripts/cloud_modal.py doctor
```

If the cloud environment has `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, run
Modal directly from the checkout:

```bash
python scripts/cloud_modal.py smoke --backend direct
python scripts/cloud_modal.py preset --backend direct --preset qrico_key16 --tag cloud_qrico_key16
```

If the cloud environment has `GH_TOKEN`, `GITHUB_TOKEN`, or `GITHUB_PAT`, but
not Modal secrets, dispatch the GitHub Actions workflow. The workflow receives
the repo's Modal secrets:

```bash
python scripts/cloud_modal.py smoke --backend workflow
python scripts/cloud_modal.py preset --backend workflow --preset qrico_key16 --tag cloud_qrico_key16
```

`--backend auto` picks direct Modal when Modal secrets are present, otherwise
GitHub workflow dispatch when a GitHub token is present:

```bash
python scripts/cloud_modal.py preset --preset qrico_key16 --tag cloud_qrico_key16
```

There is one hard boundary: GitHub repo secrets are not exposed to arbitrary
Codex cloud shells. They are only injected into GitHub Actions jobs. A normal
cloud shell therefore needs either Modal env vars for direct runs or a GitHub
token for workflow dispatch.

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
