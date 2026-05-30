# Qwen3.5 Benchmark Targets

Date: 2026-05-30

These are the new local targets for speeding up the activation-writing research
loop and adding an interpretable validation track.

## Cached Targets

- `Qwen/Qwen3.5-0.8B-Base`
  - Snapshot: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
  - Role: fast screening model.
  - Verified locally with `AutoModelForCausalLM`.
  - Text stack: 24 layers, hidden size 1024, MLP intermediate size 3584,
    max position embeddings 262144.
  - First MLP `down_proj` shape: `[1024, 3584]`.

- `Qwen/Qwen3.5-2B-Base`
  - Snapshot: `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`
  - Role: interpretable Qwen-Scope target.
  - Config verified locally.
  - Text stack: 24 layers, hidden size 2048, MLP intermediate size 6144,
    max position embeddings 262144.

- `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`
  - Snapshot: `132ea3697b591df9ee46d738aa1d528e3c6082f7`
  - Role: residual-stream SAE inspection for the 2B model.
  - Downloaded layer subset: `0, 4, 8, 12, 16, 20, 23`.
  - Full repo is about 12 GiB, so the full 24-layer SAE set should be pulled
    only after freeing disk or moving the Hugging Face cache.
  - Verified `layer0.sae.pt` keys: `W_enc`, `W_dec`, `b_enc`, `b_dec`.
  - `layer0` shapes: `W_enc [32768, 2048]`, `W_dec [2048, 32768]`.

## Local Runtime

Qwen3.5 needs a newer HF stack than the previous Qwen3 experiments:

- `torch==2.12.0`
- `transformers==5.9.0`
- `huggingface-hub==1.17.0`

The released `transformers==4.57.6` does not recognize `model_type:
qwen3_5`. The `5.9.0` line does, and `torch>=2.12` is needed for its
generation imports on this machine.

## Benchmark Requirements

Before using a model for write experiments, run a no-write teacher screen:

1. Fixed seed, fixed lesson JSONL, fixed eval JSONL.
2. Baseline must be near zero on the selected eval set.
3. In-context teacher should be near ceiling before we trust write results.
4. Report both exact-match accuracy and per-item score margins.
5. Report expanded sentinel correct-to-wrong flips and margin drops.
6. All-layer write tests are mandatory for safety claims.
7. Publish exact model IDs, snapshot hashes, dependency versions, prompts,
   eval items, and random seeds.

Recommended tracks:

- Fast screen: `Qwen/Qwen3.5-0.8B-Base`.
- Interpretability validation: `Qwen/Qwen3.5-2B-Base` plus Qwen-Scope SAE
  layers.
