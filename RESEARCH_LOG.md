# CAIC Research Log

This log records the current empirical state of the prototype. It is intentionally conservative: validation-only gains are treated as suspect unless held-out accuracy and contamination guards agree.

## Current Best Pre-Registered Stage 1

Run: `runs/tokenwindow_stage1_qwen17`

Configuration:

- model: `Qwen/Qwen3-1.7B`
- synthetic difficulty: `easy`
- accepted domain: `Nemril-002`
- rule: a valid chain must contain at least one `salted zhek`
- layer: MLP down-projection at layer `20`
- write window: final `24` tokens per selected probe
- selected probes: `24`
- selected write keys: `576`
- memory gate threshold: `0.95`
- write search choice: `eta=10.0`, `max_update_norm=500.0`

Result:

| metric | value |
| --- | ---: |
| no-document held-out accuracy | 0.50 |
| document-in-context held-out accuracy | 0.80 |
| CAIC held-out accuracy | 0.60 |
| internalization ratio | 0.333 |
| contamination accuracy before | 1.00 |
| contamination accuracy after | 1.00 |
| sentinel KL | 0.000746 |
| wall time on local MPS run | 89 s |

This is a real but small positive result: CAIC recovered one third of the context teacher's improvement on held-out questions without decreasing the labeled contamination guards.

Diagnostics from the refreshed run:

| metric | value |
| --- | ---: |
| paper-loop forward calls | 591 |
| paper-loop input tokens | 57,057 |
| total forward calls including teacher gate | 713 |
| total input tokens including teacher gate | 74,729 |
| capture seconds | 25.8 |
| causal-filter seconds | 9.5 |
| safe-write-search seconds | 34.1 |
| eval question gate-hit rate | 1.00 |
| eval token gate-hit rate | 0.863 |

The per-question diagnostics are in `runs/tokenwindow_stage1_qwen17/diagnostics.jsonl`. The write flipped two held-out invalid-chain questions from `Yes` to `No`, improving accuracy from 10/20 to 12/20. Many remaining failures are still a strong `Yes` bias on invalid chains.

## Layer/Window Sweep

Run: `runs/sweeps/layer_window_stage1`

This was an automated grid over layers `18, 20, 22` and suffix windows `12, 24, 48`. Results are collated in `runs/sweeps/layer_window_stage1/sweep_summary.csv`.

| layer/window | held-out acc | validation acc delta | contamination acc | note |
| --- | ---: | ---: | ---: | --- |
| 22 / 12 | 0.70 | 0.083 | 1.00 | best held-out audit |
| 18 / 12 | 0.65 | 0.104 | 1.00 | strongest validation-tied setting |
| 18 / 24 | 0.65 | 0.083 | 1.00 | clean |
| 22 / 24 | 0.65 | 0.083 | 1.00 | clean |
| 20 / 24 | 0.60 | 0.104 | 1.00 | original pre-registered setting |
| 20 / 12 | 0.55 | 0.083 | 1.00 | clean but weak |
| 20 / 48 | 0.55 | 0.021 | 1.00 | clean but weak |
| 18 / 48 | 0.50 | 0.042 | 1.00 | validation-only false positive |
| 22 / 48 | 0.50 | 0.000 | 1.00 | no-op |

Do not treat `22 / 12` as a confirmed improvement yet, because it was identified by a held-out audit after the grid was run. The right next test is to select layer/window by validation on one seed, then audit on fresh domains or seeds.

## Negative Results

- `Qwen/Qwen3-0.6B` often failed the teacher gate on this benchmark. If the context teacher cannot solve the domain, there is no reliable behavior to consolidate.
- Final-token-only writes produced validation gains that did not transfer to held-out evaluation. The run in `runs/gated_stage1_qwen17` selected a write with validation accuracy 0.604 but held-out accuracy stayed 0.50.
- A 48-token write window was clean but weaker than 24 tokens: `runs/tokenwindow48_stage1_qwen17` reached held-out accuracy 0.55.
- A two-layer write to layers `14` and `20` failed the safe-write search even on the first paper: `runs/tokenwindow_2paper_qwen17_layers14_20`. More layers are not automatically better under the current update rule.
- In the two-paper sequential run `runs/tokenwindow_2paper_qwen17_safe`, paper 1 wrote cleanly, but paper 2 no-oped because no candidate write improved validation accuracy while preserving guards. This is the desired failure mode for now.
- The layer/window sweep found one validation-only false positive: layer `18`, window `48` improved validation but stayed at 0.50 held-out.

## Sequential Check

Run: `runs/tokenwindow_2paper_qwen17_safe`

| paper | context acc | pre/no-doc acc | CAIC acc | write |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.80 | 0.50 | 0.60 | applied |
| 1 | 0.70 | 0.55 | 0.55 | no-op |

Retention mean after paper 2 was `0.575`, and contamination accuracy stayed at `1.00`.

Paper 2 diagnostics:

- The safe search tried 4 writes and accepted 0.
- Candidate writes moved validation margins but did not improve validation accuracy.
- Because no write was committed, held-out margins and predictions were unchanged.
- Existing paper-1 memory gate keys still showed high similarity to paper-2 prompts, which means the current key gate is not domain-isolating enough by itself. It prevents general contamination, but it does not distinguish similar synthetic domains.

## Sequential Check With Label-Balanced Writes

Run: `runs/negweight_relaxed_l22w12_2paper_qwen17`

This run adds two controls aimed at the observed two-paper failure:

- validation tracks positive and negative labels separately;
- selection and RLS fitting upweight negative examples with `--negative-label-weight 4.0`;
- safe-write search requires negative-label validation accuracy to improve with `--validation-min-negative-delta 0.05`;
- the layer/window setting is the strongest previous held-out audit setting: layer `22`, suffix window `12`.

| paper | context acc | pre/no-doc acc | CAIC acc | negative held-out acc | write |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.80 | 0.50 | 0.75 | 0.60 | applied |
| 1 | 0.80 | 0.55 | 0.60 | 0.20 | applied |

Retention mean after paper 2 was `0.70`; since paper 2 latest accuracy was `0.60`, this implies paper 0 retained at about `0.80` in this run. Contamination accuracy stayed at `1.00`.

This is the first run where two papers write sequentially and the second write improves held-out accuracy. It is still a weak result: paper 2 improves by one held-out question (`0.55 -> 0.60`) and validation negative margins worsened even while validation negative accuracy improved. Treat it as "multi-paper smoke test works under tuned settings," not as robust continual learning.

The stricter predecessor `runs/negweight_l22w12_2paper_qwen17` improved paper 1 to `0.75` but no-oped paper 2 because the best candidate's validation accuracy gain was `+0.0375`, below the then-configured `+0.05` threshold. Relaxing the threshold to `+0.02` allowed that candidate to commit and it transferred modestly to held-out.

## Bias-Rival Gauntlet And Slot-Latch Runs

Following the external research critique, the harness now includes:

- inverse-polarity questions, e.g. "should this chain be rejected?";
- one-edit minimal pairs whose labels differ;
- a scalar Yes-bias baseline tuned on validation probes;
- optional per-paper slot memory with lexical domain-title routing multiplied by the suffix behavior gate;
- optional gauntlet-gated safe-write acceptance.

Run: `runs/slot_latch_gauntlet_2paper_qwen17`

This was the first slot-memory/product-gate smoke test without gauntlet-gated acceptance.

| paper | pre/no-doc acc | context acc | CAIC acc | ordinary gauntlet | minimal pair | inverse polarity | write |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.50 | 0.80 | 0.70 | 0.65 | 0.45 | 0.50 | applied |
| 1 | 0.50 | 0.80 | 0.35 | 0.25 | 0.25 | 0.70 | applied |

This is a useful negative result. The second slot write passed ordinary validation but damaged held-out behavior badly. The inverse-polarity gain on paper 2 is not a success; it is a signature of overcorrecting the answer prior.

Run: `runs/slot_latch_gauntlet_accept_fixed_2paper_qwen17`

This reran the slot-memory setup with `--gauntlet-validation --gauntlet-min-bucket-delta 0.0`, after fixing no-op cleanup for rejected slot trials.

| paper | pre/no-doc acc | context acc | CAIC acc | ordinary gauntlet | minimal pair | inverse polarity | write |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.50 | 0.80 | 0.75 | 0.65 | 0.60 | 0.35 | applied |
| 1 | 0.50 | 0.80 | 0.50 | 0.50 | 0.50 | 0.50 | no-op |

The stricter gauntlet acceptance prevented the harmful paper-2 slot write. It did not solve paper-2 acquisition. This supports the current diagnosis: the actuator is real, but the method still lacks a robust binder/target for rule acquisition. The scalar Yes-bias baseline remained weak on latest heldout (`0.45` on paper 0, `0.50` on paper 1), so CAIC's paper-0 gain is not explained by a scalar bias alone, but the inverse-polarity failure means it is still not clean rule consolidation.

## Product-Key Routing Instrumentation

Implemented the next architectural step from the critique:

- a near-collision gauntlet bucket, where chains are selected because a same-vocabulary rival domain would assign the opposite label;
- activation-domain-latch diagnostics that pool residual activations over domain/content tokens, whiten them with a diagonal covariance, and report paper-ID classification accuracy/margins in `domain_latch.jsonl`;
- optional `--activation-slot-routing`, which uses that activation latch to choose the per-paper slot while the existing suffix behavior gate still decides when the MLP memory fires.

This does not yet claim multi-paper success. It is instrumentation and routing plumbing for testing the binding hypothesis directly. Unit coverage now checks near-collision generation and the activation-router override path. Full model smoke runs were blocked locally by `transformers` model construction hanging before CAIC code executed, so the next substantive result should be a Qwen run in an environment where `AutoModelForCausalLM.from_pretrained(...)` is healthy.

Run: `runs/activation_latch_cached_2paper_qwen17`

This run used Qwen3-1.7B, activation slot routing, near-collision gauntlets, gauntlet-gated safe search, and scalar-bias rivals. Qwen3-0.6B was tried first in `runs/activation_latch_2paper_qwen06`, but teacher gating accepted `0/2` domains within 80 candidates; the best context-teacher accuracy was only `0.60`, so the 0.6B run was correctly rejected at Stage 0.

The activation-router implementation needed batching and caching; without it, paper 2 spent excessive time doing one full activation-latch forward per prompt during capture. After caching, the full two-paper run completed.

| paper | pre/no-doc acc | context acc | AIC acc | ordinary | minimal pair | inverse polarity | near collision | write |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.50 | 0.80 | 0.40 | 0.40 | 0.35 | 0.55 | 0.35 | applied |
| 1 | 0.50 | 0.80 | 0.50 | 0.50 | 0.50 | 0.50 | 0.65 | no-op |

This is a failed consolidation result. Paper 0's accepted write had only `+0.0125` validation accuracy, `0.0` minimum fold delta, and `0.0` minimum gauntlet-bucket delta. It passed the current acceptance rule but failed held-out latest and near-collision/minimal-pair gauntlets. Paper 1 correctly no-oped. Domain-latch diagnostics were strong by paper 1 (`domain_latch_accuracy=1.0`, `active_rate=1.0`, mean margin about `0.87`), so the immediate failure is not obviously domain binding. It is the target/acceptance rule: the current MLP delta still looks like brittle answer steering, and the validator is too permissive.

Next acceptance change: require a real validation effect, not a one-example wobble. Use at least `--validation-min-accuracy-delta 0.05`, `--validation-min-negative-delta 0.05`, and `--gauntlet-min-bucket-delta 0.05` for the next run, or switch safe search to a worst-bucket score that prioritizes minimal-pair and near-collision transfer.

## Strict Routing and Target Diagnostics

Runs:

- `runs/activation_latch_singlefix_strict_2paper_qwen17`
- `runs/activation_latch_projectyn_strict_2paper_qwen17`
- `runs/answerdir_augmented_strict_2paper_qwen17`
- `runs/key_separability_layers_augmented_qwen17`
- `runs/key_separability_validity_layers_augmented_qwen17`
- `runs/key_output_separability_validity_layers_qwen17`
- `runs/validityprobe_l8_final_strict_2paper_qwen17`

Changes implemented:

- domain reuse via `--domains-jsonl`, so strict reruns use identical papers and held-out sets;
- a single-slot activation-router fix, so the first paper routes on content-token hits instead of rejecting the only prototype by an arbitrary whitened cosine threshold;
- `--project-answer-direction`, which removes the raw Yes-vs-No unembedding direction from doc/null deltas;
- augmented candidate pools: `--candidate-inverse-probes`, `--candidate-minimal-pair-probes`, and `--candidate-near-collision-probes`;
- `--target-mode answer_direction`, a supervised answer-target control;
- `scripts/key_separability.py`, which fits offline ridge probes to no-document MLP keys/outputs;
- `--target-mode validity_probe`, which fits a validity probe in MLP output space and writes the minimal target delta toward a validity margin.

Strict acceptance fixed the false positive. With `--validation-min-accuracy-delta 0.05`, `--validation-min-positive-delta 0.0`, `--validation-min-negative-delta 0.05`, and `--gauntlet-min-bucket-delta 0.05`, the original doc/null target accepted no writes. The best candidates still had the same signature: large negative-label gains and positive-label regressions. Projecting away the raw answer direction reduced some inverse-polarity damage, but did not produce a valid write.

The supervised answer-direction control also failed under strict guards:

| paper | target | pre/no-doc acc | context acc | AIC acc | write | best validation delta | positive delta | negative delta | min gauntlet delta |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | answer direction | 0.50 | 0.80 | 0.50 | no-op | +0.2125 | -0.075 | +0.500 | -0.050 |
| 1 | answer direction | 0.50 | 0.80 | 0.50 | no-op | +0.1750 | -0.225 | +0.575 | -0.300 |

This is important: even when the target is explicit DSL supervision, the current layer-22 suffix actuator mostly learns a negative-answer correction field, not a balanced rule computation.

The key-separability probe sharpened the diagnosis. For answer labels, no-document suffix keys overfit training probes perfectly but generalized poorly to inverse-polarity questions. For chain-validity labels, the same model activations were much better:

| feature source | label | best mean heldout | inverse-polarity mean | near-collision mean |
| --- | --- | ---: | ---: | ---: |
| layer-22 keys | answer | ~0.65 | ~0.40 | ~0.72 |
| layer-8 keys | validity | ~0.95 | ~0.83 | ~0.95 |
| layer-8 MLP outputs | validity | ~0.95 | ~0.90 | ~0.90 |

So the model already carries a usable validity representation in early no-document activations. The failure is not simply "the rules are absent from activations." The failure is that the current write target and late actuator do not convert that validity representation into answer behavior.

The first validity-probe write tried to exploit that representation directly at layer 8:

| paper | target | layer | write rows | pre/no-doc acc | context acc | AIC acc | write |
| ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| 0 | validity probe | 8 | final token | 0.50 | 0.80 | 0.50 | no-op |
| 1 | validity probe | 8 | final token | 0.50 | 0.80 | 0.50 | no-op |

Small validity-probe updates were no-ops; larger updates again improved some negatives by damaging positives and gauntlet buckets. Current conclusion: validity is readable in the relevant spaces, but not yet writable through a single final-token MLP-down actuator. The next architecture should write an intermediate state over a suffix/chain window and separately learn/read out question polarity, or patch residual-after-block trajectories rather than only MLP down-projection outputs.

## Direct Patch and Balanced Oracle Controls

Runs:

- `runs/target_patch_sweep_paper0_l8_l22_qwen17`
- `runs/target_patch_sweep_highscale_paper0_l22_qwen17`
- `runs/answerdir_scale128_l22_paper0_qwen17`
- `runs/answerdir_scale128_balanced_l22_paper0_qwen17`
- `runs/answerdir_scale128_balanced_uniform_l22_paper0_qwen17`

Implemented:

- `scripts/target_patch_sweep.py`, a direct activation-patch diagnostic for proposed targets;
- `--balanced-write-selection`, which forces D-optimal write-probe selection to include both positive and negative labels.

The direct patch sweep answered an important controllability question. Normal-scale activation patches moved margins but did not flip decisions. High-scale explicit answer-direction patches at layer 22 did flip decisions:

| group | baseline acc | patched acc at scale 128 | positive acc | negative acc |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.85 | 1.00 | 0.70 |
| ordinary | 0.50 | 0.80 | 1.00 | 0.60 |
| minimal pair | 0.50 | 0.95 | 1.00 | 0.90 |
| inverse polarity | 0.50 | 1.00 | 1.00 | 1.00 |
| near collision | 0.50 | 0.90 | 1.00 | 0.80 |

So the MLP-output actuator can move answers, but only with a much stronger intervention than the earlier target scales. Validity-probe patches were weaker and sometimes harmful on inverse-polarity prompts, which reinforces the need to separate chain-validity state from question-polarity readout.

The first high-scale closed-form oracle write failed for a simple reason: write-probe selection picked `0` positive probes and `32` negative probes. The causal weighting had amplified exactly the examples that repair the model's Yes prior. That created a No-only correction field.

Balanced selection fixed that experimental bug:

| run | selected positives | selected negatives | best validation delta | positive delta | negative delta | min gauntlet delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high-scale answer oracle, unbalanced | 0 | 32 | -0.0375 | -0.925 | +0.850 | -0.200 |
| high-scale answer oracle, balanced | 16 | 16 | +0.0500 | -0.050 | +0.150 | -0.050 |
| high-scale answer oracle, balanced + uniform weights | 32 | 32 | +0.0500 | -0.025 | +0.125 | 0.000 |

This is not yet a strict success, but it is a much better failure. With balanced probes and uniform weights, the write no longer smears across gauntlet buckets; it narrowly misses strict acceptance because one positive validation item regresses. The next architecture should make label selection explicit rather than asking one linear memory map to infer it from suffix keys:

- domain latch selects the paper;
- a learned validity probe/classifier estimates chain validity;
- a question-polarity detector maps validity to the desired Yes/No answer;
- the MLP memory acts as an answer actuator only after that routing decision.

That is less elegant than a single linear map, but it matches the evidence: validity is readable, answer actuation is possible, and the current failure is binding/routing between them.

## Baselines

Baselines were run against the same `runs/tokenwindow_stage1_qwen17/domains.jsonl` domain and held-out questions.

| baseline | steps | lr | held-out acc | contamination acc | train seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q&A LoRA | 10 | 5e-5 | 0.50 | 0.375 | 7.2 |
| text LoRA | 10 | 5e-5 | 0.50 | 1.00 | 5.7 |
| Q&A LoRA | 100 | 5e-5 | 0.50 | 1.00 | 27.0 |
| Q&A LoRA | 100 | 5e-4 | 0.50 | 0.375 | 25.6 |
| naive full text FT | 3 | 1e-6 | 0.50 | 0.625 | 156.8 |

These are not exhaustive baselines, but they are useful checks: a straightforward LoRA baseline did not reproduce the CAIC held-out improvement in this setup. The high learning-rate Q&A LoRA run also damaged contamination guards.

The full-model naive text fine-tune was attempted only for 3 steps because it is very slow locally. It produced NaN margins and reduced contamination accuracy, so it is currently an unstable baseline in this MPS/float16 setup.

## Compute Notes

The closed-form write itself is cheap relative to model forwards, but the current research harness is not yet a compute-optimized learner. A Stage 1 run spends most of its time on:

- teacher/no-document/context gating and evaluation;
- three activation-capture passes over candidate probes: document, null document, and query-only;
- causal patching forwards;
- safe-write-search forwards across validation folds, sentinels, and contamination guards.

The accepted Stage 1 run took about 90 seconds locally on MPS. It used no backward passes for CAIC, but it did use hundreds of short forward calls because the safety harness is deliberately strict.

A rough comparison:

- Core CAIC acquisition cost is about `3 * probes` model forwards plus one RLS solve, before causal filtering and safety search.
- Gradient training costs roughly one forward plus backward per step; a backward pass is commonly around two forward-pass costs, so a training step is roughly `3x` a forward pass at similar token length.
- In this unoptimized prototype, CAIC is not yet cheaper than a tiny 5-20 step LoRA baseline. The research bet becomes compute-interesting if it can get similar acquisition with tens of probe forwards instead of hundreds of gradient steps, while using less optimizer memory and preserving retention.
- On the current Stage 1 run, CAIC used 591 paper-loop forwards and 57k input tokens, but no backward passes. The 100-step Q&A LoRA baseline used 168 counted forwards and 9k input tokens plus 100 backward passes, but did not acquire held-out accuracy. Counting a backward pass as roughly two forward passes, that baseline is still materially cheaper in raw compute, but it failed behaviorally here.
- The current CAIC bottleneck is not the RLS write; it is safety instrumentation: three activation-capture passes, causal patching, and four safe-write-search trials. A production version would need to reduce those forwards or amortize them.

## TSOC Pivot: Source-Term Writes Instead of Raw Deltas

After the GPT-5.5 Pro research critique, we added a new diagnostic path for
Trajectory-Source Operator Consolidation (TSOC). The core change is to stop
treating `teacher - student` activation deltas as the thing to write. TSOC
instead estimates a block-local source term:

```text
source ~= (full_context_block_out - null_block_out)
        - (full_context_block_in  - null_block_in)
```

Implemented:

- `caic.modeling.capture_block_io`, which captures residual-stream inputs and outputs at decoder blocks;
- `caic.tsoc.block_source_targets`, `protected_ridge_update`, nuisance projection helpers, and reentry metrics;
- `scripts/tsoc_source_write.py`, a standalone TSOC runner that logs:
  - behavior metrics;
  - update fit statistics;
  - trigger-overlap diagnostics;
  - state-reentry distance to the context teacher.

Tests:

```text
.venv/bin/python -m pytest -q tests/test_tsoc.py tests/test_memory.py tests/test_synthetic.py
23 passed
```

### Paper-Use Trace

Run:

- `runs/tsoc_source_overlap_l8_10_12_paper0_qwen17`

Trace source:

- `P || U`, where `P` is the rule section and `U` is the paper's worked-example/use section.

Result:

- behavior did not improve;
- state reentry was essentially unchanged;
- trigger-overlap showed why:
  - future question keys had only moderate cosine to paper-use trace keys;
  - update effect norms on eval questions were tiny compared with trace target norms.

Representative trigger numbers:

| layer | heldout max cosine to trace | eval update-effect norm | trace target norm |
| ---: | ---: | ---: | ---: |
| 8 | 0.676 | 0.167 | 12.39 |
| 10 | 0.644 | 0.320 | 18.13 |
| 12 | 0.557 | 0.559 | 26.96 |

Interpretation:

This is the identifiability failure Pro predicted. The paper continuation did
not expose the same trigger basin as future Yes/No questions, so the write fit
the trace but barely fired where evaluation needed it.

### Question-Shaped Trace Diagnostic

Run:

- `runs/tsoc_questionprobe_l8_10_12_paper0_qwen17`

Trace source:

- generated question-shaped use-sites without answer labels.

Important caveat:

This is not a final deployment method. It is a diagnostic scaffold to test
whether TSOC can work when applicability states are actually observed.

Result:

- future keys now matched the trace almost perfectly;
- update effects were large;
- layer-8 state moved closer to the teacher;
- later layers moved away;
- behavior still did not improve.

Representative trigger numbers:

| layer | heldout max cosine to trace | eval update-effect norm | trace target norm |
| ---: | ---: | ---: | ---: |
| 8 | 0.999 | 5.85 | 7.21 |
| 10 | 0.998 | 9.42 | 11.59 |
| 12 | 0.997 | 13.78 | 16.26 |

Layer-8 reentry improved:

| group | pre teacher L2 | post teacher L2 | ratio |
| --- | ---: | ---: | ---: |
| heldout | 8.70 | 8.14 | 0.935 |
| minimal pair | 8.75 | 8.15 | 0.931 |
| inverse polarity | 8.66 | 8.21 | 0.948 |

But behavior remained at baseline. This means TSOC can move an early internal
state toward the context teacher, but that state alone is not enough to repair
the model's answer behavior.

### Layer-8 Only

Run:

- `runs/tsoc_questionprobe_l8only_paper0_qwen17`

Result:

- layer-8 state reentry again improved;
- behavior still did not improve;
- margins moved only slightly.

Conclusion:

Layer-8 reentry is a real signal, but it is not sufficient. We need either a
readout bridge, a better target, or a multi-layer trajectory that does not
damage downstream semantics.

### Sequential Replay

Run:

- `runs/tsoc_questionprobe_seq_l8_10_12_paper0_qwen17`

Change:

- write layers one at a time;
- each later-layer target is computed against the currently edited no-context
  trajectory, not the original null trajectory.

Result:

- later-layer state reentry improved substantially;
- behavior still did not improve;
- negative examples got worse while positives stayed correct.

State reentry:

| group | layer | pre teacher L2 | post teacher L2 | ratio |
| --- | ---: | ---: | ---: | ---: |
| heldout | 10 | 13.23 | 11.03 | 0.834 |
| heldout | 12 | 19.96 | 15.06 | 0.755 |
| minimal pair | 10 | 13.29 | 11.03 | 0.830 |
| minimal pair | 12 | 20.03 | 14.96 | 0.747 |

Behavior:

- heldout stayed `0.50`;
- ordinary stayed `0.50`;
- minimal pair stayed `0.50`;
- inverse polarity stayed `0.50`;
- near collision stayed `0.583`.

Interpretation:

This is a sharper failure than before. TSOC can now re-enter substantial parts
of the teacher trajectory, but that trajectory component is still not the right
decision variable. The next bottleneck is target purification/readout, not just
triggering or source replay.

Current diagnosis:

- Paper-use TSOC is underdetermined because use-site keys do not match future
  question keys.
- Question-shaped TSOC proves the write can fire and move teacher-state
  geometry.
- Sequential replay can improve later-layer teacher proximity.
- None of these yet improve rule-correct behavior; the source term still seems
  contaminated by style/prompt/answer-prior components, or it reconstructs
  teacher states that are not causally sufficient for the correct answer.

Next research step:

Use TSOC as an internal-state engine, but add a readout/target-purification
test. Specifically, compare:

- source targets projected away from answer/template PCs;
- source targets selected by causal effect on hidden validity-state probes;
- a two-stage write: early TSOC reentry plus late calibrated answer actuator;
- teacher-state reconstruction at layer 8/10/12 followed by causal erasure to
  test whether the reentered directions are actually load-bearing.

## TSOC Target Purification and Component Sweep

Implemented:

- target nuisance projection in `scripts/tsoc_source_write.py`;
- reusable PCA/projection helpers in `caic.tsoc`;
- `scripts/tsoc_component_sweep.py`, which decomposes TSOC source targets into
  PCs, fits a closed-form update for each component, and directly patches the
  predicted MLP-output effect into eval prompts without writing weights.

Tests:

```text
.venv/bin/python -m pytest -q
24 passed
```

### Purified Sequential TSOC

Run:

- `runs/tsoc_questionprobe_seq_purified_l8_10_12_paper0_qwen17`

Settings:

- question-shaped trace probes;
- sequential replay across layers 8, 10, 12;
- projected away direct Yes/No unembedding direction;
- removed top 8 PCs from:
  - generic answer-control MLP outputs;
  - null-document style deltas.

Result:

- behavior still did not improve;
- negative examples still got worse in margins;
- state reentry still improved at layers 10/12 but not layer 8;
- the selected nuisance basis explained very little target energy:

| layer | nuisance energy removed | target norm before | target norm after | update norm |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0337 | 271.63 | 266.97 | 29.50 |
| 10 | 0.0301 | 276.91 | 272.55 | 15.95 |
| 12 | 0.0191 | 366.49 | 362.81 | 18.46 |

Interpretation:

This specific nuisance filter is too weak. The bad component is not mostly in
the direct Yes/No direction, generic answer-control PCs, or simple null-document
style PCs. The contaminating directions are likely task/readout-specific and
must be found by causal component testing.

### Layer-8 Component Sweep

Run:

- `runs/tsoc_component_sweep_l8_paper0_qwen17`

Settings:

- layer 8 only;
- top 4 source-target PCs plus full target;
- scales 0.5, 1.0, 2.0;
- same target purification as above.

Result:

- no accuracy changes in any bucket;
- best components only nudged margins.

Interpretation:

Layer 8 can show state reentry, but its source components are not directly
read out by the model's answer machinery at these scales.

### Layer-12 High-Scale Component Sweep

Run:

- `runs/tsoc_component_sweep_l12_highscale_paper0_qwen17`

Settings:

- layer 12 only;
- top 4 source-target PCs plus full target;
- scales 4, 8, 16.

Small-slice result:

- PC1 at scale 8 improved heldout from `0.50` to `0.833` on a 6-question slice;
- PC1 at scale 8 also improved near-collision from `0.667` to `0.833`;
- other components had clear negative-repair / positive-damage behavior.

Because that was a small slice, we validated PC1 on larger buckets.

### Layer-12 PC1 Validation

Run:

- `runs/tsoc_component_sweep_l12_pc1_validate_paper0_qwen17`

Settings:

- layer 12;
- full target, PC0, and PC1;
- scale 8;
- up to 20 questions per bucket.

PC1 validation result:

| bucket | baseline acc | patched acc | acc delta | pos acc delta | neg acc delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.60 | +0.10 | 0.00 | +0.20 |
| ordinary | 0.50 | 0.40 | -0.10 | -0.20 | 0.00 |
| minimal pair | 0.50 | 0.60 | +0.10 | 0.00 | +0.20 |
| inverse polarity | 0.50 | 0.55 | +0.05 | 0.00 | +0.10 |
| near collision | 0.50 | 0.65 | +0.15 | 0.00 | +0.30 |

This is the first component-level result that is not pure collapse. It improves
several buckets without positive accuracy regression in those buckets. However,
it still strongly reduces positive margins, and it hurts the ordinary bucket.
So PC1 is not a clean rule component. It is closer to a partially useful
negative-correction/readout component.

## Mini-Language Consolidation Pivot

The strongest current behavioral benchmark is no longer the Yes/No mini-paper
task. We added `scripts/minilang_write.py`, which teaches Qwen3-1.7B a tiny
invented translation language called Lyran through a sequence of lessons, then
tries to write the context-learned translation skill into weights.

Why this benchmark is better:

- the model has no pretrained knowledge of the invented token mapping;
- the context teacher can be checked directly;
- held-out prompts require composing tense, verb, subject, object, adjective,
  and English word-order rules;
- evaluation scores whole answer options by normalized option log-probability,
  not just a single answer letter.

The current strongest write configuration is:

- model: `Qwen/Qwen3-1.7B`;
- 4 Lyran lessons, 8 examples per lesson;
- token-level teacher-forced translation traces;
- target: context/no-context residual output delta;
- write site: layer 20;
- modules: attention output projection (`o_proj`) plus MLP down-projection;
- no Q/K/V projection writes;
- device: local Mac MPS, `float16`.

### Device Correction

Earlier local runs were often explicitly launched on CPU with `--device cpu
--dtype float32`. The machine does have Apple MPS available:

```text
torch 2.12.0
mps built True
mps available True
cuda available False
machine arm64
```

Defaults in `caic/modeling.py`, `scripts/minilang_lora_baseline.py`, and
`scripts/minilang_continual_triangle.py` now use `auto`, which selects MPS on
this machine. Current main mini-language runs use `--device mps --dtype
float16`.

### Focused Single-Write Results

Run:

- `runs/minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_stricteval_eval16_qwen17_mps`

Result on the first stricter de-duplicated eval:

| condition | correct | accuracy |
| --- | ---: | ---: |
| no context | 0/12 | 0.000 |
| full lessons in context | 12/12 | 1.000 |
| closed-form weight write | 7/12 | 0.583 |

This rerun removed 4 duplicate eval questions and logged overlap metadata. It
still allowed overlap with lesson examples/trace probes, so we ran a stronger
non-overlap audit.

Run:

- `runs/minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`

Settings:

- unique random eval questions;
- lesson-example overlaps excluded;
- trace-probe overlaps excluded.

Result:

| condition | correct | accuracy |
| --- | ---: | ---: |
| no context | 1/16 | 0.0625 |
| full lessons in context | 15/16 | 0.9375 |
| closed-form weight write | 7/16 | 0.4375 |

Internalization ratio on the no-overlap audit was `0.429`. This is the most
important current positive result: the effect survives an explicit no-overlap
check, though it is materially weaker than the easiest strict eval.

The strongest ensemble variant also survived a no-overlap audit:

- run: `runs/minilang_ensemble4_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- no context: `1/17`
- context: `16/17`
- edited: `7/17`
- internalization ratio: `0.400`

So the ensemble/shared-pattern path is not a duplicate artifact either, but it
still does not beat the focused write.

Interpretation:

- The closed-form mechanism can install some context-learned translation skill.
- It is not just memorizing duplicate questions.
- It is still not a clean full language learner. Errors show subject/object,
  verb, and tense confusions remain.

### Module Ablations

Old loose-eval ablations found:

| modules written | edited accuracy |
| --- | ---: |
| MLP down only | 6/16 |
| attention output only | 2/16 |
| attention output + MLP down | 11/16 |

Under stricter eval, attention output + MLP down remains the best tested
combination. This supports the user's intuition that attention and MLP effects
interact: O-projection writes alone are too weak, MLP writes alone are useful
but incomplete, and the joint actuator is stronger.

### Q/K Projection Writes

Implemented Q/K/V projection wrapping and capture plumbing in the mini-language
runner:

- `--write-attention-q`
- `--write-attention-k`
- `--write-attention-qk`
- `--write-attention-v`
- `--attention-projection-eta-scale`

The first objective for Q/K was deliberately local: make no-context Q/K
projection outputs match the context-conditioned Q/K projection outputs at the
same trace sites. MLP/O still used the residual output-delta target.

Full-strength Q/K result:

- run: `runs/minilang_write_4lesson_toktftrace_l20_qk_attno_mlp_outputdelta_stricteval_eval16_qwen17_mps`
- no context: `0/12`
- context: `12/12`
- edited: `2/12`

Small Q/K result:

- run: `runs/minilang_write_4lesson_toktftrace_l20_qk01_attno_mlp_outputdelta_stricteval_eval16_qwen17_mps`
- Q/K eta scale: `0.1`
- no context: `0/12`
- context: `12/12`
- edited: `3/12`

Conclusion: this Q/K target is currently harmful. It does not falsify attention
writes in general; it says raw Q/K projection-output matching is too invasive or
the wrong objective. Attention-output writes are still useful. Future Q/K work
should target head attention patterns, value retrieval locality, or use a much
stricter head-specific trust region rather than direct projection-output
matching.

### Ensemble / Shared-Pattern Experiments

The user proposed a pretraining-like idea: run several independent Lyran lesson
contexts and keep only weight-write directions shared across them, hoping random
context noise cancels while the true language skill reinforces.

Implemented:

- `--ensemble-corpora`
- `--ensemble-reduction mean|sum|snr|directional`
- `--ensemble-shared-probes`
- `--ensemble-per-lesson`

One-shot ensemble writes over a full corpus were weak:

| run | reduction | edited accuracy |
| --- | --- | ---: |
| `runs/minilang_ensemble4_mean_l20_attno_mlp_outputdelta_eval16_qwen17_cpu` | mean | 4/16 |
| `runs/minilang_ensemble4_snr_sharedprobes_l20_attno_mlp_outputdelta_eval16_qwen17_mps` | SNR | 4/16 |
| `runs/minilang_ensemble4_directional_sharedprobes_l20_attno_mlp_outputdelta_eval16_qwen17_mps` | directional | 3/12 |

The directional one-shot run had very high proposal alignment:

- attention proposal cosine mean about `0.848`;
- MLP proposal cosine mean about `0.829`;
- directional agreement about `0.94` for attention and `0.93` for MLP.

So there is a real shared direction across independent lesson renderings, but a
single full-corpus shared-direction write is not enough behaviorally.

Per-lesson ensemble writes were better:

| run | reduction | edited accuracy |
| --- | --- | ---: |
| `runs/minilang_ensemble4_directional_perlesson_l20_attno_mlp_outputdelta_stricteval_eval16_qwen17_mps` | directional | 7/12 |
| `runs/minilang_ensemble4_mean_perlesson_l20_attno_mlp_outputdelta_stricteval_eval16_qwen17_mps` | mean | 4/12 |
| `runs/minilang_ensemble4_snr_perlesson_l20_attno_mlp_outputdelta_stricteval_eval16_qwen17_mps` | SNR | 6/12 |

Per-lesson directional matched the best focused strict result, while per-lesson
SNR was close and per-lesson mean did not. Interpretation:

- The shared signal is real and stable across independent lesson contexts.
- Simple averaging keeps too much context-specific noise.
- SNR/coordinate gating is better than mean, which supports the "shared pattern"
  hypothesis.
- Directional consensus removes the most noise and matches the focused write,
  but it also appears to discard useful lexical binding/content, so it has not
  beaten the focused write.

The likely next architecture is a two-channel consolidation:

- a shared grammar/operator direction learned across many lesson contexts;
- separate lexical/content binding writes that are not averaged away.

### Anchored Shared-Plus-Residual Denoising

Implemented a first two-channel denoising variant:

- `--ensemble-include-anchor`
- `--ensemble-reduction anchored_directional`
- `--ensemble-anchor-residual-scale`

Mechanism:

1. Estimate a shared update from four independently rendered Lyran lesson
   corpora using directional consensus.
2. Compute the actual anchor update from the real lesson trace.
3. Keep the shared update at full strength.
4. Add only the anchor component orthogonal to the shared direction, scaled by a
   residual coefficient.

This explicitly tests the hypothesis that the stable cross-context direction
contains reusable grammar/translation structure, while the anchor residual
contains useful content binding plus some noise.

No-overlap eval after fixing per-lesson trace-overlap accounting:

| run | anchor residual scale | no context | context | edited | internalization ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `runs/minilang_ensemble4_anchor025_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 0.25 | 0/15 | 14/15 | 6/15 | 0.429 |
| `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 0.50 | 0/15 | 14/15 | 7/15 | 0.500 |
| `runs/minilang_ensemble4_anchor075_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 0.75 | 0/15 | 14/15 | 5/15 | 0.357 |

This is the first denoising variant that improves the no-overlap internalization
ratio over the focused write and pure directional ensemble. The effect is small
but directionally important:

- too little anchor residual under-adds useful actual-trace content;
- too much anchor residual reintroduces noise;
- the current sweet spot is around `0.5`.

The run also logs useful geometry. Anchor/shared cosines are usually high
(`~0.8-0.9`), but the anchor residual norms are still large. That supports the
two-channel picture: the anchor is mostly aligned with the shared update, but
there is a substantial residual that can help or hurt depending on scale.

### Option-Negative Dead End

Implemented ensemble support for `--option-negative-keys`, where wrong answer
prefixes are added as protected negative keys. This was motivated by the
remaining mini-language errors, which are often plausible wrong translations.

Two tested forms:

1. Attention + MLP option negatives, 64 wrong-prefix prompts, default negative
   weight.
2. MLP-only option negatives, 24 wrong-prefix prompts, negative weight `0.1`.

The first completed run:

- `runs/minilang_ensemble4_anchor05_optneg_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- no context: `0/15`
- context: `14/15`
- edited: `1/15`

The second was stopped early because the same pathology appeared: MLP proposals
hit the update norm cap and proposal alignment collapsed.

Interpretation:

- wrong-answer states are probably the right kind of negative information;
- adding them directly as equal-class ridge negatives is far too strong;
- future versions need a separate soft option penalty or post-solve projection,
  not a large block of wrong-answer states inside the same protected solve.

Follow-up post-solve projection test:

- `runs/minilang_ensemble4_anchor05_mlpoptproj_s03_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- mode: `--option-negative-mode project`
- projection strength: `0.3`
- option negative prompts: 24 MLP-only wrong-prefix states
- no context: `0/15`
- context: `14/15`
- edited: `7/15`
- internalization ratio: `0.500`

This matched the anchor-only `4-support / residual=0.5` result exactly on
accuracy, while slightly improving the edited mean margin (`-0.287` vs
`-0.310`). Projection removed meaningful MLP update mass
(`~4.7` Frobenius norm on average), but it did not change the behavioral score.
Conclusion: wrong-option suppression is not the main remaining bottleneck in
this setup.

Current denoising status:

> Shared directional update + half-strength anchor residual is the best current
> "less noisy" write. Option negatives are promising diagnostically, but ridge
> option negatives are harmful and soft projection is neutral so far.

### Shared-Support Count and Probe-Coverage Sweep

User hypothesis tested: if several independent contexts teach the same
underlying object, the useful update directions should be shared while local
trace noise should cancel. I swept the number of support corpora and a small
anchor-residual adjustment.

Seed 1, strict no-overlap eval:

| run | support corpora | anchor residual | no context | context | edited | internalization ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 4 | 0.50 | 0/15 | 14/15 | 7/15 | 0.500 |
| `runs/minilang_ensemble8_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 8 | 0.50 | 0/15 | 14/15 | 8/15 | 0.571 |
| `runs/minilang_ensemble12_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 12 | 0.50 | 0/15 | 14/15 | 4/15 | 0.286 |
| `runs/minilang_ensemble8_anchor065_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | 8 | 0.65 | 0/15 | 14/15 | 5/15 | 0.357 |

Interpretation:

- 8 support corpora improved seed 1 by one strict heldout item.
- 12 support corpora was worse, despite lower update norms and high
  anchor/shared cosine. More averaging is not monotonically better.
- Increasing the anchor residual from `0.50` to `0.65` was worse, so the failure
  was not simply under-adding the anchor trace.
- The likely explanation is a real bias-variance tradeoff: a few independent
  lesson contexts reduce local trace noise; too many support contexts pull the
  update toward a generic shared translation basin and wash out fragile lexical
  bindings.

I also tested deterministic balanced trace probes:

- `runs/minilang_ensemble8_anchor05_baltrace_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- no context: `1/15`
- context: `14/15`
- edited: `1/15`
- internalization ratio: `0.000`

Balanced probes lowered proposal alignment (`attention_o` agreement `~0.849`
vs `~0.889`; MLP agreement `~0.831` vs `~0.873`) and collapsed behavior. In
this tiny-probe regime, hand-balanced coverage is worse than the random shared
probes. The balanced probes probably become too unrepresentative of the actual
lesson/use distribution.

Cross-seed validation of the current best family:

| seed | support corpora | no context | context | edited | internalization ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0/15 | 14/15 | 7/15 | 0.500 |
| 1 | 8 | 0/15 | 14/15 | 8/15 | 0.571 |
| 2 | 4 | 0/17 | 11/17 | 8/17 | 0.727 |
| 2 | 8 | 0/17 | 11/17 | 7/17 | 0.636 |

This is the strongest anti-false-positive result so far:

- The anchored closed-form write transfers across a second seed.
- Internalization ratio is actually higher on seed 2, though the context upper
  bound is weaker (`11/17`).
- The 8-support improvement does not replicate on seed 2; 4 support corpora
  are better there.

Updated conclusion:

> The less-noisy mechanism is real enough to survive a second seed, but the
> support-count optimum is unstable. Current best default should be
> `anchored_directional`, residual scale `0.5`, 4 support corpora for cheaper
> runs, and 8 support corpora only as an exploratory variant. Do not claim that
> more context averaging monotonically improves consolidation.

### Module Ablation: MLP-Only vs Attention+MLP

Ran the anchored denoising recipe with only the MLP down-projection write, no
attention `o_proj` write:

- `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_mlp_only_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- no context: `0/15`
- context: `14/15`
- edited: `4/15`
- internalization ratio: `0.286`

Same seed/eval with attention `o_proj` + MLP:

- `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- edited: `7/15`
- internalization ratio: `0.500`

Interpretation:

- MLP-only is not enough for this mini-language write.
- Attention `o_proj` is contributing real acquisition in the current recipe.
- This is consistent with the user's intuition that the whole residual update
  path matters: the MLP write alone moves some lexical cases correctly, but it
  also produces large wrong-choice margins and loses several cases that
  attention+MLP gets right.
- Earlier raw Q/K projection-output writes were harmful, so the next attention
  direction should not be "write every attention matrix." The useful attention
  actuator so far is `o_proj`; Q/K/V need a different objective before being
  included.

### General Content-Binding Channel Attempt

Important correction from the user: we should not overfit the method to the
mini-language benchmark. I first implemented a task-specific lexical channel
from the hidden mini-language word inventory, then stopped that run before
using it as evidence. That channel remains in the script as a diagnostic flag,
but it should not be used for claims about general context consolidation.

I then implemented a more general `--context-span-channel`:

- It reads only the rendered context text.
- It extracts self-supervised completion traces from context spans such as
  `x=y`, `x -> y`, and ordinary line continuations.
- It does not inspect hidden DSL objects, word lists, labels, or the benchmark
  generator.
- It writes an auxiliary MLP channel from context-with-span to span-only traces.

First run, raw context-span update:

- `runs/minilang_ensemble4_anchor05_ctxspan010_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- context-span mode: `raw`
- context-span scale: `0.1`
- no context: `0/15`
- context: `14/15`
- edited: `6/15`
- internalization ratio: `0.429`

Baseline without context-span augmentation:

- `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- edited: `7/15`
- internalization ratio: `0.500`

Diagnostics:

- context-span update norm: `~22.5`
- scaled context-span update norm: `~2.25`
- cosine with the sentence/use-trace update: only `~0.13`

Interpretation: generic context-span completion mostly points somewhere other
than the useful sentence-level consolidation direction. Even a small raw
addition is enough to flip one correct heldout item wrong.

I then added `--context-span-channel-mode aligned`, which projects the
context-span update onto the positive direction of the main sentence/use-trace
write before applying it. This tests the user's "only reinforce shared
patterns" idea in a more general form.

Aligned run:

- `runs/minilang_ensemble4_anchor05_ctxspanaligned100_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- context-span mode: `aligned`
- context-span scale: `1.0`
- no context: `0/15`
- context: `14/15`
- edited: `6/15`
- internalization ratio: `0.429`

Diagnostics:

- raw span update norm: `~23.2`
- aligned component norm: `~3.0`
- cosine with sentence/use-trace update: `~0.13`

Conclusion:

> Generic context-span completion is the right kind of general machinery, but
> the current selector is too surface-level. The useful binding signal is not
> "complete arbitrary spans from the context"; it is probably "complete or
> restate spans that are causally coupled to later use-trace computations." Do
> not use the context-span channel by default until span selection is improved.

Current safe default remains:

- `--ensemble-reduction anchored_directional`
- `--ensemble-anchor-residual-scale 0.5`
- 4 support corpora for cheaper validation
- sentence/use-trace attention `o_proj` + MLP writes
- no Q/K/V writes
- no generic context-span channel

### Packed Use-Trace Consolidation

User hypothesis: instead of writing from one noisy final answer state, create a
small quiz/use episode after the lesson and capture many pre-answer positions.
This should expose multiple facets of the same latent understanding, like a
movie rather than one snapshot.

Implemented:

- `--packed-use-trace`
- `--packed-use-mode clean|curriculum`
- `--packed-use-span-items N`

Mechanism:

- Build packed quiz items from the existing trace translation probes.
- Teacher prompts include the lesson plus quiz/use item(s).
- Student prompts include the same quiz/use item(s) without the lesson.
- `clean` mode captures each item independently.
- `curriculum` mode keeps previous Q/A pairs in context before later answer
  boundaries, approximating one long quiz transcript.

Results on the seed-1 strict no-overlap mini-language eval:

| run | trace mode | no context | context | edited | internalization ratio | seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | separate eval-style use traces | 0/15 | 14/15 | 7/15 | 0.500 | 339 |
| `runs/minilang_ensemble4_anchor05_packedclean_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | packed clean | 0/15 | 14/15 | 3/15 | 0.214 | 415 |
| `runs/minilang_ensemble4_anchor05_packedcurr_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | packed curriculum | 0/15 | 14/15 | 2/15 | 0.143 | 1072 |

Diagnostics:

- Packed curriculum had very high proposal alignment, even higher than the
  baseline, but much worse behavior. That means the packed transcript creates a
  stable update direction, but it is the wrong direction for no-context
  deployment.
- Packed clean was cheaper than curriculum but still much worse than the
  separate eval-style trace.
- Curriculum mode is much less compute efficient because prompt length grows
  with previous Q/A pairs.

Interpretation:

> The packed-use idea is conceptually strong, but this implementation changed
> the key distribution too much. We wrote "quiz transcript state" rather than
> the state that appears during deployment translation prompts.

The lesson is not that many use-sites are bad. The lesson is that use-sites
must be activation-compatible with deployment. Current separate use-trace
prompts work better because they use the same translation prompt format as
evaluation. A stronger packed version should either:

- capture multiple answer boundaries in a single natural episode without
  changing the per-item prompt format; or
- pack eval-style prompts while preserving the exact `Translate... English:`
  answer boundary; or
- use true multi-position hooks over one forward pass rather than simulating
  packed positions as many altered prefix prompts.

### Deployment-Compatible Perspective Traces

Follow-up on the same user principle: the model can think about an idea from
many perspectives, but we must preserve deployment compatibility and remove
non-understanding noise.

Implemented perspective trace prompts:

- `--trace-perspectives direct grammar lexicon roles`
- `--teacher-perspectives-only`

All perspective prompts preserve the final answer boundary:

```text
Lyran: ...
English:
```

The perspectives only change the instruction before that boundary:

- `direct`: translate directly.
- `grammar`: use word-order/grammar rules.
- `lexicon`: recall word meanings.
- `roles`: identify tense/action/subject/object/modifiers internally.

Tested variants:

| run | perspective handling | no context | context | edited | internalization ratio | seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/minilang_ensemble4_anchor05_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | direct only baseline | 0/15 | 14/15 | 7/15 | 0.500 | 339 |
| `runs/minilang_ensemble4_anchor05_persp2_roles_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | direct+roles, both teacher and student keys vary | 0/15 | 14/15 | 2/15 | 0.143 | 917 |
| `runs/minilang_ensemble4_anchor05_persp4_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | four perspectives, both teacher and student keys vary | 0/15 | 14/15 | 6/15 | 0.429 | 1539 |
| `runs/minilang_ensemble4_anchor05_teacherpersp4_directional_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps` | four teacher perspectives, direct deployment student keys | 0/15 | 14/15 | 4/15 | 0.286 | 1761 |

Diagnostics:

- Four-perspective concat multiplied trace rows from `~26` to `~104`,
  increased update norms substantially, and lowered proposal agreement.
- Two-perspective direct+roles was even worse, suggesting the roles prompt
  creates harmful non-deployment structure when used as a student key.
- Teacher-perspectives-only restored update geometry close to the direct
  baseline:
  - attention update norm `~30.9` vs baseline `~31.4`;
  - MLP update norm `~18.2` vs baseline `~18.9`;
  - proposal cosine and directional agreement roughly matched baseline.
- Despite the better geometry, behavior dropped to `4/15`. This means averaging
  multiple teacher perspective targets into the same direct key diluted the
  useful direct target rather than purifying it.

Updated interpretation:

> Perspective traces are useful conceptually, but not as extra targets to
> average into the write. The better use is probably as a filter: compute the
> direct deployment update first, then keep only components of that update that
> are supported by perspective-induced teacher deltas.

Next engineering direction:

- Compute a direct update `U_direct` using deployment-style teacher/student
  traces.
- Compute perspective updates `U_p` using perspective teacher prompts but direct
  student keys.
- Do not average `U_p` into the write.
- Instead, project or gate `U_direct` by agreement with the perspective family:
  keep components of the direct update whose direction is supported by grammar,
  lexicon, and role perspectives; shrink components that only appear in the
  direct answer posture.

This would turn perspectives into a nuisance-removal mechanism rather than a
larger noisy target set.

### Perspective Filtering Follow-Up

Implemented in `scripts/minilang_write.py`:

- `--perspective-filter`
- `--perspective-filter-granularity {update,target}`
- `--perspective-filter-mode {project,cosine_scale}`
- `--perspective-target-filter-*`

The first implementation used perspectives as filters, not as additional write
targets.

Update-level projection filter:

- run:
  `runs/minilang_ensemble4_anchor05_pfilter_project025_persp4_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- setup: same direct baseline, plus `direct grammar lexicon roles`;
- filter: compute direct deployment-shaped update, compute perspective support
  updates using direct student keys, project direct update into the support
  direction plus `0.25` direct residual.

Result:

| run | edited | internalization ratio | edited mean margin | seconds |
| --- | ---: | ---: | ---: | ---: |
| direct baseline | 7/15 | 0.500 | -0.310 | 339 |
| update-level perspective filter | 4/15 | 0.286 | -0.381 | 1007 |

Geometry:

- applied attention update norm: `30.39` vs baseline `31.38`;
- applied MLP update norm: `18.02` vs baseline `18.86`;
- perspective support updates were highly self-consistent, but this did not
  improve behavior.

Target-row filter:

- permissive run with threshold `0.25` was stopped early because it was nearly
  a no-op:
  - mean target agreement was about `0.97`;
  - mean gate was about `0.997`.
- aggressive run:
  `runs/minilang_ensemble4_anchor05_pfilter_target_t097_floor025_persp4_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- setup: threshold `0.97`, temperature `20`, floor `0.25`.

Result:

| run | edited | internalization ratio | edited mean margin | mean gate | seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| target-row perspective filter | 4/15 | 0.286 | -0.514 | 0.567 | 942 |

Geometry:

- applied attention update norm fell to `17.54`;
- applied MLP update norm fell to `10.75`;
- mean target cosine across perspectives was still high at `0.935`.

Interpretation:

Perspective prompts were not a good decomposition of "understanding" versus
noise. They mostly agreed with the direct target on broad translation-answer
structure. When the filter was permissive, it did almost nothing. When it was
aggressive, it removed useful write strength without improving purity.

This does not falsify the user's broader idea. It falsifies the simple version:
"ask the model for grammar/lexicon/role perspectives and filter by agreement."
The perspectives are too close in activation/target space to isolate the
load-bearing language skill.

### Low-Rank Shared-Subspace Ensemble

Implemented in `scripts/minilang_write.py`:

- `--ensemble-reduction subspace`
- `--ensemble-reduction anchored_subspace`
- `--ensemble-subspace-rank`

Motivation:

Directional consensus keeps one shared update direction across independently
rendered corpora. The user's hypothesis suggests the reusable language skill may
span multiple directions, while idiosyncratic per-render noise should lie
outside the shared subspace.

Test:

- run:
  `runs/minilang_ensemble4_anchor05_anchsubspace_r2_perlesson_l20_attno_mlp_outputdelta_unique_nooverlap_eval24_qwen17_mps`
- same config as the direct anchored-direction baseline;
- only changed reducer to `anchored_subspace`, rank `2`.

Result:

| run | edited | internalization ratio | edited mean margin | seconds |
| --- | ---: | ---: | ---: | ---: |
| anchored directional baseline | 7/15 | 0.500 | -0.310 | 339 |
| anchored subspace rank 2 | 5/15 | 0.357 | -0.335 | 374 |

Geometry:

- applied attention update norm: `33.27`;
- applied MLP update norm: `19.78`;
- mean anchor energy captured by support subspace: `0.714`.

Interpretation:

The second shared direction was not a better "understanding" direction. It
added back stable-but-not-useful structure. For this runner, the older
single-direction anchored consensus remains the best reducer.

Updated conclusion:

The "many perspectives / many renderings" idea is still plausible, but the
filter needs a better definition of shared substance. Raw agreement across
prompted perspectives or low-rank update geometry is insufficient. Agreement
must be measured in a causally relevant coordinate system: minimal-pair
behavior, inverse-polarity preservation, near-collision locality, or internal
rule-state re-entry. Plain target/update agreement is too coarse.

### LoRA Baseline

Run:

- `runs/minilang_lora_l20_attno_mlp_r8_trace_lr1e3_eval4_eval16_qwen17_mps`

This trained a small LoRA-style baseline on trace data. It was stopped at step
40 because the curve was flat and it was tying up the MPS device.

| step | eval accuracy |
| ---: | ---: |
| 0 | 0/16 |
| 4 | 0/16 |
| 8 | 1/16 |
| 12 | 4/16 |
| 16 | 2/16 |
| 20 | 2/16 |
| 24 | 2/16 |
| 28 | 1/16 |
| 32 | 2/16 |
| 36 | 2/16 |
| 40 | 2/16 |

This is not a definitive LoRA baseline, but it is a useful compute-matched
smoke test: the closed-form write gets materially more acquisition from a small
number of forward traces than this tiny gradient baseline did.

### Current Mini-Language Diagnosis

The closed-form write mechanism is working in a narrow but real sense:

- the model with context can translate Lyran;
- the no-context model mostly cannot;
- the closed-form write transfers a nontrivial fraction of the context
  teacher's behavior into no-context performance;
- the effect survives de-duplication and explicit overlap exclusion.

But the write is still noisy:

- Q/K projection matching damages performance;
- mean ensemble averaging does not cleanly isolate the shared language skill;
- directional consensus finds shared update geometry but does not improve over
  the focused write;
- edited margins are often shallow, so many correct answers are only barely
  correct.

Current best claim:

> Forward-pass closed-form writes can partially consolidate an in-context
> learned mini-language translation skill into weights. The remaining bottleneck
> is target decomposition: separate reusable grammar dynamics from lexical
> bindings and from prompt/answer noise.

Tests currently pass:

```text
.venv/bin/python -m pytest -q
28 passed
```

### Repeated-Lesson Scale Check

Naively expanding Lyran beyond 4 lessons failed because the context teacher got
weak before consolidation could be tested:

| setup | no-context acc | context acc | note |
| --- | ---: | ---: | --- |
| 5 lessons, expanding language | 2/16 | 9/16 | teacher too weak |
| 6 lessons, expanding language | 5/23 | 9/23 | teacher too weak |
| 8 lessons, expanding language | 2/22 | 11/22 | teacher too weak; run stopped before write |

This is not a CAIC failure. It is Stage 0: Qwen3-1.7B did not reliably learn the
larger language in context.

To test the user's "many perspectives on the same object" hypothesis, we froze
the language after lesson 4 and added more lessons as additional examples/views
of the same underlying mini-language. To keep a real held-out set, we reduced
lesson examples and trace probes:

- 8 lessons;
- language frozen after lesson 4;
- 4 examples per lesson;
- 2 trace probes per lesson;
- exhaustive modified eval;
- exact lesson-example and trace-probe overlaps excluded.

Run:

- `runs/minilang_write_8lesson_freeze3_ex4_trace2_l20_attno_mlp_outputdelta_exhaustive_nooverlap_qwen17_mps`

Result:

| condition | correct | accuracy |
| --- | ---: | ---: |
| no context | 1/12 | 0.083 |
| full repeated lessons in context | 10/12 | 0.833 |
| closed-form weight write | 5/12 | 0.417 |

Internalization ratio was `0.444`, similar to the 4-lesson no-overlap audit.
This is a useful positive scale check: repeated lessons about the same frozen
object remain consolidatable at roughly the same fraction of the context
teacher's gain. The bottleneck is not simply number of lessons; it is whether
the teacher can hold the expanded rule system in context and whether held-out
coverage remains clean.

Current diagnosis:

- Simple nuisance projection is insufficient.
- Layer-8 components are internally interesting but behaviorally weak.
- Layer-12 contains components that can improve some adversarial buckets, but
  they are still entangled with answer calibration.
- The next architecture should not write the full source field. It should
  select or synthesize a small component mixture using validation constraints:
  improve negatives, preserve positives, preserve inverse polarity, preserve
  near-collision locality, and improve margins without broad positive-margin
  collapse.

Next concrete experiment:

Build a constrained component-combination solver over layer-12 components, using
direct patching as the oracle:

- candidate components: layer-12 PC0-PCk plus maybe layer-10 PCs;
- objective: improve worst-bucket margin/accuracy;
- constraints: no positive accuracy regression, no inverse-polarity regression,
  no near-collision regression;
- then only if a patched mixture passes, fit the equivalent closed-form memory
  write.

## Constrained TSOC Component-Combination Search

Implemented:

- `scripts/tsoc_component_combo_search.py`

What it does:

- extracts TSOC source targets from question-shaped traces;
- decomposes targets into principal components per layer;
- fits protected ridge down-projection updates for each component;
- directly patches mixtures of predicted component effects into evaluation
  prompts;
- searches on one half of each evaluation bucket and reports the other half as
  a held-out split;
- applies constraints for no group-level accuracy regression, no positive
  accuracy regression, and bounded positive-margin damage.

This is still a diagnostic patching oracle, not a persistent weight-write run.
The point is to avoid writing mixtures that only look good because they repair
the model's negative-label errors by damaging positive cases.

### Layer 10 + Layer 12 Coarse Mixture Search

Run:

- `runs/tsoc_combo_l10_12_paper0_qwen17`

Settings:

- layers 10 and 12;
- top 2 source PCs per layer;
- coefficients `-8, -4, 0, 4, 8`;
- 12 random sparse two-component trials;
- answer-direction projection;
- 8 nuisance PCs from answer controls and null-document deltas;
- 12 questions per bucket, split into search/test halves.

Result:

- 23 candidate mixtures scored;
- 13 accepted on the search split;
- accepted candidates mostly improved margins without changing accuracy;
- accuracy-improving candidates were rejected because they damaged positive
  margins or regressed at least one gauntlet bucket.

Representative accepted candidate:

- `l10_pc1=-8`
  - search mean accuracy delta: `0.00`
  - test mean accuracy delta: `0.00`
  - test mean margin delta: `+1.45`
  - no group-level accuracy regression

Representative rejected pattern:

- positive `l10_pc1` or `l12_pc1` coefficients improved some negative-heavy
  buckets, but strongly reduced positive margins and sometimes hurt ordinary or
  minimal-pair buckets.

Interpretation:

The multi-layer search did not reveal a clean rule-consolidation mixture. It
did show that some source components can cleanly reshape confidence/margins,
but the components that cross answer thresholds are still entangled with
answer-polarity steering.

### Smaller Positive-Coefficient Search

Run:

- `runs/tsoc_combo_l10_12_smallpos_paper0_qwen17`

Settings:

- layers 10 and 12;
- top 2 PCs per layer;
- coefficients `0, 1, 2, 3, 4, 5, 6`;
- 8 random sparse two-component trials;
- same projection and gauntlet setup as above.

Result:

- 29 candidate mixtures scored;
- 14 accepted on the search split.

Notable candidate:

- `l10_pc0=6`
  - search mean accuracy delta: `0.00`
  - test mean accuracy delta: `+0.033`
  - test minimum group accuracy delta: `0.00`
  - test minimum positive-margin delta: `+0.193`

Bucket detail for `l10_pc0=6` on the small test split:

- heldout: `0.833 -> 1.000`
- ordinary: `0.500 -> 0.500`
- minimal pair: `0.833 -> 0.833`
- inverse polarity: `0.500 -> 0.500`
- near collision: `0.500 -> 0.500`

This looked cleaner than the layer-12 PC1 result because it preserved positive
margins instead of crushing them.

### Layer-10 PC0 Larger Validation

Run:

- `runs/tsoc_combo_l10_pc0_validate_paper0_qwen17`

Settings:

- layer 10 only;
- top source PC only;
- coefficients `0, 4, 5, 6, 7, 8`;
- 20 questions per bucket, split into search/test halves;
- same projection and gauntlet setup.

Result:

| candidate | search mean acc delta | test mean acc delta | test min acc delta | test min positive-margin delta |
| --- | ---: | ---: | ---: | ---: |
| `l10_pc0=4` | 0.00 | 0.00 | 0.00 | +0.400 |
| `l10_pc0=5` | 0.00 | 0.00 | 0.00 | +0.416 |
| `l10_pc0=6` | 0.00 | +0.020 | 0.00 | +0.456 |
| `l10_pc0=7` | 0.00 | +0.020 | 0.00 | +0.362 |
| `l10_pc0=8` | 0.00 | 0.00 | 0.00 | +0.320 |

Bucket detail for `l10_pc0=6` on the larger held-out split:

- heldout: `0.60 -> 0.70`
- ordinary: `0.50 -> 0.50`
- minimal pair: `0.80 -> 0.80`
- inverse polarity: `0.60 -> 0.60`
- near collision: `0.50 -> 0.50`

Interpretation:

Layer 10 PC0 appears to be a modest, clean confidence/readout direction. It can
flip a small number of heldout negatives without damaging positive cases or the
gauntlet. However, the gain is tiny and appears only on heldout, not on minimal
pairs or near-collision tests. This is not yet evidence of rule consolidation.

Current diagnosis after component-combo search:

- We have a clean diagnostic pipeline for constrained component mixtures.
- We found one mild layer-10 component that is safer than the previous layer-12
  PC1 result.
- The components that produce larger accuracy gains still look like
  negative-answer steering.
- The next step should move from unsupervised source PCs to a target basis
  aligned with hidden DSL rule state, because ordinary source-energy PCs are
  not isolating the variable we actually care about.

Next concrete experiment:

Use the executable DSL labels to build evaluation-only rule-state probes, then
derive target directions that align with rule validity or violation state rather
than source-target variance. Test whether patching those probe-aligned source
components improves minimal pairs and near-collision buckets without the
positive-margin collapse seen in answer-polarity components.

## Rule-Probe Target Diagnostic

Extended:

- `scripts/tsoc_component_combo_search.py`

New mode:

- `--basis-mode rule_probe`

This mode uses the synthetic DSL labels as an instrumentation-only diagnostic.
It fits a linear validity direction on the no-document trace activations, then
constructs a minimal target that should push trace states across a rule-validity
margin. The target is still projected away from the answer unembedding direction
and nuisance PCs when those flags are enabled.

This is not a deployable consolidation rule, because real contexts do not come
with hidden DSL labels. It is a test of actuator capacity: if even a supervised
rule-state direction cannot move behavior, then unsupervised source PCs are not
the only bottleneck.

### Rule-Probe Layers 10 and 12

Run:

- `runs/tsoc_ruleprobe_l10_12_validate_paper0_qwen17`

Settings:

- layers 10 and 12;
- rule-probe target basis only;
- scales `1, 2, 4, 8`;
- answer-direction projection;
- 8 nuisance PCs from answer controls and null-document deltas;
- 20 questions per bucket, split into search/test halves.

Result:

- all candidates preserved accuracy;
- no candidate changed accuracy in any bucket;
- layer 10 rule-probe at scale 8 had the best test score, but only via margin
  movement:
  - test mean accuracy delta: `0.00`
  - test mean margin delta: `-0.055`
  - test minimum positive-margin delta: `+0.275`

### Rule-Probe High Scale

Run:

- `runs/tsoc_ruleprobe_l10_12_highscale_paper0_qwen17`

Settings:

- same setup as above;
- scales `16, 32`.

Result:

| candidate | search mean acc delta | test mean acc delta | test min acc delta | test min positive-margin delta |
| --- | ---: | ---: | ---: | ---: |
| `l10_rule_probe=16` | 0.00 | 0.00 | 0.00 | +0.627 |
| `l10_rule_probe=32` | 0.00 | 0.00 | 0.00 | +1.373 |
| `l12_rule_probe=16` | 0.00 | 0.00 | 0.00 | -0.359 |
| `l12_rule_probe=32` | 0.00 | 0.00 | 0.00 | -0.569 |

Even at high scale, rule-probe targets did not flip decisions. They can move
margins, but they do not produce usable heldout or gauntlet accuracy gains.

Interpretation:

- The bottleneck is not just that source PCA selected the wrong direction.
- A supervised validity-aligned target is still not sufficient when injected
  through these MLP down-projection actuator sites.
- The model may linearly encode rule validity in hidden states while the final
  Yes/No decision does not actually read that variable out for these prompts.
- The next useful diagnostic is causal mediation: patch the rule-probe direction
  at several residual/block sites and ask which later layer, if any, converts it
  into answer logits. If no site does, then this benchmark/model pair has a
  readable-but-not-used rule-state feature.

Updated diagnosis:

We have separated three phenomena:

1. Late answer-direction patches can move behavior, but are not rule
   consolidation.
2. TSOC source writes can move teacher-state geometry, but often do not reach
   the final decision.
3. Supervised rule-probe targets are readable but not behaviorally decisive at
   layers 10 or 12.

The next mechanism should include a readout bridge, not only a source-state
write. Concretely: find the layer/path where the context teacher converts the
rule-state feature into answer evidence, then write both the rule-state source
and a small downstream readout adapter/source term, while keeping the answer
prior controls.

## Residual Readout Bridge Diagnostics

Implemented:

- `scripts/residual_readout_sweep.py`

What it tests:

- direct residual-stream patching after decoder blocks;
- supervised validity-probe patches as an actuator/readout diagnostic;
- direct answer-direction residual patches as a controllability control;
- teacher replay targets:
  - `teacher_delta = block_output(doc+q) - block_output(null+q)`;
  - `teacher_source = (doc_out-null_out) - (doc_in-null_in)`.

This is a diagnostic patching script, not a weight-write method. The point is
to locate where context-created readout signals are causally useful.

### Residual Validity-Probe Sweep

Run:

- `runs/residual_validity_readout_layers8_24_paper0_qwen17`

Settings:

- layers 8, 10, 12, 16, 20, 24;
- target mode `validity_probe`;
- patch modes `final` and `suffix`;
- scales `8, 16, 32`;
- 12 questions per bucket.

Result:

- Direct residual validity patches can move margins strongly.
- Layer 8 suffix scale 32 improved margins across normal buckets:
  - heldout margin delta: `+7.69`;
  - ordinary margin delta: `+6.84`;
  - minimal-pair margin delta: `+5.64`;
  - near-collision margin delta: `+7.15`.
- But these layer-8 margin gains did not flip accuracy.
- Later layers improved inverse-polarity accuracy, e.g.:
  - layer 10 final scale 32: inverse `0.50 -> 0.75`;
  - layer 12 suffix scale 32: inverse `0.50 -> 0.667`;
  - layer 16 final scale 32: inverse `0.50 -> 0.667`.
- Normal heldout/ordinary/minimal/near-collision did not get clean accuracy
  gains from validity-probe residual patches.

Interpretation:

The hidden validity variable is not by itself the missing readout bridge. It is
readable and margin-active, but injecting it does not reliably cause the model
to answer using the rule. This supports the hypothesis that the context teacher
contains an additional downstream answer-use/readout signal.

### Residual Answer-Direction Control

Runs:

- `runs/residual_answer_control_layers8_24_paper0_qwen17`
- `runs/residual_answer_control_l16_highscale_paper0_qwen17`

Settings:

- answer-direction residual patches;
- initial scales up to `8`, then high-scale layer-16 suffix controls at `16,
  32, 64`.

Result:

- Scales up to 8 mostly moved margins without changing accuracy.
- Layer 16 suffix high-scale patches were behaviorally decisive:
  - scale 32 heldout: `0.625 -> 0.750`;
  - scale 64 heldout: `0.625 -> 0.875`;
  - scale 64 minimal-pair: `0.500 -> 0.750`;
  - scale 64 inverse-polarity: `0.500 -> 0.875`;
  - scale 64 near-collision: `0.625 -> 0.750`.

Interpretation:

The residual patching path can flip behavior, but it needs a large direct
answer-evidence vector. This is a control only; it is not rule consolidation.

### Teacher Residual Replay

Run:

- `runs/residual_teacher_replay_layers8_24_paper0_qwen17`

Settings:

- layers 8, 12, 16, 20, 24;
- target modes `teacher_delta` and `teacher_source`;
- patch modes `final` and `suffix`;
- scales `0.5, 1, 2, 4`;
- 8 questions per bucket.

Strong small-bucket result:

- `teacher_delta`, layer 20, final-token patch, scale 2:
  - heldout: `0.625 -> 1.000`;
  - ordinary: `0.500 -> 0.625`;
  - minimal-pair: `0.500 -> 0.875`;
  - inverse-polarity: `0.500 -> 0.500`;
  - near-collision: `0.625 -> 0.750`.

Teacher-source replay also worked, especially at layer 20:

- `teacher_source`, layer 20, final-token patch, scale 4:
  - heldout: `0.625 -> 1.000`;
  - minimal-pair: `0.500 -> 0.875`.

This was the strongest diagnostic signal so far, but the first run used only
8-question buckets, so we validated the best layer.

### Layer-20 Teacher Replay Larger Validation

Run:

- `runs/residual_teacher_replay_l20_validate_paper0_qwen17`

Settings:

- layer 20 only;
- target modes `teacher_delta` and `teacher_source`;
- patch modes `final` and `suffix`;
- scales `1, 2, 4`;
- 20 questions per bucket.

Best balanced row:

- `teacher_delta`, layer 20, final-token patch, scale 2:

| bucket | baseline acc | patched acc | acc delta | pos acc delta | neg acc delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | +0.30 | 0.00 | +0.60 |
| ordinary | 0.50 | 0.70 | +0.20 | 0.00 | +0.40 |
| minimal pair | 0.50 | 0.70 | +0.20 | -0.10 | +0.50 |
| inverse polarity | 0.50 | 0.50 | 0.00 | 0.00 | 0.00 |
| near collision | 0.50 | 0.70 | +0.20 | 0.00 | +0.40 |

Other strong rows:

- `teacher_delta`, layer 20, final/suffix scale 4:
  - near-collision: `0.50 -> 0.80`;
- `teacher_source`, layer 20, final scale 4:
  - heldout: `0.50 -> 0.80`;
  - ordinary: `0.50 -> 0.70`;
  - near-collision: `0.50 -> 0.75`.

Interpretation:

This is the first robust result that looks like the next mechanism has a real
target. The context teacher has a replayable residual readout signal around
layer 20. Patching that signal into the no-document run improves heldout,
ordinary, minimal-pair, and near-collision questions.

Caveats:

- The gains still mostly repair negative-label errors; positive accuracy is
  usually unchanged and can drop slightly on minimal pairs.
- Inverse-polarity does not improve. This means the replayed teacher delta is
  probably an answer-use/readout signal tied to the prompt's current answer
  policy, not a pure reusable rule-state variable.
- This is still activation patching, not a persistent weight write.

Updated diagnosis:

The earlier source-PC and rule-probe targets failed because they missed the
teacher's downstream readout bridge. The promising next direction is not
"amplify a readable validity feature." It is:

1. capture the teacher residual readout delta at the layer where it is causally
   useful, currently around layer 20;
2. factor it against no-document student keys;
3. write a protected closed-form memory update that recreates this teacher
   readout delta;
4. pair it with earlier TSOC source-state writes only if the readout write alone
   fails to generalize.

Next concrete experiment:

Implement a persistent closed-form write for the layer-20 teacher replay target:

- keys: no-document MLP down-projection inputs at layer 20;
- targets: `teacher_delta` or `teacher_source` residual targets at layer 20;
- fit through protected ridge;
- write into layer-20 MLP down-projection memory;
- evaluate against the same 20-question gauntlets;
- compare to direct patching upper bound.

This is much closer to the original goal: the target is derived from the
context-conditioned teacher trajectory, not hidden labels or generated Q&A.

## Persistent Layer-20 Teacher-Replay Writes

The direct patch result was strong enough to test the actual weight-write
version.

Existing script used:

- `scripts/tsoc_source_write.py`

The relevant modes:

- `target_mode=raw_block_delta` writes the teacher residual delta:
  `block_output(doc+q) - block_output(null+q)`;
- `target_mode=source_delta` writes the local source proxy:
  `(doc_out-null_out) - (doc_in-null_in)`.

Setup:

- layer 20;
- question-shaped trace probes;
- 32 trace probes;
- 12 suffix tokens;
- persistent additive memory on the layer-20 MLP down-projection;
- behavior gate enabled over trace keys;
- 20-question gauntlets.

### Teacher Delta Write, Eta 2

Run:

- `runs/tsoc_l20_teacher_delta_write_eta2_paper0_qwen17`

Result:

| bucket | baseline acc | context acc | edited acc | edited delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.60 | +0.10 |
| ordinary | 0.50 | 0.70 | 0.55 | +0.05 |
| minimal pair | 0.50 | 0.60 | 0.50 | 0.00 |
| inverse polarity | 0.50 | 0.50 | 0.50 | 0.00 |
| near collision | 0.50 | 0.75 | 0.50 | 0.00 |

Diagnostics:

- trace/eval key overlap was very high:
  - heldout mean max cosine to trace keys: `0.9975`;
  - ordinary: `0.9977`;
  - minimal-pair: `0.9974`;
  - near-collision: `0.9975`.
- mean update-effect norm was large:
  - heldout: `260.1` vs mean trace target norm `158.2`;
  - ordinary: `267.2`;
  - minimal-pair: `267.3`.
- state reentry worsened:
  - heldout teacher L2 ratio: `2.09`;
  - ordinary: `2.15`;
  - minimal-pair: `2.13`;
  - near-collision: `2.11`.

Interpretation:

The memory fires on the right family of prompts and can flip a few negative
examples, but it overshoots/misaligns the teacher trajectory. This is a partial
weight-write transfer of the direct patch result, not a clean consolidation.

### Teacher Delta Write, Eta 0.5 and Eta 1

Runs:

- `runs/tsoc_l20_teacher_delta_write_eta05_paper0_qwen17`
- `runs/tsoc_l20_teacher_delta_write_eta1_paper0_qwen17`

Result:

- eta 0.5:
  - no accuracy gains;
  - mean update-effect norm around `65`, below the mean trace target norm
    `158`;
  - teacher L2 ratio roughly `1.02`, so it avoids overshoot but is too weak.
- eta 1.0:
  - no accuracy gains;
  - mean update-effect norm around `130`, closer to the target norm;
  - teacher L2 ratio roughly `1.25` to `1.30`, still moving in an imperfect
    direction.

Interpretation:

There is a narrow controllability window:

- eta 0.5 is too weak;
- eta 1.0 moves margins but not decisions;
- eta 2.0 flips some negatives but overshoots and does not generalize to
  minimal-pair/near-collision.

This points to direction/geometry error, not merely scale.

### Teacher Source Write, Eta 4

Run:

- `runs/tsoc_l20_teacher_source_write_eta4_paper0_qwen17`

Result:

| bucket | baseline acc | context acc | edited acc | edited delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.60 | +0.10 |
| ordinary | 0.50 | 0.70 | 0.55 | +0.05 |
| minimal pair | 0.50 | 0.60 | 0.50 | 0.00 |
| inverse polarity | 0.50 | 0.50 | 0.50 | 0.00 |
| near collision | 0.50 | 0.75 | 0.50 | 0.00 |

Diagnostics:

- mean update-effect norm was around `304` to `313`;
- mean trace target norm was only `92.7`;
- teacher L2 ratio worsened to about `2.1` to `2.26`.

Interpretation:

The source proxy can also transfer a small heldout/ordinary gain, but the
closed-form MLP write is much less faithful than direct residual patching.

Current state of the mechanism:

- Direct teacher residual replay works well.
- Persistent MLP down-projection writes recover only a small fraction of the
  direct-patch effect.
- The current protected ridge solve produces effects with the right trigger
  family and enough magnitude, but likely the wrong direction in residual
  space.

Next concrete experiment:

Add effect-target alignment diagnostics and then change the write geometry:

- for eval probes, compute cosine between `MLP_key @ update.T` and the actual
  teacher replay target;
- whiten/regularize MLP keys with a background covariance rather than ordinary
  ridge;
- solve in a low-rank target basis, especially the teacher replay components
  that direct patching proved causal;
- optionally add an output-space projection/trust region so the memory effect
  lands in the causal teacher-delta subspace instead of a high-norm
  off-target residual move.

## 2026-05-18 Continuation: Token Locality, Probe Count, and Residual Operator Diagnostics

### Code Added

Changed:

- `scripts/tsoc_source_write.py`
  - added `--memory-gate-final-token-only`;
  - added positive/negative effect-target alignment diagnostics;
  - added answer-direction projection diagnostics for alignment rows.
- `caic/modeling.py`
  - added a diagnostic final-token-only gate mode to
    `AdditiveMemoryLinear`.
- `scripts/residual_operator_write.py`
  - new diagnostic runner that fits a linear operator directly on the
    residual stream at decoder block outputs/inputs, then applies it with a
    forward hook.
  - This is not the final weight-write mechanism. It tests whether the failure
    is caused by the MLP down-projection actuator or by the harder mapping from
    no-doc states to teacher deltas.

Validation:

- full test suite passes: `25 passed`.

### Final-Token MLP Write

Question:

Could the MLP write be failing because it fires across too many suffix tokens,
while the direct residual replay worked best at the final answer-use token?

Runs:

- `runs/tsoc_l20_teacher_delta_write_finaltok_eta1_paper0_qwen17`
- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_paper0_qwen17`
- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_gate099_paper0_qwen17`
- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_gatefinal_paper0_qwen17`
- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_labeldiag_paper0_qwen17`

Best final-token result, eta 2:

| bucket | baseline acc | context acc | edited acc | edited delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.65 | +0.15 |
| ordinary | 0.50 | 0.70 | 0.60 | +0.10 |
| minimal pair | 0.50 | 0.60 | 0.60 | +0.10 |
| inverse polarity | 0.50 | 0.50 | 0.45 | -0.05 |
| near collision | 0.50 | 0.75 | 0.55 | +0.05 |

Interpretation:

- final-token training is better than the earlier 12-token suffix write;
- it recovers half of the context upper-bound gap on heldout and ordinary
  buckets, and it finally moves minimal pairs;
- it still mostly repairs negative examples and still fails inverse polarity;
- the artificial final-token-only gate produced the same behavior as the normal
  final-token-key write, so broad firing on earlier tokens is not the main
  bottleneck.

Scale result:

- eta 1 had near-target effect norm but little behavior:
  - heldout `0.50 -> 0.50`;
  - ordinary `0.50 -> 0.55`;
  - minimal-pair `0.50 -> 0.55`.
- eta 2 moved behavior but overshot:
  - effect-target norm ratio was about `1.90` to `1.99`;
  - MLP-output state-reentry metrics worsened.

Important caveat:

The state-reentry metric here compares MLP down-projection outputs to teacher
MLP down-projection outputs, while the target is a block-output residual delta.
So the MLP-output reentry metric is not the right faithfulness score for this
target. The effect-target diagnostic is more relevant.

### Label-Wise Alignment

Run:

- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_labeldiag_paper0_qwen17`

Findings:

- aggregate effect-target cosine was high:
  - heldout `0.899`;
  - ordinary `0.901`;
  - minimal-pair `0.887`;
  - near-collision `0.893`;
  - inverse-polarity only `0.577`.
- positive and negative rows both had decent cosine in ordinary/minimal/near
  buckets, so the failure is not just a total sign flip.
- norm ratios were near `2x`, matching the scale-2 direct replay setting but
  with enough angular/off-target error to matter.

Interpretation:

The write is not random and it is not purely using the wrong trigger. It is a
rough approximation to the teacher delta, but late-layer answer behavior is
sharp enough that a `0.89` cosine approximation can still fail adversarial
buckets.

### Probe Count Sweep

Run:

- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_96probes_paper0_qwen17`
- `runs/tsoc_l20_teacher_delta_write_finaltok_eta2_96probes_neg10_paper0_qwen17`

Result with 96 probes, negative weight 1:

| bucket | edited acc | edited delta |
| --- | ---: | ---: |
| heldout | 0.60 | +0.10 |
| ordinary | 0.55 | +0.05 |
| minimal pair | 0.55 | +0.05 |
| inverse polarity | 0.45 | -0.05 |
| near collision | 0.65 | +0.15 |

Alignment improved:

- heldout effect-target cosine `0.916`;
- ordinary `0.918`;
- minimal-pair `0.911`;
- near-collision `0.917`;
- inverse-polarity `0.770`.

But guard leakage worsened:

- negative RMSE rose from about `0.09` in the 32-probe run to `1.54`.

Stronger negative protection (`negative_weight=10`) reduced guard leakage
(`negative_rmse=0.27`) but also suppressed useful behavior:

- heldout only `+0.05`;
- ordinary `+0.05`;
- minimal `+0.05`;
- near-collision `+0.05`;
- inverse still `-0.05`.

Interpretation:

More probes help the target interpolation geometry, especially near-collision,
but the ordinary protected ridge solve starts spending capacity in unprotected
directions. This points toward a better metric/trust region, not just "more
probes."

### Residual Operator Diagnostic

Question:

Is the MLP down-projection actuator the main bottleneck?

Method:

Fit a direct residual-stream operator:

- key: no-doc block output at layer 20;
- target: `block_output(doc+q) - block_output(null+q)`;
- apply: add `W key` to the block output at the final token.

This skips MLP down-projection features. If this works much better than the MLP
write, the MLP actuator is the bottleneck. If it does not, the mapping from
no-doc state to teacher delta is the bottleneck.

Runs:

- `runs/residual_operator_l20_teacher_delta_final_eta2_paper0_qwen17`
- `runs/residual_operator_l20_teacher_delta_final_eta2_labeldiag_paper0_qwen17`
- `runs/residual_operator_l20_teacher_delta_final_eta2_nogate_paper0_qwen17`
- `runs/residual_operator_l20_teacher_delta_final_eta2_96probes_nogate_paper0_qwen17`

Layer-20 residual operator, 32 probes, eta 2:

| bucket | edited acc | edited delta |
| --- | ---: | ---: |
| heldout | 0.50 | 0.00 |
| ordinary | 0.60 | +0.10 |
| minimal pair | 0.50 | 0.00 |
| inverse polarity | 0.45 | -0.05 |
| near collision | 0.60 | +0.10 |

Despite this weak behavior, effect-target cosine was very high:

- heldout `0.969`;
- ordinary `0.970`;
- minimal-pair `0.973`;
- near-collision `0.965`;
- inverse-polarity `0.751`.

Removing the residual-operator gate did not materially change the result.

With 96 probes:

| bucket | edited acc | edited delta |
| --- | ---: | ---: |
| heldout | 0.60 | +0.10 |
| ordinary | 0.50 | 0.00 |
| minimal pair | 0.55 | +0.05 |
| inverse polarity | 0.45 | -0.05 |
| near collision | 0.55 | +0.05 |

Effect-target cosine rose further:

- heldout `0.982`;
- ordinary `0.983`;
- minimal-pair `0.981`;
- near-collision `0.981`;
- inverse-polarity `0.887`.

Interpretation:

This is an important negative result. The MLP down-projection actuator is not
the only bottleneck. Even a residual-stream operator that looks very close to
the teacher delta under Euclidean/cosine metrics can fail behaviorally.

The likely reason: the late answer readout is sensitive to small,
behaviorally-relevant residual components. Cosine is dominated by large common
teacher-delta structure, while the Yes/No decision depends on a small subspace.
We need a downstream-behavior-weighted metric, not plain residual L2/cosine.

### Oracle Patch Check

Run:

- `runs/residual_teacher_replay_l20_exact30_paper0_qwen17`
- `runs/residual_teacher_replay_l20_exact30_batch1_paper0_qwen17`

Layer-20 final-token teacher-delta patch, scale 2, exact same heldout/gauntlet
sizes:

| bucket | baseline acc | patched acc | delta |
| --- | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | +0.30 |
| ordinary | 0.50 | 0.70 | +0.20 |
| minimal pair | 0.50 | 0.70 | +0.20 |
| inverse polarity | 0.50 | 0.50 | 0.00 |
| near collision | 0.50 | 0.70 | +0.20 |

The result survives `batch_size=1`, so it is not a padding/batching artifact.

Interpretation:

The teacher delta itself is causal and useful when supplied exactly for each
eval prompt. The hard unsolved part is learning a persistent operator that
predicts the behaviorally-important part of that delta from no-context hidden
states.

### Three-Layer Residual Operator

Run:

- `runs/residual_operator_l12_16_20_teacher_delta_final_eta1_paper0_qwen17`

Setup:

- layers 12, 16, 20;
- target: teacher residual delta;
- key: block output;
- eta 1 per layer;
- final-token only.

Result:

All buckets stayed at or near baseline accuracy, and negative margins became
much worse:

- heldout negative mean margin: `-8.68 -> -20.84`;
- ordinary: `-9.82 -> -20.49`;
- minimal-pair: `-6.76 -> -22.18`;
- near-collision: `-7.85 -> -21.04`.

Layerwise effect-target cosines were excellent:

- layer 12 heldout `0.993`;
- layer 16 heldout `0.983`;
- layer 20 heldout `0.969`.

Interpretation:

Naively distributing cumulative teacher deltas across multiple layers strongly
reinforces the model's existing Yes prior. This supports the TSOC diagnosis:
raw teacher deltas are downstream symptoms. Multi-layer raw-delta replay is not
source isolation.

### Current Diagnosis

What now seems solid:

- The context teacher contains a causal residual signal.
- Exact layer-20 final-token teacher-delta replay can recover a meaningful
  fraction of context performance.
- Closed-form persistent writes can partially transfer that effect.
- The failure is not primarily trigger-key overlap or broad earlier-token
  firing.
- The failure is not solely the MLP down-projection actuator.

What is not solved:

- The persistent operator does not yet predict the small behaviorally-critical
  part of the teacher delta.
- Plain residual-space ridge regression optimizes the wrong metric.
- Mean cosine/euclidean target alignment can look excellent while answer
  behavior fails.
- Multi-layer cumulative-delta writing worsens answer-prior contamination.

Next best research step:

Estimate a downstream-behavior metric for candidate residual targets.

Concrete version:

1. For each probe, compute the gradient/Jacobian of the Yes-vs-No margin with
   respect to the layer-20 final residual state.
2. Score target-error not by residual L2, but by its projection onto this
   behaviorally-sensitive subspace.
3. Fit a closed-form operator with a weighted loss that prioritizes matching
   the teacher delta in the margin-sensitive directions while suppressing the
   broad common teacher/style delta.
4. Compare:
   - Euclidean residual operator;
   - gradient-weighted residual operator;
   - MLP down-projection write using the same behavior-weighted target;
   - exact oracle teacher-delta patch.

This is still compatible with the original goal. The update remains cheap and
closed-form, but the metric must know which part of the context-induced vector
field is actually load-bearing.

### Gradient-Weighted Residual Operator

Code:

- `scripts/residual_operator_write.py` now supports
  `--solve-mode margin_gradient`.

Method:

Instead of solving `W key ~= teacher_delta` in residual-vector space, solve
scalar constraints:

- compute the gradient of the correct Yes/No margin with respect to the
  layer-20 final residual state;
- compute how much the exact teacher delta would move that margin;
- fit a minimum-norm residual operator whose effect matches those scalar
  margin changes on write probes;
- add zero-margin-change constraints on guard prompts.

This is not the desired final method because it uses labeled answer probes, but
it is a diagnostic for whether downstream-behavior weighting helps.

Runs:

- `runs/residual_operator_l20_margin_gradient_eta2_paper0_qwen17`
- `runs/residual_operator_l20_margin_gradient_eta4_paper0_qwen17`

Eta 2 result:

| bucket | edited acc | edited delta | negative margin change | positive margin change |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.00 | `-8.68 -> -4.43` | `6.72 -> 4.98` |
| ordinary | 0.50 | 0.00 | `-9.82 -> -4.24` | `6.45 -> 4.82` |
| minimal pair | 0.50 | 0.00 | `-6.76 -> -4.69` | `8.62 -> 4.79` |
| inverse polarity | 0.50 | 0.00 | `-6.98 -> -5.43` | `8.19 -> 5.02` |
| near collision | 0.50 | 0.00 | `-7.85 -> -4.58` | `8.71 -> 4.82` |

Eta 4 result:

| bucket | edited acc | edited delta | negative margin change | positive margin change |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.00 | `-8.68 -> -3.16` | `6.72 -> 3.59` |
| ordinary | 0.50 | 0.00 | `-9.82 -> -3.06` | `6.45 -> 3.44` |
| minimal pair | 0.50 | 0.00 | `-6.76 -> -3.24` | `8.62 -> 3.51` |
| inverse polarity | 0.50 | 0.00 | `-6.98 -> -3.81` | `8.19 -> 3.56` |
| near collision | 0.50 | 0.00 | `-7.85 -> -3.07` | `8.71 -> 3.54` |

Interpretation:

The behavior-weighted solve moves margins in a controlled way, and it does
reduce the severe negative-case error, but it also weakens positives and does
not flip answers. At this stage it looks more like a calibrated margin-control
baseline than context-rule consolidation.

Updated diagnosis:

- raw residual-vector matching preserves too much irrelevant teacher-delta
  mass;
- margin-gradient matching overcorrects toward a generic uncertainty/decision
  boundary;
- neither alone identifies the rule-conditioned source field.

The next architecture should combine:

- early/mid source extraction;
- a learned or estimated semantic/rule-state basis;
- downstream behavior weighting only as a small trust-region metric, not as the
  whole target.

### Generic Teacher-Functional Filters

Follow-up on the scalable version of the sharper direction: avoid
translation-specific validation and use only the context teacher's own output
behavior to decide what part of a write is load-bearing.

Code added in `scripts/residual_operator_write.py`:

- `--solve-mode teacher_kl_gradient`
- `--solve-mode teacher_kl_weighted_vector`
- `--solve-mode teacher_logit_jacobian`

These modes are label-free at write time:

- `teacher_kl_gradient` computes the gradient of KL from the no-context student
  distribution to the context teacher distribution, then fits scalar constraints
  so the update matches the teacher delta's first-order KL effect.
- `teacher_kl_weighted_vector` keeps the original residual teacher-vector
  target, but weights trace rows by whether the target points in a
  KL-reducing direction.
- `teacher_logit_jacobian` selects the context teacher's top-k next-token logits
  and fits scalar constraints through their local residual Jacobians.

Small diagnostic setup:

- domain: paper 0 from
  `runs/activation_latch_singlefix_strict_2paper_qwen17/domains.jsonl`;
- layer 20 residual operator;
- final token only;
- 16 trace probes;
- 20 heldout questions and 10-question gauntlet buckets;
- MPS, Qwen3-1.7B.

Runs:

- `runs/residual_operator_l20_teacher_klgrad_top128_eta2_small_paper0_qwen17`
- `runs/residual_operator_l20_teacher_klweighted_top128_eta2_small_paper0_qwen17`
- `runs/residual_operator_l20_vector_ridge_eta2_small_paper0_qwen17`
- `runs/residual_operator_l20_teacher_logitjac_top4_eta1_small_paper0_qwen17`
- `runs/residual_operator_l20_teacher_logitjac_top4_eta2_small_paper0_qwen17`
- `runs/residual_operator_l12_16_20_teacher_logitjac_top4_eta1_small_paper0_qwen17`

Results:

| method | heldout | ordinary | minimal pair | inverse polarity | near collision | main failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vector ridge eta2 | +0.10 | 0.00 | 0.00 | -0.20 | +0.10 | inverse damage |
| KL scalar eta2 | 0.00 | 0.00 | 0.00 | +0.20 | -0.10 | collapsed positives |
| KL-weighted vector eta2 | +0.10 | 0.00 | 0.00 | -0.20 | +0.10 | same as vector ridge |
| logit-Jacobian eta1 | +0.05 | 0.00 | 0.00 | 0.00 | 0.00 | safe but weak |
| logit-Jacobian eta2 | +0.10 | +0.10 | +0.10 | 0.00 | +0.10 | positive damage |
| logit-Jacobian layers 12/16/20 eta1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | no acquisition |

Important diagnostics:

- KL scalar did exactly what we feared: it learned a generic decision-boundary
  shove. It made negatives correct but broke positives.
- KL-weighted vector had mean gate `0.72` but behaved almost identically to
  ordinary vector ridge. The KL relevance score did not isolate a new useful
  subset of rows.
- Logit-Jacobian eta1 was the first generic functional objective that preserved
  inverse polarity, but it only flipped one heldout item.
- Logit-Jacobian eta2 gained one item in heldout/ordinary/minimal/near-collision
  but started damaging positives. It is a softer version of the same
  answer-boundary problem.
- Applying logit-Jacobian writes at layers 12/16/20 did not accumulate useful
  acquisition.

Interpretation:

Generic functional coordinates are useful, but not sufficient as the whole
objective. Scalar functional objectives tend to optimize the nearest behavioral
lever, which is answer calibration. Vector objectives preserve teacher
trajectory shape but still miss the behaviorally tiny readout components. The
promising role for functional gradients is as a trust-region / sentinel metric:

- keep the vector/source-field write as the main candidate;
- use teacher-functional Jacobians to reject or shrink components that cause
  broad answer-boundary movement;
- do not let the functional gradient itself become the target.

Compute note:

The generic functional diagnostics add local backward passes, but no optimizer
loop. Top-k logit Jacobian with `k=4` was still a small diagnostic run on MPS.
This remains plausibly much cheaper than gradient training, but it is too
expensive to apply densely at every layer/token unless we first learn which
sites need it.

### Functional Trust-Region Follow-Up

Implemented additional modes in `scripts/residual_operator_write.py`:

- `--solve-mode vector_logit_calibrated`
- `--solve-mode vector_kl_orthogonalized`

The goal was to keep the ordinary teacher-vector write as the candidate and use
functional gradients only as a trust-region/sentinel.

`vector_logit_calibrated`:

- fits the ordinary vector-ridge teacher-delta write;
- computes top-k teacher-logit Jacobian constraints;
- chooses a global scale for the candidate write by least-squares fit to those
  functional constraints.

Runs:

- `runs/residual_operator_l20_vector_logitcal_top4_eta2_small_paper0_qwen17`
- `runs/residual_operator_l20_vector_logitcal_top4_nocenter_eta2_small_paper0_qwen17`

Result:

| method | heldout | ordinary | minimal pair | inverse polarity | near collision | scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vector ridge eta1 | +0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 implied |
| vector ridge eta2 | +0.10 | 0.00 | 0.00 | -0.20 | +0.10 | 1.00 |
| logit-calibrated centered | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.187 |
| logit-calibrated uncentered | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.356 |

Interpretation:

Teacher-logit calibration was too conservative. It avoided inverse-polarity
damage, but scaled away the useful heldout/near-collision gain. The top-k
functional constraints are not yet a good scale oracle.

`vector_kl_orthogonalized`:

- fits the ordinary vector-ridge teacher-delta write;
- fits the scalar KL-gradient nuisance update;
- removes the projection of the vector write onto that KL-gradient direction.

Run:

- `runs/residual_operator_l20_vector_klorth_top128_eta2_small_paper0_qwen17`

Result:

| method | heldout | ordinary | minimal pair | inverse polarity | near collision |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector ridge eta2 | +0.10 | 0.00 | 0.00 | -0.20 | +0.10 |
| KL-orthogonalized vector eta2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Diagnostics:

- KL nuisance update norm: `0.889`;
- vector write norm: `30.763`;
- projection removed: `1.249`;
- cosine between vector write and nuisance direction: only `0.041`.

Interpretation:

This is a very useful negative result. The behaviorally active part of the
vector write is tiny in Frobenius/cosine terms. Removing a 4%-cosine generic KL
component removed both the inverse-polarity damage and the heldout/near-collision
gains. That strongly supports the current diagnosis:

> The current residual/operator writes are mostly behaviorally powered by a
> tiny answer-boundary component, while the large teacher-vector mass is mostly
> inert or cosmetic under the final decision.

The next method should not merely remove the answer-boundary component; that
throws away the only component currently crossing answer thresholds. It needs a
replacement behaviorally active component tied to rule/content state rather
than generic answer calibration.

Next concrete direction:

- Use paired prompts with the same answer format but different rule/content
  state to estimate a content-conditioned readout direction.
- In the synthetic rule benchmark, this can be built without using hidden labels
  at write time by contrasting context-teacher traces for matched near-neighbor
  chains inside the same domain.
- The write target should be the part of the teacher delta that differs across
  matched content states while the answer-format/logit-boundary component is
  held roughly constant.

### Wide Sequential Final-Token MLP Writes

Bug fix:

- `scripts/tsoc_source_write.py` had an indentation bug introduced while adding
  alignment diagnostics: with `--alignment-diagnostics` off, it wrote only
  `config.json` and skipped the write/eval block.
- Fixed and verified with the full test suite: `25 passed`.

Hypothesis:

The user's "edit every layer" intuition may be right if the edit is a small
sequential source correction, not a copied cumulative teacher delta.

Setup:

- layers `6, 8, 10, 12, 14, 16, 18, 20`;
- final-token only;
- question-probe traces;
- sequential replay;
- target: raw block-output teacher delta against the current edited trajectory;
- persistent additive MLP down-projection memory;
- memory gate final-token-only.

Single-paper eta 0.5:

Run:

- `runs/tsoc_seq_even_l6_20_final_eta05_paper0_qwen17_rerun`

| bucket | baseline acc | context acc | edited acc | delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.70 | +0.20 |
| ordinary | 0.50 | 0.70 | 0.70 | +0.20 |
| minimal pair | 0.50 | 0.60 | 0.50 | 0.00 |
| inverse polarity | 0.50 | 0.50 | 0.50 | 0.00 |
| near collision | 0.50 | 0.75 | 0.75 | +0.25 |

Single-paper eta 1:

Run:

- `runs/tsoc_seq_even_l6_20_final_eta1_paper0_qwen17`

| bucket | baseline acc | context acc | edited acc | delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.80 | +0.30 |
| ordinary | 0.50 | 0.70 | 0.85 | +0.35 |
| minimal pair | 0.50 | 0.60 | 0.55 | +0.05 |
| inverse polarity | 0.50 | 0.50 | 0.45 | -0.05 |
| near collision | 0.50 | 0.75 | 0.85 | +0.35 |

Interpretation:

This is the strongest single-paper result so far. It supports a revised version
of the "write many layers" idea:

- one late layer was too restrictive;
- many small final-token source writes across layers can recover the context
  upper bound on heldout and beat the measured context condition on ordinary
  and near-collision buckets;
- but inverse-polarity is still weak, and minimal-pair improvement remains
  small, so this is still not clean rule consolidation.

Two-paper eta 1 with retention:

Run:

- `runs/tsoc_seq_even_l6_20_final_eta1_2paper_retention_qwen17`

Immediate per-paper metrics:

| paper | bucket | baseline | context | edited | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | heldout | 0.50 | 0.80 | 0.80 | +0.30 |
| 0 | ordinary | 0.50 | 0.70 | 0.85 | +0.35 |
| 0 | minimal pair | 0.50 | 0.60 | 0.55 | +0.05 |
| 0 | inverse polarity | 0.50 | 0.50 | 0.45 | -0.05 |
| 0 | near collision | 0.50 | 0.75 | 0.85 | +0.35 |
| 1 | heldout | 0.50 | 0.65 | 0.80 | +0.30 |
| 1 | ordinary | 0.55 | 0.70 | 0.85 | +0.30 |
| 1 | minimal pair | 0.50 | 0.55 | 0.65 | +0.15 |
| 1 | inverse polarity | 0.50 | 0.50 | 0.45 | -0.05 |
| 1 | near collision | 0.55 | 0.70 | 0.60 | +0.05 |

Retention:

| after write | evaluated paper | heldout retention acc | context acc |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0.80 | 0.85 |
| 1 | 0 | 0.65 | 0.65 |
| 1 | 1 | 0.80 | 0.55 |

Interpretation:

- multiple sequential writes now work in the weak sense: paper 1 learns strongly
  after paper 0 has already been written;
- paper 0 does not collapse after paper 1, but it drops from `0.80` to `0.65`,
  so interference is real;
- this is a better multiple-paper result than the earlier single-layer CAIC
  attempts.

Two-paper eta 0.5 with retention:

Run:

- `runs/tsoc_seq_even_l6_20_final_eta05_2paper_retention_qwen17`

Retention:

| after write | evaluated paper | heldout retention acc | context acc |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0.70 | 0.85 |
| 1 | 0 | 0.55 | 0.65 |
| 1 | 1 | 0.80 | 0.55 |

Interpretation:

Lower eta does not fix retention in this tiny sample. The interference problem
is probably not just write strength; it needs routing/covariance/control over
which memory subspaces get reused.

Current best mechanism:

- wide sequential final-token MLP down-projection source writes across layers
  `6..20`, eta around `1`;
- this gives real acquisition on two sequential papers, but only partial
  retention and still fails inverse-polarity robustness.

Next attention direction:

We have not yet touched attention weights. The most conservative first
attention write should target attention `o_proj`/output contributions, not Q/K
attention probabilities. The deployment paper tokens are absent, so directly
copying teacher attention maps is likely wrong. The first test should instead
ask whether a closed-form `o_proj` write can recreate the context-conditioned
working-memory contribution of attention at the final token.

### Attention Output-Projection Writes

Code:

- `caic/modeling.py`
  - added attention `o_proj` discovery/wrapping;
  - added `install_additive_attention_memory`;
  - added `capture_attention_io`.
- `scripts/attention_source_write.py`
  - attention-only `o_proj` closed-form source writes.
- `scripts/joint_attention_mlp_write.py`
  - per-layer sequential write:
    1. attention `o_proj` write;
    2. MLP down-projection write for the remaining block-output delta.

Tests:

- full test suite passes: `25 passed`.

Rationale:

Do not start by editing attention probabilities. In the teacher run, document
tokens exist; in deployment they do not. So copying teacher attention maps would
probably teach the model to chase unavailable context positions. The safer
first attention write targets the residual contribution after attention has
already integrated information.

Attention-only run:

- `runs/attention_oproj_seq_even_l6_20_final_eta1_paper0_qwen17`

| bucket | baseline | context | edited | delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.55 | +0.05 |
| ordinary | 0.50 | 0.70 | 0.50 | 0.00 |
| minimal pair | 0.50 | 0.60 | 0.55 | +0.05 |
| inverse polarity | 0.50 | 0.50 | 0.55 | +0.05 |
| near collision | 0.50 | 0.75 | 0.55 | +0.05 |

Interpretation:

Attention `o_proj` writes alone are weak but not inert. They improve negative
margins and move a few buckets by one item, including inverse-polarity. That is
consistent with attention output being a working-memory support component, not
the whole consolidation write.

Joint attention+MLP, eta 1 / eta 1:

- `runs/joint_attn_mlp_seq_even_l6_20_final_eta1_paper0_qwen17`

| bucket | baseline | context | edited | delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.75 | +0.25 |
| ordinary | 0.50 | 0.70 | 0.80 | +0.30 |
| minimal pair | 0.50 | 0.60 | 0.60 | +0.10 |
| inverse polarity | 0.50 | 0.50 | 0.40 | -0.10 |
| near collision | 0.50 | 0.75 | 0.80 | +0.30 |

Compared with MLP-only eta 1:

- heldout worsened: `0.80 -> 0.75`;
- ordinary worsened: `0.85 -> 0.80`;
- minimal-pair improved: `0.55 -> 0.60`;
- inverse worsened: `0.45 -> 0.40`;
- near-collision worsened: `0.85 -> 0.80`.

Joint attention eta 0.5 / MLP eta 1:

- `runs/joint_attn05_mlp1_seq_even_l6_20_final_paper0_qwen17`

| bucket | baseline | context | edited | delta |
| --- | ---: | ---: | ---: | ---: |
| heldout | 0.50 | 0.80 | 0.75 | +0.25 |
| ordinary | 0.50 | 0.70 | 0.85 | +0.35 |
| minimal pair | 0.50 | 0.60 | 0.55 | +0.05 |
| inverse polarity | 0.50 | 0.50 | 0.40 | -0.10 |
| near collision | 0.50 | 0.75 | 0.90 | +0.40 |

Interpretation:

Attention output writes are affecting real behavior. They can boost
near-collision and slightly change minimal-pair behavior, but the naive joint
write is not uniformly better than MLP-only and makes inverse-polarity worse.

Updated attention conclusion:

- yes, attention should probably be part of a whole-model consolidation method;
- the first `o_proj` implementation works technically and has behavioral
  effects;
- but attention writes need their own objective/gating, not simply
  "teacher attention output delta plus MLP delta";
- Q/K pattern writes remain untested and are higher risk.

Next attention experiment:

Use attention writes as a constrained auxiliary term, not a full teacher-delta
copy:

- either only write selected layers where attention-only improved inverse or
  minimal-pair behavior;
- or use attention writes to reduce the MLP target's residual error but clip
  them by inverse-polarity sentinel margins;
- then rerun two-paper retention.

## 2026-05-18 mini-language consolidation benchmark

User proposed a better aligned test for the real goal: instead of unrelated
synthetic papers, teach one invented language across lessons and ask whether
the write makes the no-context model translate. I added
`scripts/minilang_write.py`.

Benchmark shape:

- invented language: Lyran;
- grammar for the first clean stage: `TENSE VERB SUBJECT OBJECT`, adjectives
  after nouns, English output in normal subject-verb-object order;
- evaluation is multiple-choice over full English translations, but scored by
  average log-probability of the whole candidate translation, not by answer
  letter. This was important because letter scoring produced answer-letter
  priors and false positives;
- tokenizer truncation is now set to left-truncation in `caic/modeling.py` so
  long contexts keep the question suffix;
- token-level teacher-forcing trace prompts were added so the write sees the
  states used during actual translation scoring, not only the `English:` final
  prompt state.

Early evaluator bug found:

- 8-lesson context check with letter scoring looked misleading; full-candidate
  scoring showed that the initial 8-lesson task was not reliably context
  learnable.
- Qwen3-1.7B on a 4-lesson present-tense subset is context learnable:
  baseline `0/16`, full-context `16/16` for seed 1.

Important mini-language runs:

| run | write | target | edited | context | notes |
| --- | --- | --- | ---: | ---: | --- |
| `minilang_write_4lesson_4layer_cachedscore_eta1_qwen17_cpu` | MLP, layers 8/12/16/20 | final-prompt only | 0/8 | 7/8 | no movement; objective did not match translation scoring |
| `minilang_write_4lesson_tftrace_4layer_nogate_eta1_eval16_details_qwen17_cpu` | MLP, 8/12/16/20 | word teacher-forcing output delta | 5/16 | 16/16 | real but partial; mostly learned some verb choices |
| `minilang_write_4lesson_toktftrace_l20_nogate_eta1_eval16_qwen17_cpu` | MLP only, layer 20 | token teacher-forcing output delta | 6/16 | 16/16 | layer 20 alone beat the wider 4-layer MLP write |
| `minilang_write_4lesson_toktftrace_l20_attno_only_outputdelta_nogate_eta1_eval16_qwen17_cpu` | attention `o_proj` only, layer 20 | token teacher-forcing output delta | 2/16 | 16/16 | attention alone weak |
| `minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_nogate_eta1_eval16_qwen17_cpu` | attention `o_proj` + MLP, layer 20 | token teacher-forcing output delta | 11/16 | 16/16 | strongest language result |
| `minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_nogate_shuffle_eta1_eval16_qwen17_cpu` | same, shuffled targets | token teacher-forcing output delta | 8/16 | 16/16 | control improves too; real target still better |
| `minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_nogate_eta1_eval16_seed0_qwen17_cpu` | same best config, seed 0 | token teacher-forcing output delta | 7/16 | 15/16 | weaker but still above baseline |
| `minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_nogate_probes8_eta1_eval16_qwen17_cpu` | same but 8 trace probes | token teacher-forcing output delta | 0/16 | 16/16 | more probes caused a clipped/noisy attention update and destroyed the gain |

Most important finding:

The user's intuition that attention and MLP writes can positively interact was
validated on the language task. Attention `o_proj` alone was weak (`2/16`);
MLP-only layer 20 was partial (`6/16`); the joint layer-20 attention+MLP write
hit `11/16` with the same context upper bound of `16/16`. The effect is not
purely target-specific because shuffled targets reached `8/16`, but the real
target improved verb/translation choices beyond shuffled.

Detailed behavior from the best seed-1 run:

- baseline usually reversed subject/object or preferred common English verbs;
- context solved every heldout item;
- joint write fixed many `lum=sees`, `narp=helps`, and `vek=likes` cases;
- remaining failures were mostly subject/object binding errors and a few verb
  confusions.

Interpretation:

This is the first result in the repo that looks like partial baking-in of a
context-taught skill rather than only Yes/No answer prior repair. It is still
not full rule consolidation:

- the task is small;
- shuffled targets are uncomfortably strong, so part of the gain is a broad
  translation-mode correction;
- subject/object binding remains fragile;
- scaling from 4 lessons to 10 lessons failed under the current sequential
  write.

10-lesson coherent corpus check:

- `minilang_contextcheck_10lesson_freeze3_eval8_lefttrunc_qwen17_cpu`:
  baseline `0/8`, context `6/8`;
- `minilang_write_10lesson_freeze3_toktftrace_l20_attno_mlp_outputdelta_nogate_eta1_eval8_qwen17_cpu`:
  edited `0/8`;
- full-context final-only write on the same 10-lesson corpus:
  edited `1/8`.

That says the current writer does not yet benefit automatically from many
repeated coherent lessons. Sequential interference and context-teacher weakness
both matter. The next clean direction is not more random layers; it is a
better routed/contrastive language objective with explicit subject/object and
verb binding diagnostics.

Current mini-language conclusion:

- closed-form writes can move a context-taught translation behavior into the
  no-context model;
- attention+MLP is materially better than either part alone on this task;
- the write is still noisy and partly mode-shaping;
- the main unsolved problem is binding: the update learns pieces of the
  translation system but does not reliably preserve who did what to whom.

Audit update, 2026-05-19:

Reproducible artifact:

```bash
.venv/bin/python scripts/minilang_audit_saved.py \
  --run runs/minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_nogate_eta1_eval16_qwen17_cpu \
  --control runs/minilang_write_4lesson_toktftrace_l20_attno_mlp_outputdelta_nogate_shuffle_eta1_eval16_qwen17_cpu \
  --output runs/minilang_saved_audit_l20_attno_mlp_vs_shuffle.json
```

Treat the `11/16` layer-20 attention+MLP result as a promising lead, not as a
many-sigma confirmation that the model learned Lyran's rules. A stress audit of
the saved run found several false-positive paths:

- the nominal 16-question eval contains only 12 unique source sentences;
- the edited score collapses from `11/16` to `7/12` when duplicate source
  sentences are counted once;
- two of those 12 unique items are exact lesson examples, both counted correct;
  excluding exact lesson-example overlaps leaves `5/10` unique heldout
  sentences correct;
- the same-eval shuffled-target control reached `8/16`; paired against the
  real target this is only 5 real-only wins vs 2 shuffle-only wins
  (`p ~= 0.23` one-sided by exact sign/McNemar-style binomial);
- on unique sentences the paired comparison is 4 real-only wins vs 2
  shuffle-only wins (`p ~= 0.34`);
- across 32 saved `eval16` mini-language variants, this run is the single best
  result, so the headline result is post-selection over a broad local search;
- `--option-negative-keys` dropped the same setup to `2/16`, which suggests the
  token teacher-forcing/full-candidate scorer can be exploited by candidate
  answer-prefix effects rather than clean source-sentence understanding;
- increasing trace probes to 8 dropped the same setup to `0/16`, so the effect
  is not stable to more trace evidence.

The strict interpretation is: layer-20 attention `o_proj` + MLP writes can
cause a real behavioral movement on this tiny Lyran task, but the saved
`11/16` result does not yet prove robust rule internalization. The next
mini-language benchmark should use unique eval generation, exclude exact lesson
and trace overlaps, report duplicate-collapsed accuracy, use option-negative
controls by default, and evaluate an exhaustive heldout grid before claiming
that the intervention wrote the underlying translation rule.

Additional attention note:

Earlier paper-task Q/K experiments were also implemented after the first
attention section above:

- Q/K-only writes on the synthetic paper task produced no accuracy movement;
- Q-only was inert and slightly worsened some margins;
- K-only slightly improved some negative margins but did not flip answers;
- Q+K+O+MLP did not beat MLP-only on the paper benchmark.

So far, raw Q/K projection matching is not the right way to write attention.
The mini-language result instead points to `o_proj` as a useful residual
actuator when paired with MLP writes.

All-layer mini-language test:

User asked what happens if we write the translation-token trace across all
layers. I patched `scripts/minilang_write.py` with `--cache-current-captures`
so an all-layer simultaneous write is tractable.

All-layer run:

- `runs/minilang_write_4lesson_toktftrace_all28_attno_mlp_outputdelta_cache_eta1_eval16_qwen17_cpu`
- layers: 0 through 27;
- modules: attention `o_proj` plus MLP down-proj;
- trace: token-level teacher-forced translation prefixes;
- target: output delta;
- baseline: `0/16`;
- context: `16/16`;
- edited: `0/16`.

The update was too aggressive:

- attention `o_proj`: 42/112 writes hit the norm cap;
- MLP down: 16/112 writes hit the norm cap.

The edited model mostly chose wrong tense/verb variants such as `saw/liked`
instead of present-tense `sees/likes`, so the all-layer write introduced a
global translation/tense distortion rather than improving binding.

Smaller all-layer write:

- `runs/minilang_write_4lesson_toktftrace_all28_attno_mlp_outputdelta_cache_eta01_eval16_qwen17_cpu`
- same setup, eta `0.1`;
- baseline: `0/16`;
- context: `16/16`;
- edited: `1/16`.

No norm clipping occurred at eta `0.1`, but it still did not recover the
layer-20 result (`11/16`). Current conclusion: naive all-layer writing is worse
than a focused layer-20 attention+MLP write. The issue is not only update scale;
the all-layer objective is poorly coordinated. A real whole-model version needs
layer-specific trust regions and causal layer selection, not equal-strength
copies of the same output-delta target everywhere.

## 2026-05-19 Continual Mini-Language Triangle Smoke

User asked whether the layer-20 mini-language write supports sequential
continual learning across multiple complex tasks. I added
`scripts/minilang_continual_triangle.py`, which creates independent invented
translation tasks, applies one layer-20 attention `o_proj` + MLP output-delta
write per task, and evaluates the triangular retention matrix after each write.

Engineering notes:

- Apple MPS works for Qwen3-1.7B only when it has a quiet GPU window. Concurrent
  Qwen MPS jobs from other agents caused silent exits or severe stalls.
- The runner uses `--merge-updates` on MPS: the closed-form update is merged
  directly into the wrapped projection's base weight, which is algebraically
  equivalent to ungated additive memory and avoids the nonzero memory-buffer
  MPS forward path.
- Explicit `torch.mps.empty_cache()` / GC after scoring and capture phases made
  repeated MPS scoring materially more stable.

Partial run:

- `runs/minilang_continual3_l20_attno_mlp_eval6_qwen17_mps_merged_cacheclear`
- model: `Qwen/Qwen3-1.7B`
- tasks: 3 generated mini-languages, 4 lessons each
- eval questions per task: 6
- write: layer 20, attention `o_proj` + MLP down-proj, token teacher-forcing
  output-delta target, eta `1.0`, max update norm `50`
- run was stopped during task 2 capture after another MPS job stalled the GPU;
  the first two write/eval steps completed.

Initial context learnability:

| task | language | baseline | context |
| ---: | --- | ---: | ---: |
| 0 | Lyran | 0/6 | 6/6 |
| 1 | Vomar | 0/6 | 5/6 |
| 2 | Seldic | 2/6 | 2/6 |

Triangle result through two writes:

| after write | evaluated task | edited | note |
| ---: | ---: | ---: | --- |
| 0 | 0 Lyran | 2/6 | partial acquisition, one third of context gap |
| 1 | 0 Lyran | 1/6 | retention regressed after Vomar write |
| 1 | 1 Vomar | 0/6 | no acquisition despite 5/6 context upper bound |

Interpretation:

This is negative for clean continual learning in the current configuration.
The first task acquires weakly, the second write interferes with task-0
retention, and the second task itself does not acquire. Seldic is not
context-learnable in this generated profile, so it is not a meaningful
consolidation target.

Current diagnosis:

- The layer-20 write can move a single mini-language behavior, but independent
  languages reuse the same projection subspace and interfere.
- The current task generator can produce profiles that the context teacher does
  not solve; a real five-task test needs a preflight stage that samples tasks
  until each has high context accuracy and low baseline accuracy.
- The next clean continual-learning test should run with exclusive GPU access,
  context-learnability prefiltering, and probably per-task routing/orthogonal
  subspace constraints rather than one shared ungated layer-20 update.

## 2026-05-19 Content-Contrastive Residual Operator Tests

User's hypothesis: writing one final answer state is too noisy; use multiple
perspectives or matched contrasts so only the shared understanding survives.

I added contrastive residual-operator modes to
`scripts/residual_operator_write.py`:

- `--contrastive-trace-pairs`: use opposite-label one-edit minimal pairs as the
  trace set and replace raw teacher deltas with pairwise contrast deltas.
- `--contrastive-auxiliary-pairs`: add a small number of pairwise contrast rows
  to an ordinary candidate-probe solve.
- `--paper-offset`: skip earlier rows in `domains.jsonl` so targeted paper/layer
  sweeps do not repeatedly rerun paper 0.

The goal was not to overfit the synthetic benchmark. This is a diagnostic for a
general idea: if two traces are matched in answer format and prompt style but
differ in rule-relevant content, subtracting them should cancel more nuisance
state than raw doc-minus-null deltas.

Single-paper paper-0 result, layer 20, final token, vector ridge:

| run | heldout delta | minimal-pair delta | inverse delta | near-collision delta | note |
| --- | ---: | ---: | ---: | ---: | --- |
| raw vector eta 1 | +0.05 | 0.00 | 0.00 | 0.00 | weak but safe |
| raw vector eta 2 | +0.10 | 0.00 | -0.20 | +0.10 | acquisition with polarity damage |
| pure contrast pairs eta 2 | +0.05 | +0.10 | 0.00 | -0.10 | first minimal-pair gain without inverse collapse, but harms near-collision |
| raw + contrast auxiliary eta 1 | 0.00 | 0.00 | 0.00 | 0.00 | too weak/inert |
| raw + contrast auxiliary eta 2 | +0.15 | +0.10 | 0.00 | +0.10 | best paper-0 setting so far |

Best paper-0 run:

- `runs/residual_operator_l20_vector_eta2_contrastaux8_s05_small_paper0_qwen17`
- heldout: `10/20 -> 13/20`, internalization ratio about `0.50`
- heldout positives stayed `10/10`
- heldout negatives moved `0/10 -> 3/10`
- minimal pairs moved `5/10 -> 6/10`
- inverse polarity stayed `5/10 -> 5/10`
- near-collision moved `6/10 -> 7/10`

This was the first setting in this residual-operator family that improved the
ordinary heldout and minimal-pair bucket without recreating the eta-2 inverse
polarity collapse.

Two-paper robustness check:

- `runs/residual_operator_l20_vector_eta2_contrastaux8_s05_2papers_qwen17`
- Same settings, no retuning.

| paper | heldout delta | ordinary delta | minimal delta | inverse delta | near delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.15 | 0.00 | +0.10 | 0.00 | +0.10 |
| 1 | 0.00 | -0.10 | -0.10 | -0.40 | 0.00 |

Plain raw vector eta-2 on the same two papers:

- `runs/residual_operator_l20_vector_eta2_2papers_qwen17`

| paper | heldout delta | ordinary delta | minimal delta | inverse delta | near delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.10 | 0.00 | 0.00 | -0.20 | +0.10 |
| 1 | 0.00 | 0.00 | 0.00 | -0.20 | 0.00 |

Interpretation: the contrastive auxiliary improves paper 0, but it does not
solve the generality problem. Paper 1 is a hard case for the layer-20 final-token
actuator either way, and the auxiliary can make its inverse-polarity damage
worse.

Paper-1 layer and position tests:

| run | heldout delta | ordinary delta | minimal delta | inverse delta | near delta | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| layer 8 only, eta 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | high trace alignment, no behavior |
| layers 8/12/16/20, eta 2 | 0.00 | +0.10 | 0.00 | +0.20 | 0.00 | multi-layer stack changes polarity behavior but still misses heldout |
| layers 8/12/16, eta 2 | 0.00 | 0.00 | 0.00 | +0.20 | 0.00 | mid-stack alone handles inverse bucket, not acquisition |
| layers 8/12/16, trace last 4, eta 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | multi-position traces make the write smaller/inert |
| layers 8/12/16, trace last 4, eta 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | scaling the cleaner trace still does not move behavior |
| layers 8/12/16, trace last 4, eta 8, all tokens | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | applying beyond final token does not reveal hidden gain |
| layers 8/12/16, source target, eta 2 | -0.10 | -0.10 | -0.20 | 0.00 | 0.00 | current source target hurts positives |

Important diagnostics:

- Layer 8 often has very high effect-target cosine on paper-1 heldout and
  minimal-pair traces, but no answer movement. That suggests it is shaping an
  upstream state, not acting as a direct behavioral actuator.
- Multi-layer 8/12/16 edits can improve inverse-polarity behavior without
  improving heldout negatives. That is evidence of a distinct state/behavior
  component, but not yet rule consolidation.
- Four-position trace rows produce much smaller updates. Even at eta 8, they
  align with teacher deltas but remain behaviorally inert. So "more nearby
  perspectives" is not automatically enough; the selected positions must be
  causally useful, not just less noisy.
- The current `teacher_source` implementation did not fix the problem on paper
  1. In this residual-operator form, source targets still remove or distort
  useful positive behavior.

Current conclusion:

Matched contrast is a useful diagnostic and can reduce one failure mode
(paper-0 inverse collapse), but it is not yet a general filter for
"understanding." The current residual-operator writes still depend heavily on
which layer is a behavioral actuator for the specific domain.

The sharper next direction is not more eta sweeps. It is causal perspective
selection:

- generate many in-context use-sites or quiz-style perspectives;
- measure which positions/layers actually causally move the teacher toward the
  correct functional behavior;
- fit the closed-form update only on the stable intersection of those causal
  source directions;
- keep matched contrast rows as a nuisance-removal term, not as the whole
  target.

This remains aligned with the final goal because the method being tested is
generic: use context traces, matched contrasts, and causal filtering to isolate
the context-induced computation before doing a cheap closed-form weight write.
The current benchmark uses DSL labels to create clean contrasts, so the present
contrastive rows are diagnostic scaffolding rather than the final deployment
protocol.

## 2026-05-19: Clean Mini-Language Benchmark, Modal, and Surprise/Locality Tests

Goal for this phase: stop evaluating on tasks where the context teacher is weak.
The yes/no mini-paper benchmark often had base around chance and context only
modestly better, so apparent write gains could be answer-prior repair rather
than consolidation. We built a teacher-gated mini-language control where:

- no-context base must fail;
- context teacher must solve the selected items;
- write probes and eval items can be separated by seed/overlap checks;
- unrelated sentinel tasks are evaluated before and after the write.

Modal setup:

- Modal profile `tedshachtman` is authenticated.
- Added `scripts/modal_caic_runner.py`.
- Remote A10 smoke passed: Torch sees CUDA and an NVIDIA A10.
- This made Qwen3-1.7B ablations practical: many runs complete in roughly
  25-55 seconds after image/model cache.

Benchmark builder:

- Added `scripts/build_minilang_teacher_gated.py`.
- Strict first-attempt seed search with `baseline=0/20` and `context=20/20`
  found no perfect 4-lesson seeds in 50 tries.
- Increasing examples to 12 improved teacher ceiling to 19/20 but did not
  reliably produce strict 20/20.
- Added `--selection-mode teacher_filter`: evaluate a larger heldout candidate
  pool, then select only items where base is wrong and context is right. This is
  not used for writing; it is an evaluation control.

Accepted clean benchmark:

- Remote artifact:
  `/modal-runs/minilang_teacher_filtered_6lesson_ex8_cand80_eval20_qwen17_cuda`
- Local copy:
  `runs/modal_minilang_teacher_filtered_6lesson_ex8_cand80_eval20_qwen17_cuda`
- Settings: 6 lessons, 8 examples/lesson, 80 candidate questions, select 20.
- Result: base `0/20`, context `20/20`, 43 eligible teacher-correct/base-wrong
  candidates out of 76.

Core clean-benchmark write results:

| run | eval | sentinel | interpretation |
| --- | ---: | ---: | --- |
| layer-20 MLP + attention-O, trace4, output delta | 9/20 | not measured | attention-O damaged the cleaner MLP effect |
| layer-20 MLP only, trace4, original seed | 14/20 | not measured | strong, but had one trace/eval overlap |
| layer-20 attention-O only | 5/20 | not measured | attention-O alone is weak |
| MLP only, eta 2 | 11/20 | not measured | stronger write corrupts more than it helps |
| MLP only, trace16, no eval overlap | 11/20 | not measured | more probes add noisy/incompatible constraints |
| MLP only, trace4 offset 5555, no eval overlap | 9/20 | not measured | the 14/20 result was not robust to probe seed |
| MLP target-filtered across direct/grammar/lexicon/roles perspectives | 9/20 | not measured | current perspective target filter did not improve |

Conclusion: the closed-form write has a real transfer signal on a clean task
(`0/20 -> 9/20` without context), but it is highly trace-selection dependent.
The current implementation does not yet reliably isolate the shared language
skill from local probe state.

Surprise/energy implementation:

- Added `positive_weights` to `caic.tsoc.protected_ridge_update`.
- Added `--activation-energy-weighting` to `scripts/minilang_write.py` with
  modes `block_action`, `source_norm`, `action_excess`, `mlp_key`, and
  `combined`.
- On the older six-paper benchmark, activation-energy weighting improved
  heldout by about `+0.083` over base, but that benchmark had weak teachers.
- On the clean mini-language benchmark:
  - `combined` energy: 9/20, same as no-energy.
  - sharper `source_norm` top-k: 13/20 in an overlapping-seed run, but this did
    not resolve the core no-overlap instability.

Interpretation: the current energy proxy is not the true surprise mechanism. It
selects large rows, but does not cleanly separate "new object configuration"
from "generic active task object."

Sentinel drift tests:

- Added `--sentinel-eval`: 10 unrelated MC tasks before/after the write,
  including arithmetic, grammar, common knowledge, and an explicit
  English-vs-Lyran contamination item.
- Baseline sentinel accuracy is low (`6/10`) because the scoring setup is rough,
  but before/after drift is still informative.

| run | eval | sentinel after | sentinel drift | margin drift |
| --- | ---: | ---: | ---: | ---: |
| no gate, no energy | 9/20 | 5/10 | -1/10 | -0.825 |
| energy combined | 9/20 | 5/10 | -1/10 | -0.834 |
| sentinel negative keys | 8/20 | 5/10 | -1/10 | -0.719 |
| memory gate threshold .95 | 7/20 | 6/10 | 0/10 | ~0 |
| memory gate threshold .90 | 8/20 | 6/10 | 0/10 | -0.001 |

This strongly supports the user hypothesis qualitatively:

- the raw write can make the model "live in translation world" too broadly;
- simple energy weighting does not prevent that;
- explicit locality/routing protects unrelated states almost perfectly;
- the current gate is too narrow, so it under-generalizes to heldout Lyran
  sentences.

Current best interpretation:

The closed-form write is doing something real, but the central unsolved problem
is conditionality. We need a gate or surprise geometry that is broad over the
new object/class (Lyran translation states) and narrow against ordinary task
states. Raw full-matrix writes generalize more but drift. Explicit similarity
gates preserve old behavior but miss many heldout cases. The next direction is
therefore not "more eta" or "write all layers"; it is learning/deriving a
domain/content trigger manifold or surprise-locality metric.

Concrete next experiments:

- Build a better domain/content gate using Lyran token/content positions rather
  than only suffix answer-state keys.
- Evaluate a two-factor gate: content/domain trigger decides whether this is the
  learned object; suffix/behavior state decides where to apply the actuator.
- Improve sentinels so baseline is closer to 10/10; current drift signal is
  useful but the sentinel benchmark itself is weak.
- Test sequential multi-language or same-language lesson streams with
  acquisition, retention, and sentinel drift after every write.
- Treat energy/surprise as "default-object protected geometry" rather than row
  magnitude: preserve normal task-object states while allowing context-induced
  source terms in the novel object subspace.

### Follow-up: Routing and All-Layer Surprise Diagnostics

User objection accepted: literal term/content slot routing is hacky and should
not be treated as a proposed final method. It is useful only as a diagnostic
that conditional routing matters.

Term slot diagnostic:

- Added `--term-slot-gate`, first with substring matching, then fixed to
  whole alphanumeric term boundaries after noticing `pa` matched `Paris`.
- Word-boundary term slot result on layer 20:
  - edited: `9/20`, same as no-gate;
  - sentinel: `6/10 -> 6/10`;
  - sentinel margin drift: `-0.213`, much smaller than no-gate `-0.825`.
- Interpretation: object/domain routing can preserve unrelated behavior without
  losing acquisition, but string routing is not general enough for the goal.

All-layer non-string runs:

| run | edited | sentinel | sentinel drift | interpretation |
| --- | ---: | ---: | ---: | --- |
| layer 20 no gate | 9/20 | 5/10 | -1/10 | stable single-layer signal, some drift |
| all 28 MLP, energy weighting, eta .25, norm 10 | 11/20 | 3/10 | -3/10 | learns more but smears badly |
| all 28 MLP, energy + sentinel negative keys | 11/20 | 5/10 | -1/10 | negatives reduce but do not solve drift |
| all 28 MLP, energy + behavior gate .90 | 4/20 | 6/10 | 0/10 | protects old behavior but under-generalizes |
| all 28 MLP, energy + key-separation filter | 9/20 | 4/10 | -2/10 | suppresses default-like layers but still broad |
| layers 17-23 only, energy + sentinel negatives | 5/20 | 5/10 | -1/10 | separability alone does not identify useful layers |

Activation-derived separability diagnostic:

- For each layer, measured max cosine from ordinary/sentinel negative keys to
  Lyran trace keys.
- Early layers were default-like: negative-to-positive mean max around `0.90`.
- Mid/late layers were more separated: roughly `0.27-0.55` around layers 17-23.
- Final layer collapsed back toward generic state.

This supports the "surprise/default state" intuition but also shows row
magnitude and simple key separability are not sufficient. All-layer writes can
capture more of the context behavior, but without a better surprise/routing rule
they impose the task world too broadly.

Closed-book interpretation:

The context teacher's `20/20` is an open-book upper bound. It includes both
internalized understanding and the ability to reread/recompute from the lesson.
The no-context weight-written model should not be expected to always reach
`20/20` from one consolidation pass. The right metric is closed-book
internalization ratio: how much of the context upper bound remains when the
book is removed, while unrelated abilities are preserved.

Item-level behavior for the stable no-string layer-20 write:

- Base: `0/20`.
- Context: `20/20`.
- Edited: `9/20`.
- The edited model often gets simple lexical/choice structure right, but remains
  inconsistent on role binding, subject/object swaps, and tense. This suggests
  the write captured part of the language object but not the full executable
  grammar.

Updated next direction:

- Split evaluation into closed-book concept recall, simple application, and
  compositional/open-book-style recomputation. Do not use one score as if it
  measured one thing.
- Replace literal term gates with activation-derived object gates: infer whether
  the relevant learned object is instantiated from content/use-site activations,
  then apply the write only under that latent gate.
- Treat surprise as "deviation from the model's default instantiation of the
  current object," not as large activation norm. The implementation needs a
  default-state model or covariance/prototype geometry per object class.

### Activation Object-Gate Implementation

Implemented the next routing primitive, without string matching:

- `caic.modeling.AdditiveMemoryLinear` now supports a sequence-level
  `object_gate_keys` gate. It computes whether the learned object is present
  anywhere in the current sequence, then multiplies the memory by one scalar per
  batch item. This is separate from the existing token-local behavior gate.
- `scripts/minilang_write.py` adds `--activation-object-gate`, using source
  sentence activation keys from prompts that stop on the invented-language
  sentence rather than on `English:`. This gives a two-factor gate:
  object/content state decides whether the write is active; behavior keys can
  still decide where in the answer posture it fires.
- The ensemble mini-language path now logs sentinel before/after metrics when
  `--sentinel-eval` is enabled, matching the non-ensemble path.
- `scripts/minilang_continual_triangle.py` now logs sentinel drift after each
  write and retention deltas from each task's immediate post-acquisition score.
  This directly measures the new goal: after later tasks, a previous task
  should not perform worse than it did right after it was learned.

Validation:

```text
.venv/bin/python -m pytest -q
30 passed
```

No model result is claimed yet for the activation object gate. The first run to
try is the clean teacher-filtered mini-language artifact with layer-20 MLP-only,
`--activation-object-gate`, `--sentinel-eval`, and optionally the existing
token-local `--memory-gate` as the second factor.

### Contrastive Density Object Gate (CLOD v1)

Implemented the first version of the proposed contrastive likelihood-ratio
router:

- Added `caic/contrastive_gate.py`.
- `fit_contrastive_density_gate` takes row-major activation matrices
  `[n, d]`, performs diagonal calibration whitening, builds a compact SVD basis,
  solves the protected generalized-eigen problem
  `S_pos v = lambda S_neg v`, and fits shrinkage Gaussian densities in the
  resulting low-rank subspace.
- Runtime score is
  `log p_pos(q) - max_g log p_neg_g(q)`, with an added positive-support floor.
  The support floor matters: the first unit test exposed an open-space failure
  where a token outside both positive and negative Gaussians received a high
  likelihood ratio.
- `AdditiveMemoryLinear` now supports density-ratio sequence object gates in
  addition to the older cosine object gate. Multiple density gates are combined
  by max/OR and can use the same `object_gate_floor` dampening parameter.
- `scripts/minilang_write.py` adds `--object-gate-mode density_ratio` for
  MLP writes. The first implementation uses source/content object keys as
  positives, and default answer plus sentinel activations as protected
  negatives. Attention writes are intentionally rejected for density gates until
  the MLP path is understood.

Validation:

```text
.venv/bin/python -m pytest -q
33 passed
```

Clean teacher-filtered mini-language, layer-20 MLP, behavior gate `.90`,
output-delta target, trace probes `4`, sentinel eval:

| run | edited | sentinel | sentinel drift | margin drift | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| behavior gate only `.90` | `8/20` | `6/10 -> 6/10` | `0/10` | `-0.001` | current best non-string safe baseline |
| CLOD uncalibrated | `5/20` | `6/10 -> 6/10` | `0/10` | `-0.0048` | density score separated trace positives, but sigmoid temp was dominated by huge positive outliers; protected negatives gated at about `0.49`, making this mostly half-strength behavior-gated writing |
| CLOD calibrated, floor `0.0` | `2/20` | `6/10 -> 6/10` | `0/10` | `-0.0020` | protected negatives gated low, but heldout acquisition collapsed; positive manifold under-covered deployment prompts |
| CLOD calibrated, floor `0.50` | `4/20` | `6/10 -> 6/10` | `0/10` | `-0.0020` | dampened write recovers some acquisition |
| CLOD calibrated, floor `0.75` | `6/20` | `6/10 -> 6/10` | `0/10` | `-0.0066` | better acquisition/safety tradeoff, still behind behavior gate only |

Continual six-task behavior-gate `.90` result:

- Teacher-filtered eval selected `8/80` context-correct questions for each of
  six tasks; context was `8/8` on every selected task.
- Immediate acquisition by task: `1/8`, `1/8`, `4/8`, `1/8`, `1/8`, `4/8`.
- Final after all six tasks: `4/8`, `1/8`, `3/8`, `2/8`, `2/8`, `4/8`.
- Sentinel stayed `6/10 -> 6/10` with final margin drift `-0.0013`.
- One task degraded relative to immediate acquisition (`Seldic`, `4/8 -> 3/8`).

Updated diagnosis:

- The generalized-eigen density-ratio object is mathematically useful and
  safe, but CLOD v1 is not yet the winning router.
- The hard/strict object manifold is too narrow when fit from the current
  `trace_probes=4` source-token positives. It separates training positives from
  protected negatives but does not cover heldout deployment instantiations well
  enough.
- The blocker is now sharper: positive applicability-manifold construction,
  not the ridge write, not scale, and not sentinel protection in isolation.

Immediate next experiments:

- Add heldout object-gate diagnostics: score distributions for trace positives,
  eval source-token prompts, sentinel prompts, default answer prompts, and rival
  task prompts. The current updates log only shows fit-time positive/protected
  separation.
- Increase object-gate positive coverage independently from write trace rows:
  allow `--object-gate-probes` separate from `--trace-probes`, and generate
  unlabeled source/application prompts for gate fitting.
- Add rival mini-language negative groups to the density gate. The current CLOD
  run only used default answer and sentinel negatives.
- After the gate distribution is sane, add the protected contrastive key
  projection into the closed-form solve. Do not implement that before verifying
  heldout gate coverage.

Follow-up: expanded object-gate positives and diagnostics.

- Added `--object-gate-probes` so the object/applicability manifold can use
  many unlabeled source/application prompts without increasing the number of
  write-target traces.
- Added `--object-gate-rival-negatives` using a rival invented-language prompt
  family as protected negatives.
- Added `--object-gate-diagnostics`, logging fitted gate behavior on training
  positives, write keys, heldout eval source keys, default answer keys, sentinel
  keys, and rival keys.
- Added density-gate top-k sequence pooling. This tests the intended two-factor
  semantics: content/source tokens should establish sequence-level object
  presence, while behavior keys decide readout/use positions.

Layer-20 CLOD, behavior gate `.90`, write traces `4`, object-gate probes `64`,
rival negatives `64`:

| object floor / pooling | edited | sentinel | margin drift | interpretation |
| --- | ---: | ---: | ---: | --- |
| floor `.75`, original pooling | `7/20` | `6/10 -> 6/10` | `-0.0020` | expanded positive manifold improves CLOD from `6/20` to `7/20`, but still relies on floor/dampening |
| floor `0.0`, original pooling | `1/20` | `6/10 -> 6/10` | `-0.0020` | strict object gate still kills acquisition |
| floor `0.0`, top-k `8` | `1/20` | `6/10 -> 6/10` | `-0.0020` | top-k pooling alone does not fix runtime underfire |
| floor `0.0`, top-k `1` | `0/20` | `6/10 -> 6/10` | `-0.0066` | max pooling is worse |

Diagnostic clue:

- With floor `.75`, heldout eval source-token gate means were high
  (`~0.88-0.98` on later lessons), and protected negative fitted gates were low.
- With floor `0.0`, heldout source-token diagnostics remained moderate/high, but
  write-key/readout diagnostics were very low (`~0.01-0.03`), and behavior
  acquisition collapsed.

Updated interpretation:

- More gate positives helped, which supports the manifold-coverage hypothesis.
- However, the strict density object gate is still not a correct implementation
  of two-factor routing. It recognizes source/content activations in diagnostic
  captures, but the runtime memory output is still effectively suppressed at
  readout unless a large floor is present.
- The next implementation should explicitly separate sequence-level object
  evidence from token-local memory application, rather than relying only on
  scoring the same MLP-down activation tensor inside the wrapper.
- A second, non-translation benchmark is now important. A synthetic
  persona/user-model task would test whether the same object/use/protected
  machinery can consolidate arbitrary contextual understanding rather than
  translation-specific structure.

### Multi-Layer CLOD Safety Check

Ran CLOD v1 beyond layer 20 to test the constraint that a real
surprise/locality system should be safe when available throughout the model.

Settings:

- Clean teacher-filtered mini-language artifact.
- MLP output-delta writes.
- Behavior gate `.90`.
- Density object gate with floor `.75`.
- Trace probes `4`, seed offset `5555`.
- Sentinel eval enabled.

Results:

| layers | edited | sentinel | margin drift | interpretation |
| --- | ---: | ---: | ---: | --- |
| `20` | `6/20` | `6/10 -> 6/10` | `-0.0066` | best CLOD v1 single-layer result |
| `8,12,16,20,24` | `3/20` | `6/10 -> 6/10` | `+0.0107` | safe, but acquisition collapses |
| `16,20,24` | `4/20` | `6/10 -> 6/10` | `+0.0102` | early layers were not the only issue |
| all `0..27` MLP | `3/20` | `6/10 -> 7/10` | `+0.1671` | all-layer CLOD is sentinel-safe here, but not acquisition-useful |

Layer diagnostics on the all-28 run showed the density object gate suppressing
protected negatives strongly (`neg_gate_mean` around `0.018`) across layers.
Trace positive gate means were high, often `0.85-0.99`. Therefore the failure is
not simply that the fitted gate is inactive on training positives or active on
sentinels.

Updated interpretation:

- CLOD v1 passes a weak safety test across all MLP layers on this sentinel set:
  it does not break the sentinel benchmark.
- It fails the stronger desideratum: adding more layers should add useful source
  fields. Instead, multi-layer writes are less useful than the single known-good
  layer.
- This points to layerwise source incoherence / overconstrained readout rather
  than just insufficient layer coverage. Independent output-delta writes at many
  layers are not composing into the teacher trajectory.
- Next acquisition work should prioritize broader positive applicability
  coverage and sequential layerwise replay/source targets before further
  all-layer scaling.

### One-Pass Intrinsic Surprise Writes

This round returned to the stricter research constraint:

- lesson/context forward only;
- no generated quiz/probe pass for the write;
- no null/default prompt pass;
- no runtime object router or behavior gate;
- no next-token prediction target;
- the prior must come from current weights plus the single lesson activation
  trace.

Implemented:

- `caic/intrinsic_surprise.py`
  - weight-relative MLP feature surprise,
  - one-pass feature persistence,
  - generic down-value projection,
  - associative feature binding writes,
  - state feature-birth writes,
  - conjunctive SwiGLU feature-birth writes,
  - final-aligned token selection.
- `scripts/minilang_write.py`
  - `--intrinsic-surprise-write`,
  - `--intrinsic-surprise-target-mode associative_binding|feature_birth`,
  - `--intrinsic-surprise-token-mode last|top|all|final_aligned`,
  - feature-birth/conjunction options,
  - effective-target normalization and LM-head generic projection options.
- Unit coverage in `tests/test_intrinsic_surprise.py`.

Validation: `.venv/bin/python -m pytest -q` passed with `42 passed`.

Key result:

The new final-aligned selector proves that the one-pass signal is real. It
selects lesson tokens whose intrinsic surprise overlaps the final integrated
lesson state, avoiding the worst boilerplate failures from raw top-energy
selection.

| method | edited | sentinel | margin drift | interpretation |
| --- | ---: | ---: | ---: | --- |
| associative, final token, effective, scale `.5` | `5/20` | `6/10 -> 5/10` | `-2.81` | first lesson-only signal, but toxic |
| feature birth, final token, fresh neurons | `0/20` | `6/10 -> 6/10` | `+0.06` | safe but too specific |
| conjunctive feature birth, final token, no reuse | `1/20` | `6/10 -> 6/10` | `+0.05` | safe, weak |
| conjunctive feature birth, top-8 tokens | `0/20` | `6/10 -> 6/10` | `-0.07` | raw top surprise selects boilerplate |
| associative, final-aligned top-8, effective, scale `.5` | `10/20` | `6/10 -> 2/10` | `-3.71` | acquisition target reached, unsafe |
| same, scale `.1` | `2/20` | `6/10 -> 6/10` | `-0.20` | safe but weak |
| same, scale `.2` | `5/20` | `6/10 -> 5/10` | `-1.00` | stronger but unsafe |
| same, 4 layers `12,16,20,24`, scale `.1` | `2/20` | `6/10 -> 6/10` | `-0.11` | multi-layer small writes remain safe but weak |
| same, LM-head projection rank `64`, scale `.5` | `10/20` | `6/10 -> 4/10` | `-3.22` | projection helps but not enough |
| same, LM-head projection rank `256`, scale `.5` | `10/20` | `6/10 -> 4/10` | `-2.84` | similar acquisition, still unsafe |
| same, LM-head projection rank `1024`, scale `.5` | `9/20` | `6/10 -> 4/10` | `-3.79` | too much projection loses some acquisition, not safety |

Important diagnostic:

- Raw top-energy/persistence token selection chose `"Ly"`, `" VER"`, `"Ad"`,
  `"=p"`, and other lesson-format tokens. Energy alone was identifying unusual
  document structure, not the latent object.
- Final-aligned surprise moved selected positions into the example/application
  span and produced the first `10/20` lesson-only acquisition result.
- The unsafe `10/20` runs show runaway effective targets: target Frobenius norm
  grows from hundreds to tens of thousands across six lessons. Lowering scale
  removes the damage but also removes most acquisition.

Updated diagnosis:

- The blocker is no longer "can a single forward pass expose useful
  context-understanding signal?" It can: final-aligned associative surprise
  reached `10/20`.
- The blocker is closed-form protection/locality without a second pass or
  runtime gate. Existing weight-only protections are insufficient:
  down-value generic projection, base-norm effective targets, stronger ridge,
  and LM-head projection all fail to preserve sentinels at the acquisition
  point.
- Down-only associative writes are too broad but generalize. Feature-birth and
  conjunction writes are safe but under-generalize. The missing mathematical
  object is a weight-only protected solve that keeps the final-aligned
  acquisition direction while suppressing globally reusable/output-policy
  directions.

Next research move:

- Replace scalar ridge with a weight-derived protected metric, not a runtime
  gate. The natural candidate is a closed-form solve with an input-side and
  output-side penalty:

  \[
  \min_{\Delta W}
  \|\Delta W A - R\|^2
  + \lambda\|\Delta W\|^2
  + \alpha\|H\Delta W A\|^2
  + \beta\|\Delta W G\|^2
  \]

  where `H` is a weight-only output/readout sensitivity operator derived from
  the unembedding and later layer weights, and `G` is a weight-only generic-key
  subspace derived from MLP feature geometry. This keeps the write rule
  closed-form and one-pass but gives it a real notion of "do not write generic
  capability/readout directions."


## 2026-05-20 - Exponential Surprise and Expanded Sentinel Metric

Implemented two updates to the intrinsic surprise write path:

- Added exponential surprise row weighting in `caic/intrinsic_surprise.py` and
  `scripts/minilang_write.py`:
  - `--intrinsic-surprise-weight-mode linear|exponential`
  - `--intrinsic-surprise-exp-temperature`
  - `--intrinsic-surprise-exp-cap`
- Added an expanded sentinel suite and harm-oriented sentinel shift metrics:
  - `--sentinel-suite core|expanded`
  - `sentinel_correct_to_wrong`
  - `sentinel_wrong_to_correct`
  - `sentinel_preservation_rate`
  - `sentinel_mean_margin_drop`
  - `sentinel_max_margin_drop`
  - `sentinel_before_correct_mean_margin_drop`
  - `sentinel_severe_margin_drop_count`

Validation: `.venv/bin/python -m pytest -q` passed with `44 passed`.

Motivation:

The hypothesis was that consolidation should be nonlinear in surprise: ordinary
small mismatches in a familiar context should produce little update, while a
large violation of the model's latent object prior should dominate the update.
This was implemented as robust-standardized exponential weighting over selected
surprise rows in the closed-form solve.

Main result:

Naive exponential weighting is not yet the missing tool. It made the write more
selective, but the highest raw-surprise rows are not reliably the safest or most
semantic rows. Strong exponential weighting under-acquired and still damaged the
expanded sentinel suite. Gentler exponential weighting tied the linear protected
baseline but did not improve the Pareto frontier.

Expanded sentinel comparison on `final_aligned` top-8 tokens, `key_feature_top_k=16`,
layer 20, protected metric solve, output penalty rank 256/weight 10, input penalty
256/weight 20/usage power 1:

| weighting | scale | edited | expanded sentinel | correct->wrong | wrong->correct | preservation | mean margin drop | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | `.50` | `12/20` | `12/25 -> 9/25` | `6` | `3` | `0.500` | `3.613` | `7.171` |
| linear | `.35` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `3.550` | `7.124` |
| exp temp `2.0`, cap `20` | `.50` | `12/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `3.567` | `7.126` |
| exp temp `1.5`, cap `20` | `.35` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `3.503` | `7.073` |
| exp temp `1.0`, cap `100` | `.50` | `7/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.252` | `6.558` |
| exp temp `.7`, cap `100` | `.50` | `9/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.466` | `6.765` |

Interpretation:

- The user hypothesis is directionally plausible, but raw feature-surprise
  magnitude is not purified enough. Exponential weighting amplifies both semantic
  novelty and high-surprise nuisance rows.
- The better sentinel metric shows why raw accuracy was too coarse: some writes
  gain sentinel items while simultaneously flipping many previously correct
  items and causing large margin drops.
- The best current operating points are still not safe enough under the expanded
  sentinel metric. `11-12/20` acquisition comes with `5` or more correct-to-wrong
  sentinel flips on the expanded suite.

Updated diagnosis:

The next mathematical object should not be plain exponential surprise. It should
be exponential weighting of a *purified* surprise score, where the score discounts
features whose activation can be explained by generic answer posture, option
formatting, or high-impact globally reused MLP features. In other words:

\[
  w_i \propto \exp(\operatorname{purified\_surprise}_i / T)
\]

not

\[
  w_i \propto \exp(\operatorname{raw\_feature\_surprise}_i / T).
\]

A likely next implementation is an intrinsic contrast score computed from the
single lesson forward plus weights:

\[
  \operatorname{purified\_surprise}_{t,j}
  = \operatorname{feature\_surprise}_{t,j}
  \cdot \operatorname{lesson\_persistence}_{j}
  \cdot \operatorname{down\_specificity}_{j}
  \cdot (1 - \operatorname{generic\_readout\_alignment}_{j})
\]

then apply exponential weighting to that purified score. The current experiments
show that the exponential nonlinearity should come after purification, not before.


## 2026-05-20 - Purified Exponential Surprise Sweep

Implemented a weight-only readout-specificity factor for intrinsic surprise scoring:

- Added `down_output_basis_specificity(...)` in `caic/intrinsic_surprise.py`.
- Added `--intrinsic-surprise-readout-specificity-power` in `scripts/minilang_write.py`.
- This lets the feature score used for token/feature selection and row weighting be purified by:
  - lesson persistence from the single lesson forward,
  - down-value specificity from current MLP down weights,
  - readout/output specificity against the LM-head generic basis.

Validation: `.venv/bin/python -m pytest -q` passed with `45 passed`.

Test setup:

All runs below used layer 20, `final_aligned` top-8 tokens, `key_feature_top_k=16`,
protected metric solve, output penalty rank 256/weight 10, input penalty
256/weight 20/usage power 1, and the expanded 25-question sentinel suite.

| method | edited | expanded sentinel | correct->wrong | wrong->correct | preservation | mean margin drop | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear, scale `.35` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `3.550` | `7.124` |
| exp temp `1.5`, cap `20`, scale `.35` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `3.503` | `7.073` |
| purified linear, scale `.35` | `12/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.470` | `6.616` |
| purified exp temp `1.5`, scale `.35` | `12/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.476` | `6.693` |
| purified exp temp `2.0`, scale `.25` | `9/20` | `12/25 -> 7/25` | `7` | `2` | `0.417` | `3.550` | `6.007` |
| purified exp temp `2.0`, scale `.35` | `12/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.520` | `6.662` |
| purified exp temp `3.0`, scale `.35` | `12/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.514` | `6.622` |
| readout-only exp temp `2.0`, scale `.50` | `11/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `3.599` | `7.127` |
| purified exp temp `2.0`, scale `.50` | `11/20` | `12/25 -> 9/25` | `5` | `2` | `0.583` | `3.416` | `5.961` |

Conclusion:

Purified exponential surprise is not enough in this form. It can reduce the
average margin harm on previously correct sentinel items, but it does not reduce
correct-to-wrong sentinel flips below the best linear/exponential baseline. In
some settings it worsens preservation despite good acquisition.

Interpretation:

- The score purification factors are not isolating semantic novelty sharply
  enough. Multiplying persistence, down-value specificity, and readout
  specificity changes which rows dominate, but the selected high-score rows still
  include behaviorally dangerous directions.
- Exponential surprise appears useful only after the *right* surprise coordinate
  exists. Applying it to the current intrinsic feature score mostly sharpens an
  imperfect ranking.
- The expanded sentinel metric is now doing its job: acquisition alone looks
  promising (`11-12/20`), but correct-to-wrong flips show the write is still
  damaging pre-existing capabilities.

Next mathematical move:

The blocker is now the feature/row score, not the exponential nonlinearity. The
next candidate should estimate a local *surprise residual* inside the lesson
itself, rather than scoring features independently. One concrete direction is a
single-pass within-lesson predictor:

\[
  \hat a_{t,j} = f_j(\text{nearby/contextual feature history before } t)
\]

using only weight-derived/simple closed-form statistics from the same lesson,
then score:

\[
  s_{t,j}=\frac{(a_{t,j}-\hat a_{t,j})^2}{\sigma_j^2+\epsilon}
\]

This would distinguish "feature is globally/highly active" from "feature is
locally unexpected given the object's current trajectory." If the bagel-store
object is mostly normal, most features are predicted and receive little update;
only the locally unexplained feature-binding residual gets exponentiated.


## 2026-05-20 - Single-Pass Predictive Residual Surprise Write

Implemented a no-SAE predictive-residual write over native MLP channels:

- Added `select_intrinsic_predictive_residual_write(...)` in `caic/intrinsic_surprise.py`.
- Added `--target-mode predictive_residual` and `--intrinsic-surprise-prediction-ridge` in `scripts/minilang_write.py`.
- Added regression coverage in `tests/test_intrinsic_surprise.py`.

Validation: `.venv/bin/python -m pytest -q` passed with `46 passed`.

Theory tested:

The write should not reward raw high activation. It should reward local
prediction error inside the lesson. For selected surprising token rows, the
algorithm splits active MLP features into cheap "cause" channels and candidate
"residual" channels, fits a tiny ridge predictor from the lesson's previous
rows, then writes only the residual feature mass that was not predictable from
the current local object trajectory.

This is the cheap predictive-coding analogue:

\[
  \epsilon_{t,j}
  =
  a_{t,j}
  -
  \hat a_{t,j}(a_{<t,\text{cause}})
\]

with exponential weighting applied to the residual surprise, not to raw feature
energy.

Test setup:

All runs used the clean teacher-filtered 6-lesson mini-language artifact,
`final_aligned` top-8 tokens, `key_feature_top_k=16`, protected metric solve,
output penalty rank 256/weight 10, input penalty rank 256/weight 20/usage
power 1, and the expanded 25-question sentinel suite.

| method | edited | expanded sentinel | correct->wrong | wrong->correct | preservation | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| predictive linear, scale `.50`, layer 20 | `1/20` | `12/25 -> 12/25` | `2` | `2` | `0.833` | `+0.348` | `1.233` |
| predictive exp temp `2.0`, scale `.50`, layer 20 | `1/20` | `12/25 -> 12/25` | `2` | `2` | `0.833` | `+0.455` | `1.609` |
| predictive exp temp `2.0`, scale `.75`, layer 20 | `3/20` | `12/25 -> 8/25` | `6` | `2` | `0.500` | `-1.374` | `3.813` |
| predictive exp temp `2.0`, scale `1.00`, layer 20 | `3/20` | `12/25 -> 9/25` | `5` | `2` | `0.583` | `-1.121` | `3.544` |
| predictive exp temp `2.0`, scale `1.50`, layer 20 | `1/20` | `12/25 -> 11/25` | `4` | `3` | `0.667` | `-0.296` | `2.157` |
| predictive exp temp `2.0`, scale `.50`, layers 18,20 | `1/20` | `12/25 -> 14/25` | `2` | `4` | `0.833` | `+1.872` | `2.492` |
| predictive exp temp `2.0`, scale `.50`, layers 16,18,20 | `1/20` | `12/25 -> 11/25` | `3` | `2` | `0.750` | `-1.522` | `3.938` |

For comparison, the earlier associative binding write at layer 20 reached
`11/20`, but expanded sentinel moved `12/25 -> 10/25` with `5` correct-to-wrong
flips and before-correct margin drop around `7.1`.

Update-scale diagnostics:

| method | mean selected rows | mean target Frobenius | mean update Frobenius | mean negative RMSE |
| --- | ---: | ---: | ---: | ---: |
| predictive exp `.50`, layer 20 | `128` | `912` | `17.8` | `0.0063` |
| predictive exp `.75`, layer 20 | `128` | `2486` | `31.1` | `0.0519` |
| predictive exp `1.00`, layer 20 | `128` | `3663` | `33.1` | `0.0615` |
| predictive exp `1.50`, layer 20 | `128` | `8111` | `41.3` | `0.1117` |
| predictive exp `.50`, layers 18,20 | `128/layer` | `344`, `510` | `10.6`, `6.7` | `0.0020`, `0.0007` |
| predictive exp `.50`, layers 16,18,20 | `128/layer` | `186`, `411`, `878` | `10.1`, `13.2`, `11.9` | `0.0010`, `0.0030`, `0.0133` |

Conclusion:

The cheap predictive-residual rule is substantially safer than the raw
associative write, but it under-acquires. At scale `.50`, it mostly preserves
or improves the expanded sentinel score, including the two-layer run
`12/25 -> 14/25`, but acquisition is only `1/20`. Raising scale can reach
`3/20`, but sentinel damage returns immediately.

Interpretation:

- The predictive-residual idea is doing something real: it drastically reduces
  target/update size and protected-negative RMSE relative to the high-acquisition
  associative write.
- The current local predictor is too conservative. It throws away too much of
  the behaviorally useful lesson signal along with the dangerous generic
  activation mass.
- Adding more layers does not automatically improve acquisition. Layer 18 plus
  20 is safe at this scale, but not more capable. Layer 16 plus 18 plus 20 is
  less safe.
- The current bottleneck is not SAE-style feature discovery. It is the
  cause/error split: the write needs to identify surprising *relations* among
  native MLP channels, not only unpredicted residual channel magnitudes.

Next mathematical move:

Keep the single-forward, no-SAE constraint, but upgrade the predictor from
"current token residual feature after previous-token feature history" to a
native-channel binding residual:

\[
  \epsilon_{t,jk}
  =
  a_{t,j}a_{t,k}
  -
  \widehat{a_{j}a_{k}}(\text{lesson history}, W_{\text{up}}, W_{\text{down}})
\]

or an equivalent low-rank approximation. The target should remain protected by
the input/output metric penalties, but the surprise coordinate should be
relational/conjunctive. The likely missing object is not "feature j is
surprising"; it is "feature j and feature k are co-instantiated in a way the
current weight geometry did not predict." That is closer to the user's purple
giraffe example: not purple alone, not giraffe alone, but the unexpected binding
between otherwise familiar latent objects.


## 2026-05-20 - Relational Native-Channel Surprise

Implemented a no-SAE relational surprise write:

- Added `select_intrinsic_relational_residual_write(...)` in `caic/intrinsic_surprise.py`.
- Added `--intrinsic-surprise-target-mode relational_residual`.
- Added `--intrinsic-surprise-pair-top-k`.
- Added `--intrinsic-surprise-bidirectional-pairs`.
- Added unit coverage for unexpected pair binding and reciprocal pair rows.

Validation: `.venv/bin/python -m pytest -q` passed with `48 passed`.

The mathematical move:

Instead of treating a single MLP channel as the surprise coordinate, score
unexpected co-instantiation between native MLP channels. For a selected token,
fit a tiny within-lesson predictor:

\[
  a_i a_j \approx f(a_i, a_j)
\]

using only the same lesson forward. The write keeps pair rows where:

\[
  |a_i a_j| - |\widehat{a_i a_j}| > 0
\]

and then writes sparse associative rows from one member of the surprising pair
to the other member's down-projection value. This is still a single-forward,
closed-form, no-SAE method. No extra probes, no null pass, no next-token
training, and no runtime router.

Main results on the clean teacher-filtered 6-lesson mini-language artifact with
expanded 25-question sentinel:

| method | edited | expanded sentinel | correct->wrong | wrong->correct | preservation | margin delta | mean margin drop | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| predictive residual, exp `.50`, layer 20 | `1/20` | `12/25 -> 12/25` | `2` | `2` | `0.833` | `+0.455` | `1.002` | `1.609` |
| relational pair16, exp `.35`, layer 20 | `1/20` | `12/25 -> 12/25` | `0` | `0` | `1.000` | `+0.010` | `0.081` | `0.113` |
| relational pair16, exp `5.0`, layer 20 | `1/20` | `12/25 -> 13/25` | `2` | `3` | `0.833` | `+0.493` | `0.720` | `0.838` |
| relational pair16, exp `10.0`, layer 20 | `2/20` | `12/25 -> 14/25` | `3` | `5` | `0.750` | `+0.425` | `1.386` | `2.083` |
| relational pair64, exp `5.0`, layer 20 | `2/20` | `12/25 -> 15/25` | `3` | `6` | `0.750` | `+0.534` | `1.215` | `1.403` |
| relational pair64, exp `7.5`, layer 20 | `4/20` | `12/25 -> 13/25` | `3` | `4` | `0.750` | `-0.416` | `2.046` | `3.239` |
| relational pair64, exp `10.0`, layer 20 | `6/20` | `12/25 -> 12/25` | `4` | `4` | `0.667` | `-1.001` | `2.621` | `4.502` |
| bidirectional pair64, exp `2.5`, layer 20 | `3/20` | `12/25 -> 15/25` | `2` | `5` | `0.833` | `+1.011` | `0.865` | `1.137` |
| bidirectional pair64, exp `5.0`, layer 20 | `3/20` | `12/25 -> 13/25` | `3` | `4` | `0.750` | `-0.170` | `1.641` | `2.274` |
| bidirectional pair64, exp `7.5`, layer 20 | `2/20` | `12/25 -> 14/25` | `2` | `4` | `0.833` | `-0.603` | `2.029` | `3.156` |
| associative binding baseline, linear `.35`, layer 20 | `11/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `-1.763` | `3.550` | `7.124` |

Update diagnostics:

| method | rows/layer/lesson | target Frobenius | update Frobenius | negative RMSE |
| --- | ---: | ---: | ---: | ---: |
| relational pair16, exp `.35` | `128` | `130` | `1.7` | `0.0000` |
| relational pair16, exp `10.0` | `128` | `3704` | `31.8` | `0.0334` |
| relational pair64, exp `5.0` | `512` | `2928` | `19.0` | `0.0221` |
| relational pair64, exp `10.0` | `512` | `5856` | `31.5` | `0.0647` |
| bidirectional pair64, exp `2.5` | `1024` | `2393` | `16.9` | `0.0145` |
| bidirectional pair64, exp `5.0` | `1024` | `4786` | `28.5` | `0.0678` |

Interpretation:

- Relational surprise is much safer than raw associative binding at comparable
  update norms. At low/moderate scales it can preserve or improve expanded
  sentinel score with far smaller margin drops.
- The first real acquisition point is relational pair64 scale `10.0`, which
  reaches `6/20` while keeping net expanded sentinel accuracy unchanged
  (`12/25 -> 12/25`). This is not safe enough by the stricter flip/margin
  metric because it still causes `4` correct-to-wrong flips and a before-correct
  margin drop of `4.50`.
- The best safe-ish point is bidirectional pair64 scale `2.5`: `3/20`,
  expanded sentinel `12/25 -> 15/25`, only `2` correct-to-wrong flips, and
  before-correct margin drop `1.14`.
- Reciprocal orientation helps at the safety end but does not improve peak
  acquisition. This suggests orientation is a secondary issue, not the main
  bottleneck.

Current diagnosis:

The native-channel relational coordinate is directionally better than single
feature surprise: it gives a much cleaner safety/acquisition curve. But it is
still not isolating the behaviorally central bindings strongly enough to hit the
near-term goal of about `10/20` while preserving sentinel behavior.

The next blocker is likely *which relation rows become values*, not whether the
surprise should be relational. The current target copies the down-projection
value of one paired channel. For language learning, the useful object may be a
higher-rank phrase/role binding that requires aggregating several pair-selected
value channels into one row per trigger feature, closer to associative binding,
but with relational residual weights choosing the value mixture.

Next implementation candidate:

Aggregate relational residual pairs by trigger feature:

\[
  v_i
  =
  \sum_j
  \omega_{ij}
  a_j e_j
\]

where \(\omega_{ij}\) is the pair residual score. Then write one sparse row per
trigger \(i\) to the aggregated target \(v_i W_{\text{down}}^\top\). This keeps
the relational surprise purifier but restores the richer multi-value target that
made associative binding acquire well. The current pair-row version is too
atomized.


## 2026-05-20 - Relational Aggregate and Context-Value Writes

Implemented two follow-up variants after relational pair rows:

- `relational_aggregate`: group surprising relation pairs by trigger feature,
  writing one richer value target per trigger instead of one row per pair.
- `--intrinsic-surprise-relation-value-mode`:
  - `residual`: write only the unexpected residual component of paired values.
  - `full`: use relational surprise to select/weight bindings, but write the
    full paired value channel.
  - `context`: use relational surprise to select trigger rows, but write the
    full local value context, closer to associative binding.

Validation: `.venv/bin/python -m pytest -q` passed with `50 passed`.

Key results:

| method | edited | expanded sentinel | correct->wrong | wrong->correct | preservation | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aggregate residual, scale `.50` | `1/20` | `12/25 -> 15/25` | `1` | `4` | `0.917` | `+0.299` | `0.241` |
| aggregate residual, scale `1.0` | `1/20` | `12/25 -> 16/25` | `0` | `4` | `1.000` | `+0.758` | `0.388` |
| aggregate residual, scale `2.5` | `2/20` | `12/25 -> 14/25` | `2` | `4` | `0.833` | `+0.470` | `1.408` |
| aggregate full, scale `.50` | `1/20` | `12/25 -> 15/25` | `1` | `4` | `0.917` | `+0.522` | `0.383` |
| aggregate full, scale `1.0` | `3/20` | `12/25 -> 15/25` | `2` | `5` | `0.833` | `+0.689` | `0.917` |
| aggregate full, scale `1.5` | `2/20` | `12/25 -> 14/25` | `2` | `4` | `0.833` | `+0.407` | `1.477` |
| aggregate full, scale `2.5` | `1/20` | `12/25 -> 14/25` | `2` | `4` | `0.833` | `-0.510` | `3.023` |
| bidirectional aggregate full, scale `1.0` | `6/20` | `12/25 -> 11/25` | `4` | `3` | `0.667` | `-1.751` | `4.427` |
| bidirectional aggregate full, scale `1.0`, stronger protection | `6/20` | `12/25 -> 9/25` | `5` | `2` | `0.583` | `-1.866` | `4.710` |
| context-value, scale `.20` | `1/20` | `12/25 -> 15/25` | `0` | `3` | `1.000` | `+1.009` | `0.071` |
| context-value, scale `.35` | `2/20` | `12/25 -> 16/25` | `0` | `4` | `1.000` | `+1.227` | `0.389` |
| context-value, scale `.50` | `2/20` | `12/25 -> 17/25` | `0` | `5` | `1.000` | `+1.478` | `0.633` |
| context-value, scale `1.0` | `3/20` | `12/25 -> 13/25` | `3` | `4` | `0.750` | `+1.040` | `1.883` |
| context-value, scale `2.0` | `4/20` | `12/25 -> 12/25` | `4` | `4` | `0.667` | `-0.014` | `3.892` |
| context-value, scale `5.0` | `4/20` | `12/25 -> 12/25` | `4` | `4` | `0.667` | `-0.797` | `5.443` |
| relational pair64, scale `10.0` | `6/20` | `12/25 -> 12/25` | `4` | `4` | `0.667` | `-1.001` | `4.502` |
| associative baseline, scale `.35` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `0.583` | `-1.763` | `7.124` |

Interpretation:

- Aggregating pair values did not produce the hoped-for jump to `10/20`.
- Writing full paired values improves acquisition modestly (`3/20`) while
  staying much safer than raw associative binding.
- Bidirectional aggregate full reaches `6/20`, but it reintroduces sentinel
  damage. Stronger protected penalties did not fix that; it preserved acquisition
  but made sentinel worse.
- Context-value mode is the cleanest safety result so far. Up through scale
  `.50`, it causes zero correct-to-wrong sentinel flips and improves expanded
  sentinel accuracy as high as `12/25 -> 17/25`. But acquisition remains only
  `2/20`.

Current diagnosis:

Relational surprise can identify safe update rows, but those rows do not carry
enough task-solving leverage. The high-acquisition behavior still appears to
come from broader answer/translation-mode directions. Relational selection
successfully avoids those directions, which preserves sentinel behavior, but
also leaves acquisition weak.

The next bottleneck is not row safety. It is missing task-relevant readout
structure: the write needs to create or modify a representation that the model
can use to answer translation questions, not merely bind native-channel
coactivations. Within the no-SAE, single-forward constraint, the most plausible
next move is a low-rank readout bridge derived from the lesson itself:

\[
  \Delta W
  =
  \Delta W_{\text{safe relational context}}
  +
  \alpha \Delta W_{\text{lesson readout actuator}}
\]

where the readout actuator is not trained on future questions, but is inferred
from the lesson's own source/translation alignments or final-token continuation
structure. The current pure-surprise update is safe but does not yet expose a
strong enough path from learned bindings to the model's existing answer
machinery.


## 2026-05-20 - All-Layer Constraint and Surprise-Field Normalization

Hard constraint added: the intervention should not be treated as a tuned
layer-20 actuator. It should be safe and meaningful when applied across all
decoder layers. Layer-20-only results are now diagnostic history, not an
acceptable target.

Implementation changes:

- Added `--intrinsic-span-readout-bridge`, which derives a small lesson-span
  readout write from spans such as `x=y` and `x -> y`.
- Added `--intrinsic-surprise-readout-specificity-power` support for negative
  powers, so surprise feature selection can prefer existing readout-coupled
  MLP features.
- Added `--intrinsic-surprise-target-row-norm-cap`, a layer-local target row
  norm cap to prevent late-layer coordinate scale from dominating all-layer
  writes.
- Added `--intrinsic-surprise-center-targets`, which subtracts each layer's
  weighted mean target before solving, intended to remove global posture
  components.

Validation: `.venv/bin/python -m pytest -q` passed with `53 passed`.

Key layer-20 and bridge results:

| method | edited | expanded sentinel | correct->wrong | wrong->correct | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| context-value `.35`, no bridge | `2/20` | `12/25 -> 16/25` | `0` | `4` | `+1.227` | `0.389` |
| context-value `.35` + span bridge `.50` | `2/20` | `12/25 -> 16/25` | `0` | `4` | `+1.246` | `0.375` |
| context-value `.35` + span bridge `1.0` | `2/20` | `12/25 -> 16/25` | `0` | `4` | `+1.229` | `0.392` |
| context-value `.35` + span bridge `25` | `2/20` | `12/25 -> 16/25` | `0` | `4` | `+1.297` | `0.428` |
| context-value `.35` + span bridge `100` | `3/20` | `12/25 -> 16/25` | `1` | `5` | `+1.541` | `0.543` |
| context-value `.35`, readout specificity `-1` | `2/20` | `12/25 -> 15/25` | `0` | `3` | `+1.138` | `0.446` |
| context-value `.35`, readout specificity `-2` | `1/20` | `12/25 -> 17/25` | `0` | `5` | `+1.136` | `0.528` |

The span readout bridge was too small at low scales and only reached `3/20`
when scaled to `100`, where it introduced sentinel churn. Negative
readout-specificity feature selection did not improve acquisition.

Associative-binding protection and clipping:

| method | edited | expanded sentinel | correct->wrong | wrong->correct | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| associative effective, input protection `20` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `-1.763` | `7.124` |
| associative effective, input protection `22` | `11/20` | `12/25 -> 10/25` | `5` | `3` | `-1.780` | `7.095` |
| associative effective, input protection `30` | `11/20` | `12/25 -> 9/25` | `5` | `2` | `-1.775` | `7.063` |
| associative effective, norm cap `30` | `11/20` | `12/25 -> 8/25` | `6` | `2` | `-2.176` | `6.441` |
| associative effective, norm cap `20` | `10/20` | `12/25 -> 8/25` | `7` | `3` | `-1.636` | `5.873` |

Generic protected negatives and simple trust-region clipping do not tame the
high-acquisition associative write. The damaging component is not caught by the
current weight-derived negative keys, and reducing norm does not remove it.

All-layer results:

| method | edited | expanded sentinel | correct->wrong | wrong->correct | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all 28 layers, context-value `.05` | `1/20` | `12/25 -> 15/25` | `1` | `4` | `+0.259` | `2.610` |
| all 28 layers, context-value `.10` | `4/20` | `12/25 -> 12/25` | `3` | `3` | `-0.459` | `4.733` |
| all 28 layers, context-value `.20` | `4/20` | `12/25 -> 10/25` | `6` | `4` | `-0.474` | `5.647` |
| all 28 layers, context-value `.10`, persistence `1` | `6/20` | `12/25 -> 12/25` | `3` | `3` | `-0.466` | `4.688` |
| all 28 layers, context-value `.20`, persistence `1` | `7/20` | `12/25 -> 10/25` | `7` | `5` | `-0.673` | `5.437` |
| all 28 layers, `.10`, persistence `1`, target cap `20` | `2/20` | `12/25 -> 13/25` | `3` | `4` | `-0.910` | `4.542` |
| all 28 layers, `.10`, persistence `1`, target cap `10` | `1/20` | `12/25 -> 11/25` | `4` | `3` | `-1.262` | `4.364` |
| all 28 layers, `.20`, persistence `1`, target cap `20` | `3/20` | `12/25 -> 12/25` | `5` | `5` | `-0.874` | `5.541` |
| all 28 layers, `.10`, persistence `1`, centered targets | `0/20` | `12/25 -> 8/25` | `5` | `1` | `-1.443` | `3.303` |
| all 28 layers, `.20`, persistence `1`, centered targets | `1/20` | `12/25 -> 5/25` | `7` | `0` | `-2.833` | `6.561` |

Interpretation:

- The current all-layer surprise write does not satisfy the hard constraint.
  It can reach `6-7/20` acquisition, but it causes sentinel churn and large
  before-correct margin drops.
- Persistence weighting is useful for acquisition in all-layer mode, raising
  context-value from `4/20` to `6-7/20`, but it does not solve safety.
- Target row-norm caps reduce acquisition and do not remove sentinel churn.
  Therefore late-layer coordinate magnitude is not the only harmful component.
- Centering targets is a strong negative result: it kills acquisition and makes
  sentinel behavior worse. The common target component carries behaviorally
  useful signal, but it is entangled with unsafe global posture movement.

Updated diagnosis:

The blocker under the all-layer constraint is not simply "which layer" or
"how much norm." The useful acquisition signal currently lives in a component
that behaves like a broad mode/readout shift when written everywhere. Safe
native-channel relational writes are too weak; high-acquisition associative
writes are unsafe; all-layer persistence recovers some acquisition but still
fails sentinel preservation.

The next mathematical object should be an all-layer local source rule that
separates common semantic source movement from common answer/posture movement
without using sentinels, generated probes, or a runtime router. A promising
direction is a layer-local predictive-coding decomposition:

\[
  r_{l,t}
  =
  \text{surprising persistent component}
  -
  \text{ordinary common-mode relaxation component}
\]

where "ordinary" must come from the cold weights and the single lesson forward,
not an extra null pass. The failed centering experiment says the common mode
cannot simply be deleted; it has to be decomposed into semantic common mode and
posture common mode.


## 2026-05-20 - All-Layer Protection Sweep

Follow-up after the all-layer failure: test whether broader cold-weight
protection in MLP key space can keep the useful all-layer context-value signal
while reducing sentinel churn.

Additional all-layer probes:

| method | edited | expanded sentinel | correct->wrong | wrong->correct | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline all 28, context `.10`, persistence `1`, input `256`, weight `20` | `6/20` | `12/25 -> 12/25` | `3` | `3` | `-0.466` | `4.688` |
| generic output projection rank `64` | `0/20` | `12/25 -> 12/25` | `5` | `5` | `-0.535` | `4.560` |
| generic output projection rank `64`, specificity `1` | `1/20` | `12/25 -> 12/25` | `4` | `4` | `-0.570` | `4.789` |
| generic output projection rank `128` | `1/20` | `12/25 -> 12/25` | `3` | `3` | `-0.379` | `4.126` |
| predictive residual `.20`, persistence `1` | `1/20` | `12/25 -> 11/25` | `3` | `2` | `-0.488` | `3.206` |
| predictive residual `.50`, persistence `1` | `1/20` | `12/25 -> 13/25` | `2` | `3` | `-0.668` | `5.059` |
| input protection `512`, weight `20` | `5/20` | `12/25 -> 14/25` | `3` | `5` | `-0.052` | `4.481` |
| input protection `1024`, weight `20` | `6/20` | `12/25 -> 17/25` | `1` | `6` | `+0.191` | `4.205` |
| input protection `512`, weight `40` | `7/20` | `12/25 -> 14/25` | `2` | `4` | `-0.279` | `4.544` |
| input protection `1024`, weight `40` | `7/20` | `12/25 -> 17/25` | `1` | `6` | `+0.398` | `3.847` |

The attempted `2048` negative-key run was aborted as impractically slow for
the current solver shape. It created duplicate Modal clients after a Codex
restart and did not produce an after-write row before being killed locally.

Interpretation:

- Cold-weight generic output projection is not the right nuisance estimate.
  It destroys acquisition while leaving sentinel churn.
- Predictive residual is safer but does not acquire. The task-relevant signal
  still appears to require the fuller context-value/common component.
- Broader input protection is the first all-layer improvement in the right
  direction. Increasing the protected key basis from `256` to `1024` and
  penalty from `20` to `40` gives the current best all-layer result:
  `7/20`, sentinel `12/25 -> 17/25`, only `1` sentinel correct-to-wrong flip,
  and before-correct margin drop reduced from `4.688` to `3.847`.

Updated working baseline:

```bash
python scripts/minilang_write.py \
  --model Qwen/Qwen3-1.7B \
  --device cuda \
  --dtype float16 \
  --lessons 6 \
  --lesson-examples 8 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
  --intrinsic-surprise-write \
  --intrinsic-surprise-target-mode relational_aggregate \
  --intrinsic-surprise-token-mode final_aligned \
  --intrinsic-surprise-relation-value-mode context \
  --intrinsic-surprise-target-scale 0.10 \
  --intrinsic-surprise-weight-mode exponential \
  --intrinsic-surprise-exp-temperature 2.0 \
  --intrinsic-surprise-exp-cap 20 \
  --intrinsic-surprise-persistence-power 1.0 \
  --intrinsic-surprise-output-penalty-rank 256 \
  --intrinsic-surprise-output-penalty-weight 10 \
  --intrinsic-surprise-input-penalty-features 1024 \
  --intrinsic-surprise-input-penalty-weight 40 \
  --intrinsic-surprise-input-penalty-usage-power 1
```

This still does not meet the hard goal. The headline sentinel accuracy improves,
but one previously-correct sentinel flips and the before-correct margin drop is
still large. However, it is the first all-layer setting that improves
acquisition and the sentinel headline at the same time, so the next research
move should refine the protected key basis rather than return to layer choice.

Next candidate:

Replace the current one-hot protected feature keys with a low-rank protected
key subspace built from the same cold-weight geometry. The current protected
keys are axis-aligned feature probes. The result above suggests that protecting
more of key space helps, but doing it by thousands of one-hot rows is slow and
crude. A low-rank basis could make protection denser, faster, and more
layer-general.

Follow-up implementation:

- Added `--intrinsic-surprise-input-penalty-mode {onehot,svd,hybrid}`.
- `svd` uses top right-singular vectors of the current MLP down-projection as
  dense protected key directions.
- `hybrid` concatenates SVD protected directions with the previous one-hot
  high-score feature probes.
- Changed the TSOC ridge solve from ordinary `pinv` to Hermitian `pinv` via
  `solve_ridge_system(...)`, because dense SVD protected keys produced
  ill-conditioned symmetric systems where default SVD pseudoinverse failed.

Validation after implementation: `.venv/bin/python -m pytest -q` passed with
`55 passed`.

Dense-basis result:

- The first `svd` run failed with `torch._C._LinAlgError` from the old default
  pseudoinverse.
- After switching to Hermitian `pinv`, the `svd256w40` all-layer run remained
  too slow to be practical and was killed locally.

Interpretation:

Dense protected key bases are still mathematically plausible, but the current
row-space solver is the wrong implementation path for them. The practical
baseline remains one-hot input protection `1024`, weight `40`. If we revisit
dense protection, use a smaller dual/kernel solve or pre-orthogonalize protected
keys so the solve is not dominated by ill-conditioned dense negative rows.


## 2026-05-20 - All-Layer Continual Diagnostic And Last-Token Check

Implemented `scripts/minilang_intrinsic_continual.py`, a continual-learning
diagnostic for the current lesson-only intrinsic-surprise write. Unlike the
older triangle script, it writes from the lessons themselves using the
single-forward intrinsic writer rather than answer traces. It evaluates:

- teacher-filtered heldout tasks;
- the triangular retention matrix after each write;
- expanded sentinel flip and margin metrics after each write.

Also extended `run_intrinsic_surprise_writes(...)` so later tasks can include
previously selected write keys as protected negatives. This is not a runtime
gate; it is a closed-form protection term in the solve.

Validation: `.venv/bin/python -m pytest -q tests/test_minilang.py
tests/test_tsoc.py` passed with `16 passed`.

Two-task all-layer continual diagnostic, using the current working all-layer
baseline (`relational_aggregate`, context value mode, scale `.10`, exponential
surprise, persistence `1`, output penalty `256/10`, input penalty `1024/40`):

| setting | task 0 after task 0 | task 0 after task 1 | task 1 after task 1 | sentinel after task 0 | sentinel after task 1 |
| --- | ---: | ---: | ---: | --- | --- |
| no old-task protection | `2/8` | `0/8` | `2/8` | `12/25 -> 11/25`, `7` c2w | `12/25 -> 9/25`, `8` c2w |
| old selected keys as negatives, max `256` rows/layer | `2/8` | `1/8` | `0/8` | `12/25 -> 11/25`, `7` c2w | `12/25 -> 12/25`, `6` c2w |

Detailed interpretation:

- The ungated all-layer write fails on all three axes: weak acquisition,
  sentinel damage, and catastrophic retention.
- Old-task negative protection improves the specific retention/sentinel
  interference mode, but mostly by suppressing the second write. Task-1
  acquisition falls to `0/8`.
- Therefore the sequence failure is upstream of continual protection. The
  selected surprise keys/targets are still too generic. When protected, the
  solve cannot find a clean task-specific acquisition direction; when
  unprotected, it writes into shared translation/readout/posture subspace and
  erases task 0.

This is important for the theory: continual learning is not blocked only by a
missing old-task protection term. The current surprise object itself is not
purified enough.

Last-token-only check:

The user hypothesis was that the lesson-final token may carry a compressed
"what changed in this context" state. I tested all-layer lesson-only writes
using `--intrinsic-surprise-token-mode last` rather than `final_aligned`.

| scale | edited | context | sentinel | correct->wrong | wrong->correct | margin delta | before-correct margin drop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `.10` | `6/39` | `21/39` | `12/25 -> 16/25` | `1` | `5` | `+1.899` | `1.208` |
| `.30` | `4/39` | `21/39` | `12/25 -> 17/25` | `0` | `5` | `+2.018` | `3.465` |

The last-token runs use far fewer rows per layer (`~7.5` selected rows on
average versus `~118` for final-aligned continual runs). They are much safer by
sentinel headline and flip metrics, especially `.30` with `0` correct-to-wrong
flips, but they do not acquire the language at all (`baseline 6/39`; edited
`6/39` or worse).

Interpretation:

- The final lesson token alone is not a sufficient consolidation state for this
  mini-language task, at least with the current relational-aggregate target.
- It may be a useful safety/protection signal, but not the acquisition signal.
- The useful acquisition signal appears spread across content/use positions,
  while the safest compressed summary is too weak to teach role-binding and
  lexical mappings.

Current high-level status:

We do not yet have a technique that both learns one context and provably avoids
brain damage. We have a real all-layer learning signal, but its useful component
is entangled with broad mode/readout movement. The next mathematical step should
not be more scale, more examples, or a runtime object gate. It should be a
single-pass, weight-intrinsic definition of object-level surprise that separates
semantic latent-object mismatch from generic task/readout/posture activity.


## 2026-05-20 - WICR Same-Token v1

Implemented the first slice of GPT-5.5 Pro's WICR proposal:
`compatibility_residual` intrinsic surprise mode.

Files changed:

- `caic/intrinsic_surprise.py`
  - Added `mlp_activation_normals(...)` for selected SwiGLU channels.
  - Added `select_intrinsic_compatibility_residual_write(...)`.
  - Same-token WICR scores active feature pairs by whether the source feature's
    down-value is compatible with the target feature's activation normal under
    the current weights.
- `scripts/minilang_write.py`
  - Added `--intrinsic-surprise-target-mode compatibility_residual`.
  - Added WICR flags:
    - `--wicr-compatibility-threshold`
    - `--wicr-compatibility-temperature`
    - `--wicr-posture-pcs`
    - `--wicr-target-vector-mode normal|value`
- `scripts/minilang_intrinsic_continual.py`
  - Added matching parser flags so WICR can be used in the continual runner.
- `tests/test_intrinsic_surprise.py`
  - Added tests for activation normals and WICR selection/value-target mode.

Validation: `.venv/bin/python -m pytest -q` passed with `59 passed`.

Implementation note:

The first WICR implementation is **same-token only**. It does not yet implement
attention-edge compatibility. For each selected token, it:

1. Selects top MLP features by weight-relative feature surprise.
2. Computes activation normals for selected target features.
3. Scores feature pairs by co-instantiation times incompatibility between
   source down-value and target activation normal.
4. Aggregates surprising pairs into one key/target row per selected token.
5. Solves the same protected closed-form MLP-down update as the other intrinsic
   writers.

Results, all-layer, final-aligned 8 tokens, expanded sentinel suite:

| method | edited | context | sentinel | c2w | w2c | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WICR normal target, scale `.10` | `3/39` | `21/39` | `12/25 -> 13/25` | `1` | `2` | `+0.645` | `0.218` |
| WICR normal target, scale `.50` | `4/39` | `21/39` | `12/25 -> 9/25` | `3` | `0` | `-1.690` | `4.644` |
| WICR value target, scale `.03` | `4/39` | `21/39` | `12/25 -> 13/25` | `0` | `1` | `+0.164` | `0.249` |
| WICR value target, scale `.10` | `4/39` | `21/39` | `12/25 -> 15/25` | `1` | `4` | `+0.878` | `1.411` |
| WICR value target, scale `.10`, no output/posture quotient | `6/39` | `21/39` | `12/25 -> 16/25` | `1` | `5` | `+1.743` | `1.492` |

Interpretation:

- Same-token WICR v1 is much safer than the old high-acquisition
  relational/context-value write by margin metrics. The `.03` value-target run
  had zero sentinel correct-to-wrong flips and low before-correct margin drop.
- But it does not acquire the mini-language. The best WICR result is baseline
  level (`6/39`) only after removing the output/posture quotient; the safer
  projected runs are below baseline (`3-4/39`).
- Removing output/posture projection did **not** recover acquisition, so the
  failure is not mainly over-projection. Same-token compatibility selection is
  safer but not behaviorally sufficient.

Updated diagnosis:

WICR's compatibility coordinate is promising as a safety filter, but same-token
MLP feature compatibility is too local/abstract for this task. Mini-language
role binding and translation behavior likely require cross-token attention-edge
compatibility or a different target construction that reaches existing
readout/composition circuits without broad posture movement.

Next step if continuing WICR:

Implement attention-edge WICR, not more same-token scale. The key missing piece
is source-token feature to target-token feature compatibility through the frozen
attention value/output path:

\[
F_{l,t,s}v_{l,i}
\quad\text{versus}\quad
n_{l,t,j}.
\]

The same-token result already tells us the target should probably stay in value
mode for readout leverage, but pair selection needs cross-token structure to
capture subject/object/tense binding.


## 2026-05-20 - Attention-Edge WICR Smoke Test

Implemented the cross-token attention-edge slice of WICR.

Files changed:

- `caic/intrinsic_surprise.py`
  - Added attention-module helpers for Qwen-style attention projections.
  - Added `attention_flow_values(...)`, which maps selected MLP feature
    down-values through the frozen attention value/output path.
  - Extended `select_intrinsic_compatibility_residual_write(...)` with
    attention edges, optional edge-only mode, and attention flow modes.
- `scripts/minilang_write.py`
  - Added `--attn-implementation`.
  - Added attention capture for WICR when `--wicr-attention-edges > 0`.
  - Added WICR edge flags:
    - `--wicr-attention-edges`
    - `--wicr-attention-flow-mode`
    - `--wicr-no-same-token-edges`
- `scripts/minilang_intrinsic_continual.py`
  - Added the matching WICR edge flags.
- `caic/modeling.py`
  - Added optional `attn_implementation` passthrough when loading models.
- `tests/test_intrinsic_surprise.py`
  - Added tests for attention value flow and attention-edge WICR selection.

Implementation note:

Qwen's default SDPA attention path does not return attention maps when
`output_attentions=True`. Real attention-edge WICR requires
`--attn-implementation eager`. The first attention-edge run without eager
attention should be treated as a same-token/no-attention control.

Results, all-layer, value target, final-aligned 8 tokens, expanded sentinel
suite:

| method | edited | context | sentinel | c2w | w2c | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-token WICR, scale `.03` | `4/39` | `21/39` | `12/25 -> 13/25` | `0` | `1` | `+0.164` | `0.249` |
| Attention flag without eager, scale `.03` | `4/39` | `21/39` | `12/25 -> 13/25` | `0` | `1` | `+0.175` | `0.260` |
| Attention edges + same-token, eager, scale `.03` | `4/39` | `20/39` | `12/25 -> 9/25` | `6` | `3` | `-1.548` | `5.092` |
| Attention edges only, eager, scale `.01` | `5/39` | `20/39` | `12/25 -> 10/25` | `2` | `0` | `-0.566` | `1.157` |

Selected-row and update-size diagnostics:

| method | update rows | selected rows mean/min/max | target Fro mean/max | update Fro mean/max |
| --- | ---: | ---: | ---: | ---: |
| Attention edges + same-token, eager | `168` | `23.7 / 15 / 33` | `58.034 / 478.511` | `1.006 / 6.044` |
| Attention edges only, eager | `168` | `18.39 / 10 / 25` | `17.625 / 123.292` | `0.351 / 6.584` |

Interpretation:

- Real attention-edge WICR increases row count and update magnitude, but does
  not improve acquisition over same-token WICR.
- Adding edge structure reintroduced sentinel damage. The same+edge run caused
  `6` correct-to-wrong sentinel flips and a `5.092` before-correct margin drop.
- Edge-only WICR is less damaging, but still below the safety bar and still not
  acquiring (`5/39`, below baseline `6/39`).
- Therefore attention-edge WICR v1 is not the missing mathematical tool. It is
  a useful falsifier: cross-token compatibility through frozen attention maps,
  as currently constructed, selects broader harmful movement without finding the
  behaviorally useful readout/composition component.

Updated recommendation:

Do not spend more time scaling this WICR implementation. The attention-edge
smoke test was the high-value check, and it failed in the informative way: same
token is safe but too weak; cross-token is stronger but unsafe and still not
useful. The next move should be to go back to GPT-5.5 Pro with this new
negative result and ask for a revised one-pass, weight-intrinsic surprise
coordinate that explains why compatibility-edge selection did not isolate the
causal acquisition component.


## 2026-05-20 - CORI v1: Conditional Object-Relation Innovation

Implemented the first minimal CORI path suggested by GPT-5.5 Pro:
`conditional_relation_innovation`.

The implementation is deliberately a new intrinsic target mode, not a runtime
router. It still uses one lesson forward pass and a closed-form MLP-down write.

Files changed:

- `caic/intrinsic_surprise.py`
  - Added `relation_edge_matrix(...)`.
  - Added `default_relation_prior(...)`.
  - Added `select_intrinsic_conditional_relation_innovation_write(...)`.
  - CORI builds a feature-relation field over selected native MLP channels,
    conditions out empirical feature marginals and a weight-induced
    compatibility prior, then writes dense relation-state keys from the top SVD
    innovation modes.
- `scripts/minilang_write.py`
  - Added `--intrinsic-surprise-target-mode conditional_relation_innovation`.
  - Added CORI flags:
    - `--cori-feature-top-k`
    - `--cori-relation-rank`
    - `--cori-beta`
    - `--cori-edge-top-k`
    - `--cori-edge-attention-scale`
    - `--cori-sinkhorn-steps`
    - `--cori-target-mode`
- `scripts/minilang_intrinsic_continual.py`
  - Added matching CORI flags so the mode can be used in the continual runner.
- `tests/test_intrinsic_surprise.py`
  - Added a CORI selector test.

Validation: `.venv/bin/python -m pytest -q` passed with `62 passed`.

Implementation note:

This first CORI version reuses the existing protected closed-form update path.
It does not yet add a separate Schur-projected dual solve backend. The
important tested variable here is the surprise coordinate and row construction:
dense relation keys plus marginal-conditioned relation innovation. The existing
input/output protection remains the same as the current all-layer baseline.

I also fixed a sign issue after the first calibrated run. SVD relation
components have arbitrary paired signs, while the dense CORI key is sign
invariant because it multiplies left and right component activations. Each CORI
target is now oriented toward the lesson's own down-projection value for the
dense relation key.

Results, all-layer, no attention edges, p=`128`, rank=`16`, beta=`3`, expanded
sentinel suite:

| method | edited | context | sentinel | c2w | w2c | margin delta | before-correct margin drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CORI `svd_value`, scale `.10` | `6/39` | `21/39` | `12/25 -> 12/25` | `0` | `0` | `-0.000` | `0.006` |
| CORI `svd_value`, scale `3.0`, pre-orientation | `2/39` | `21/39` | `12/25 -> 12/25` | `0` | `0` | `+0.296` | `0.099` |
| CORI `svd_value`, scale `3.0`, oriented | `3/39` | `21/39` | `12/25 -> 11/25` | `1` | `0` | `+0.218` | `0.404` |
| CORI `innovation_value`, scale `3.0`, oriented | `3/39` | `21/39` | `12/25 -> 13/25` | `0` | `1` | `+0.274` | `0.044` |

Update-size diagnostics:

| method | selected rows/layer | target Fro mean/max | update Fro mean/max |
| --- | ---: | ---: | ---: |
| CORI `svd_value`, scale `.10` | `16` | `0.043 / 0.159` | `0.008 / 0.058` |
| CORI `svd_value`, scale `3.0`, pre-orientation | `16` | `1.285 / 4.756` | `0.248 / 1.742` |
| CORI `svd_value`, scale `3.0`, oriented | `16` | `1.296 / 4.756` | `0.248 / 1.765` |
| CORI `innovation_value`, scale `3.0`, oriented | `16` | `0.807 / 2.726` | `0.124 / 0.600` |

Interpretation:

- CORI v1 is extremely safe at its natural scale, but it is behaviorally a
  no-op. The natural probability-normalized relation targets are far smaller
  than WICR and the current relational aggregate baseline.
- Scaling CORI into the WICR-size update regime does not recover acquisition.
  It mostly worsens the mini-language score while leaving sentinel metrics
  good or mildly mixed.
- The sign-orientation fix was theoretically necessary, but it did not change
  the conclusion. The target direction still is not the behaviorally useful
  readout/composition component.
- `innovation_value` is safer than oriented `svd_value` at the calibrated
  scale, but still drops acquisition from baseline `6/39` to `3/39`.

Current CORI diagnosis:

CORI's "same marginals + weight prior" relation residual is a strong safety
filter, but in this first implementation it filters out the acquisition signal
along with the task/posture junk. The dense relation keys are not enough by
themselves; the target being fitted is still not the causal component that
improves translation behavior.

This means CORI v1 should not be taken to attention-edge or continual
diagnostics yet. The single-task acquisition bar is not met. The next research
question is narrower:

What target should a purified relation innovation key write?

The current target choices write residual value mixtures derived from the
relation field itself. That is safer, but apparently not useful. The earlier
all-layer relational aggregate acquired because it wrote a much fuller
context-value/readout component, but that component was contaminated. The open
problem is to extract the useful readout/composition actuator conditioned on
the CORI relation keys without reintroducing broad posture movement.


## 2026-05-20 - Updated GPT-5.5 Pro Request After CORI

Updated `PRO_RESEARCH_HANDOFF.md` and the Downloads request document with the
post-WICR/post-CORI research state.

The new request frames the blocker more sharply:

- high-acquisition writes exist, but are contaminated with broad
  readout/posture/mode movement;
- WICR and CORI show that safer one-pass surprise coordinates are possible;
- CORI at natural scale is essentially behaviorally inert:
  - baseline `6/39`;
  - context `21/39`;
  - edited `6/39`;
  - predictions changed `0/39`;
  - context-only opportunities captured `0/19`;
  - edited score movement only about `0.37%` of the context-induced score
    movement;
  - sentinel `12/25 -> 12/25` with `0` correct-to-wrong flips;
- scaling CORI does not recover acquisition.

The request asks GPT-5.5 Pro not for another router, scale sweep, or safer key
selector, but for the missing mathematical target:

> Given a purified one-pass surprise/object/relation key, what closed-form
> target should it write so that the update contains the causal
> readout/composition actuator without copying answer posture?

The request explicitly asks for equations, tensor shapes, a closed-form solve,
sequential protection math, first implementation steps, and falsification
criteria.


## 2026-05-20 - STAR v1: Schur-Transport Actuator Residual

Implemented the next GPT-5.5 Pro proposal, `schur_transport_actuator`.

The intended target is no longer CORI's local relation-value target. STAR uses
CORI to get purified dense relation keys, then writes the same-pass future
residual computation attributable to those keys after Schur-residualizing
position/finality, key magnitude, ordinary low-surprise key directions, and
readout/posture projections. The posture component of the same selected key is
returned as a zero-target negative key in the existing closed-form solve.

Files changed:

- `caic/intrinsic_surprise.py`
  - Added optional `negative_keys` and `diagnostics` to
    `IntrinsicSurpriseSelection`.
  - Added Schur row residualization helpers.
  - Added same-pass future transport capsule construction.
  - Added `select_intrinsic_schur_transport_actuator_write(...)`.
- `scripts/minilang_write.py`
  - Added `--intrinsic-surprise-target-mode schur_transport_actuator`.
  - Added STAR flags for object-summary gain, future horizon/decay, Schur
    ridge, map ridge, value projection, and posture-negative scale.
  - Threaded all captured MLP-input states into STAR so each layer can use
    future same-pass states.
  - Merged selector-provided posture negative keys into the existing protected
    ridge/metric solve.
- `tests/test_intrinsic_surprise.py`
  - Added a STAR shape/target test.

Validation:

- `.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q`
  - `22 passed`
- `.venv/bin/python -m pytest -q`
  - `63 passed`

Runs used the same eval set for comparability:

- baseline: `5/38`
- full context: `20/38`
- expanded sentinel before: `12/25`

All runs were all-layer MLP-down writes, no attention edges, p=`128`,
rank=`16`, beta=`3`, final-aligned mode, exponential weights, persistence
power `1`, output penalty rank/weight `256/10`, input protection
features/weight `1024/40`.

| run | target scale | edited | sentinel | c2w | w2c | before-correct margin drop | changed eval predictions | captured context-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/star_alllayer_scale010_qwen17_auto` | `.10` | `5/38` | `12/25 -> 12/25` | `0` | `0` | `0.025` | `0` | `0/19` |
| `runs/star_alllayer_scale100_qwen17_auto` | `1.0` | `3/38` | `12/25 -> 12/25` | `0` | `0` | `0.157` | `2` | `0/19` |
| `runs/star_alllayer_scale300_qwen17_auto` | `3.0` | `3/38` | `12/25 -> 10/25` | `2` | `0` | `0.483` | `2` | `0/19` |

Update diagnostics:

| run | update Fro mean/median/max | target Fro mean/median/max | STAR explained-ratio mean/median/max |
| --- | ---: | ---: | ---: |
| scale `.10` | `0.00217 / 0.00049 / 0.0997` | `2.95 / 1.61 / 37.09` | `0.00165 / 0.00141 / 0.00490` |
| scale `1.0` | `0.0213 / 0.00511 / 1.001` | `29.23 / 15.26 / 416.28` | `0.162 / 0.138 / 0.490` |
| scale `3.0` | `0.0660 / 0.0153 / 3.040` | `91.03 / 45.14 / 1188.95` | `1.484 / 1.308 / 5.229` |

Interpretation:

- STAR v1 is safe at scale `.10`, but it is behaviorally a no-op:
  no predictions changed and zero context-only opportunities were captured.
- Scaling STAR makes it behaviorally active in the wrong direction. At scale
  `1.0` and `3.0`, translation accuracy drops from `5/38` to `3/38`.
- Scale `1.0` is sentinel-safe by c2w, but still weakens previously-correct
  sentinel margins and loses task accuracy.
- Scale `3.0` reintroduces sentinel damage: `2` correct-to-wrong flips and
  a larger before-correct margin drop.
- The future-transport target as implemented does not recover the useful
  readout/composition actuator. It either remains too small or writes a
  non-causal/harmful component.

Current diagnosis after STAR:

The proposal correctly identified target construction as the blocker, but this
STAR target is not the missing target. Schur-residualized future residual
movement is still not aligned with the behaviorally useful actuator. The most
important empirical point is that scale `1.0` moves predictions but captures
`0/19` context-only opportunities; the direction is wrong, not merely too weak.

Next useful ablations:

1. Shuffle STAR future capsules and keys. If shuffled STAR behaves similarly,
   the current future target is just broad mode movement.
2. Replace future residual target with future **score/logit-direction-free
   readout bridge** only if it can be derived from weights and same-pass
   states without labels.
3. Try a target-orientation rule that aligns the STAR target to the lesson's
   own **downstream self-consistency** rather than raw future residual deltas.
4. Add a state-movement diagnostic: compare edited answer-option score deltas
   to context score deltas. Current STAR captured no context-only items even
   at unsafe scale, so the next target must prove it points in the
   context-teacher direction before another continual run.


## 2026-05-20 - STAR falsifiers and relational safety frontier

Added a score-space alignment diagnostic:

- `scripts/analyze_eval_alignment.py`

For each run, it compares answer-option score deltas:

\[
\Delta_{\text{context}} = s_{\text{context}} - s_{\text{baseline}},
\qquad
\Delta_{\text{edit}} = s_{\text{edited}} - s_{\text{baseline}}.
\]

It reports centered global cosine, projection ratio, correct-answer score
gain, captured context-only opportunities, and expanded sentinel drift. This
is now a better diagnostic than edited accuracy alone because it tells whether
the write is moving in the same direction as the full-context teacher.

### STAR alignment result

Reanalyzed the STAR runs:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STAR scale `.10` | `5/38` | `20/38` | `5/38` | `0` | `0` | `0` | `0/19` | `0.035` | `0.000` | `+0.001` | `0` | `0.025` |
| STAR scale `1.0` | `5/38` | `20/38` | `3/38` | `2` | `0` | `2` | `0/19` | `-0.300` | `-0.004` | `-0.008` | `0` | `0.157` |
| STAR scale `3.0` | `5/38` | `20/38` | `3/38` | `2` | `0` | `2` | `0/19` | `-0.287` | `-0.013` | `-0.017` | `2` | `0.483` |

Interpretation:

- Safe STAR is not acquiring because it is effectively a no-op.
- Active STAR is not merely under-scaled; its score movement is anti-aligned
  with the context teacher.
- STAR's same-pass future residual target is therefore not the missing target.

### STAR shuffle ablations

Implemented deterministic STAR ablation flags:

- `--star-shuffle-future-targets`
- `--star-shuffle-keys`

These roll target/key rows by one position after STAR row selection, preserving
row distributions while breaking the intended pairing.

Scale `1.0` results:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real STAR | `5/38` | `20/38` | `3/38` | `2` | `0` | `2` | `0/19` | `-0.300` | `-0.004` | `-0.008` | `0` | `0.157` |
| shuffled future targets | `5/38` | `20/38` | `5/38` | `0` | `0` | `0` | `0/19` | `-0.115` | `-0.002` | `-0.004` | `0` | `0.027` |
| shuffled keys | `5/38` | `20/38` | `5/38` | `1` | `0` | `0` | `0/19` | `0.035` | `0.001` | `-0.007` | `0` | `0.081` |

Additional update statistics:

| run | update Fro mean/median/max | target Fro mean/median/max | STAR explained-ratio mean/median/max |
| --- | ---: | ---: | ---: |
| real STAR scale `1.0` | `0.0213 / 0.00510 / 1.001` | `29.23 / 15.26 / 416.28` | `0.162 / 0.138 / 0.490` |
| shuffled future scale `1.0` | `0.0341 / 0.00583 / 1.290` | `35.75 / 20.02 / 377.36` | `0.181 / 0.163 / 0.506` |

Interpretation:

- Shuffled future targets had equal or larger update norms but became a no-op.
- Therefore STAR's real key-target pairing does matter, but it matters in the
  wrong behavioral direction.
- The failure is not just "future targets too small." It is target orientation
  and causal relevance.

### Relational aggregate still has the correct direction

Compared STAR to the best relational aggregate family using the same alignment
diagnostic.

Best prior all-layer relational aggregate run:

- all 28 layers;
- `relational_aggregate`;
- context-value target;
- target scale `.10`;
- exponential surprise weighting, temperature `2.0`, cap `20`;
- persistence power `1`;
- input penalty `1024/40`;
- output penalty `256/10`;
- eval from the teacher-filtered 20-item set.

Result:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relational aggregate `.10` | `0/20` | `20/20` | `7/20` | `11` | `7` | `0` | `7/20` | `0.589` | `0.179` | `+0.222` | `1` | `3.847` |

Interpretation:

- Unlike STAR/CORI/WICR, relational aggregate clearly moves in the
  context-teacher direction.
- The current useful target family is the unsafe one.
- The next target should probably be a purified version of this
  context-value/readout actuator, not a replacement with future residual
  movement.

### Stronger global output protection does not solve safety

Tested whether the unsafe relational aggregate component is mostly a broad
LM-head/readout basis artifact.

Runs:

1. Stronger output metric penalty:
   - output penalty rank/weight `1024/40`;
   - no LM-head target projection;
   - target scale `.10`.
2. Stronger output penalty plus LM-head target projection:
   - LM-head generic target projection rank `256`;
   - output penalty rank/weight `1024/40`;
   - target scale `.10`.
3. Half scale with strong output penalty:
   - output penalty rank/weight `1024/40`;
   - target scale `.05`.

Results:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relational aggregate `.10`, output `256/10` | `0/20` | `20/20` | `7/20` | `11` | `7` | `0` | `7/20` | `0.589` | `0.179` | `+0.222` | `1` | `3.847` |
| relational aggregate `.10`, output `1024/40` | `0/20` | `20/20` | `4/20` | `13` | `4` | `0` | `4/20` | `0.613` | `0.185` | `+0.199` | `1` | `2.355` |
| relational aggregate `.10`, LM projection `256` + output `1024/40` | `0/20` | `20/20` | `3/20` | `13` | `3` | `0` | `3/20` | `0.615` | `0.187` | `+0.193` | `1` | `2.558` |
| relational aggregate `.05`, output `1024/40` | `0/20` | `20/20` | `0/20` | `2` | `0` | `0` | `0/20` | `0.596` | `0.121` | `+0.137` | `0` | `1.126` |

Sentinel failure details for the useful relational runs:

- The same grammar sentinel item flips correct-to-wrong in the `.10`
  relational family even with output `1024/40`.
- Stronger output protection reduces broad margin damage but does not remove
  the discrete c2w failure at useful acquisition scale.
- LM-head target projection reduces acquisition further and still does not
  remove the c2w failure.
- Half scale removes c2w but also removes discrete acquisition, despite
  retaining positive context-aligned score movement.

Interpretation:

- The damaging direction is not just a global LM-head/readout basis.
- The useful and harmful parts are entangled at row/target/component level.
- Global output penalties and global target projection are too blunt.
- The next mathematical object should probably operate on the relational
  aggregate target family and decompose each row/component into:
  - context-aligned useful actuator;
  - sentinel/posture-sensitive component;
  - generic answer/mode component.

### Current diagnosis after exploration

The missing object is more specific than "target construction":

> We need a one-pass, closed-form, row/component-level purification of the
> relational/context-value actuator that preserves its context-teacher
> score-space alignment while removing the sentinel-sensitive component.

STAR's future-residual target is falsified for now because it is safe but
misoriented. CORI is safe but inert. WICR is safe/weak or strong/unsafe. The
only current target with real context alignment is relational aggregate
context-value, and it fails because the same family also carries broad
capability drift.

The next request to GPT-5.5 Pro should therefore ask for:

1. A closed-form purification of the relational aggregate target itself, not
   another unrelated key selector.
2. A weight-only or same-pass-only observability metric that predicts sentinel
   sensitivity more locally than LM-head PCs.
3. A way to decompose row targets into useful semantic actuator vs generic
   mode/readout component without sentinel examples, null prompts, labels, or
   runtime routing.
4. A sequential protection metric that protects old key-to-output
   transformations without suppressing all new acquisition.


## 2026-05-20 - KARP v1: Key-Attributable Readout Purification

GPT-5.5 Pro proposed KARP: Key-Attributable Readout Purification.

The core idea is to stop asking whether a target vector is output-sensitive in
isolation. Instead ask whether a **key x target atom** is generic:

\[
\text{risk}(p, q)
\approx
\frac{p^\top C_G p}{p^\top C_S p+\epsilon}
\cdot
\frac{q^\top F_G q}{q^\top F_S q+\epsilon}.
\]

This allows output/readout-sensitive directions when they are paired with a
specific relational key, and shrinks them when paired with generic keys.

### Implementation

Added KARP as a purifier around the existing useful target family:

- `relational_aggregate`;
- context-value target;
- current protected metric solve first;
- then KARP purifies the candidate update.

Files changed:

- `caic/intrinsic_surprise.py`
  - Added `KarpPurificationResult`.
  - Added KARP update purifier:
    `karp_purify_update(...)`.
  - Added compact risk-basis helpers.
- `scripts/minilang_write.py`
  - Added `--intrinsic-target-purifier none|karp`.
  - Added KARP hyperparameters:
    - `--karp-key-rank`;
    - `--karp-value-rank`;
    - `--karp-low-surprise-quantile`;
    - `--karp-eta-cross`;
    - `--karp-eta-key`;
    - `--karp-eta-value`;
    - `--karp-risk-ratio-cap`.
  - KARP diagnostics are logged per layer.
- `tests/test_intrinsic_surprise.py`
  - Added a synthetic KARP test showing that the same readout direction is
    preserved more when paired with a specific key than when paired with a
    generic key.

Validation:

- `.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q`
  - `23 passed`
- `.venv/bin/python -m pytest -q`
  - `64 passed`

### Implementation correction

The first KARP implementation used only the top generalized generic-risk
directions. It was too diagnostic and barely touched the actual candidate
update:

| run | edited | c2w | before-correct drop | mean removed update ratio |
| --- | ---: | ---: | ---: | ---: |
| risk-basis KARP, rank `32`, eta-cross `10` | `3/20` | `1` | `2.719` | `0.014` |

This showed that the risky basis did not cover the relational candidate map.
The useful update mostly lived outside the shrinkable block.

Fixed by switching to a mixed signal+risk basis:

- half signal-side basis from selected relational keys/targets;
- half generic-risk basis from low-surprise same-pass rows, protected input
  keys, and output/readout basis;
- risk ratio is then scored inside that mixed basis.

This makes KARP act on the actual candidate map rather than only measuring
generic side directions.

### Main KARP results

All runs below:

- all 28 layers;
- `relational_aggregate`;
- context-value target;
- target scale `.10`;
- exponential surprise weighting, temperature `2.0`, cap `20`;
- persistence power `1`;
- input penalty `1024/40`;
- output penalty `1024/40`;
- KARP rank `32/32` when enabled;
- teacher-filtered 20-question eval.

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `1024/40` | `0/20` | `20/20` | `4/20` | `13` | `4` | `0` | `4/20` | `0.613` | `0.185` | `+0.199` | `1` | `2.355` |
| relational `.05`, output `1024/40` | `0/20` | `20/20` | `0/20` | `2` | `0` | `0` | `0/20` | `0.596` | `0.121` | `+0.137` | `0` | `1.126` |
| mixed KARP, eta-cross `2.0`, eta-key/value `.15/.05` | `0/20` | `20/20` | `0/20` | `5` | `0` | `0` | `0/20` | `0.426` | `0.100` | `+0.159` | `0` | `1.859` |
| mixed KARP, eta-cross `.50`, eta-key/value `.05/.02` | `0/20` | `20/20` | `3/20` | `7` | `3` | `0` | `3/20` | `0.518` | `0.130` | `+0.200` | `0` | `2.215` |
| mixed KARP, eta-cross `.25`, eta-key/value `.02/.01` | `0/20` | `20/20` | `3/20` | `8` | `3` | `0` | `3/20` | `0.536` | `0.130` | `+0.173` | `1` | `2.418` |
| mixed KARP product-only, eta-cross `.50`, eta-key/value `0/0` | `0/20` | `20/20` | `1/20` | `7` | `1` | `0` | `1/20` | `0.548` | `0.135` | `+0.170` | `1` | `2.034` |

### KARP diagnostics

Mean diagnostics:

| run | removed update ratio | kept coeff energy ratio | cross-risk before | cross-risk after | atoms shrunk >90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| risk-basis KARP `10` | `0.014` | `0.012` | `9065.15` | `1013.18` | `1024` |
| mixed KARP `2.0` | `0.318` | `0.434` | `46.57` | `0.209` | `800` |
| mixed KARP `.50` | `0.241` | `0.608` | `45.79` | `0.392` | `618` |
| mixed KARP `.25` | `0.200` | `0.700` | `44.41` | `0.526` | `548` |
| product-only `.50` | `0.221` | `0.658` | `43.40` | `0.402` | `595` |

Interpretation:

- Mixed-basis KARP successfully acts on the candidate relational map.
- It produces the first useful KARP frontier point:
  - `3/20`;
  - zero sentinel c2w;
  - positive context alignment.
- It is not solved:
  - before-correct sentinel margin drop remains high (`2.215`);
  - acquisition is below the unsafe `4/20` strong-output relational run and
    below the older unsafe `7/20` run.
- Product-only KARP is worse:
  - acquisition drops to `1/20`;
  - the c2w flip returns.
  - The small additive key/value rails are contributing useful safety.

### Current KARP diagnosis

KARP is directionally useful but still too blunt.

It does something qualitatively different from global output projection:

- global output `1024/40` at scale `.10`:
  - `4/20`, c2w `1`;
- KARP eta `.50`:
  - `3/20`, c2w `0`.

So KARP improves the discrete c2w frontier relative to the strong-output
baseline. However, it does not yet solve margin safety, and it gives up one
acquired item.

The likely issue is the current generic-risk metric:

- generic output basis is still too close to the useful semantic readout
  component;
- low-surprise same-pass rows may not identify the particular sentinel-risk
  direction sharply enough;
- the value-risk side needs a more local observability metric than LM-head PCs
  plus low-surprise MLP outputs;
- KARP needs to rank atoms by **risk per context-aligned score movement**, not
  by generic risk alone.

Next local moves:

1. Add a KARP diagnostic using the existing eval-only score alignment:
   which KARP-removed atoms correlate with loss of context-teacher score
   movement?
2. Replace value-risk factors with a stronger local Fisher / residual-to-logit
   metric from the same pass, instead of only LM-head PCs and low-surprise MLP
   outputs.
3. Add a trust rule based on removed-update ratio and retained signal energy,
   so KARP does not over-prune layers where generic risk and useful readout are
   not separable.
4. Test KARP at scale `.15` only after margin safety improves; current margin
   drift is still too high.


## 2026-05-20 - KARP follow-up: local Fisher value risk and layer trust

After the first KARP grid, we tested two immediate refinements:

1. a same-pass local Fisher / residual-to-logit value-risk basis;
2. a KARP-derived all-layer trust scalar.

Both preserve the one-pass constraint:

- one lesson/context forward pass;
- no null prompt;
- no generated probes;
- no sentinel negatives for the write;
- no next-token optimization;
- closed-form update.

### Implementation

Updated `scripts/minilang_write.py`:

- Added `local_logit_fisher_basis(...)`.
  - For selected lesson positions, it takes the model's own top-k next-token
    distribution.
  - It builds low-rank factors:
    \[
    \sqrt{p_i}(W_U[i]-\mathbb{E}_p[W_U]).
    \]
  - This is used only as a KARP value-side risk basis, not as a training loss
    or label target.
- Added KARP flags:
  - `--karp-local-fisher-rank`;
  - `--karp-local-fisher-top-k`;
  - `--karp-local-fisher-max-positions`;
  - `--karp-layer-risk-budget`.
- Added a write-time KARP layer trust scalar:
  \[
  s_l=\min\left(1,\sqrt{\frac{b}{r_l+\epsilon}}\right),
  \]
  where \(r_l\) is `karp_cross_risk_after` and \(b\) is
  `--karp-layer-risk-budget`.

Validation:

- `.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q`
  - `23 passed`
- `.venv/bin/python -m pytest -q`
  - `64 passed`

### Results

All runs below:

- all 28 layers;
- `relational_aggregate`;
- context-value target;
- target scale `.10`, except the listed `.05` safety baseline;
- exponential surprise weighting, temperature `2.0`, cap `20`;
- input protection `1024/40`;
- expanded sentinel suite.

| run | edited | changed | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `1024/40` | `4/20` | `13` | `4/20` | `0.613` | `0.185` | `+0.199` | `1` | `2.355` |
| relational `.05`, output `1024/40` | `0/20` | `2` | `0/20` | `0.596` | `0.121` | `+0.137` | `0` | `1.126` |
| mixed KARP `.50` | `3/20` | `7` | `3/20` | `0.518` | `0.130` | `+0.200` | `0` | `2.215` |
| KARP + local Fisher rank `32` + output `1024/40` | `2/20` | `8` | `2/20` | `0.526` | `0.130` | `+0.193` | `0` | `2.247` |
| KARP + local Fisher rank `32`, no output penalty | `1/20` | `4` | `1/20` | `0.202` | `0.042` | `+0.095` | `4` | `3.148` |
| KARP + risk trust budget `.25` | `1/20` | `4` | `1/20` | `0.407` | `0.088` | `+0.148` | `0` | `1.839` |
| KARP + risk trust budget `.50` | `1/20` | `7` | `1/20` | `0.472` | `0.109` | `+0.166` | `0` | `1.871` |

### Diagnostics

Local Fisher with global output basis:

- KARP output basis rank becomes `1056` (`1024` global LM PCs + `32`
  same-pass Fisher rows).
- Mean removed-update ratio remains essentially unchanged:
  - plain KARP: `0.241`;
  - KARP + local Fisher + global output: `0.240`.
- Behavior worsens slightly:
  - acquisition `3/20 -> 2/20`;
  - sentinel drop `2.215 -> 2.247`.

Local Fisher without global output protection:

- Acquisition collapses to `1/20`.
- Sentinel c2w rises to `4`.
- Context-teacher alignment collapses:
  - centered cosine `0.202`;
  - projection ratio `0.042`.

Therefore this cheap top-k Fisher basis is not a sufficient substitute for
global output protection, and appending it to the global output basis does not
find the missing sentinel-sensitive atoms.

KARP layer trust:

- budget `.25`:
  - mean layer trust scale `0.842`;
  - median `0.968`;
  - minimum `0.307`.
- budget `.50`:
  - mean layer trust scale `0.938`;
  - median `1.000`;
  - minimum `0.434`.

Layer trust improves margin safety relative to plain KARP:

- plain KARP drop `2.215`;
- trust `.25` drop `1.839`;
- trust `.50` drop `1.871`.

But it also collapses acquisition:

- plain KARP `3/20`;
- trust `.25` `1/20`;
- trust `.50` `1/20`.

This looks like a smarter scale-down, not a new purification axis.

### Interpretation

The new results sharpen the diagnosis:

- The global output penalty is doing real safety work. Removing it and relying
  on local Fisher produces severe sentinel c2w damage.
- The top-k same-pass Fisher basis is too broad or mislocalized. It does not
  identify the specific sentinel-sensitive value atoms better than LM-head PCs.
- KARP's current cross-risk metric correlates with useful acquisition energy:
  using it as a layer trust scalar reduces sentinel margin damage but also
  removes the threshold-crossing acquisition component.
- The remaining blocker is not "add a local output-risk basis" in this simple
  form. We need a risk metric that predicts **collateral margin damage per unit
  context-teacher score movement**, not generic output observability alone.

Updated current frontier:

- Best acquisition with zero c2w remains mixed KARP `.50`:
  - `3/20`, c2w `0`, drop `2.215`.
- Best margin safety with nonzero acquisition in this family is trust `.25`:
  - `1/20`, c2w `0`, drop `1.839`.
- The old `.05` relational safety baseline still has lower drop (`1.126`) but
  no acquisition.

### Current ask for the next theory step

KARP product-risk purification is directionally right but still too blunt.
The next mathematical object should distinguish:

\[
\text{generic output observability}
\quad\text{from}\quad
\text{collateral output damage}.
\]

The write needs readout leverage to answer at all. Penalizing local Fisher or
post-KARP cross-risk removes useful threshold-crossing behavior. The missing
metric is something like:

\[
\frac{\text{predicted unrelated-margin movement}}
{\text{predicted context-teacher-aligned movement}}
\]

estimated only from the cold weights and the single lesson pass.

## 2026-05-20 - SHARP-KARP tested: same-pass shadow anchors are safe but over-prune

GPT-5.5 Pro proposed SHARP-KARP: use low-surprise, high-confidence same-pass
tokens as "shadow anchors" for collateral margin damage. The intended metric is
signed and candidate-conditioned:

\[
\text{risk}(p,q)
\approx
\frac{\text{predicted stable-margin drop from atom }p\otimes q}
{\text{predicted relational target-aligned movement from }p\otimes q+\epsilon}.
\]

This is a good refinement over plain local Fisher: it tries to penalize an atom
only when the current candidate update is predicted to lower stable margins,
not merely because the value direction can affect logits.

### Implementation

Added `--intrinsic-target-purifier sharp_karp`.

Files changed:

- `caic/intrinsic_surprise.py`
  - added `SharpKarpPurificationResult`;
  - added `sharp_karp_purify_update`;
  - added compact LM-head row lookup from precomputed top logits;
  - hardened PCA and small symmetric solves against ill-conditioned SVD/pinv
    failures;
  - made `relational_aggregate` empty all-layer selections skip instead of
    crashing.
- `scripts/minilang_write.py`
  - added SHARP flags:
    - `--sharp-shadow-anchors`;
    - `--sharp-key-rank`;
    - `--sharp-value-rank`;
    - `--sharp-signal-top-k`;
    - `--sharp-low-surprise-quantile`;
    - `--sharp-confidence-quantile`;
    - `--sharp-eta`;
    - `--sharp-shadow-weight`;
    - `--sharp-karp-kappa`;
    - `--sharp-shadow-temperature`;
    - `--sharp-solve-mode {ridge,shrink}`.
- `tests/test_intrinsic_surprise.py`
  - added a synthetic SHARP test showing that a shadow-margin-dropping atom is
    shrunk while a selected signal atom is preserved.

Validation:

```bash
.venv/bin/python -m pytest -q
# 65 passed
```

### Ridge-mode SHARP

First implementation used the full coefficient-space ridge/refit proposed by
5.5 Pro.

Run:

- all 28 layers;
- `relational_aggregate`;
- context-value target;
- target scale `.10`;
- output/input protection `1024/40`;
- SHARP rank `32/32`;
- SHARP eta `.35`;
- shadow weight `1.5`;
- KARP atom rails `.50/.05/.02`.

Initial anchor selection produced only about `9` anchors/layer on average.
Result:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHARP ridge, sparse anchors | `3/20` | `0.617` | `0.154` | `3` | `4.454` |

This was worse than KARP. We patched anchor fallback so layers use a larger
low-surprise/high-confidence set rather than starving the shadow constraint.
Anchor coverage rose to about `55` anchors/layer.

Result:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHARP ridge, anchor fallback | `3/20` | `0.596` | `0.148` | `4` | `4.589` |

Interpretation:

- Under-anchoring was not the only problem.
- The coefficient-space refit is unsafe: it rebuilds the relational map and
  can amplify harmful atoms while still satisfying the local shadow objective.
- Same-pass shadow anchors are not reliable enough to support a refit.

### Shrink-mode SHARP

Added `--sharp-solve-mode shrink`, which does not refit \(K \to R\). It only
attenuates atoms in the existing candidate map:

\[
M^\star_{ab}
=
\frac{M^0_{ab}}{1+\Lambda_{ab}}
\quad
\text{with an additional diagonalized shadow-energy shrink}.
\]

This keeps SHARP closer to KARP: purify the existing relational candidate
instead of inventing a new low-rank map.

Conservative rank-16 shrink:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHARP shrink r16, eta `.35`, shadow `1.5` | `0/20` | `0.230` | `0.038` | `0` | `0.951` |

Diagnostics:

- shadow drop before/after: `0.371 -> 0.009`;
- signal retention mean: `0.530`;
- removed update ratio mean: `0.929`;
- update Fro mean: `2.335 -> 0.482`.

Weaker shrink:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHARP shrink r16, eta `.10`, shadow `.5` | `1/20` | `0.253` | `0.043` | `0` | `0.967` |

Diagnostics:

- shadow drop before/after: `0.342 -> 0.014`;
- signal retention mean: `0.601`;
- removed update ratio mean: `0.924`;
- update Fro mean after: `0.559`.

### Interpretation

SHARP-KARP gives a clean result:

- Ridge/refit mode is unsafe and should not be pursued in its current form.
- Shrink mode is safe by the before-correct criterion, but behaves like a
  structured scale-down:
  - it reaches the target safety region (`drop < 1.0`, c2w `0`);
  - it collapses acquisition and context-teacher alignment.
- The same-pass shadow-margin proxy is highly correlated with useful
  threshold-crossing acquisition atoms. Penalizing it removes the thing we need.

This falsifies a stronger version of the shadow-anchor hypothesis:

\[
\text{same-pass stable-margin preservation}
\not\approx
\text{unrelated-capability preservation}
\]

at least not with logit-lens top-vs-runner anchors and atomwise shrink.

The current frontier remains:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `1024/40` | `4/20` | `0.613` | `0.185` | `1` | `2.355` |
| mixed KARP `.50` | `3/20` | `0.518` | `0.130` | `0` | `2.215` |
| SHARP shrink r16 `.10/.5` | `1/20` | `0.253` | `0.043` | `0` | `0.967` |
| relational `.05`, output `1024/40` | `0/20` | `0.596` | `0.121` | `0` | `1.126` |

### Updated diagnosis

The missing metric is not:

- global output sensitivity;
- top-k local Fisher sensitivity;
- same-pass stable-margin drop;
- KARP cross-risk;
- layer-level risk trust.

All of these correlate too strongly with the useful acquisition actuator.

The useful acquisition path still appears to be a small threshold-crossing
readout component inside `relational_aggregate` context-value. The safety
problem is not simply "this atom changes confident same-pass logits"; useful
readout atoms do that too.

The next theory step needs a metric closer to:

\[
\frac{
\text{predicted off-task option reordering under generic MC/question states}
}{
\text{predicted teacher-aligned option reordering under learned-object states}
}
\]

but it must be estimated without sentinel prompts, generated probes, null
prompts, labels, or runtime routers. The current same-pass lesson anchors are
not enough.

## 2026-05-21 - ORCA-KARP first pass: option-space target-parallel purification

GPT-5.5 Pro proposed ORCA-KARP: Object-Relative Contrastive Actuator
Purification.

Core idea:

- keep the existing `relational_aggregate` context-value actuator;
- decompose the candidate update into key/value atoms;
- measure each atom's same-pass option-space effect using local top-k LM-head
  contrasts;
- keep atoms whose option movement is target-parallel after nuisance
  residualization;
- shrink atoms whose effect is target-orthogonal, common posture, off-object
  readout, or KARP-generic.

This directly responds to the SHARP failure. SHARP asked whether the candidate
lowers same-pass stable margins. That deleted the acquisition path. ORCA asks
whether the atom's readout movement is explained by the relational target
itself.

### Implementation

Added:

- `OrcaKarpPurificationResult`;
- `_mixed_signal_risk_candidate_basis(...)`, which includes candidate SVD
  directions so the purifier actually covers the proposed update;
- `orca_karp_purify_update(...)`;
- `--intrinsic-target-purifier orca_karp`;
- ORCA flags:
  - `--orca-key-rank`;
  - `--orca-value-rank`;
  - `--orca-option-top-k`;
  - `--orca-object-rank`;
  - `--orca-off-object-rank`;
  - `--orca-eta-orth`;
  - `--orca-eta-posture`;
  - `--orca-eta-off-object`;
  - `--orca-eta-karp`;
  - `--orca-signal-floor-quantile`;
  - `--orca-nuisance-ridge`.

Added a synthetic unit test showing that ORCA keeps a target-parallel option
atom more than a target-orthogonal option atom.

Verification:

```bash
.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q
# 25 passed

.venv/bin/python -m pytest -q
# 66 passed
```

### Engineering notes

The literal ORCA quotient was too aggressive with the cheap logit-lens option
map. Early `1024/40` rank-48 diagnostics saturated:

- `orca_atom_diag_mean` near the cap (`~96.7`);
- all `2304` atoms shrunk by more than 50%;
- `orca_signal_retention ~0.0007`.

That was SHARP-style over-pruning in a new coordinate. We added a robust signal
floor:

\[
\operatorname{denom}_{ab}=S_{ab}+\operatorname{quantile}(S_+, q)
\]

and defaulted the first useful local run to `q=.50`.

The first implementation was also too slow because it recomputed large SVDs over
protected input/output bases per layer. We changed ORCA to use compact sketches
of negative keys and to reuse the existing output basis directly for the
off-object readout cloud.

### Results

All runs below use:

- `relational_aggregate`;
- `relation_value_mode=context`;
- all 28 layers;
- 6 lessons, 8 examples each;
- teacher-filtered 20-question eval;
- expanded 25-question sentinel suite;
- scale `.10`;
- exponential surprise temperature `2.0`;
- persistence power `1.0`.

Baseline weak-protection relational run:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `256/10`, input `256/20` | `6/20` | `11` | `0.578` | `0.183` | `3` | `4.688` |

ORCA rank-16 soft:

```bash
--intrinsic-target-purifier orca_karp
--orca-key-rank 16
--orca-value-rank 16
--orca-option-top-k 8
--orca-object-rank 64
--orca-off-object-rank 128
--orca-eta-orth 0.02
--orca-eta-posture 0.01
--orca-eta-off-object 0.02
--orca-eta-karp 0.05
--orca-signal-floor-quantile 0.50
--karp-eta-key 0.01
--karp-eta-value 0.005
```

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORCA r16 soft, output `256/10`, input `256/20` | `6/20` | `11` | `0.544` | `0.166` | `1` | `2.956` |

Diagnostics over 168 updates:

| metric | mean | median |
| --- | ---: | ---: |
| `orca_removed_update_ratio` | `0.541` | `0.520` |
| `orca_signal_retention` | `0.302` | `0.268` |
| `orca_candidate_capture_ratio` | `0.696` | `0.683` |
| `orca_atom_diag_mean` | `34.47` | `34.97` |
| `orca_signal_mean` | `0.0076` | `0.0057` |

ORCA rank-16 medium:

```bash
--orca-eta-orth 0.05
--orca-eta-posture 0.02
--orca-eta-off-object 0.05
--orca-eta-karp 0.10
```

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORCA r16 medium, output `256/10`, input `256/20` | `4/20` | `9` | `0.479` | `0.155` | `2` | `2.008` |

Diagnostics over 168 updates:

| metric | mean | median |
| --- | ---: | ---: |
| `orca_removed_update_ratio` | `0.606` | `0.587` |
| `orca_signal_retention` | `0.166` | `0.106` |
| `orca_candidate_capture_ratio` | `0.698` | `0.678` |
| `orca_atom_diag_mean` | `44.79` | `43.97` |
| `orca_signal_mean` | `0.0072` | `0.0052` |

### Interpretation

ORCA is not yet safe, but it is meaningfully different from SHARP:

- SHARP shrink reached safety by deleting acquisition (`0-1/20`).
- ORCA r16 soft preserved the full `6/20` acquisition of the weak-protection
  relational baseline while cutting c2w from `3` to `1` and before-correct drop
  from `4.688` to `2.956`.

That is the first purifier result that reduces collateral damage without
collapsing the acquisition score in the weak-protection setting.

However:

- it still fails the hard criterion (`c2w` must be `0`);
- stronger ORCA penalties did not monotonically improve discrete safety:
  before-correct drop improved to `2.008`, but c2w worsened to `2` and
  acquisition fell to `4/20`;
- the cheap option-space map sees most atoms as target-orthogonal
  (`orca_orthogonal_mean ~0.99`), so the absolute ORCA scores are not yet
  well-calibrated;
- candidate capture at rank 16 is only about `0.70`, so important update
  energy still sits outside the inspected atom basis.

Current diagnosis:

\[
\text{ORCA's object-relative option coordinate is directionally useful,}
\]

but the current cheap top-k logit-lens implementation is still too blunt for
the final safety frontier. It improves safety/acquisition tradeoff relative to
the `256/10` relational baseline, but not relative to the hard target.

### Next experiments

1. Run ORCA r32 or r48 with the optimized basis code on the `1024/40` stack,
   ideally off local MPS if possible. This is the direct comparison to:
   - mixed KARP `.50`: `3/20`, c2w `0`, drop `2.215`;
   - relational `.10`, output `1024/40`: `4/20`, c2w `1`, drop `2.355`.
2. Add an ORCA atom ablation mode:
   - kept atoms only;
   - removed atoms only;
   - top-risk removed atoms only;
   - top-signal kept atoms only.
   This will tell whether ORCA's removed atoms actually carry sentinel damage
   or whether the shrink is mostly cosmetic.
3. Improve the option map:
   - compare current top-k LM-head contrasts to exact local RMSNorm/logit VJP
     for a small layer subset;
   - test whether option-space centering should be per-question/option-family
     rather than same-pass top-k only.
4. If ORCA atom ablations validate the decomposition, add a closed-form
   low-rank solve directly in ORCA coefficient space rather than post-hoc
   shrink.

## 2026-05-21 - ORCA Atom Ablations Show The Useful Signal Lives In The Basis Residual

We added explicit ORCA ablation modes:

- `kept_only`: apply only the projected atoms kept after ORCA shrinkage;
- `removed_only`: apply only the projected atoms removed by ORCA shrinkage;
- `residual_only`: apply only the candidate update component outside the
  mixed ORCA key/value basis;
- `top_signal_kept` and `top_risk_removed` for follow-up component tests.

Verification:

```bash
.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q
# 25 passed

.venv/bin/python -m pytest -q
# 66 passed
```

All runs below use:

- all 28 layers;
- 6 lessons, 8 examples;
- teacher-filtered 20-question eval;
- expanded 25-question sentinel suite;
- `relational_aggregate`, context value;
- scale `.10`;
- ORCA r16 soft settings.

### Weak-protection decomposition

The matched weak-protection stack is output `256/10`, input `256/20`.

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline relational `.10` | `6/20` | `11` | `0.578` | `0.183` | `3` | `4.688` |
| ORCA purified | `6/20` | `11` | `0.544` | `0.166` | `1` | `2.956` |
| ORCA kept-only | `1/20` | `3` | `-0.041` | `-0.006` | `1` | `1.011` |
| ORCA removed-only | `1/20` | `1` | `0.377` | `0.046` | `1` | `1.129` |
| ORCA residual-only | `9/20` | `13` | `0.603` | `0.157` | `2` | `1.287` |

Mean diagnostics over 168 updates:

| run | update Fro | basis residual Fro | projected Fro | kept projected Fro | removed projected Fro |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept-only | `0.420` | `1.532` | `1.410` | `0.420` | `1.087` |
| removed-only | `0.536` | `0.693` | `0.796` | `0.309` | `0.536` |
| residual-only | `1.679` | `1.679` | `1.582` | `0.488` | `1.228` |

Interpretation:

- The ORCA-kept projected atoms are mostly behaviorally inert.
- The ORCA-removed projected atoms are also mostly behaviorally inert, with
  weak raw score movement but almost no centered teacher-aligned movement.
- The acquisition-heavy component is the basis residual: the part of the
  relational/context-value candidate outside the mixed ORCA atom subspace.

This is a sharper failure than "ORCA removes the wrong atoms." The current
mixed key/value basis itself appears to be the wrong inspection coordinate.
Projecting away that low-rank basis acts like a useful deflation step, but
trying to classify atoms inside the basis does not isolate the useful actuator.

### Strong-protection residual-only check

We also tested residual-only on the stronger output/input protection stack
`1024/40`.

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORCA residual-only, output/input `1024/40` | `2/20` | `8` | `0.618` | `0.155` | `1` | `0.426` |

Sentinel accuracy improved overall (`12/25 -> 15/25`), but one previously
correct sentinel flipped: "Which animal is known for barking?" moved from
`Dog` to `Cloud`. This is not safe enough, but it shows strong protection can
turn the residual-only component into a small teacher-aligned nudge with much
lower average margin damage.

### Updated diagnosis

The current frontier is no longer simply "find better unsafe atoms inside the
candidate map." The evidence now says:

\[
\text{relational/context-value contains the useful actuator,}
\]

but:

\[
\text{the current low-rank atom bases mostly capture inert or generic projected
movement, while the useful acquisition lives in their complement.}
\]

The next mathematical move should probably treat low-rank generic/readout
subspaces as things to quotient out first, then construct a new positive basis
inside the residual complement. In other words:

1. keep the one-pass relational/context-value target family;
2. deliberately deflate the mixed ORCA projected subspace;
3. learn or derive a new residual-coordinate purifier, instead of asking the
   old ORCA basis to classify atoms.

Open question for the next Pro round:

> What is the right closed-form coordinate for the basis residual that carries
> acquisition, and how do we remove the remaining single sentinel c2w without
> crushing the residual signal into the `1024/40` low-acquisition regime?

## 2026-05-21 - Q-RICO Tests: Safe Residual Filtering Under-Acquires

5.5 Pro proposed Q-RICO: quotient out the ORCA mixed basis, then solve in the
residual map using a target-orthogonal option-scramble metric. We implemented
Q-RICO as a purifier around the existing `relational_aggregate` context-value
write.

Implementation details:

- added `qrico_purify_update` in `caic/intrinsic_surprise.py`;
- added `--intrinsic-target-purifier qrico`;
- added `--qrico-solve-mode sylvester|residual_filter`;
- added diagnostics for quotient residual size, option-scramble quotient,
  capture ratio, and layer trust;
- avoided full update-map SVDs after the first draft proved too slow, replacing
  them with QR-only signal/generic row bases plus cheap map-probe rows;
- first option-scramble implementation uses same-pass top-logit contrast rows,
  not the full deterministic vocab sketch yet.

Verification:

```bash
.venv/bin/python -m pytest -q
# 69 passed
```

All runs below use:

- all 28 layers;
- 6 lessons, 8 examples;
- teacher-filtered 20-question eval;
- expanded 25-question sentinel suite;
- `relational_aggregate`, context value;
- output/input weak-protection stack `256/10`, `256/20`;
- top-32 option contrasts, target-parallel rank 1 unless noted.

### Results

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q-RICO CCA, deflate `16/16`, trust on, scale `.10` | `2/20` | `7` | `0.537` | `0.121` | `3` | `2.542` |
| Q-RICO map-probe CCA, deflate `16/16`, trust on, scale `.10` | `0/20` | `0` | `0.149` | `0.006` | `0` | `0.105` |
| Q-RICO map-probe CCA, deflate `4/4`, no trust, scale `.10` | `1/20` | `1` | `0.592` | `0.096` | `0` | `0.569` |
| Q-RICO map-probe CCA, no deflate, no trust, scale `.10` | `1/20` | `3` | `0.568` | `0.127` | `1` | `2.327` |
| Q-RICO residual-filter, deflate `4/4`, no trust, scale `.10` | `1/20` | `4` | `0.638` | `0.125` | `0` | `0.219` |
| Q-RICO residual-filter, deflate `4/4`, no trust, scale `.20` | `1/20` | `12` | `0.606` | `0.174` | `0` | `1.978` |

For comparison, the previous ORCA residual-only weak-protection run was:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORCA residual-only, scale `.10` | `9/20` | `13` | `0.603` | `0.157` | `2` | `1.287` |

### Interpretation

Q-RICO did not reproduce the acquisition-bearing residual-only result.

What worked:

- Q-RICO residual-filter is genuinely safer: scale `.10` had `0` c2w and low
  before-correct drop (`0.219`) while retaining positive teacher alignment
  (`centered cos 0.638`, projection ratio `0.125`).
- The option-scramble/value-side filter is not just random shrinkage; it
  preserves teacher-aligned score motion better than the inert map-probe CCA
  run.

What failed:

- Every Q-RICO variant under-acquired badly (`0-2/20`) versus ORCA
  residual-only (`9/20`).
- The CCA reconstruction route is too lossy. It can have high row capture
  diagnostics but still fail to cross behavioral thresholds.
- Direct residual filtering preserves safety but still does not recover the
  threshold-crossing component.
- Scaling the residual filter to `.20` increased prediction churn and margin
  movement but did not improve acquisition, so the missing component is not
  simply "more safe residual-filter magnitude."

Updated diagnosis:

\[
\text{The useful actuator is the direct high-rank residual map, not a low-rank
residual CCA reconstruction.}
\]

Q-RICO's option-scramble metric can reduce collateral damage, but it appears to
remove or fail to preserve the discrete threshold-crossing part of the
acquisition. The next mathematical object should preserve the full residual map
more faithfully while constraining collateral effects. A low-rank value-side
filter is not enough.

Open question for the next Pro round:

> Given that ORCA residual-only gets `9/20` but unsafe, while Q-RICO residual
> filtering is safe but stuck at `1/20`, what is the right closed-form
> full-rank or high-rank purification of the residual map that preserves
> threshold-crossing acquisition while preventing sentinel option flips?

## 2026-05-21 - Q-RICO Key-Width Correction And SPECTRA Trial

5.5 Pro proposed SPECTRA: preserve the direct high-rank ORCA residual map, then
apply a small closed-form rank-one correction that pins learned-object tail
functionals while clipping generic option-contrast hazards.

Implementation:

- added `--intrinsic-target-purifier spectra`;
- added SPECTRA tail/hazard constraints in `caic/intrinsic_surprise.py`;
- added a reusable rank-one metric projection helper;
- added focused tests for tail preservation and hazard clipping.

Verification:

```bash
.venv/bin/python -m pytest -q
# 71 passed
```

### Important correction: Q-RICO was partly underconfigured

The earlier Q-RICO runs used `--intrinsic-surprise-key-feature-top-k 8`, while
the ORCA residual-only baseline used `16`. Re-running the best Q-RICO
residual-filter configuration with `key_feature_top_k=16` changed the picture.

All rows below use:

- all 28 layers;
- `relational_aggregate`, context value;
- output/input weak stack `256/10`, `256/20`;
- Q-RICO residual-filter;
- deflate `4/4`;
- no layer trust;
- no option-scramble penalty (`scramble_weight=0.0`);
- teacher-filtered 20-question eval and expanded sentinel suite.

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q-RICO key8, scale `.10` | `2/20` | `12` | `0.615` | `0.176` | `0` | `1.918` |
| Q-RICO key16, scale `.10` | `5/20` | `11` | `0.578` | `0.162` | `0` | `0.964` |
| Q-RICO key16, scale `.15` | `4/20` | `10` | `0.515` | `0.146` | `7` | `6.738` |
| Q-RICO key16, scale `.20` | `5/20` | `10` | `0.584` | `0.163` | `6` | `5.032` |

This makes Q-RICO key16 scale `.10` the current best safe single-task point:

\[
5/20,\quad 0\text{ c2w},\quad \text{drop}=0.964.
\]

Scale pressure does not help. At `.15` and `.20`, acquisition does not improve
and sentinel damage returns hard. This means the safe frontier is not just
"make Q-RICO bigger."

### SPECTRA result

The first fast SPECTRA implementation used a cheaper Q-RICO-style deflation
basis, not the exact ORCA basis. It was safe but weak:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPECTRA fast, hazard rank `4`, budget `.25` | `2/20` | `2` | `0.292` | `0.032` | `0` | `0.262` |
| SPECTRA fast, no-hazard | `2/20` | `2` | `0.137` | `0.010` | `0` | `0.201` |
| SPECTRA fast, mild hazard rank `1`, budget `.50` | `1/20` | `2` | `-0.235` | `-0.019` | `0` | `0.130` |

Mean SPECTRA diagnostics for the main fast run:

- tail mass retention: `0.999`;
- hazard spectral ratio: `0.615`;
- correction Frobenius norm: `0.056`;
- residual map Frobenius norm: `1.532`.

The hazard correction was not the main reason for the weak behavior. The
no-hazard SPECTRA run was already weak, so the loss happened in the quotient
coordinate. We patched SPECTRA to use the exact ORCA mixed basis, but the
exact-basis all-layer jobs ran for more than 35 minutes without completing, so
that path is currently too slow for the research loop unless optimized.

### Updated diagnosis

The current reliable frontier is:

| method | acquisition | safety |
| --- | ---: | --- |
| ORCA residual-only weak stack | `9/20` | unsafe: `2` c2w, drop `1.287` |
| Q-RICO key16 residual-filter `.10` | `5/20` | safe: `0` c2w, drop `0.964` |
| SPECTRA fast | `2/20` | safe but too weak |

The key new facts:

1. Q-RICO was not fully falsified. With the same key width as ORCA, it becomes
   the best safe method so far.
2. The safe/unsafe transition is sharp: Q-RICO `.10` is safe, but `.15` and
   `.20` cause large sentinel damage without more acquisition.
3. SPECTRA's conceptual move may still be right, but the exact ORCA quotient is
   too slow in the current implementation, and the fast quotient does not
   preserve the acquisition-bearing residual.

Next moves:

1. Treat Q-RICO key16 `.10` as the current safe single-task baseline.
2. Add Q-RICO support to the continual diagnostic and run two-task retention.
3. Optimize exact SPECTRA basis construction before testing it again, or drop
   SPECTRA if the optimized no-hazard verifier still fails to reproduce ORCA
   residual-only.
4. Explore whether Q-RICO's safe frontier can be improved by row selection or
   key width, not by scale.

## 2026-05-21 - Two-Task Continual Benchmark Becomes The Main Scoreboard

We updated `scripts/minilang_intrinsic_continual.py` so it can run the same
purifier stack as `scripts/minilang_write.py`, including Q-RICO and SPECTRA
flags. We also added `scripts/summarize_continual_run.py`, a compact
scoreboard for sequential acquisition, retention, and sentinel preservation.

Verification:

```bash
.venv/bin/python -m py_compile scripts/summarize_continual_run.py
.venv/bin/python -m pytest -q
# 71 passed
```

Important implementation correction:

- `minilang_intrinsic_continual.py` had
  `--intrinsic-surprise-input-penalty-usage-power` defaulting to `1.0`.
- The safe single-task Q-RICO frontier used the write-script default `0.0`.
- We changed the continual default to `0.0` and re-ran the two-task diagnostic
  with the single-task-safe settings.

### New benchmark format

Single-task eval remains a smoke test, but the primary research scoreboard is
now:

- task 0 immediate acquisition after write 0;
- task 0 retention after write 1;
- task 1 immediate acquisition after write 1;
- sentinel correct-to-wrong after each write;
- sentinel before-correct margin drop after each write.

This better matches the actual goal: continual learning without damaging other
capabilities.

### Two-task Q-RICO key16 results

Both runs:

- two tasks: Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 8-question eval per task;
- all 28 layers;
- Q-RICO residual-filter;
- key feature top-k `16`;
- scale `.10`;
- output/input weak stack `256/10`, `256/20`;
- input-penalty usage power `0.0`;
- expanded sentinel suite.

#### No old-task protection

| step | eval task | edited | delta vs base | retention delta | edited margin |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Lyran | `3/8` | `+2/8` | `0` | `-0.337` |
| 1 | Lyran | `2/8` | `+1/8` | `-1/8` | `-0.348` |
| 1 | Vomar | `0/8` | `-1/8` | `0` | `-1.219` |

| step | sentinel acc | c2w | w2c | before-correct drop |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `18/25` | `0` | `6` | `1.717` |
| 1 | `18/25` | `1` | `7` | `3.721` |

Interpretation:

- Task 0 acquires weakly and then partially forgets.
- Task 1 does not acquire.
- Sentinel safety breaks after the second write.

#### Old selected keys as negatives

| step | eval task | edited | delta vs base | retention delta | edited margin |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Lyran | `2/8` | `+1/8` | `0` | `-0.273` |
| 1 | Lyran | `2/8` | `+1/8` | `0` | `-0.298` |
| 1 | Vomar | `1/8` | `0/8` | `0` | `-1.077` |

| step | sentinel acc | c2w | w2c | before-correct drop |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `18/25` | `0` | `6` | `0.601` |
| 1 | `19/25` | `0` | `7` | `0.628` |

Interpretation:

- Old-key negatives preserve task 0 and sentinel safety.
- They do so partly by suppressing the second write: task 1 stays at baseline.
- Update diagnostics confirm the suppression:
  - task 0 mean update Frobenius: `2.76`;
  - task 1 mean update Frobenius with old negatives: `1.37`.

### Updated diagnosis

The method is not yet a continual learner.

Q-RICO key16 `.10` can safely acquire a small amount from one context, but in
sequence:

\[
\text{no old protection} \Rightarrow \text{forgetting + sentinel damage},
\]

while:

\[
\text{old-key protection} \Rightarrow \text{retention + safety but no new acquisition}.
\]

This exactly matches the earlier theoretical concern: protecting old keys is
too crude. It forbids reuse of broad translation/readout key manifolds. The
first interpretation was that we should protect old key-to-value
transformations rather than keys alone.

However, the deployment constraint has now tightened: we are not allowed to
store old transformations either. Old-key negatives and old-transformation
metrics are diagnostics only. The final method must preserve old learning using
only the updated weights themselves, plus the next session's one forward pass.

### New research priority

Do not optimize single-task acquisition as the main hill.

The primary target is now the two-task benchmark:

\[
\text{task0 immediate} > \text{base},\quad
\text{task0 after task1} \ge \text{task0 immediate},\quad
\text{task1 immediate} > \text{base},\quad
\text{sentinel c2w}=0.
\]

Single-task runs are still useful for fast falsification, but a technique does
not count as progress unless it improves this sequential scoreboard.

## 2026-05-21 - Hard Constraint Update: No Cross-Session Stored State

New hard constraint:

After a context/session is read and the closed-form write is applied, the next
context/session receives only the updated model weights. The system cannot carry
forward any separate state from previous sessions.

Not allowed across sessions:

- stored selected keys;
- old-task negative key banks;
- activation traces;
- examples, summaries, or document memories;
- low-rank atoms;
- old key-to-value transformation records;
- Fisher/EWC/Laplace/sketch metrics;
- runtime routers, task IDs, memory slots, or adapters.

This refines the continual-learning target. We still want:

\[
\text{task0 immediate} > \text{base},\quad
\text{task0 after task1} \ge \text{task0 immediate},\quad
\text{task1 immediate} > \text{base},\quad
\text{sentinel c2w}=0.
\]

But the protection must be weight-only:

\[
\text{future write} = f(W_{\text{current}},\ \text{one new forward pass}),
\]

with no \(f\)-input from old sessions except what is already encoded in
\(W_{\text{current}}\).

Updated blocker:

> Can one-pass closed-form surprise writes make learned transformations
> self-preserving inside the weights, so later writes computed only from the
> current weights and a new context pass do not overwrite them?

If this is impossible, the right falsifier is an identifiability argument: the
current weights may not reveal which directions were previously learned versus
preexisting, unless the write itself modifies the geometry in a detectable,
self-protecting way.

## 2026-05-21 - Implemented SEAL-Q-RICO Prototype

After the no-sidecar-state constraint, GPT-5.5 Pro proposed SEAL-Q:
Symmetry-Encoded Anti-erasure Learning for Q-RICO.

Core idea:

- keep the current best safe single-task base, Q-RICO/key16;
- add a signed anti-erasure post-pass that shrinks only candidate update
  components anti-parallel to current salient down-value columns;
- after applying an accepted write, encode salience inside the weights by an
  exact SwiGLU gauge transform:

\[
U_j \leftarrow c_j U_j,\qquad D_j \leftarrow D_j/c_j.
\]

For Qwen-style SwiGLU MLPs this preserves the MLP contribution exactly because
the up branch activation scales by \(c_j\) and the matching down column scales
by \(1/c_j\). This creates a detectable up/down imbalance in the weights
without storing any external state.

Implemented prototype:

- new purifier mode: `--intrinsic-target-purifier seal_qrico`;
- helpers:
  - `mlp_gauge_salience`;
  - `signed_anti_erase_update`;
  - `compute_gauge_seal_scales`;
  - `apply_mlp_gauge_seal_`;
  - `gauge_canonical_key_scale`;
- relational aggregate selector now accepts optional `scoring_keys`, allowing
  gauge-canonical surprise scoring while keeping raw keys for the actual solve;
- `scripts/minilang_write.py` and `scripts/minilang_intrinsic_continual.py`
  expose:
  - `--seal-eta-erase`;
  - `--seal-eta-seal`;
  - `--seal-max-scale`;
  - `--seal-salience-tau`;
  - `--seal-disable-apply`;
  - `--seal-canonicalize-surprise`.

Important implementation detail:

The repo applies writes through `AdditiveMemoryLinear`, not by directly editing
only the frozen down projection. The gauge seal therefore scales:

- the MLP up-projection row;
- the frozen/base down column;
- the additive memory down column;
- all slot memory down columns.

This preserves the effective MLP output after a write, rather than only the
base model's output.

Unit tests added:

- function-preserving SwiGLU gauge transform, including additive memory;
- salience detection for sealed channels;
- signed anti-erasure shrinks negative-parallel but not positive-parallel
  updates;
- canonical activation \(a_j\|D_j\|\) is invariant under the up/down gauge;
- `seal_qrico_purify_update` smoke test.

Verification:

```text
.venv/bin/python -m pytest -q
76 passed in 1.30s
```

Next experiments:

1. Single-task sanity:
   Q-RICO/key16 `.10` + signed anti-erasure, with `--seal-disable-apply`, to
   isolate whether anti-erasure damages acquisition.
2. Two-task benchmark:
   `seal_qrico` with gauge seal and `--seal-canonicalize-surprise`.
   This is the main test because it uses no old keys, no old atoms, no old
   transforms, and no sidecar state.
3. Gauge-only ablation:
   apply the seal but set `--seal-eta-erase 0`, testing whether the seal is
   visible but unused.

Falsifiers:

- If salience is detectable but task 0 still forgets after task 1, then
  anti-erasure of current down columns is not sufficient; interference is
  coming through orthogonal additions or downstream interactions.
- If task 1 acquisition collapses, SEAL-Q has recreated old-key suppression in
  weight-only form.
- If sentinel c2w persists despite high anti-erasure removal, the unsafe
  component is additive false evidence rather than destructive erasure.

### SEAL-Q Two-Task Results

All runs used the primary two-task teacher-filtered setup:

- Qwen/Qwen3-1.7B;
- two tasks: Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 8-question eval per task;
- all 28 layers;
- `relational_aggregate`, context-value, final-aligned;
- Q-RICO residual-filter base;
- key feature top-k `16`;
- scale `.10`;
- output/input weak stack `256/10`, `256/20`;
- expanded sentinel suite;
- no old-key negatives, no old atoms, no sidecar state.

Baselines in this run:

- Lyran baseline `1/8`, context `8/8`;
- Vomar baseline `1/8`, context `8/8`;
- teacher filter found `73/120` Lyran context-correct candidates and `50/120`
  Vomar context-correct candidates.

#### Full SEAL-Q

Flags:

- `--intrinsic-target-purifier seal_qrico`;
- `--seal-canonicalize-surprise`;
- `--seal-eta-erase 2.0`;
- `--seal-eta-seal 0.05`;
- `--seal-max-scale 1.10`.

Results:

| step | eval task | edited | delta vs base | retention delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | Lyran | `0/8` | `-1/8` | `0` |
| 1 | Lyran | `1/8` | `0/8` | `+1/8` from bad acquisition |
| 1 | Vomar | `1/8` | `0/8` | `0` |

Sentinel:

| step | sentinel acc | c2w | w2c | before-correct drop | max drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `10/25` | `3` | `1` | `1.525` | `3.874` |
| 1 | `13/25` | `3` | `4` | `2.414` | `6.396` |

Diagnostics over 336 layer/lesson updates:

- mean anti-erasure ratio `0.065`;
- mean update retention after anti-erasure `0.989`;
- mean seal scale `1.00022`, max `1.05127`;
- mean scaled channels/layer update `92.9`;
- mean update Frobenius `1.90`;
- mean Q-RICO layer trust `0.630`.

Interpretation:

The exact gauge seal is numerically valid, but this operating point is worse
than Q-RICO alone. It loses single-task acquisition and still damages sentinels.
The failure is not because anti-erasure deleted the update: update norm
retention is about `99%`. The harmful component is therefore likely not a
simple destructive-parallel edit to currently salient down columns.

#### Anti-Erasure Only

Flags:

- `--intrinsic-target-purifier seal_qrico`;
- `--seal-disable-apply`;
- no canonical surprise;
- `--seal-eta-erase 2.0`.

Results:

| step | eval task | edited | delta vs base | retention delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | Lyran | `2/8` | `+1/8` | `0` |
| 1 | Lyran | `2/8` | `+1/8` | `0` |
| 1 | Vomar | `1/8` | `0/8` | `0` |

Sentinel:

| step | sentinel acc | c2w | w2c | before-correct drop | max drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `14/25` | `2` | `4` | `0.468` | `1.609` |
| 1 | `15/25` | `2` | `5` | `0.893` | `3.632` |

Diagnostics:

- mean anti-erasure ratio `0.108`;
- mean update retention `0.991`;
- no seal applied;
- mean update Frobenius `1.79`;
- mean Q-RICO layer trust `0.629`.

Interpretation:

Signed anti-erasure alone improves margin drop versus full SEAL-Q and preserves
weak task-0 acquisition, but it does not solve the benchmark. Task 1 remains at
baseline and sentinel c2w remains nonzero.

#### Gauge-Only

Flags:

- `--intrinsic-target-purifier seal_qrico`;
- `--seal-canonicalize-surprise`;
- `--seal-eta-erase 0.0`;
- seal applied.

Results:

| step | eval task | edited | delta vs base | retention delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | Lyran | `2/8` | `+1/8` | `0` |
| 1 | Lyran | `2/8` | `+1/8` | `0` |
| 1 | Vomar | `1/8` | `0/8` | `0` |

Sentinel:

| step | sentinel acc | c2w | w2c | before-correct drop | max drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `10/25` | `3` | `1` | `1.533` | `3.479` |
| 1 | `11/25` | `3` | `2` | `2.229` | `7.054` |

Diagnostics:

- anti-erasure ratio `1.0`, as expected;
- update retention `~1.0`;
- mean seal scale `1.00022`, max `1.05127`;
- mean scaled channels/layer update `93.1`;
- mean update Frobenius `1.93`;
- mean Q-RICO layer trust `0.632`.

Interpretation:

The gauge seal/canonicalization path is unsafe even without anti-erasure. Since
the gauge transform itself is function-preserving at application time, the
damage likely comes from how the subsequent write interprets sealed geometry,
or from using canonicalized activation magnitudes in a way that overselects
highly readout-sensitive channels. The current seal is a valid way to mark
weights, but it is not yet a useful protection rule.

### Updated SEAL-Q Diagnosis

SEAL-Q in its current form is falsified as the next mechanism.

What survived:

- exact SwiGLU up/down gauge sealing works mechanically;
- salience is detectable from weights;
- signed anti-erasure removes the intended destructive-parallel component;
- all of this can be done with no sidecar state.

What failed:

- no run acquired task 1 above baseline;
- no run achieved zero sentinel c2w;
- full SEAL-Q erased useful task-0 acquisition;
- gauge-only worsened sentinel safety despite being function-preserving at the
  moment of sealing;
- anti-erasure-only had the best margin behavior but still had `2` c2w and no
  task-1 acquisition.

Most likely lesson:

The remaining sentinel failure is not primarily destructive erasure of already
load-bearing down columns. It is more likely additive false evidence or
mode/readout injection into generic option directions. Also, encoding salience
by up/down gauge imbalance may be too entangled with the very activation
statistics the next writer uses for surprise.

Next question for theory:

Can a weight-only continual mechanism leave a detectable consolidation mark
that future write rules can use without distorting the native surprise
coordinate? If gauge imbalance is the wrong mark, alternatives might need to be
function-preserving rotations within MLP hidden subspaces, nullspace-preserving
orthogonalization, or a write rule whose protection is recoverable from
ordinary current-weight geometry without changing feature scale.

## 2026-05-21: OCEP/Q-RICO Follow-Up

5.5 Pro proposed **OCEP-Q**: oblique collateral-evidence projection. The
diagnosis was that SEAL-Q falsified destructive erasure as the main issue, so
the remaining failure should be additive generic evidence injection. OCEP tries
to preserve selected object-key effects while projecting out option/readout
evidence that generic same-pass/current-weight keys can access.

Implemented:

- `ocep_residual`: OCEP applied directly to the relational aggregate
  context-value candidate;
- `ocep_qrico`: Q-RICO first, then OCEP;
- no sidecar state, no old keys/transforms, no probes;
- current-weight generic anchors from down/value geometry, high-upstream-norm
  one-hot channel anchors, low-surprise same-pass keys, and protected negative
  keys already present in the solve;
- option basis from output/readout basis, target rows, and local top-logit
  contrasts from the same lesson pass.

Unit tests:

- OCEP projection reduces synthetic generic leakage while preserving object
  effects;
- OCEP wrapper builds current-weight/same-pass bases;
- full suite: `78 passed`.

### OCEP Residual, Initial Aggressive Run

Flags:

- all layers `0..27`;
- relational aggregate context-value;
- scale `.10`;
- input/output protection `256/20` and `256/10`;
- `--intrinsic-target-purifier ocep_residual`;
- `--ocep-object-rank 64`;
- `--ocep-correction-cap 0.35`.

Results:

| metric | value |
| --- | ---: |
| baseline | `2/20` |
| context | `15/20` |
| edited | `1/20` |
| sentinel before/after | `12/25 -> 10/25` |
| sentinel c2w | `6` |
| before-correct drop | `5.586` |
| mean OCEP leakage reduction | `0.704` |
| mean object delta ratio | `0.105` |

Interpretation:

This was a hard fail. OCEP removed its leakage sketch, but disturbed the
selected object effects and made sentinel behavior substantially worse.

### OCEP Residual, Object-Preserving Run

Flags changed:

- `--ocep-object-rank 256`;
- `--ocep-correction-cap 0.10`.

Results:

| metric | value |
| --- | ---: |
| baseline | `2/20` |
| context | `15/20` |
| edited | `4/20` |
| sentinel before/after | `12/25 -> 12/25` |
| sentinel c2w | `5` |
| before-correct drop | `5.636` |
| mean OCEP leakage reduction | `0.191` |
| mean object delta ratio | `0.000` |
| mean correction ratio | `0.100` |

Interpretation:

Preserving the object-key span exactly recovers some acquisition, but safety is
still unacceptable. This falsifies the current OCEP residual correction as a
sentinel-safe purifier: the dangerous component is not removed by this generic
option-leakage projection.

### OCEP-Q and Q-RICO Control

OCEP-Q flags:

- Q-RICO residual filter base;
- same scale/protection/eval split;
- `--ocep-object-rank 256`;
- `--ocep-correction-cap 0.10`.

OCEP-Q results:

| metric | value |
| --- | ---: |
| baseline | `2/20` |
| context | `15/20` |
| edited | `1/20` |
| sentinel before/after | `12/25 -> 15/25` |
| sentinel c2w | `0` |
| before-correct drop | `0.849` |
| mean OCEP leakage reduction | `0.445` |
| mean object delta ratio | `0.000` |

Q-RICO-only control on the exact same split/settings:

| metric | value |
| --- | ---: |
| baseline | `2/20` |
| context | `15/20` |
| edited | `1/20` |
| sentinel before/after | `12/25 -> 14/25` |
| sentinel c2w | `0` |
| before-correct drop | `0.742` |

Interpretation:

On this regenerated split, Q-RICO itself is safe but inert. OCEP-Q does not
make it materially worse on safety, but also does not restore acquisition. The
broader pattern remains:

- raw/relational high-rank residual maps can acquire, but are unsafe;
- Q-RICO-style filtering is safe, but often too weak;
- OCEP residual preserves object rows yet still leaves sentinel c2w;
- OCEP-Q is another safe/inert point.

Updated diagnosis:

The current OCEP hypothesis is insufficient. The unsafe acquisition component
is not captured by the generic-key × option-basis leakage sketch, even when the
object span is preserved exactly. The dangerous part may be a more specific
target-row-to-option interaction, a downstream multi-layer amplification not
visible in the local MLP-down map, or an additive evidence direction that is
only generic after propagation through later layers.

Next research priority:

Stop adding standalone local output filters. The next candidate should either:

1. build the safety metric from predicted downstream propagation of the update
   through later frozen layers, still in closed form and same-pass only; or
2. move the main benchmark to two-task + sentinels first, using the current
   safe/inert and unsafe/acquiring single-task points as controls rather than
   optimizing more single-task purifier variants.

## 2026-05-21: Benchmark Harness And SPECTRA Practicality Check

Added `scripts/continual_benchmark_grid.py` so candidate techniques can be run
against the same two-task + expanded-sentinel benchmark without hand-building
commands. It defines the current standard configuration:

- two tasks, Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 8-question eval per task;
- all layers `0..27`;
- relational aggregate context-value target;
- final-aligned token selection;
- key feature top-k `16`;
- target scale `.10`;
- output/input protection `256/10` and `256/20`;
- input penalty usage power `0.0`;
- expanded sentinel suite.

Registered presets:

- `relational_raw`: high-acquisition unsafe control;
- `qrico_key16`: current safe single-task baseline and primary continual
  control;
- `spectra_noquotient`: direct high-rank SPECTRA tail/hazard diagnostic;
- `ocep_residual`: object-preserving OCEP residual unsafe control;
- `ocep_qrico`: Q-RICO then OCEP safe/inert control;
- `seal_qrico_no_apply`: signed anti-erasure without gauge sealing.

Verification:

```bash
.venv/bin/python -m py_compile scripts/continual_benchmark_grid.py
.venv/bin/python -m pytest -q
# 78 passed
```

I also started a Modal run for the `spectra_noquotient` preset:

```text
/modal-runs/spectra_noquotient_continual_s010_20260521
```

It reached the task-0 write phase but stayed there for over 10 minutes with no
layer/eval progress, so I stopped the Modal app. This is not a behavioral
result, but it is a practical result: even without exact ORCA quotienting,
current SPECTRA is too slow for the all-layer two-task loop. That keeps it out
of the fast research scoreboard unless the implementation is substantially
optimized.

Updated practical priority:

The main loop should now use `scripts/continual_benchmark_grid.py` to compare
new methods against the same two-task sentinel scoreboard. SPECTRA-style
projection is currently disfavored on runtime grounds. The next implementable
method should probably be a cheaper downstream-propagation safety metric, not
another local output/post-solve projection.

## 2026-05-22: PRISM-Q Two-Task Result And Faster Diagnostic Harness

Implemented first-pass PRISM-Q as `--intrinsic-target-purifier prism_q`.
The purifier keeps the direct relational/context-value map, builds a same-pass
innovation basis plus a local option/readout hazard basis, residualizes hazard
against innovation, then clips generic-key -> hazard singular modes while
preserving signal retention. This is still a cheap approximation to the 5.5 Pro
proposal: it uses captured downstream MLP-output rows and local top-k LM-head
contrasts, not exact `J_{l->r}` or attention V/O transport.

Verification:

```bash
python -m py_compile caic/intrinsic_surprise.py \
  scripts/minilang_write.py \
  scripts/minilang_intrinsic_continual.py \
  scripts/continual_benchmark_grid.py \
  tests/test_intrinsic_surprise.py
python -m pytest tests/test_intrinsic_surprise.py::test_prism_q_clips_generic_hazard_preserving_signal -q
python -m pytest tests/test_intrinsic_surprise.py -q
# 79 passed
```

Full two-task PRISM-Q, loose budget:

```bash
python scripts/continual_benchmark_grid.py \
  --modal --run --preset prism_q \
  --tag prism_q_20260522_121758
```

Result:

- baseline: task0 `1/8`, task1 `1/8`;
- full-context teacher: task0 `8/8`, task1 `8/8`;
- after task0: task0 edited `1/8`;
- after task1: task0 edited `2/8`, task1 edited `2/8`;
- sentinel after task0: `7` correct-to-wrong, before-correct mean drop `6.607`;
- sentinel after task1: `9` correct-to-wrong, before-correct mean drop `7.712`;
- PRISM diagnostics: mean correction Frobenius `0.093`, mean hazard ratio
  `0.635`, signal retention `~1.0`.

Interpretation: loose PRISM was mostly inactive. Its spectral budget was often
larger than the measured hazard, so it left the unsafe direct map effectively
unchanged.

Full two-task PRISM-Q, strict budget:

```bash
python scripts/continual_benchmark_grid.py \
  --modal --run --preset prism_q_strict \
  --tag prism_q_strict2_20260522_131144
```

Result:

- after task0: task0 edited `1/8`;
- after task1: task0 edited `1/8`, task1 edited `1/8`;
- sentinel after task0: `5` correct-to-wrong, before-correct mean drop `5.862`;
- sentinel after task1: `7` correct-to-wrong, before-correct mean drop `6.368`;
- PRISM diagnostics: mean correction Frobenius `0.292`, mean hazard ratio
  `0.207`, signal retention `~1.0`.

Interpretation: strict PRISM did perform the intended hazard clipping, but it
still did not protect sentinels and it collapsed acquisition to baseline. That
falsifies this first PRISM approximation as the relevant safety coordinate.
Either the real downstream hazard is not captured by the cheap MLP-output /
top-k-logit basis, or the harmful component is not generic-key -> propagated
option evidence in the form tested here.

Runtime/harness update:

The full all-layer two-task benchmark is too slow for first-pass purifier
debugging. Added:

- `--early-stop-c2w-over`;
- `--early-stop-task0-min-edited-correct`;
- `prism_q_fast` preset in `scripts/continual_benchmark_grid.py`.

`prism_q_fast` uses 40 teacher-filter candidates, 4 eval questions, a
representative 7-layer band, and early-stops after task 0 when sentinel c2w is
nonzero or task-0 acquisition is below `1/4`. This is a diagnostic filter only:
final claims still require the full all-layer, no-sidecar, two-task +
expanded-sentinel benchmark.

Updated diagnosis:

PRISM-Q as implemented is not the missing object. The current evidence favors
Q-RICO/key16-style filtering as the safest available weight-only continual
baseline, despite low acquisition. The next move should focus on improving
acquisition without reintroducing the broad sentinel-damaging component, and it
should use fast diagnostics before promotion to the full two-task benchmark.

## 2026-05-22: TRACE-Q Local Approximation Failed Fast Gate

5.5 Pro proposed TRACE-Q: keep the Q-RICO/key16 scaffold, but build an
object-vs-ambient downstream tangent quotient. The intended full version uses
same-pass VJP rows through later frozen layers:

\[
L_{\ell,e}=C_e W_U J_{\mathrm{rms}}(h_e)J_{\ell\rightarrow e}.
\]

The implementation here is a cheap first approximation, not full TRACE-Q:

- added `--intrinsic-target-purifier trace_q`;
- used Q-RICO residual-filter (`deflate 4/4`, no layer trust) as the scaffold;
- selected object endpoints from high-weight relational rows;
- selected ambient endpoints from low-surprise, high-confidence same-pass
  positions;
- built local option contrast rows from endpoint top-k LM-head rows;
- built object and ambient residual bases from those local contrast rows;
- residualized generic keys against object keys;
- applied an object-predominant target projector plus a two-sided
  generic-key -> ambient-contrast collateral shrink.

Verification:

```bash
python -m py_compile caic/intrinsic_surprise.py \
  scripts/minilang_write.py \
  scripts/minilang_intrinsic_continual.py \
  scripts/continual_benchmark_grid.py \
  tests/test_intrinsic_surprise.py
python -m pytest tests/test_intrinsic_surprise.py -q
# 39 passed
```

Added fast presets:

- `qrico_key16_fast`: Q-RICO control on the same reduced fixture;
- `trace_q_fast`: projector + collateral TRACE-local;
- `trace_q_projector_fast`: target projector only;
- `trace_q_collateral_fast`: no target projection, stronger collateral shrink.

All use 40 teacher-filter candidates, 4 eval items, layers
`4,8,12,16,20,24,27`, and early-stop after task 0 if c2w is nonzero or task0
does not reach at least `1/4`.

### Fast Diagnostic Results

All runs used the same reduced split:

- sentinel before: `12/25`, mean margin `0.712`;
- Lyran baseline `1/4`, context `4/4`;
- Vomar baseline `1/4`, context `4/4`.

| Preset | Task0 edited | Task0 delta | Sentinel c2w | w2c | Before-correct drop | Sentinel acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qrico_key16_fast` | `2/4` | `+1/4` | `3` | `1` | `2.999` | `-0.08` |
| `trace_q_fast` | `1/4` | `0` | `2` | `6` | `1.117` | `+0.16` |
| `trace_q_projector_fast` | `1/4` | `0` | `3` | `6` | `1.489` | `+0.12` |
| `trace_q_collateral_fast` | `1/4` | `0` | `2` | `3` | `2.089` | `+0.04` |

TRACE diagnostics:

- `trace_q_fast`: mean measured collateral went from `0.0683` to
  `~7.6e-7`, but c2w remained `2` and acquisition fell to baseline.
- `trace_q_projector_fast`: no collateral correction, c2w remained `3`.
- `trace_q_collateral_fast`: collateral-only reduced c2w from Q-RICO's `3` to
  `2`, but still killed discrete acquisition.

Interpretation:

The local TRACE approximation is not a frontier move. It reproduces the
familiar tradeoff: reducing the measured local collateral coordinate lowers
sentinel damage somewhat but removes the threshold-crossing task acquisition.
The measured local object/ambient contrast quotient is therefore not the
missing safety coordinate.

This does **not** fully falsify TRACE-Q as originally stated, because the
original proposal required exact or semi-exact downstream tangent transport
through later frozen blocks. What is falsified is the cheap local endpoint
contrast version. The fast gate did its job: do not promote `trace_q` local to
the full all-layer two-task benchmark without implementing a materially better
`J_{\ell\rightarrow e}` transport.

Current practical baseline:

- Q-RICO/key16 remains the only scaffold with a known safe single-task frontier
  (`5/20`, c2w `0`, drop `0.964` on the earlier full single-task split);
- on the reduced two-task fixture it is acquisition-positive but unsafe;
- local downstream-ish filters keep failing by deleting acquisition before
  they solve c2w.

Next research implication:

The fastest useful ladder is now:

1. keep `qrico_key16_fast` as the reduced-fixture control;
2. only test new safety coordinates on the fast fixture first;
3. require `>=2/4` task0, `0` c2w, and drop `<1.0` before a full run;
4. if revisiting TRACE, implement actual downstream VJP/frozen-attention
   transport rather than another local LM-head contrast sketch.

## 2026-05-22: TDMI-Q Hidden-Manifold Row Weighting Failed Fast Gate

5.5 Pro proposed TDMI-Q: keep the Q-RICO/key16 high-rank residual scaffold, but
score each selected relational row by where its proposed effect is transported
in the same-pass hidden trajectory. The intended full version uses exact VJP
rows through the frozen downstream stack. The implementation here is a fast
hidden-manifold proxy, not full exact transport:

- added `--intrinsic-target-purifier tdmi_q`;
- computed a preliminary Q-RICO residual-filter update;
- row effect: `u_i = k_i @ update.T`;
- object basis from high-weight selected targets, row effects, current hidden
  states at selected object tokens, and optional downstream captured hidden
  states at the same tokens;
- ambient/default basis from low-surprise same-pass hidden states plus
  low-norm downstream captured hidden rows;
- ambient basis residualized against the object basis;
- row trust:
  `floor + (1-floor) * sigmoid((log signal - log ambient - threshold) / temp)`;
- reweighted the original positive row weights by TDMI trust, recomputed the
  protected update, then ran Q-RICO residual filtering with the TDMI weights;
- added fast presets `tdmi_q_fast` and `tdmi_q_local_fast`.

Verification:

```bash
python -m py_compile caic/intrinsic_surprise.py \
  scripts/minilang_write.py \
  scripts/minilang_intrinsic_continual.py \
  scripts/continual_benchmark_grid.py \
  tests/test_intrinsic_surprise.py
python -m pytest tests/test_intrinsic_surprise.py::test_tdmi_q_trusts_object_transport_over_default_manifold -q
# 1 passed
```

All TDMI runs used the same reduced split as the TRACE fast gate:

- sentinel before: `12/25`, mean margin `0.712`;
- Lyran baseline `1/4`, context `4/4`;
- Vomar baseline `1/4`, context `4/4`;
- layers `4,8,12,16,20,24,27`;
- weak input/output protection stack `256/20`, `256/10`;
- no old-key negatives, old atoms, Fisher sidecar, or external state.

### Fast Diagnostic Results

| Preset | TDMI settings | Task0 edited | Task0 delta | Sentinel c2w | w2c | Before-correct drop | Sentinel acc delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qrico_key16_fast` | control | `2/4` | `+1/4` | `3` | `1` | `2.999` | `-0.08` |
| `tdmi_q_fast` | future hidden rows, threshold `0`, floor `.15` | `3/4` | `+2/4` | `3` | `1` | `2.404` | `-0.08` |
| `tdmi_q_local_fast` | current-layer only, threshold `0`, floor `.15` | `3/4` | `+2/4` | `1` | `3` | `1.725` | `+0.08` |
| `tdmi_q_tuned_local_strict` | current-layer only, threshold `1.0`, floor `.05`, temp `.35` | `1/4` | `0` | `1` | `2` | `1.012` | `+0.04` |
| `tdmi_q_tuned_local_hard` | current-layer only, threshold `1.5`, floor `.02`, temp `.35` | `1/4` | `0` | `1` | `2` | `1.269` | `+0.04` |
| `tdmi_q_tuned_future_strict` | future hidden rows, threshold `1.0`, floor `.05`, temp `.35` | `2/4` | `+1/4` | `2` | `2` | `1.792` | `0.00` |

TDMI diagnostics:

- `tdmi_q_fast` kept mean signal fraction `0.956` and ambient fraction `0.773`;
- `tdmi_q_local_fast` kept mean signal fraction `0.900` and ambient fraction
  `0.673`;
- strict local TDMI reduced ambient kept fraction to `0.492`, but acquisition
  collapsed to baseline and c2w remained `1`;
- hard local TDMI reduced ambient kept fraction to `0.369`, but again stayed
  at baseline with c2w `1`.

Interpretation:

TDMI hidden-manifold row weighting found a more acquisition-positive row
ranking than local TRACE: loose TDMI reached `3/4` on task0. But the ranking is
not sentinel-safe. Tightening the trust gate behaves like previous shrink
coordinates: acquisition disappears before the last c2w is removed. The future
hidden-state proxy was worse than the local-only proxy on this split, which is
evidence against the current cheap transport approximation.

This does not falsify exact downstream VJP transport. It does falsify the
implemented TDMI proxy as a fast frontier move. Do not promote `tdmi_q` to the
full benchmark without replacing the proxy with materially more faithful
transport, and do not treat hidden-state row weighting alone as the missing
allocation mechanism.

Current implication:

- Q-RICO/key16 remains the practical single-task safe frontier.
- On the reduced two-task gate, every acquisition-positive method so far is
  sentinel-unsafe.
- Every safety coordinate that reduces c2w toward zero has collapsed task0 back
  to baseline on this fast split.
- The next useful proposal should either implement exact downstream transport
  with a clear causal falsifier, or shift away from row/atom/output-filtering
  toward an allocation mechanism that preserves the threshold-crossing
  component without using old sidecar state.

## 2026-05-22: DICE Diverse Invariant Context Ensemble

User hypothesis: the earlier ensemble experiments mostly averaged multiple
translation-lesson renderings, so the invariant across contexts was not just
the invented language. It also included "translation lesson / answer posture."
The stronger test is to create many intentionally different context worlds
whose only designed common factor is the same underlying mini-language, then
keep only update coordinates that recur across those diverse contexts.

Implementation:

- added `render_task_lesson_variant(...)` with 10 distinct surface frames:
  ordinary lesson, field note, cipher memo, stage directions, recipe notes,
  game rules, dialogue notes, map legend, correction sheet, and story caption
  key;
- added `--dice-diverse-contexts N` to the continual runner, replacing
  progressive same-format lessons with `N` diverse renderings of the final
  task lesson;
- added `--dice-defer-apply` to collect per-context proposal updates without
  applying them immediately;
- added `dice_support_consensus_update(...)`, which computes coordinate-level
  sign support across proposal updates and applies a high-support gate:

  \[
  M_\mathrm{dice}=\mathrm{mean}(M_c)\cdot
  \sigma((\mathrm{support}-\tau)T)
  \cdot \min(\exp(\alpha(\mathrm{support}-\tau)), c_\max).
  \]

- added fast presets:
  - `dice_qrico_fast`: 8 diverse contexts, support threshold `.75`;
  - `dice_qrico_strict_fast`: 12 diverse contexts, support threshold `.875`;
  - manual 12-context sweeps at thresholds `.75`, `.80`, and boosted `.875`.

Verification:

```bash
python -m py_compile scripts/minilang_write.py \
  scripts/minilang_intrinsic_continual.py \
  scripts/minilang_continual_triangle.py \
  scripts/continual_benchmark_grid.py \
  tests/test_minilang.py
python -m pytest tests/test_minilang.py -q
# 12 passed
```

All DICE runs used the reduced two-task fast fixture:

- Qwen/Qwen3-1.7B;
- two tasks: Lyran then Vomar;
- 4 teacher-filtered eval questions per task from 40 candidates;
- layers `4,8,12,16,20,24,27`;
- `relational_aggregate`, context-value, final-aligned;
- Q-RICO residual-filter scaffold (`deflate 4/4`, no layer trust);
- weak protection stack `256/20` input, `256/10` output;
- no old-key negatives or sidecar state.

### Fast Diagnostic Results

| Preset | Contexts | Support threshold | Gate mean | Task0 after task0 | Task1 after task1 | Sentinel c2w after task0 | Sentinel c2w after task1 | Before-correct drop after task1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qrico_key16_fast` | same-format | n/a | n/a | `2/4` | n/a | `3` | n/a | n/a |
| `dice_qrico_fast` | `8` | `.75` | `0.334` | `1/4` | n/a | `2` | n/a | n/a |
| `dice_qrico_relaxed12` | `12` | `.75` | `0.276` | `1/4` | n/a | `2` | n/a | n/a |
| `dice_qrico_mid` | `12` | `.80` | `0.177` | `1/4` | `3/4` | `0` | `2` | `0.492` |
| `dice_qrico_strict_fast` | `12` | `.875` | `0.076` | `1/4` | `3/4` | `0` | `0` | `0.126` |
| `dice_qrico_boosted` | `12` | `.875` | `0.082` | `1/4` | `3/4` | `0` | `1` | `0.222` |

Notes:

- `dice_qrico_strict_fast` completed the reduced two-task run with zero
  sentinel c2w after both tasks and very small before-correct drop
  (`0.104` after task0, `0.126` after task1).
- However, task0 did not improve over baseline: baseline `1/4`, edited `1/4`.
- Task1 did improve: baseline `2/4`, edited `3/4`, context `4/4`.
- Relaxing the support threshold to `.80` preserved task1 acquisition but
  reintroduced c2w after task1.
- Relaxing to `.75` failed at task0 with c2w `2`.
- Boosting strict high-support coordinates increased update norm modestly but
  reintroduced one c2w after task1 without improving task0.

Interpretation:

DICE is qualitatively different from TRACE/TDMI/PRISM. It can almost eliminate
sentinel churn on the reduced two-task fixture without old-key protection or
sidecar state, and it preserves task1 acquisition in the strict setting. The
tradeoff is that the first task under-acquires: the high-support intersection
appears to remove most of the threshold-crossing Lyran component along with the
translation/posture contaminant.

So the user hypothesis is partially validated:

- deliberate context diversity plus high-support filtering strongly suppresses
  unsafe broad movement;
- more contexts make accidental commonalities less likely to survive;
- but the linearly common coordinate across diverse worlds is still too weak
  for reliable task0 acquisition on this fast fixture.

This is not the same failure as the local safety filters. DICE is not merely
detecting a bogus hazard coordinate; it is selecting a very small invariant
intersection. The next DICE-specific work should avoid blunt coordinate
intersection and instead align proposals in a semantic/key-conditioned basis:

1. support over relational rows or Q-RICO residual modes rather than raw matrix
   coordinates;
2. anti-support contexts that share translation format but use a different
   language, to explicitly subtract translation posture;
3. context clusters rather than all-context unanimity, so lexical bindings that
   appear in different surface circuits can still contribute;
4. a final scale sweep only after a support coordinate passes task0 with zero
   c2w.

Current status:

DICE is promising as a safety/consolidation idea but not yet a new frontier.
It deserves one more targeted implementation if we can define support in a
better coordinate system. Raw coordinate high-support is too conservative.

## 2026-05-22: DICE Coordinate Follow-Up - SVD Support And Rival Anti-Support

Follow-up question: if raw matrix-coordinate support is too conservative, can
DICE recover acquisition by voting in a better coordinate, or by explicitly
subtracting translation-posture support from rival-language contexts?

Implemented two variants:

1. `--dice-support-space svd`
   - stack diverse-context proposal updates;
   - compute SVD of the proposal matrix;
   - score sign support over shared proposal coefficients instead of raw
     weight entries;
   - reconstruct only gated high-support modes.

2. rival-language anti-support
   - added `--dice-anti-contexts`;
   - render same-style diverse contexts for a different invented language;
   - build anti-proposal updates during the same write construction;
   - suppress positive coordinates whose common sign also appears in the
     anti-support proposals;
   - no anti-support state is stored after the write.

Added presets:

- `dice_qrico_svd_fast`;
- `dice_qrico_svd_strict_fast`;
- `dice_qrico_anti_fast`;
- `dice_qrico_anti_strict_fast`.

Verification:

```bash
python -m py_compile scripts/minilang_write.py \
  scripts/minilang_intrinsic_continual.py \
  scripts/continual_benchmark_grid.py \
  tests/test_minilang.py
python -m pytest tests/test_minilang.py -q
# 14 passed
```

### Fast Results

Same reduced two-task fixture as the prior DICE sweep.

| Preset | Support space | Anti contexts | Task0 after task0 | Task1 after task1 | Sentinel c2w after task0 | Sentinel c2w after task1 | Drop after task1 | Mean update Fro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dice_qrico_strict_fast` | raw coordinate | `0` | `1/4` | `3/4` | `0` | `0` | `0.126` | `0.912` |
| `dice_qrico_svd_fast` | proposal SVD rank 6 | `0` | `1/4` | n/a | `2` | n/a | n/a | `2.308` |
| `dice_qrico_svd_strict_fast` | proposal SVD rank 4 | `0` | `1/4` | n/a | `2` | n/a | n/a | `2.042` |
| `dice_qrico_anti_fast` | raw coordinate | `8` | `1/4` | `0/4` | `0` | `0` | `0.048` | `0.418` |
| `dice_qrico_anti_strict_fast` | raw coordinate | `8` | `1/4` | `2/4` | `0` | `0` | `0.061` | `0.294` |
| `dice_anti_light` | raw coordinate | `8` | `1/4` | `0/4` | `0` | `2` | `0.468` | `0.858` |

SVD-support diagnostics:

- SVD rank 6 kept about `72.5%` proposal energy, gate mean `0.396`, update
  Frobenius `2.31`, and c2w `2`;
- SVD rank 4 kept about `57.7%` proposal energy, gate mean `0.317`, update
  Frobenius `2.04`, and c2w `2`;
- neither improved task0 over baseline.

Interpretation: shared proposal SVD modes are too broad. They recover update
mass, but the mass is exactly the unsafe component raw coordinate DICE was
removing. This does not solve the "support coordinate" problem.

Anti-support diagnostics:

- default anti-support drove gate mean to `0.067` / update Frobenius `0.418`;
- stricter anti-support drove gate mean to `0.031` / update Frobenius `0.294`;
- both were sentinel-clean but inert;
- light anti-support raised gate mean to `0.192` / update Frobenius `0.858`,
  but c2w returned after task1.

Interpretation: raw coordinate anti-support does subtract shared
translation/posture movement, but it also subtracts much of the useful mapping
signal. There is not an obvious scalar anti-support band in this raw
coordinate. It behaves as another safety shrinker.

Current DICE conclusion:

- The user's diverse-context idea is partially validated: high-support
  intersection and rival anti-support both strongly suppress sentinel damage.
- Raw matrix coordinates are the wrong support unit for acquisition.
- Proposal SVD modes are too broad and unsafe.
- Rival anti-support in raw coordinates is too blunt and inert.

Next DICE-specific move, if continuing this family:

Support and anti-support should be computed over **key-conditioned
input-output maplets**, not raw coordinates or global SVD modes. Concretely,
for each proposal, decompose the update by the selected relational key rows:

\[
e_{c,i}=k_{c,i}M_c,\qquad
\phi_{c,i}=(\mathrm{key\ cluster}, \mathrm{target/value\ direction}).
\]

Then support should count recurring key-target effects for the same lexical or
relation cluster across diverse contexts, while anti-support should subtract
effects that recur in rival-language contexts with different source tokens.
That is more expensive to implement but is the coordinate that matches the
hypothesis. Raw DICE has now done enough to show the idea is safety-relevant
but underpowered in weight-entry space.
