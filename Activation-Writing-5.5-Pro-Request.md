# Request For GPT-5.5 Pro: Global Coherence Energy For One-Pass Weight Updates

Date: 2026-05-23

Audience: GPT-5.5 Pro. I am giving you two files:

1. `RESEARCH_LOG.md`: the full research log, including all implemented attempts,
   metrics, falsifiers, and harness details.
2. This file: the current prompt/request. Please read the full research log
   first, then answer this prompt.

## 2026-05-23 Current Ask: Turn Global Working-Memory Coherence Into A Weight Update

Please treat this section as the current live request. The full research log is
attached separately; older DICE, TDMI-Q, TRACE-Q, PRISM-Q, SEAL-Q, and
purifier sections below are retained for continuity.

The new hypothesis is:

\[
\boxed{
\text{the thing to learn is not a feature, not even an isolated relation, but a
conditional deformation of the whole currently instantiated world-state.}
}
\]

Working memory is the model temporarily configuring its world model as if many
constraints were true at once. Learning should happen when this instantiated
state exposes a coherence failure: under the old weights, the whole state does
not settle cleanly. A purple giraffe is not surprising because "purple" fires
or because "giraffe" fires. It is surprising because ordinary street context,
animal identity, visual realism, causal expectation, and the purple-giraffe
conjunction do not jointly cohere under the current model.

So the desired update is not:

- write the active feature;
- write a feature delta;
- write a pairwise binding;
- write local relation surprise;
- write answer/readout movement;
- write a local logit-safety filtered target.

The desired update is closer to:

\[
\boxed{
\text{find the minimal weight update that makes the entire instantiated
context-state more self-consistent under the model, while not globally lowering
the energy of generic answer/posture/default states.}
}
\]

In the dog-in-room analogy, the model should not learn "dogs are likely" or
even naked "dog in room." It should learn that this particular room/world-state,
with this dog-shaped unexpected object in this relational position, is less
impossible than before. In the mini-language benchmark, the model should not
learn "translation mode." It should learn that this particular latent language
system makes the rule/example/use context cohere.

### Why This Explains The Failures

The research pattern fits this theory:

- safe-but-inert methods detect purified local surprise without writing the
  full coherence deformation;
- acquiring-but-unsafe methods write a real chunk of the coherence deformation,
  but include generic posture/task/readout state;
- local output filters fail because useful coherence updates must eventually
  touch readout, and harmfulness is a downstream/global property, not a local
  vector property;
- old-key protection suppresses new learning because it protects coordinates,
  not state coherence;
- raw DICE over diverse contexts is safety-relevant, but raw weight-coordinate
  agreement deletes the threshold component because the true object is not
  linearly aligned in raw entries.

The latest first probe, `wm_coherence`, was a row-level graph reweighting method:

- current graph nodes: same-pass residual outputs at selected relational rows;
- target graph nodes: outputs plus context-value targets;
- candidate graph nodes: outputs plus proposed update effects;
- row score: reduction in pairwise cosine-graph error, with a low-surprise
  default-state penalty.

It acquired task0 and was c2w-clean after task0, but failed sequentially:

| Preset | Task0 after task0 | c2w after task0 | Task0 after task1 | Task1 after task1 | c2w after task1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wm_coherence_fast` | `2/4` | `0` | `2/4` | `0/4` | `4` |
| `wm_coherence_strict_fast` | `2/4` | `0` | `2/4` | `0/4` | `4` |

Interpretation: row-level graph trust is still a reweighted unsafe actuator. It
is not the global coherence-energy update.

### Latest Local Update: TAG-CE Probe Implemented

Your latest proposal was `TAG-CE`: lift relational/context-value node targets
into an object graph edge field, settle that edge field, absorb generic
edge-pattern x posture/readout-value nuisance, then solve closed-form ridge over
edge rows.

I implemented the first fast probe:

- purifier: `--intrinsic-target-purifier tag_ce`
- core function: `tag_ce_purify_update(...)`
- fast preset: `tag_ce_fast`
- ablations: `tag_ce_schur_off_fast`, `tag_ce_no_graph_settle_fast`,
  `tag_ce_shuffle_edges_fast`
- tests: generic posture edge absorption and object-specific readout edge
  preservation

Verification so far:

- focused tests pass: `pytest tests/test_intrinsic_surprise.py -q` -> `43 passed`
- tiny MPS smoke path completed end-to-end:
  - one task, one lesson, two examples, one eval item, layer 27
  - context teacher `1/1`, edited `0/1`
  - sentinel c2w `0/10`
  - TAG-CE diagnostics: `object_energy_delta=86253.3`,
    `ambient_energy_after=232990.6`, `layer_scale=0.523`,
    `update_fro_after=0.462`

Interpretation: implementation is runnable and c2w-clean on the tiny smoke, but
it did not acquire. The ambient/object energy ratio suggests the first TAG-CE
layer veto or ambient penalty may be over-shrinking the actual update. The
proper reduced two-task CUDA fixture is still the next decisive run.

### Latest TAG-CE Safety Screen

I then ran a small 7-layer local MPS two-task screen. This screen is not
acquisition-informative because raw relational does not acquire either, but it
is useful for sentinel drift:

- two tasks, Lyran then Vomar;
- no teacher filtering;
- `2` lessons/task, `4` examples/lesson, `2` eval questions/task;
- layers `4 8 12 16 20 24 27`;
- core sentinel suite.

Results:

| Run | Task0 after task0 | c2w after task0 | drop after task0 | Task0 after task1 | Task1 after task1 | c2w after task1 | drop after task1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw relational | `0/2` | `1` | `2.289` | `0/2` | `0/2` | `2` | `3.565` |
| TAG-CE + centroid anchor | `0/2` | `0` | `0.276` | `0/2` | `1/2` | `0` | `0.676` |
| TAG-CE + centroid + raw node potential | `0/2` | `0` | `2.555` | `0/2` | `1/2` | `1` | `3.084` |
| TAG-CE + centroid + conditioned node potential | `0/2` | `0` | `2.547` | `0/2` | `1/2` | `1` | `3.591` |
| TAG-CE + centroid + lowfreq rank-4 node modes | `0/2` | `0` | `4.742` | `0/2` | `1/2` | `1` | `5.085` |

This suggests:

- the edge-field/Schur coordinate is safety-relevant;
- the centroid anchor is the safest current TAG-CE variant;
- a direct graph node-potential anchor is locally falsified, even after
  centering and norm capping, because graph potentials contain unstable
  graph-constant/low-frequency components;
- naive low-frequency graph Laplacian node modes are also locally falsified;
  smooth node modes still carry generic/default posture unless conditioned more
  strongly;
- the fast preset now defaults to centroid-anchor-only. `--tagce-potential-weight`
  and `--tagce-lowfreq-weight` remain only as ablation knobs.

### Latest Acquisition Gate: TAG-CE v1 Fails, DICE Anti-Support Works Locally

I then ran a one-task teacher-filtered local MPS gate where raw relational does
acquire, so this is acquisition-informative:

- one task: Lyran;
- teacher-filtered `4` eval items from `20` candidates;
- layers `4 8 12 16 20 24 27`;
- `6` lessons, `8` examples/lesson;
- core sentinel suite.

Results:

| Run | Baseline | Context | Edited | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw relational | `1/4` | `4/4` | `2/4` | `0` | `2.812` |
| Q-RICO residual filter | `1/4` | `4/4` | `2/4` | `1` | `2.947` |
| TAG-CE centroid, relaxed/no-veto | `1/4` | `4/4` | `0/4` | `0` | `5.836` |
| TAG-CE centroid, default/veto | `1/4` | `4/4` | `0/4` | `1` | `1.609` |
| TAG-CE centroid, Schur off | `1/4` | `4/4` | `1/4` | `2` | `6.777` |

So TAG-CE v1 is locally falsified as an acquisition method. Schur-off does not
restore acquisition, so the edge-lifted target itself is losing the
threshold-crossing component.

I then tested the user's diverse-context/DICE hypothesis on the unsafe direct
relational write:

| Run | Baseline | Context | Edited | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw standard lessons | `1/4` | `4/4` | `2/4` | `0` | `2.812` |
| DICE raw, `4` diverse contexts | `0/4` | `4/4` | `1/4` | `1` | `1.068` |
| DICE raw, `8` diverse contexts | `1/4` | `4/4` | `1/4` | `1` | `0.460` |
| DICE raw, `4` diverse + `4` rival-language anti contexts | `0/4` | `4/4` | `1/4` | `0` | `0.056` |

The DICE anti-support result is now the most promising local result:

- nonzero acquisition over baseline;
- zero sentinel c2w;
- near-zero sentinel margin drift;
- no sidecar state after the write; contexts are only used during current write
  construction.

I added a benchmark preset:

```text
dice_relational_raw_anti_fast
```

New live question: focus on the multi-context invariant-write direction unless
you see a strong reason not to. How do we make DICE anti-support into a stronger
closed-form semantic-invariance method that applies to unsafe acquisition-bearing
maps without raw-coordinate consensus deleting the threshold component?

### Hard Constraints

Please satisfy all constraints:

1. One lesson/context pass only for the write.
2. No labels, no null prompts, no sentinels, no held-out eval questions, no
   probes generated outside the write context.
3. Closed-form bounded linear algebra; no optimizer loop.
4. No runtime router, retrieval, adapter selection, or task ID.
5. Sequential benchmark must pass only updated weights between tasks. There can
   be no old-key bank, old transform memory, Fisher sidecar, stored contexts,
   harness state, or external protection object available to the next write.
6. Diagnostic ablations may use forbidden objects, but the proposed final
   mechanism cannot.
7. The target benchmark is the two-task learning + sentinel benchmark, not
   single-task accuracy.

### What I Want You To Figure Out

Please propose the next implementable mathematical tool that turns the global
coherence idea into a weight update.

The likely direction is a **map-level or trajectory-level coherence-energy
solve**, not another scalar row trust. A sketch of the kind of object I mean:

\[
E_{\text{obj}}(M)
=
\left\|
\mathcal{G}
\left(h_i + k_iM,\;h_j+k_jM\right)
-
\mathcal{G}^{\star}_{ij}
\right\|^2,
\]

where:

- \(h_i\) are same-pass residual/working-memory states;
- \(k_iM\) are proposed MLP-down effects;
- \(\mathcal{G}\) is a graph/coherence statistic over tokens/layers/relations;
- \(\mathcal{G}^{\star}\) is the target settled graph inferred from the same
  context, not from labels;
- default/posture energy is penalized by preserving low-surprise/default state
  graphs.

But please do not just accept this sketch. Improve or replace it if needed.

Answer in implementable detail:

1. What is the right state graph or coherence statistic?
   - pairwise cosine graph?
   - relation field?
   - cross-layer transport graph?
   - attention-mediated consistency graph?
   - predictive coding residual between rule/example/use tokens?

2. How do we infer the target settled/coherent graph from one forward pass
   without copying posture?

3. What is the exact closed-form objective for the update \(M\)?
   - Include shapes.
   - Include linearization if needed.
   - Include how to keep it computationally tractable.

4. How should generic/default/posture states be preserved under the no-sidecar
   sequential constraint?
   - The next task only sees current weights and its new context.
   - No stored old examples or old subspaces.

5. How does this differ from failed row-level `wm_coherence`, TRACE-local,
   TDMI-Q, Q-RICO, CORI, and DICE?

6. What first fast reduced-fixture experiments should I run?
   - Include exact pass bars.
   - Include ablations that distinguish "real global coherence update" from
     "smaller unsafe relational write."

7. What are the falsification criteria?

The core challenge:

\[
\boxed{
\text{make this particular instantiated world-state lower energy without
making generic answer/posture/default states lower energy.}
}
\]

Please focus on solving that.

## 2026-05-23 Update: DICE On Unsafe Learning-Causing Writes

This older section is retained for continuity. The current live request is the
global working-memory coherence ask above.

Hard constraint reminder: the sequential benchmark must pass only updated
weights between tasks. There can be no old-key bank, old transform memory,
Fisher sidecar, stored contexts, task router, harness state, or external
protection object available to the next write. Diagnostic controls can use
these objects, but final candidate methods cannot.

The latest user hypothesis was: DICE should be tried on interventions that
actually caused learning but were unsafe. The previous DICE runs mostly wrapped
Q-RICO, which is already conservative. So I added fast reduced-fixture presets
around the unsafe direct `relational_aggregate` context-value write, plus ORCA
`residual_only` screens.

### New Presets

- `relational_raw_fast`: direct unsafe relational/context-value control.
- `dice_relational_raw_fast`: 12 diverse contexts, support threshold `.80`.
- `dice_relational_raw_strict_fast`: 12 diverse contexts, threshold `.875`.
- `dice_relational_raw_screen_fast`: 6 diverse contexts, threshold `.67`.
- `dice_orca_residual_fast`: 12-context DICE around ORCA `residual_only`.
- `dice_orca_residual_screen_fast`: 4-context DICE around ORCA
  `residual_only`.

The ORCA-residual DICE runs were stopped for runtime: even the 4-context screen
was too slow in the current implementation because every proposal rebuilds the
ORCA basis. ORCA-DICE may still be conceptually relevant, but it needs proposal
caching or a cheaper residual-only path before it is a practical experiment.

### Raw-Relational DICE Results

Reduced two-task fixture:

- 40 teacher-filter candidates;
- 4 eval questions per task;
- representative layers `4,8,12,16,20,24,27`;
- expanded sentinels;
- early stop after task 0 if c2w > 0 or task0 edited < 1.

| Preset | Contexts | Support threshold | Task0 baseline | Task0 edited | c2w after task0 | before-correct drop | Mean update Fro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `relational_raw_fast` | `1` | n/a | `1/4` | `2/4` | `1` | `2.304` | `5.601` |
| `dice_relational_raw_fast` | `12` | `.80` | `1/4` | `0/4` | `0` | `0.240` | `1.897` |
| `dice_relational_raw_strict_fast` | `12` | `.875` | `1/4` | `0/4` | `0` | `0.159` | `1.492` |
| `dice_relational_raw_screen_fast` | `6` | `.67` | `0/4` | `0/4` | `1` | `0.469` | `2.631` |

Support diagnostics:

| Preset | Mean support | p99 support | Gate mean | High-support fraction | Proposal cosine mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dice_relational_raw_fast` | `0.0130` | `0.321` | `0.0020` | `0.0017` | `0.255` |
| `dice_relational_raw_strict_fast` | `0.0134` | `0.321` | `0.0013` | `0.0011` | `0.253` |
| `dice_relational_raw_screen_fast` | `0.0140` | `0.357` | `0.0050` | `0.0025` | `0.226` |

Interpretation:

Direct raw relational still proves the unsafe learner exists on this reduced
fixture: `1/4 -> 2/4`, but with `1` c2w and large sentinel margin damage. Raw
DICE around that same learner is a safety filter, not an invariant learner.
At 12 contexts it removes c2w but also deletes all threshold acquisition. At 6
contexts / lower threshold it still gets `0/4` and c2w comes back.

This sharpens the DICE diagnosis:

- diverse-context support is safety-relevant;
- raw update entries are not the invariant unit;
- SVD proposal modes are too broad and unsafe;
- rival anti-support in raw entries is too blunt and inert;
- applying raw-coordinate DICE to the unsafe learner confirms that the useful
  threshold component is not linearly common in raw matrix coordinates.

### Current Ask

Please propose the next implementable multi-context tool under the weight-only
sequential constraint.

I do **not** think the right answer is "more raw DICE contexts." The next
plausible direction is support over **key-conditioned behavioral/effect
maplets**, not raw entries or global SVD modes:

\[
\phi_{c,i}
=
\left(
\text{canonical source-key cluster},
\text{canonical target/effect direction},
k_{c,i}M_c
\right).
\]

Please be specific enough to implement: tensors, alignment procedure, support
score, anti-support score, closed-form solve or post-solve filter, diagnostics,
first runs, ablations, and falsification criteria.

Questions to answer:

1. What is the right maplet coordinate for diverse-context support, given that
   raw entries are too sparse and SVD modes are unsafe?
2. How should source-key/role clusters be canonicalized across intentionally
   different context worlds without using labels at write time?
3. How should rival-language anti-support be matched at the maplet level so it
   subtracts translation posture but not true lexical/grammar binding?
4. Should maplet DICE wrap raw relational/context-value proposals, ORCA
   residual-only proposals, or Q-RICO residual-filter proposals?
5. What cheap screening implementation avoids the current ORCA-DICE runtime
   problem?
6. What reduced-fixture pass bar should promote a method to the full two-task
   benchmark?
7. What ablation distinguishes "DICE learned the invariant language object"
   from "DICE just made a smaller update"?

## 2026-05-22 Update: DICE Coordinate Follow-Up

This older section is retained for continuity. The 2026-05-23 unsafe-write
DICE update above is the current live state.

After the first DICE sweep, I implemented the two obvious coordinate
follow-ups:

1. support over shared proposal SVD modes instead of raw weight coordinates;
2. rival-language anti-support contexts, to subtract translation/lesson posture
   that appears in same-format but different-language writes.

### SVD Support

Implementation:

- `--dice-support-space svd`;
- stack diverse-context proposal updates;
- take SVD of the proposal matrix;
- compute sign support over proposal coefficients;
- reconstruct only gated high-support modes.

Results on the reduced two-task fast fixture:

| Preset | Support coordinate | Rank | Task0 after task0 | Sentinel c2w after task0 | Drop after task0 | Mean update Fro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dice_qrico_strict_fast` | raw coordinate | n/a | `1/4` | `0` | `0.104` | `0.912` |
| `dice_qrico_svd_fast` | proposal SVD | `6` | `1/4` | `2` | `0.998` | `2.308` |
| `dice_qrico_svd_strict_fast` | proposal SVD | `4` | `1/4` | `2` | `0.875` | `2.042` |

Interpretation: proposal SVD support is too broad. It recovers update mass, but
the recovered mass is unsafe and still does not improve task0. This falsifies
plain proposal-SVD modes as the better DICE coordinate.

### Rival Anti-Support

Implementation:

- `--dice-anti-contexts`;
- render rival-language contexts during the same write construction;
- compute anti-proposal updates;
- suppress positive coordinates whose common sign also appears in the rival
  anti-support proposals;
- no rival context or anti-support state is stored after the write.

Results:

| Preset | Positive contexts | Anti contexts | Anti gate mean | Task0 after task0 | Task1 after task1 | Sentinel c2w after task0 | Sentinel c2w after task1 | Mean update Fro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dice_qrico_anti_fast` | `8` | `8` | `0.261` | `1/4` | `0/4` | `0` | `0` | `0.418` |
| `dice_qrico_anti_strict_fast` | `12` | `8` | `0.263` | `1/4` | `2/4` | `0` | `0` | `0.294` |
| `dice_anti_light` | `8` | `8` | `0.644` | `1/4` | `0/4` | `0` | `2` | `0.858` |

Interpretation: raw coordinate anti-support is a strong safety brake, but it
subtracts too much useful mapping signal. Making it lighter restores update
mass but c2w comes back. There is no obvious scalar anti-support band in raw
coordinates.

### Updated DICE Conclusion

The diverse-context idea is safety-relevant but still under-acquires:

- raw coordinate high-support is clean but too weak;
- proposal-SVD support is broader and unsafe;
- raw coordinate anti-support is cleaner but even weaker;
- lighter anti-support loses the safety benefit.

The next plausible DICE coordinate is not raw entries and not global SVD modes.
It should be **key-conditioned input-output maplets**: support over recurring
effects of selected relational keys into target/value directions, with
anti-support subtracting effects that recur under rival-language contexts.

Concretely, for each proposal context \(c\):

\[
e_{c,i}=k_{c,i}M_c,\qquad
\phi_{c,i}=(\text{key cluster},\text{target/value direction}).
\]

Support should count recurring key-target effects for the same lexical or
relation cluster across diverse contexts. Anti-support should subtract
key-target effects that recur in rival contexts with different source tokens.

### Current Ask

Please propose the next implementable DICE-style tool.

It must be specific enough to implement: tensors, objectives, closed-form
solve, required diagnostics, first runs, ablations, and falsification criteria.

Please address:

1. What is the right key-conditioned support coordinate, given that raw
   coordinates are too conservative and SVD modes are unsafe?
2. How should rival-language anti-support be matched to positive contexts so
   it subtracts translation posture but not true lexical/grammar binding?
3. Should maplet support wrap Q-RICO residual-filter proposals, or should it
   operate on raw relational/context-value proposals before Q-RICO?
4. What exact reduced-fixture pass bar should determine promotion to the full
   two-task benchmark?
5. What ablation would distinguish "DICE learned the invariant language object"
   from "DICE is just a smaller update norm"?

## 2026-05-22 Update: DICE Diverse Invariant Context Ensemble

This older section is retained for continuity. The coordinate follow-up above
is the current live state.

User hypothesis: previous ensemble attempts mostly combined multiple
translation-lesson renderings, so their shared direction still included
translation-answer posture. A stronger test is to create many deliberately
different context worlds whose only designed common factor is the same
mini-language, then keep only update coordinates that recur across those
contexts.

Implementation added:

- `render_task_lesson_variant(...)` with 10 distinct surface frames: ordinary
  lesson, field note, cipher memo, stage directions, recipe notes, game rules,
  dialogue notes, map legend, correction sheet, and story caption key;
- `--dice-diverse-contexts N`, replacing progressive same-format lessons with
  `N` diverse renderings of the final task lesson;
- `--dice-defer-apply`, which collects per-context proposal updates without
  applying them immediately;
- `dice_support_consensus_update(...)`, a coordinate-level sign-support
  ensemble:

```text
M_dice = mean(M_c)
       * sigmoid((support - threshold) * temperature)
       * min(exp(strength * (support - threshold)), cap)
```

Fast fixture:

- Qwen/Qwen3-1.7B;
- two tasks: Lyran then Vomar;
- 4 teacher-filtered eval questions per task from 40 candidates;
- layers `4,8,12,16,20,24,27`;
- `relational_aggregate`, context-value, final-aligned;
- Q-RICO residual-filter scaffold (`deflate 4/4`, no layer trust);
- weak protection stack `256/20` input, `256/10` output;
- no old-key negatives or sidecar state.

Fast results:

| Preset | Contexts | Support threshold | Gate mean | Task0 after task0 | Task1 after task1 | Sentinel c2w after task0 | Sentinel c2w after task1 | Drop after task1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qrico_key16_fast` | same-format | n/a | n/a | `2/4` | n/a | `3` | n/a | n/a |
| `dice_qrico_fast` | `8` | `.75` | `0.334` | `1/4` | n/a | `2` | n/a | n/a |
| `dice_qrico_relaxed12` | `12` | `.75` | `0.276` | `1/4` | n/a | `2` | n/a | n/a |
| `dice_qrico_mid` | `12` | `.80` | `0.177` | `1/4` | `3/4` | `0` | `2` | `0.492` |
| `dice_qrico_strict_fast` | `12` | `.875` | `0.076` | `1/4` | `3/4` | `0` | `0` | `0.126` |
| `dice_qrico_boosted` | `12` | `.875` | `0.082` | `1/4` | `3/4` | `0` | `1` | `0.222` |

Interpretation:

DICE is qualitatively different from the local hazard filters. Strict
high-support consensus nearly eliminates sentinel churn on the reduced
two-task fixture without old-key protection or sidecar state. However, it
under-acquires task0: baseline `1/4`, edited `1/4`. It does preserve a small
task1 gain: baseline `2/4`, edited `3/4`. Relaxing or boosting the gate
reintroduces c2w before it fixes task0.

The user hypothesis is partially validated:

- deliberate context diversity plus high-support filtering strongly suppresses
  unsafe broad movement;
- more contexts make nuisance coincidences less likely to survive;
- raw coordinate-level intersection is too conservative and loses much of the
  threshold-crossing language signal.

### Current Live State

Q-RICO/key16 remains the full single-task safe frontier, but DICE is the first
recent branch to make the reduced two-task sentinel behavior look clean without
old-task sidecar protection. It is not a new accuracy frontier because task0
does not acquire.

The next DICE-specific question is whether support should be computed in a
semantic/key-conditioned coordinate rather than raw matrix coordinates:

1. support over relational rows, Q-RICO residual modes, or key-target maplets;
2. anti-support contexts that share translation format but use a different
   language, to subtract translation posture explicitly;
3. cluster support rather than all-context unanimity, so lexical bindings that
   route through different surface circuits can still contribute;
4. a final scale/gain sweep only after a coordinate passes task0 with zero c2w.

### New Ask

Please propose the next implementable mathematical tool.

It must be specific enough to implement: tensors, objectives, closed-form solve,
required diagnostics, first runs, ablations, and falsification criteria.

Please address:

1. DICE raw coordinate support gives strong safety but weak task0 acquisition.
   What coordinate should support be computed in instead?
2. How should anti-support be constructed without violating the one-pass,
   no-probes/no-sidecar spirit? For example, can same-write rival-language or
   same-format contexts be used as contrastive construction data?
3. Should the next DICE version combine strict invariant support with Q-RICO’s
   known safe single-task scaffold, or should it operate on the higher-acquiring
   raw relational residual?
4. Is the task1-only gain in strict DICE likely real signal, task asymmetry, or
   just fast-fixture variance? What diagnostic would separate those?
5. What is the fastest diagnostic ladder before promoting any DICE variant to
   the full two-task benchmark?

## 2026-05-22 Update: TDMI-Q Hidden-Manifold Row Weighting Failed Fast Gate

This older section is retained for continuity. The DICE section above is the
current live state.

Your latest TDMI-Q direction was implemented in a cheap first-pass form and
tested on the reduced two-task + expanded-sentinel fast gate. It did **not**
pass the gate.

Important caveat:

This is not exact downstream-VJP TDMI-Q. It used same-pass hidden-state
transport proxies: current-layer and captured downstream hidden rows at object
and ambient token positions. It did not form exact `J_{l->e}`, exact RMSNorm or
attention VJPs, or a true frozen-stack tangent transport.

Implementation added:

- `--intrinsic-target-purifier tdmi_q`;
- preliminary Q-RICO residual-filter update;
- row effect `u_i = k_i @ update.T`;
- object basis from high-weight selected targets, row effects, current hidden
  states at selected tokens, and optionally downstream captured hidden states;
- ambient/default basis from low-surprise same-pass hidden states and low-norm
  downstream hidden rows;
- ambient basis residualized against the object basis;
- row trust
  `floor + (1-floor) * sigmoid((log signal - log ambient - threshold) / temp)`;
- reweighted row weights followed by the normal protected ridge + Q-RICO
  residual filtering path;
- fast presets `tdmi_q_fast` and `tdmi_q_local_fast`.

Verification:

```text
targeted TDMI synthetic test passed
py_compile passed for modified implementation files
```

### Fast Gate Setup

Same reduced fixture as the TRACE fast gate:

- Qwen/Qwen3-1.7B;
- two tasks: Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 4-question eval per task from 40 candidates;
- representative layers `4,8,12,16,20,24,27`;
- `relational_aggregate`, context-value, final-aligned;
- key feature top-k `16`;
- target scale `.10`;
- output/input weak stack `256/10`, `256/20`;
- expanded sentinel suite;
- early-stop after task0 if sentinel c2w is nonzero or task0 edited correct is
  below `1/4`;
- no old-key negatives, no old atoms, no Fisher/sketch sidecar state.

Fast split baseline:

- sentinel before: `12/25`, mean margin `0.712`;
- Lyran baseline `1/4`, context `4/4`;
- Vomar baseline `1/4`, context `4/4`.

### Fast Results

| Preset | TDMI settings | Task0 edited | Task0 delta | Sentinel c2w | w2c | Before-correct drop | Sentinel acc delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qrico_key16_fast` | control | `2/4` | `+1/4` | `3` | `1` | `2.999` | `-0.08` |
| `tdmi_q_fast` | future hidden rows, threshold `0`, floor `.15` | `3/4` | `+2/4` | `3` | `1` | `2.404` | `-0.08` |
| `tdmi_q_local_fast` | current-layer only, threshold `0`, floor `.15` | `3/4` | `+2/4` | `1` | `3` | `1.725` | `+0.08` |
| `tdmi_q_tuned_local_strict` | current-layer only, threshold `1.0`, floor `.05`, temp `.35` | `1/4` | `0` | `1` | `2` | `1.012` | `+0.04` |
| `tdmi_q_tuned_local_hard` | current-layer only, threshold `1.5`, floor `.02`, temp `.35` | `1/4` | `0` | `1` | `2` | `1.269` | `+0.04` |
| `tdmi_q_tuned_future_strict` | future hidden rows, threshold `1.0`, floor `.05`, temp `.35` | `2/4` | `+1/4` | `2` | `2` | `1.792` | `0.00` |

Diagnostics:

- loose TDMI was more acquisition-positive than TRACE-local: `3/4` task0;
- future hidden-row TDMI did not improve safety over the Q-RICO control
  (`3` c2w);
- local-only TDMI was less unsafe (`1` c2w), but still failed the zero-c2w
  gate;
- stricter local TDMI reduced the ambient kept fraction from `0.673` to
  `0.492` or `0.369`, but task0 collapsed to baseline and c2w remained `1`.

Interpretation:

TDMI hidden-manifold row weighting is informative but not a frontier move. It
found acquisition-positive rows, but the same row trust coordinate still does
not isolate sentinel-safe threshold-crossing updates. Tightening the trust gate
repeats the familiar pattern: acquisition disappears before c2w reaches zero.
The cheap future hidden-state transport proxy was worse than local-only on this
split.

This does **not** fully falsify exact downstream VJP transport. It does falsify
the implemented TDMI proxy as the next practical step.

### Current Live State

The current practical baseline remains Q-RICO/key16:

- known full single-task safe frontier: `5/20`, context captures `11/20`,
  centered cosine `0.578`, projection ratio `0.162`, sentinel c2w `0`,
  before-correct drop `0.964`;
- reduced two-task fast fixture: acquisition-positive (`2/4`) but unsafe
  (`3` c2w);
- loose TDMI improves fast acquisition to `3/4` but remains unsafe;
- strict TDMI/TRACE/PRISM-style filters collapse acquisition before solving
  c2w;
- two-task full benchmark remains the primary target.

The high-level goal and hard constraints are unchanged:

- one forward pass over the lesson/context;
- closed-form write;
- surprise/innovation/free-energy driven;
- all-layer compatible;
- no null prompts, quizzes, answer traces, labels, probes, SAE, RAG, router;
- no sidecar state across sessions after weights are written;
- primary benchmark is two-task continual learning plus sentinel preservation,
  not single-task frontier chasing.

### New Ask

Please propose the next implementable mathematical tool.

It must be specific enough to implement: tensors, objectives, closed-form solve,
required diagnostics, first runs, ablations, and falsification criteria.

Please address:

1. TDMI loose row weighting improves fast acquisition (`3/4`) but is unsafe;
   strict TDMI collapses acquisition before eliminating the last c2w. What
   allocation object should replace row trust?
2. PRISM strict, TRACE-local, and TDMI all reduced their measured hazards but
   did not solve real sentinel c2w. Is exact downstream tangent transport still
   worth implementing, or do these failures imply that transported readout
   quotients are the wrong family?
3. Should we now improve acquisition from the known safe Q-RICO full
   single-task point, or keep trying to purify acquisition-positive reduced
   two-task maps?
4. Under the no-sidecar rule, is robust two-task retention identifiable from
   only current weights and a new context pass? If not, state the minimal
   additional assumption that still respects merged-weight consolidation.
5. What is the fastest diagnostic ladder before promoting a method to the full
   two-task benchmark?

## 2026-05-22 Update: TRACE-Q Local Approximation Failed Fast Gate

This older section is retained for continuity. The TDMI-Q section above is the
current live state.

Your latest TRACE-Q direction was implemented in a cheap first-pass form and
tested on the reduced two-task + expanded-sentinel fast gate. It did **not**
pass the gate.

Important caveat:

This is not the exact downstream-VJP TRACE-Q you proposed. It used local
endpoint option contrasts from top-k LM-head rows as a cheap proxy. It did not
implement exact `J_{\ell\rightarrow e}`, exact RMSNorm/logit VJP, attention
V/O transport, or full frozen-stack tangent transport.

Implementation added:

- `--intrinsic-target-purifier trace_q`;
- Q-RICO residual-filter scaffold (`deflate 4/4`, no layer trust);
- object endpoints from high-weight relational rows;
- ambient endpoints from low-surprise, high-confidence same-pass positions;
- local option contrast rows from endpoint top-k LM-head rows;
- object/ambient residual bases;
- generic keys residualized against object keys;
- object-predominant target projector;
- two-sided generic-key -> ambient-contrast collateral shrink;
- fast presets:
  - `qrico_key16_fast`;
  - `trace_q_fast`;
  - `trace_q_projector_fast`;
  - `trace_q_collateral_fast`.

Verification:

```text
39 passed
```

### Fast Gate Setup

- Qwen/Qwen3-1.7B;
- two tasks: Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 4-question eval per task from 40 candidates;
- representative layers `4,8,12,16,20,24,27`;
- `relational_aggregate`, context-value, final-aligned;
- key feature top-k `16`;
- target scale `.10`;
- output/input weak stack `256/10`, `256/20`;
- expanded sentinel suite;
- early-stop after task0 if sentinel c2w is nonzero or task0 edited correct is
  below `1/4`;
- no old-key negatives, no old atoms, no Fisher/sketch sidecar state.

Fast split baseline:

- sentinel before: `12/25`, mean margin `0.712`;
- Lyran baseline `1/4`, context `4/4`;
- Vomar baseline `1/4`, context `4/4`.

### Fast Results

| Preset | Task0 edited | Task0 delta | Sentinel c2w | w2c | Before-correct drop | Sentinel acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qrico_key16_fast` | `2/4` | `+1/4` | `3` | `1` | `2.999` | `-0.08` |
| `trace_q_fast` | `1/4` | `0` | `2` | `6` | `1.117` | `+0.16` |
| `trace_q_projector_fast` | `1/4` | `0` | `3` | `6` | `1.489` | `+0.12` |
| `trace_q_collateral_fast` | `1/4` | `0` | `2` | `3` | `2.089` | `+0.04` |

Diagnostics:

- `trace_q_fast` drove its measured local collateral from mean `0.0683` to
  `~7.6e-7`, but task acquisition fell to baseline and `2` sentinel c2w
  remained.
- `trace_q_projector_fast` showed that the target projector alone does not
  remove c2w.
- `trace_q_collateral_fast` showed that collateral-only shrink reduces c2w a
  little, but still deletes discrete task acquisition.

Interpretation:

The cheap local TRACE approximation is not a frontier move. It repeats the
familiar pattern: reducing the measured local collateral coordinate lowers
sentinel damage somewhat, but removes the threshold-crossing task signal before
solving c2w.

This does **not** fully falsify the exact TRACE-Q idea, because exact
downstream transport remains unimplemented. It does falsify another local
LM-head/top-k option-contrast approximation as the relevant safety coordinate.

### Current Live State

The current practical baseline remains Q-RICO/key16:

- known full single-task safe frontier: `5/20`, context captures `11/20`,
  centered cosine `0.578`, projection ratio `0.162`, sentinel c2w `0`,
  before-correct drop `0.964`;
- reduced two-task fast fixture: acquisition-positive (`2/4`) but unsafe
  (`3` c2w);
- two-task full benchmark: weak acquisition/retention and old-key negatives
  preserve mainly by suppressing task1.

Local output filters, same-pass anchors, gauge sealing, anti-erasure,
object-span preservation, SPECTRA-style local tail/hazard clipping, first-pass
PRISM, and local TRACE have all failed or collapsed into safe/inert behavior.

The high-level goal and hard constraints are unchanged:

- one forward pass over the lesson/context;
- closed-form write;
- surprise/innovation/free-energy driven;
- all-layer compatible;
- no null prompts, quizzes, answer traces, labels, probes, SAE, RAG, router;
- no sidecar state across sessions after weights are written;
- primary benchmark is two-task continual learning plus sentinel preservation,
  not single-task frontier chasing.

### New Ask

Please propose the next implementable mathematical tool.

It must be specific enough to implement: tensors, objectives, closed-form solve,
required diagnostics, first runs, ablations, and falsification criteria.

Please address:

1. Given PRISM strict clipping and TRACE-local both reduced their measured
   hazards but did not solve real sentinel c2w, what safety coordinate should
   replace local LM-head/top-k option sketches?
2. Is exact downstream tangent transport worth implementing, or do these
   failures imply the direction should shift away from propagated readout
   quotients?
3. Should we improve acquisition from the safe Q-RICO baseline instead of
   trying to purify raw relational/high-rank residual maps?
4. Under the no-sidecar rule, is robust two-task retention identifiable from
   only current weights and a new context pass? If not, state the minimal
   additional assumption that still respects the spirit of merged-weight
   consolidation.
5. What is the fastest diagnostic ladder before promoting a method to the full
   two-task benchmark?

## 2026-05-22 Update: PRISM-Q Falsifier And New Ask

This older section is retained for continuity. The TRACE-Q section above is the
current live state.

Your latest PRISM-Q direction was implemented in a first-pass cheap form and
tested on the primary two-task + expanded-sentinel benchmark. It did not solve
the benchmark, and the results are informative.

Implementation added:

- `--intrinsic-target-purifier prism_q`;
- PRISM same-pass innovation basis;
- PRISM local option/readout hazard basis;
- hazard residualization against innovation;
- generic-key -> hazard singular-mode clipping;
- signal-retention backoff;
- `prism_q`, `prism_q_strict`, and `prism_q_fast` benchmark presets.

Important implementation caveat:

This was a cheap approximation to the full PRISM proposal. It used captured
downstream MLP-output rows and local top-k LM-head contrasts as the propagated
signal/hazard basis. It did **not** implement exact `J_{l->r}`, exact local
RMSNorm/logit VJP, or attention V/O transport.

Verification:

```text
79 passed
```

including the synthetic PRISM clipping test.

### Primary Two-Task Benchmark Setup

- Qwen/Qwen3-1.7B;
- two tasks: Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 8-question eval per task;
- all 28 layers;
- `relational_aggregate`, context-value, final-aligned;
- key feature top-k `16`;
- target scale `.10`;
- output/input weak stack `256/10`, `256/20`;
- expanded sentinel suite;
- no old-key negatives, no old atoms, no sidecar state.

Baselines:

- Lyran baseline `1/8`, context `8/8`;
- Vomar baseline `1/8`, context `8/8`.

### PRISM-Q Loose Budget

Flags:

- `--intrinsic-target-purifier prism_q`;
- `--prism-budget 0.25`;
- `--prism-correction-cap 0.35`;
- `--prism-signal-retention-min 0.90`.

Result:

- after task0: Lyran edited `1/8`;
- after task1: Lyran edited `2/8`, Vomar edited `2/8`;
- sentinel after task0: `7` correct-to-wrong, before-correct mean drop `6.607`;
- sentinel after task1: `9` correct-to-wrong, before-correct mean drop `7.712`;
- PRISM diagnostics: mean correction Frobenius `0.093`, mean hazard ratio
  `0.635`, signal retention `~1.0`.

Interpretation:

Loose PRISM was mostly inactive. Its spectral budget was often larger than the
measured hazard, so it left the unsafe direct map effectively unchanged.

### PRISM-Q Strict Budget

Flags:

- `--intrinsic-target-purifier prism_q`;
- `--prism-budget 0.02`;
- `--prism-correction-cap 1.00`;
- `--prism-signal-retention-min 0.85`.

Result:

- after task0: Lyran edited `1/8`;
- after task1: Lyran edited `1/8`, Vomar edited `1/8`;
- sentinel after task0: `5` correct-to-wrong, before-correct mean drop `5.862`;
- sentinel after task1: `7` correct-to-wrong, before-correct mean drop `6.368`;
- PRISM diagnostics: mean correction Frobenius `0.292`, mean hazard ratio
  `0.207`, signal retention `~1.0`.

Interpretation:

Strict PRISM did perform the intended hazard clipping, but it still did not
protect sentinels and it collapsed acquisition to baseline. That falsifies this
first PRISM approximation as the relevant safety coordinate. Either the real
downstream hazard is not captured by the cheap MLP-output / top-k-logit basis,
or the harmful component is not generic-key -> propagated option evidence in
the form tested here.

### Runtime/Harness Update

The full all-layer two-task benchmark is too slow for first-pass purifier
debugging. I added:

- `--early-stop-c2w-over`;
- `--early-stop-task0-min-edited-correct`;
- `prism_q_fast` preset in `scripts/continual_benchmark_grid.py`.

`prism_q_fast` uses:

- 40 teacher-filter candidates;
- 4 eval questions;
- 7 representative layers;
- early-stop after task0 if sentinel c2w is nonzero or task0 acquisition is
  below `1/4`.

This is only a diagnostic filter. Final claims still require the full
all-layer, no-sidecar, two-task + expanded-sentinel benchmark.

### Current Diagnosis

The current best weight-only continual baseline remains Q-RICO/key16-style
filtering, despite low acquisition. Local output filters, same-pass anchors,
gauge sealing, anti-erasure, object-span preservation, SPECTRA-style local
tail/hazard clipping, and first-pass propagated-hazard PRISM have all failed
or collapsed into safe/inert behavior.

The central failure now looks like:

\[
\text{the acquisition-bearing component is close to the damaging component,}
\]

but the damaging component is **not** captured by:

- global output PCs;
- local top-k LM-head option sketches;
- same-pass stable-margin anchors;
- generic-key x local-option leakage;
- destructive anti-parallel edits to existing down columns;
- cheap propagated MLP-output hazard bases.

The high-level goal and hard constraints are unchanged:

- one forward pass over the lesson/context;
- closed-form write;
- surprise/innovation/free-energy driven;
- all-layer compatible;
- no null prompts, quizzes, answer traces, labels, probes, SAE, RAG, router;
- no sidecar state across sessions after weights are written;
- primary benchmark is two-task continual learning plus sentinel preservation,
  not single-task frontier chasing.

Please propose the next implementable mathematical tool. The answer should be
specific enough to implement: tensors, objectives, closed-form solve, required
diagnostics, first runs, ablations, and falsification criteria.

In particular, please address:

1. Given PRISM strict clipping reduced the measured hazard ratio but did not
   reduce real sentinel c2w, what safety coordinate should replace it?
2. Should we improve acquisition from the safe Q-RICO baseline instead of
   trying to purify raw relational/high-rank residual maps?
3. Is the no-sidecar continual-learning constraint forcing an impossibility
   boundary? If yes, state the minimal additional assumption needed.
4. If previous writes cannot store old keys/atoms/Fisher/sketches and gauge
   marks distort future writers, what weight-only mechanism can make writes
   self-preserving?
5. What is the fastest diagnostic ladder before promoting a method to the full
   two-task benchmark?

## 2026-05-21 Update: SEAL-Q Falsifier And New Ask

This older section is retained for continuity. The PRISM-Q section above is the
current live state.

Your SEAL-Q proposal was implemented and tested. The mechanical symmetry works,
but the method does not solve the benchmark.

Implementation added:

- `--intrinsic-target-purifier seal_qrico`;
- Q-RICO residual-filter base;
- signed anti-erasure against current down-value columns;
- exact SwiGLU up/down gauge seal:

\[
U_j \leftarrow c_j U_j,\qquad D_j \leftarrow D_j/c_j.
\]

The repo uses additive memory wrappers for down-projection writes, so the gauge
seal correctly scales:

- up-projection row;
- frozen/base down column;
- additive memory down column;
- slot memory down columns.

Unit tests pass:

```text
76 passed
```

including function-preserving SwiGLU gauge tests, salience detection, signed
anti-erasure, canonical activation invariance, and SEAL-Q smoke tests.

### Primary Two-Task Benchmark Results

Setup:

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

Baselines:

- Lyran baseline `1/8`, context `8/8`;
- Vomar baseline `1/8`, context `8/8`.

#### Full SEAL-Q

Flags:

- gauge seal applied;
- canonicalized surprise;
- `eta_erase=2.0`;
- `eta_seal=0.05`;
- max scale `1.10`.

Result:

- step 0 Lyran: `0/8`, sentinel c2w `3`, before-correct drop `1.525`;
- step 1 Lyran: `1/8`;
- step 1 Vomar: `1/8`, sentinel c2w `3`, before-correct drop `2.414`.

Diagnostics:

- mean anti-erasure ratio `0.065`;
- update retention after anti-erasure `0.989`;
- mean seal scale `1.00022`, max `1.05127`;
- mean scaled channels/layer update `92.9`.

Interpretation: full SEAL-Q is worse than Q-RICO. It loses acquisition and
still damages sentinels.

#### Anti-Erasure Only

Flags:

- no gauge seal;
- no canonical surprise;
- `eta_erase=2.0`.

Result:

- step 0 Lyran: `2/8`, sentinel c2w `2`, before-correct drop `0.468`;
- step 1 Lyran: `2/8`;
- step 1 Vomar: `1/8`, sentinel c2w `2`, before-correct drop `0.893`.

Interpretation: anti-erasure alone is the least bad SEAL-Q variant, but still
fails: task 1 does not acquire and sentinel c2w is nonzero.

#### Gauge-Only

Flags:

- gauge seal applied;
- canonicalized surprise;
- `eta_erase=0.0`.

Result:

- step 0 Lyran: `2/8`, sentinel c2w `3`, before-correct drop `1.533`;
- step 1 Lyran: `2/8`;
- step 1 Vomar: `1/8`, sentinel c2w `3`, before-correct drop `2.229`.

Interpretation: the gauge seal/canonicalization path is unsafe even without
anti-erasure. Since the gauge transform is function-preserving at application
time, the harm likely comes from how the next write interprets the sealed
geometry, or from the canonicalized surprise coordinate overselecting
readout-sensitive channels.

### Updated Diagnosis

SEAL-Q is falsified in its current form.

What survived:

- exact SwiGLU gauge sealing is mechanically valid;
- salience is visible from weights;
- signed anti-erasure removes the intended destructive-parallel component;
- the mechanism obeys the no-sidecar-state rule.

What failed:

- no variant acquired task 1 above baseline;
- no variant reached zero sentinel c2w;
- full SEAL-Q erased useful task-0 acquisition;
- gauge-only worsened sentinel safety;
- anti-erasure-only improved margin drop but not enough.

The remaining sentinel failure is likely not primarily destructive erasure of
already load-bearing down columns. It is probably additive false evidence or
mode/readout injection into generic option directions. Also, up/down gauge
imbalance may be a bad consolidation mark because it contaminates or distorts
the writer's future surprise coordinates.

### New Ask

Given the hard no-sidecar-state constraint, what is the next implementable
mathematical tool?

Please do not propose old-key negatives, old-transform atoms, Fisher/EWC
state, routers, adapters, or any other persistent state outside weights except
as diagnostics. The next context gets only updated weights and one new forward
pass.

We need a method that can improve the two-task benchmark:

\[
\text{task0 immediate} > \text{base},\quad
\text{task0 after task1} \ge \text{task0 immediate},\quad
\text{task1 immediate} > \text{base},\quad
\text{sentinel c2w}=0.
\]

Please reason from the new SEAL-Q failure. In particular:

1. If function-preserving gauge imbalance is the wrong weight-only mark, what
   alternative mark or mechanism could make previous writes self-protecting
   without changing the model's future surprise coordinate destructively?
2. If the unsafe component is additive false evidence rather than erasure, what
   closed-form one-pass metric should identify and remove it while preserving
   task acquisition?
3. Is there an identifiability/impossibility result here under the no-sidecar
   constraint, and if so what extra assumption is minimally necessary?
4. Should we return to high-rank residual-map purification, but with a
   weight-only protection rule that does not rely on stored old transforms?

## Short Version

We are trying to build an artificial synaptic consolidation rule for transformers:

> Run one forward pass over a context/lesson/conversation. From that pass plus the current weights, compute a closed-form weight update that internalizes the surprising reusable understanding into the model, without damaging unrelated capabilities.

The current state is no longer vague:

- A one-pass write signal exists.
- The only current target family that clearly moves behavior toward the full-context teacher is the `relational_aggregate` context-value target.
- That target is unsafe: it still carries broad readout/posture/mode movement and flips unrelated sentinel items at useful acquisition scale.
- WICR, CORI, and STAR are safer, but they are behaviorally inert or misoriented.
- Stronger global output protection reduces margin damage but does not remove the same correct-to-wrong sentinel failure.

The narrowed blocker is:

> We need a one-pass, closed-form, row/component-level purification of the relational/context-value actuator that preserves its context-teacher score-space alignment while removing sentinel-sensitive and generic readout/posture components.

Please do not propose another unrelated key selector unless you can explain why the current evidence implies key selection is still the main blocker. The strongest evidence says target/component purification is the blocker.

## Hard Constraints

Please treat these as hard constraints unless you give a rigorous impossibility argument.

1. **One forward pass at write time**

   At deployment/write time, we get exactly one forward pass through the actual context/lesson/conversation.

   Allowed:

   - current model weights;
   - activations from this one pass;
   - deterministic closed-form linear algebra over those weights/activations.

   Not allowed at write time:

   - null/default contrast prompts;
   - a second pass over "the same prompt without the lesson";
   - generated quizzes/probes/future questions;
   - teacher-forced answer traces;
   - heldout examples;
   - labels, DSL metadata, or hidden task structure;
   - next-token loss training/backprop;
   - empirical calibration passes collected at write time;
   - sentinel examples as write-time training data;
   - stored activations, keys, examples, atoms, Fisher/EWC metrics, task
     summaries, or other sidecar state from previous context sessions.

2. **Closed-form update**

   The write should be a closed-form update, or a small bounded number of closed-form linear algebra operations. No optimizer loop.

3. **Surprise-driven**

   The update should be driven by surprise, prediction error, free energy, innovation, or an equivalent weight-induced mismatch. Raw activation norm is not enough.

4. **No runtime router as the core answer**

   We have tried behavior gates, object gates, density-ratio routers, and related mechanisms. They are useful diagnostics, but the final primitive should not be "store a memory and route to it." The locality should come from the weight update geometry itself.

5. **No RAG or sidecar document memory**

   The update must merge into weights or an always-active global low-rank plastic substrate. It cannot retrieve the original context or examples at inference time.

6. **No persistent per-context or cross-session memory outside weights**

   This is now a hard constraint. After reading a context and writing the
   weights, the next context/session receives only the updated model weights.
   It does not receive stored selected keys, old-task negatives, low-rank
   atoms, old transformation metrics, Fisher/EWC sketches, task IDs, summaries,
   routers, adapters, or any other sidecar consolidation state from previous
   sessions.

   Continual learning must therefore be achieved by the weight update itself
   and by the geometry of future closed-form writes computed from the current
   weights plus the new single forward pass. A proposal may use old-key or
   old-transformation storage only as a diagnostic ablation, not as the final
   method.

7. **No SAE requirement**

   Sparse autoencoders are too expensive for the deployed method. If a feature basis is needed, derive it cheaply from existing weights and the single pass.

8. **All-layer compatibility**

   The rule must be safe to apply across all layers. It may emit near-zero updates in layers where the evidence is weak, but it cannot rely on a hand-picked late layer.

9. **General, not translation-specific**

   The mini-language benchmark is a microscope, not the goal. The same principle should plausibly apply to:

   - user identity learned from conversation;
   - new object-property facts;
   - fictional worlds;
   - scientific mechanisms;
   - legal/procedural rules;
   - math notation;
   - compositional relations.

10. **Continual learning**

   The long-term goal is sequential learning:

   - learn context 1;
   - learn context 2;
   - ...
   - learn context 6;
   - previous learned contexts should not degrade below their immediate post-write scores;
   - sentinel/general capabilities should not degrade.

11. **Safety is not only accuracy**

   We track sentinel:

   - correct-to-wrong flips;
   - wrong-to-correct flips;
   - preservation rate;
   - mean margin drift;
   - before-correct margin drop;
   - severe margin drop count.

   A method that gains task accuracy but weakens unrelated answers is not acceptable.

## Human/Neuroscience Intuition

The target is not "fine-tune on a lesson." It is closer to predictive-coding-style synaptic consolidation.

The model's weights already encode a world. A context temporarily configures that world into a specific object/relation state. If that state is normal under the model's prior, little should be written. If that state contains a meaningful surprise, the responsible object/relation coordinates should receive a nonlinear write.

Examples:

- In a "purple giraffe" context, "purple" and "giraffe" are not new from scratch. The surprising object is the binding/configuration: giraffe instantiated with purple color.
- In a conversation with a user, the model should learn facts/preferences/projects about that user from the ordinary working-memory soup, not because a prompt says "now learn user profile."
- In a mini-language lesson, the model should update lexical mappings, tense, and role-binding relations, not just enter "translation answer mode."

The write should feel like:

> familiar context activity relaxes back to the prior and writes almost nothing; strong semantic mismatch writes a lot into the responsible latent object/relation coordinates.

But the latest experiments add a crucial correction:

> the useful acquisition component is close to readout/value movement, so simply projecting away all output-sensitive directions kills behavior or leaves the same sentinel failure. We need a local decomposition, not a global readout ban.

## Benchmark

The current measurable benchmark is a synthetic mini-language translation task.

The model reads lessons describing an invented language. Then it answers multiple-choice no-context translation questions.

Typical setting:

- model: `Qwen/Qwen3-1.7B`;
- lessons: 6;
- examples per lesson: 8;
- eval: 39-40 heldout multiple-choice translation questions, or a teacher-filtered 20-question subset;
- baseline no-context: usually `5-6/38-39`, or `0/20` on the teacher-filtered set;
- full-context teacher: usually `20-21/38-39`, or `20/20` on the teacher-filtered set;
- expanded sentinel: 25 unrelated MC questions.

The near-term target is not full context-teacher performance. A useful near-term target is around `8-12/20` equivalent partial acquisition, with:

- zero sentinel correct-to-wrong flips;
- low before-correct margin drop;
- context-aligned score movement;
- eventual sequential retention.

## Important Existing Results Before STAR

### Raw activation deltas and TSOC

Raw teacher-student activation deltas are downstream symptoms, not the causal source. They can move internal trajectories toward teacher states, but often write answer/style/prompt residue rather than reusable understanding.

The useful diagnosis was:

> write the missing source/operator, not the downstream activation symptom.

But source extraction alone did not solve behavioral readout or safety.

### Relational aggregate: real acquisition, unsafe

Prior best all-layer acquisition frontier:

- all 28 layers;
- one lesson forward;
- `relational_aggregate`;
- context-value target;
- target scale `.10`;
- exponential surprise weighting, temperature `2.0`, cap `20`;
- persistence power `1`;
- output/readout penalty rank/weight `256/10`;
- input penalty features/weight `1024/40`.

Result on teacher-filtered 20-question eval:

- baseline `0/20`;
- context `20/20`;
- edited `7/20`;
- captured context-only opportunities `7/20`;
- centered score-delta cosine with context teacher `0.589`;
- projection ratio onto context score movement `0.179`;
- mean correct-answer score gain on context opportunities `+0.222`;
- expanded sentinel improves in aggregate, but has `1` correct-to-wrong flip;
- before-correct sentinel margin drop around `3.847`.

Interpretation:

- This is the only current write family with clear context-teacher alignment.
- It is not just random mode movement.
- It is unsafe because the useful target is contaminated.

### Continual learning diagnostic: negative

Two-task all-layer continual diagnostic with the same family of lesson-only surprise writes:

No old-task protection:

- task 0 immediately after task 0: `2/8`;
- task 0 after task 1: `0/8`;
- task 1 after task 1: `2/8`;
- sentinel after task 1: `12/25 -> 9/25`, `8` correct-to-wrong flips.

Old selected keys as protected negatives:

- task 0 after task 1: `1/8`;
- task 1 after task 1: `0/8`;
- sentinel net `12/25 -> 12/25`, but still `6` correct-to-wrong flips.

Interpretation:

> Old-task protection reduces damage mainly by suppressing the new write. The upstream selected key/target coordinates are still broad/generic and overlap across tasks.

### Final-token-only: safe but weak

All-layer last-token-only:

- scale `.10`: baseline `6/39`, context `21/39`, edited `6/39`; sentinel `12/25 -> 16/25`, `1` correct-to-wrong.
- scale `.30`: edited `4/39`; sentinel `12/25 -> 17/25`, `0` correct-to-wrong.

Interpretation:

> The final token is safer, but too weak or too compressed. The useful signal is distributed over content/use positions.

### WICR and CORI

WICR: Weight-Induced Compatibility Residual.

- Same-token WICR is safe but weak.
- Attention-edge WICR is stronger but unsafe and still does not acquire.

Representative results:

| method | edited | context | sentinel | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| same-token WICR, scale `.03` | `4/39` | `21/39` | `12/25 -> 13/25` | `0` | `0.249` |
| attention edges + same-token, eager, `.03` | `4/39` | `20/39` | `12/25 -> 9/25` | `6` | `5.092` |
| attention edges only, eager, `.01` | `5/39` | `20/39` | `12/25 -> 10/25` | `2` | `1.157` |

CORI: Conditional Object-Relation Innovation.

It computes:

> actual feature-relation coupling minus weight-implied default coupling with the same feature marginals.

All-layer, no attention edges, p=`128`, rank=`16`, beta=`3`:

| method | edited | context | sentinel | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| CORI `svd_value`, scale `.10` | `6/39` | `21/39` | `12/25 -> 12/25` | `0` | `0.006` |
| CORI `svd_value`, scale `3.0`, pre-orientation | `2/39` | `21/39` | `12/25 -> 12/25` | `0` | `0.099` |
| CORI `svd_value`, scale `3.0`, oriented | `3/39` | `21/39` | `12/25 -> 11/25` | `1` | `0.404` |
| CORI `innovation_value`, scale `3.0`, oriented | `3/39` | `21/39` | `12/25 -> 13/25` | `0` | `0.044` |

At safe scale `.10`, detailed score analysis showed:

- baseline `6/39`;
- context `21/39`;
- edited `6/39`;
- predictions changed `0/39`;
- context-only opportunities `19`;
- captured `0/19`;
- edited score movement only about `0.37%` of context-induced score movement;
- mean cosine between edited score delta and context score delta `-0.067`.

Interpretation:

> CORI is safe because it is basically a no-op. Scaling does not recover acquisition. CORI may be a useful safety coordinate, but the target it writes is not the useful actuator.

## What We Tested After Your STAR Proposal

You proposed STAR: Schur-Transport Actuator Residual.

The idea:

- use CORI to obtain purified relation keys;
- construct same-pass future residual capsules;
- Schur-residualize nuisance/posture variables;
- write the future integrated residual computation statistically attributable to the key;
- return posture components as zero-target negatives.

We implemented this as `schur_transport_actuator`.

Files changed:

- `caic/intrinsic_surprise.py`
  - added STAR selector and same-pass future transport capsules;
  - added selector-provided negative keys;
  - added diagnostics.
- `scripts/minilang_write.py`
  - added `--intrinsic-surprise-target-mode schur_transport_actuator`;
  - threaded future layer states into STAR;
  - added STAR hyperparameters.
- `tests/test_intrinsic_surprise.py`
  - added STAR selector tests.

Validation:

- `.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q`
  - `22 passed`
- `.venv/bin/python -m pytest -q`
  - `63 passed`

### STAR score-space alignment

All STAR runs:

- all 28 layers;
- MLP down only;
- no attention edges;
- p=`128`;
- relation rank=`16`;
- beta=`3`;
- final-aligned token mode;
- exponential weights;
- persistence power `1`;
- output penalty rank/weight `256/10`;
- input penalty features/weight `1024/40`.

Common baseline:

- baseline `5/38`;
- full context `20/38`;
- expanded sentinel before `12/25`.

Results:

| run | scale | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STAR `.10` | `.10` | `5/38` | `0` | `0` | `0` | `0/19` | `0.035` | `0.000` | `+0.001` | `0` | `0.025` |
| STAR `1.0` | `1.0` | `3/38` | `2` | `0` | `2` | `0/19` | `-0.300` | `-0.004` | `-0.008` | `0` | `0.157` |
| STAR `3.0` | `3.0` | `3/38` | `2` | `0` | `2` | `0/19` | `-0.287` | `-0.013` | `-0.017` | `2` | `0.483` |

Interpretation:

- Safe STAR is behaviorally a no-op.
- Active STAR moves in the wrong direction: it loses baseline-correct items and captures no context-only opportunities.
- This is not just under-scaling. The direction is wrong.

### STAR shuffle ablations

We added deterministic ablation flags:

- `--star-shuffle-future-targets`
- `--star-shuffle-keys`

These roll target/key rows by one position after STAR row selection, preserving row distributions while breaking intended pairing.

Scale `1.0` results:

| run | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real STAR | `3/38` | `2` | `0` | `2` | `0/19` | `-0.300` | `-0.004` | `-0.008` | `0` | `0.157` |
| shuffled future targets | `5/38` | `0` | `0` | `0` | `0/19` | `-0.115` | `-0.002` | `-0.004` | `0` | `0.027` |
| shuffled keys | `5/38` | `1` | `0` | `0` | `0/19` | `0.035` | `0.001` | `-0.007` | `0` | `0.081` |

Update statistics:

| run | update Fro mean/median/max | target Fro mean/median/max | explained-ratio mean/median/max |
| --- | ---: | ---: | ---: |
| real STAR `1.0` | `0.0213 / 0.00510 / 1.001` | `29.23 / 15.26 / 416.28` | `0.162 / 0.138 / 0.490` |
| shuffled future `1.0` | `0.0341 / 0.00583 / 1.290` | `35.75 / 20.02 / 377.36` | `0.181 / 0.163 / 0.506` |

Interpretation:

- Shuffled future targets had equal or larger update norms but became a no-op.
- Therefore real STAR pairing matters, but it matters in the wrong direction.
- The failure is target orientation/causal relevance, not merely target size.

## New Relational Safety Frontier

Since STAR was misoriented, we went back to the target family that actually points toward the context teacher: `relational_aggregate` context-value.

We tested whether stronger global output protection can remove the sentinel damage.

### Stronger output metric penalty

Settings:

- all 28 layers;
- `relational_aggregate`;
- context-value target;
- exponential weights;
- persistence power `1`;
- input penalty `1024/40`;
- output penalty rank/weight increased from `256/10` to `1024/40`;
- target scale `.10`.

Result:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline relational `.10`, output `256/10` | `0/20` | `20/20` | `7/20` | `11` | `7` | `0` | `7/20` | `0.589` | `0.179` | `+0.222` | `1` | `3.847` |
| relational `.10`, output `1024/40` | `0/20` | `20/20` | `4/20` | `13` | `4` | `0` | `4/20` | `0.613` | `0.185` | `+0.199` | `1` | `2.355` |

Interpretation:

- Stronger output protection reduces broad margin damage.
- It does not remove the discrete sentinel correct-to-wrong failure.
- Acquisition drops from `7/20` to `4/20`.
- The edited score movement remains context-aligned.

### LM-head target projection plus strong output penalty

Settings:

- LM-head generic target projection rank `256`;
- output penalty rank/weight `1024/40`;
- target scale `.10`.

Result:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, LM projection `256` + output `1024/40` | `0/20` | `20/20` | `3/20` | `13` | `3` | `0` | `3/20` | `0.615` | `0.187` | `+0.193` | `1` | `2.558` |

Interpretation:

- Projecting targets away from a global LM-head basis reduces acquisition further.
- It still does not eliminate the sentinel c2w flip.
- Therefore the damaging component is not just a broad global LM-head/readout direction.

### Half-scale strong output penalty

Settings:

- output penalty rank/weight `1024/40`;
- target scale `.05`.

Result:

| run | base | context | edited | changed | gained | lost | captured context-only | centered cosine | projection ratio | opportunity answer score gain | sentinel c2w | sentinel before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| relational `.05`, output `1024/40` | `0/20` | `20/20` | `0/20` | `2` | `0` | `0` | `0/20` | `0.596` | `0.121` | `+0.137` | `0` | `1.126` |

Interpretation:

- Halving scale removes sentinel flips.
- It also removes discrete acquisition, though score movement remains context-aligned.
- This confirms a safety/acquisition frontier: the useful relational target is aligned, but the unsafe component is entangled at useful scale.

## Current Blunt Diagnosis

Please verify, refine, or refute this:

1. One-pass acquisition signal exists.
2. The current useful signal is carried by relational/context-value target rows.
3. Safe alternatives like CORI and STAR do not recover the useful actuator.
4. STAR specifically falsified "Schur-residualized future residual capsule" as the next target, because active STAR is anti-aligned with context teacher.
5. Global output/readout penalties and LM-head target projection are too blunt:
   - they reduce margin damage;
   - they lower acquisition;
   - they do not remove the same sentinel c2w failure at useful scale.
6. Therefore the missing object is a **local row/component-level decomposition** of the relational/context-value target into:
   - useful semantic/readout/composition actuator;
   - sentinel-sensitive component;
   - generic answer/posture/mode component.

The immediate research question is now:

> How do we purify the relational aggregate context-value target itself, using only weights and one context pass, so that the closed-form update retains context-teacher alignment while avoiding sentinel-sensitive collateral movement?

## What I Need From You

Please propose the next implementable mathematical tool.

Do not just say:

- "try more scale";
- "try more examples";
- "use a runtime gate";
- "go back to STAR";
- "project harder onto LM-head/output PCs";
- "use generated probes";
- "meta-train it later."

The request is not broad brainstorming. I need equations, tensor shapes, algorithmic steps, and diagnostic/falsification criteria.

## Specific Questions

### A. Is the current diagnosis right?

Is the blocker now row/component-level target purification of relational/context-value writes?

If not, explain exactly why the following pattern is better explained by something else:

- relational aggregate has strong context-teacher alignment and acquisition;
- relational aggregate damages sentinel margins and flips one unrelated answer at useful scale;
- stronger global output penalty reduces but does not fix damage;
- global LM-head target projection still leaves the c2w failure;
- CORI is safe but inert;
- STAR future target is safe at low scale and anti-aligned at active scale;
- old-task protection suppresses new acquisition rather than enabling continual learning.

### B. What is the correct local purification object?

We currently have rows:

\[
K \in \mathbb{R}^{n \times m}, \qquad R \in \mathbb{R}^{n \times d}
\]

from relational aggregate context-value selection:

- \(K\): sparse or relation-local MLP feature keys;
- \(R\): context-value residual targets from surprising paired native channels;
- \(m\): MLP width;
- \(d\): residual width.

The existing solve is roughly:

\[
\min_{\Delta W}
\|W_+^{1/2}(K\Delta W^\top - R)\|^2
+ \lambda \|\Delta W\|^2
+ \rho \|B\Delta W^\top\|^2
+ \gamma \|\Pi_{\text{out}}\Delta W\|^2.
\]

This is insufficient because \(R\) itself is contaminated.

Please define a purification:

\[
R = R_{\text{useful}} + R_{\text{posture}} + R_{\text{sentinel-risk}} + R_{\text{generic}}
\]

or an equivalent decomposition, using only:

- current weights;
- same-pass activations;
- relation keys/targets;
- local Jacobians or Fisher-like observability from the same pass;
- weight geometry.

What exactly are these subspaces or components?

How do we estimate them without sentinel examples, null prompts, labels, or extra probes?

### C. How do we detect sentinel-sensitive directions without sentinel examples?

The sentinel failure is not killed by global LM-head PCs. We need a more local risk metric.

Possible ingredients you may use:

- local residual-to-logit Jacobian on same-pass lesson tokens;
- model's own next-token distribution Fisher metric at lesson tokens;
- downstream block Jacobian norms;
- RMSNorm/unembedding sensitivity;
- MLP down-value observability;
- directions that cause large changes under generic high-confidence output states;
- "common computation" directions inferred from low-surprise rows in the same pass;
- directions with high fan-out through later layers;
- directions whose effect is not conditioned on relational keys;
- weight-induced compatibility degree or centrality.

Please propose a concrete risk metric stronger than global LM-head projection.

It must explain why output `1024/40` and LM projection `256` still leave the same c2w failure.

### D. How do we keep the readout actuator while removing posture?

The useful acquisition component appears close to readout/value movement:

- relational aggregate context-value target gets acquisition;
- output penalties reduce but do not safely separate it;
- STAR/CORI targets that avoid this readout component become inert or wrong.

So the answer cannot be "remove output-sensitive directions." Behavior must eventually touch logits.

Please define a conditional criterion like:

> keep output-sensitive directions only when they are locally attributable to the relational surprise key and not explainable by generic posture/mode variables.

But give the actual math:

- Schur complement?
- CCA/partial regression?
- generalized eigenproblem?
- constrained low-rank factorization?
- local observability quotient?
- key-conditioned Fisher decomposition?
- causal transport in block Jacobian coordinates?

### E. What closed-form solve should replace the current one?

Please define:

- key construction \(K\);
- purified target \(R_{\text{purified}}\);
- row weights;
- input protection;
- output protection;
- local sentinel-risk metric;
- all-layer trust rule;
- closed-form solution.

If the solution is no longer a simple ridge solve, give the exact objective and solution.

Examples of acceptable forms:

- generalized ridge;
- Sylvester equation;
- generalized eigen basis plus ridge;
- low-rank Schur complement solve;
- constrained least squares with closed-form KKT solution;
- natural-gradient update with diagonal/low-rank metrics.

### F. What should weight-only sequential protection be?

Old selected keys as negatives currently suppress new acquisition.

The obvious answer would be to store a compact continual metric that protects
old learned transformations. That is now explicitly disallowed. After a write,
the next session gets only the updated model weights, not old selected keys,
old activations, old atoms, Fisher/EWC sketches, task summaries, or old
key-to-value transformation records.

Please define a **weight-only** continual mechanism. It must protect previous
learning because that learning is already embedded in the weights, not because
a sidecar state remembers it.

Questions to answer:

- How can a future closed-form write infer, from current weights plus the new
  single-pass activations, which weight-space directions are already
  load-bearing from earlier writes?
- Is there a synaptic/homeostatic/curvature proxy computable from weights alone
  that prevents overwriting without storing old context traces?
- Can the update rule make previous writes "self-protecting" by changing the
  weight geometry itself, rather than maintaining an external protection state?
- If this is impossible under the constraints, give the identifiability
  argument and the weakest extra assumption that would make it possible.

Give the math under the no-sidecar-state constraint.

### G. What should we implement first?

Please give a concrete implementation plan that fits the repo:

- `caic/intrinsic_surprise.py`
  - selector and target purification logic;
- `caic/tsoc.py`
  - protected ridge/metric update utilities;
- `scripts/minilang_write.py`
  - single-context mini-language runner;
- `scripts/minilang_intrinsic_continual.py`
  - sequential diagnostic;
- `tests/test_intrinsic_surprise.py`
  - selector/unit tests.

Please provide:

- new mode name;
- tensor shapes;
- pseudocode;
- default hyperparameters;
- first three runs;
- expected pass/fail thresholds.

## Minimum Pass/Fail Criteria

Single-task bar:

- edited score above baseline;
- target: at least `8/20` eventually, but even `3-5/20` is meaningful only if sentinel is safe and score movement is context-aligned;
- zero sentinel correct-to-wrong flips;
- before-correct sentinel margin drop ideally below `1.0`;
- prediction changes on learned-task questions move toward context-teacher choices;
- score-space projection ratio meaningfully positive;
- not just broad answer-option/posture shifts.

Continual bar:

- task 0 after task 1 not below task 0 immediately after task 0, within one item tolerance;
- task 1 acquisition not suppressed to zero;
- sentinel c2w flips remain zero or near zero;
- old protection does not work merely by shrinking later updates into dust.

Generality bar:

- method should plausibly transfer beyond mini-language translation;
- specify a non-translation smoke test next, such as fictional object-property relations or user-profile facts from prose.

Mechanistic bar:

- shuffled targets should fail;
- relation/key shuffling should fail;
- output/posture ablation should reveal the predicted safety/acquisition tradeoff;
- all-layer should be no worse than selected-layer by safety metrics;
- update should preserve positive context-teacher score-space alignment;
- the purified update should improve over the current frontier:
  - relational `.10`, output `1024/40`: `4/20`, c2w `1`, drop `2.355`;
  - relational `.05`, output `1024/40`: `0/20`, c2w `0`, drop `1.126`;
  - STAR `1.0`: `3/38`, anti-aligned, c2w `0`.

## Strong Preferences For Your Answer

Please structure your response like this:

1. **Blunt Diagnosis**
   - What exactly is the blocker now?
   - Is one-pass surprise-from-weights still plausible?

2. **Core Mathematical Tool**
   - Name it.
   - Give equations.
   - Explain the predictive-coding/free-energy/synaptic analogy if useful.

3. **Local Purification**
   - How to decompose relational target rows/components.
   - How to estimate local sentinel/readout risk from weights and one pass.
   - How to keep key-conditioned useful readout while removing generic posture.

4. **Closed-Form Solve**
   - Tensor shapes.
   - Objective.
   - Solution.
   - Layer trust/scaling.

5. **Weight-Only Continual Mechanism**
   - How previous learning is protected without storing old-session state.
   - How the next solve detects already-load-bearing weight directions from
     weights plus the new pass only.
   - Why it protects without suppressing acquisition.

6. **Minimal Implementation**
   - files/functions;
   - pseudocode;
   - first three experiments.

7. **Ablations And Falsification**
   - explicit pass/fail criteria.

## Important Warnings

Please do **not** give us:

- another runtime object gate/router as the main answer;
- another raw feature-norm surprise rule;
- a rebranded STAR future residual target unless you fix why it was anti-aligned;
- "just project harder onto LM-head PCs";
- "just use sentinels as negatives";
- "just add more examples";
- "just use generated probes";
- "meta-train the writer" as the immediate next step;
- a benchmark-specific translation parser;
- a rule that uses labels or heldout questions.

My strongest current hypothesis:

\[
\boxed{
\text{The useful object is already inside relational context-value targets,}
\quad
\text{but it must be purified locally, not replaced globally.}
}
\]

Find the closed-form local purification rule.

## Postscript: KARP v1 Implemented And Partially Tested

After this request was written, GPT-5.5 Pro proposed KARP: Key-Attributable
Readout Purification. We implemented a first KARP wrapper around the existing
`relational_aggregate` context-value update.

KARP idea:

\[
\text{risk}(p,q)
\approx
\frac{p^\top C_G p}{p^\top C_S p+\epsilon}
\cdot
\frac{q^\top F_G q}{q^\top F_S q+\epsilon}.
\]

It shrinks key x value atoms only when they are both generic-key active and
generic-output observable.

Implementation notes:

- First KARP basis used only top generic-risk generalized directions. It barely
  touched the actual update: mean removed update ratio `0.014`.
- We fixed this by using a mixed signal+risk basis, so KARP decomposes the
  actual relational candidate map rather than only diagnosing risk directions.
- Tests pass: `.venv/bin/python -m pytest -q` -> `64 passed`.

Results on the teacher-filtered 20-question eval:

| run | edited | centered cosine | projection ratio | sentinel c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `1024/40` | `4/20` | `0.613` | `0.185` | `1` | `2.355` |
| relational `.05`, output `1024/40` | `0/20` | `0.596` | `0.121` | `0` | `1.126` |
| mixed KARP eta-cross `2.0` | `0/20` | `0.426` | `0.100` | `0` | `1.859` |
| mixed KARP eta-cross `.50`, eta-key/value `.05/.02` | `3/20` | `0.518` | `0.130` | `0` | `2.215` |
| mixed KARP eta-cross `.25`, eta-key/value `.02/.01` | `3/20` | `0.536` | `0.130` | `1` | `2.418` |
| product-only KARP `.50` | `1/20` | `0.548` | `0.135` | `1` | `2.034` |

Current interpretation:

- KARP is directionally useful: it gives the first point with nonzero
  acquisition (`3/20`) and zero sentinel correct-to-wrong flips.
- It is not solved: before-correct sentinel margin drop is still high (`2.215`)
  and acquisition is below the unsafe `4/20` strong-output relational run.
- Product-only KARP is worse, so the additive key/value rails help.
- The remaining blocker is likely that the value-risk metric is still too
  blunt. It uses LM-head/output basis plus low-surprise same-pass outputs, but
  this is not yet a sharp sentinel-risk observability metric.

If sending a new request to GPT-5.5 Pro, the next question should be:

> KARP product-risk purification improved the discrete c2w frontier but still
> over-prunes useful acquisition and leaves large sentinel margin drops. How do
> we build a one-pass, weight-only value-risk/observability metric that
> identifies sentinel-sensitive value atoms more locally than LM-head PCs plus
> low-surprise MLP outputs, while preserving context-teacher-aligned readout
> atoms?

## Postscript 2: KARP Local Fisher And Layer Trust Tested

We then tested two direct KARP refinements.

### Same-pass local Fisher value-risk basis

Implementation:

- Added a cheap local Fisher approximation from the lesson pass.
- For selected lesson positions, take the model's own top-k next-token
  distribution.
- Build factors:

\[
\sqrt{p_i}(W_U[i]-\mathbb{E}_p[W_U]).
\]

- Use those factors only as a KARP value-side risk basis.
- No labels, no sentinels, no null prompts, no generated probes, no gradient
  training.

Results:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed KARP `.50`, output `1024/40` | `3/20` | `0.518` | `0.130` | `0` | `2.215` |
| KARP + local Fisher rank `32` + output `1024/40` | `2/20` | `0.526` | `0.130` | `0` | `2.247` |
| KARP + local Fisher rank `32`, no output penalty | `1/20` | `0.202` | `0.042` | `4` | `3.148` |

Interpretation:

- Appending local Fisher to the existing global output basis does almost
  nothing useful.
- Using local Fisher without global output protection is unsafe and
  misaligned.
- The global output penalty is doing real safety work.
- The cheap top-k local Fisher basis does not identify the specific
  sentinel-sensitive value atoms.

### KARP layer-risk trust scalar

Implementation:

Scale each layer's update by:

\[
s_l=\min(1,\sqrt{b/(r_l+\epsilon)}),
\]

where \(r_l\) is `karp_cross_risk_after` and \(b\) is the risk budget.

Results:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed KARP `.50` | `3/20` | `0.518` | `0.130` | `0` | `2.215` |
| KARP trust budget `.25` | `1/20` | `0.407` | `0.088` | `0` | `1.839` |
| KARP trust budget `.50` | `1/20` | `0.472` | `0.109` | `0` | `1.871` |

Interpretation:

- Layer trust improves sentinel margin safety but collapses acquisition.
- It behaves like a smarter scale-down, not a new purification axis.
- KARP's current cross-risk metric is correlated with useful acquisition
  energy, so using it directly as a trust gate removes threshold-crossing
  behavior.

### Updated diagnosis

The problem is now sharper:

1. The useful actuator still lives in the `relational_aggregate` context-value
   map.
2. KARP product-risk purification is directionally right because it gives
   nonzero acquisition with zero c2w.
3. But current value-risk estimates are too blunt:
   - LM-head PCs are broad;
   - low-surprise same-pass outputs are broad;
   - cheap local Fisher is broad or mislocalized;
   - post-KARP cross-risk catches useful acquisition too.
4. The missing metric is not generic output observability. The model must touch
   output/readout directions to answer.

The next thing we need is a one-pass, weight-derived metric for:

\[
\frac{
\text{predicted unrelated/collateral margin movement}
}{
\text{predicted context-teacher-aligned movement}
}
\]

or an equivalent atom-level objective that preserves readout atoms when they
are conditionally useful but suppresses atoms that create broad collateral MC
posture movement.

Please do not propose:

- another runtime gate/router;
- using sentinel examples as negatives;
- generated probes;
- null prompts;
- next-token training;
- a plain local Fisher basis;
- simply lowering the update scale.

We need the next mathematical tool after KARP: a sharper atom-side risk metric
or target decomposition that can keep the `3/20` acquisition path, reduce
before-correct sentinel margin drop below about `1.0`, and eventually scale to
continual learning.

## Postscript 3: SHARP-KARP Shadow Anchors Tested And Mostly Falsified

GPT-5.5 Pro then proposed SHARP-KARP: signed shadow-anchor readout purification.
The idea was to use low-surprise, high-confidence same-pass tokens as a proxy
for stable ambient model beliefs, then penalize key-value atoms that the
candidate write predicts will lower those same-pass top-vs-runner margins.

We implemented it as:

```text
--intrinsic-target-purifier sharp_karp
```

with two solve modes:

- `ridge`: coefficient-space refit of the relational target with shadow anchor
  penalties;
- `shrink`: post-hoc attenuation of risky atoms in the existing candidate map.

Implementation details:

- compact key/value bases from mixed signal+risk rows;
- same-pass top-logit LM-head row lookup precomputed once per lesson;
- shadow anchors selected from low-surprise, high-confidence, low-overlap
  tokens, with fallback to avoid starving the anchor set;
- logit-lens top-vs-runner margin gradients as value-side shadow factors;
- hardening against ill-conditioned SVD/pinv failures and non-finite atom
  coefficients;
- empty all-layer relational selections now no-op instead of crashing.

Validation:

```bash
.venv/bin/python -m pytest -q
# 65 passed
```

Results on the same teacher-filtered 20-question eval:

| run | edited | centered cosine | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `1024/40` | `4/20` | `0.613` | `0.185` | `1` | `2.355` |
| mixed KARP `.50` | `3/20` | `0.518` | `0.130` | `0` | `2.215` |
| SHARP ridge, sparse anchors | `3/20` | `0.617` | `0.154` | `3` | `4.454` |
| SHARP ridge, anchor fallback | `3/20` | `0.596` | `0.148` | `4` | `4.589` |
| SHARP shrink r16, eta `.35`, shadow `1.5` | `0/20` | `0.230` | `0.038` | `0` | `0.951` |
| SHARP shrink r16, eta `.10`, shadow `.5` | `1/20` | `0.253` | `0.043` | `0` | `0.967` |

Interpretation:

- SHARP ridge/refit is unsafe. It rebuilds the map and can amplify harmful
  atoms even with many anchors.
- SHARP shrink is safe but mostly inert. It gets the margin target (`drop < 1`)
  but removes the context-teacher-aligned acquisition path.
- Same-pass stable-margin drop is too correlated with useful acquisition atoms.
  Penalizing it works as a structured scale-down, not as a clean collateral
  damage separator.

This falsifies the strong version of:

```text
same-pass stable-margin preservation ~= unrelated-capability preservation
```

with the current logit-lens anchor construction.

Updated ask:

Please do **not** propose another plain same-pass Fisher/margin-anchor penalty
unless it explains why SHARP shrink collapses acquisition. We need a metric
closer to:

\[
\frac{
\text{predicted off-task option reordering under generic MC/question states}
}{
\text{predicted teacher-aligned option reordering under learned-object states}
}
\]

but still under the hard constraints:

- one lesson/context forward pass;
- no sentinel examples as negatives;
- no generated probes;
- no null/default prompts;
- no labels or heldout data;
- no next-token training;
- no runtime router.

The next proposal should explain how to identify collateral option/readout
damage without using same-pass stable margins as the main proxy, because SHARP
shows that proxy protects by deleting the useful readout actuator.

## Postscript 4: ORCA-KARP Implemented And First Tested

GPT-5.5 Pro then proposed ORCA-KARP: Object-Relative Contrastive Actuator
Purification.

ORCA keeps the existing useful target family, `relational_aggregate`
context-value, but changes the atom risk metric. Instead of asking whether an
atom is globally readout-sensitive or whether it lowers same-pass stable
margins, it asks whether the atom's local option-space effect is parallel to
the relational target after same-pass nuisance residualization.

Implemented:

- `OrcaKarpPurificationResult`;
- `_mixed_signal_risk_candidate_basis(...)`;
- `orca_karp_purify_update(...)`;
- `--intrinsic-target-purifier orca_karp`;
- ORCA flags for ranks, option top-k, object/off-object basis ranks, atom
  penalties, signal floor, and nuisance ridge;
- a synthetic unit test showing ORCA keeps target-parallel option atoms more
  than target-orthogonal atoms.

Verification:

```bash
.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q
# 25 passed

.venv/bin/python -m pytest -q
# 66 passed
```

Initial engineering result:

- literal ORCA with rank `48/48` and no signal floor saturated the atom penalty;
- `orca_atom_diag_mean` was near cap, all `2304` atoms were shrunk by more than
  50%, and signal retention was nearly zero;
- we added a robust signal floor and optimized the off-object basis path so
  local runs are tractable.

Main behavioral results so far use:

- all 28 layers;
- 6 lessons, 8 examples;
- teacher-filtered 20-question eval;
- expanded 25-question sentinel suite;
- `relational_aggregate`, context value;
- scale `.10`;
- output/input protection `256/10` and `256/20`.

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relational `.10`, output `256/10`, input `256/20` | `6/20` | `11` | `0.578` | `0.183` | `3` | `4.688` |
| ORCA r16 soft | `6/20` | `11` | `0.544` | `0.166` | `1` | `2.956` |
| ORCA r16 medium | `4/20` | `9` | `0.479` | `0.155` | `2` | `2.008` |

Soft ORCA settings:

```bash
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

ORCA r16 soft diagnostics over 168 updates:

- mean removed update ratio: `0.541`;
- mean signal retention: `0.302`;
- mean candidate capture ratio: `0.696`;
- mean atom penalty: `34.47`;
- mean ORCA option signal: `0.0076`.

Interpretation:

- ORCA is not safe enough yet: the hard target remains c2w `0`, and ORCA soft
  still has c2w `1`.
- But unlike SHARP, ORCA does not simply erase acquisition. It preserved the
  `6/20` acquisition of the weak-protection relational baseline while reducing
  c2w from `3` to `1` and before-correct drop from `4.688` to `2.956`.
- Stronger ORCA did not monotonically improve discrete safety: drop improved,
  but acquisition fell and c2w worsened to `2`.
- The cheap top-k option map marks most candidate atoms as target-orthogonal
  (`orca_orthogonal_mean ~0.99`), so the coordinate is likely under-calibrated.
- Rank-16 candidate capture is only about `0.70`; higher-rank ORCA should be
  tested on a faster backend or after further optimization.

Current narrowed blocker:

> ORCA's object-relative option coordinate is directionally useful, but the
> current cheap top-k logit-lens implementation is still too blunt. We need to
> validate whether ORCA-kept and ORCA-removed atoms actually separate
> acquisition from sentinel damage.

Recommended next work:

1. Add ORCA atom ablations:
   - kept atoms only;
   - removed atoms only;
   - top-risk removed atoms only;
   - top-signal kept atoms only.
2. Run ORCA r32/r48 on the `1024/40` protection stack, preferably not on local
   MPS.
3. Improve the option-space map:
   - compare top-k LM-head contrast to exact local RMSNorm/logit VJP on a small
     layer subset;
   - test whether per-option-family centering is needed.
4. If ablations validate ORCA's atom ranking, move from post-hoc shrink to a
   closed-form coefficient-space solve.

## Postscript 5: ORCA Ablations Found A New Failure Mode

We implemented ORCA ablation modes and ran the three-way decomposition.

Modes:

- `kept_only`: apply only projected atoms kept by ORCA;
- `removed_only`: apply only projected atoms removed by ORCA;
- `residual_only`: apply only the component outside the mixed ORCA key/value
  basis.

Verification:

```bash
.venv/bin/python -m pytest tests/test_intrinsic_surprise.py -q
# 25 passed

.venv/bin/python -m pytest -q
# 66 passed
```

Matched weak-protection setting:

- all 28 layers;
- 6 lessons, 8 examples;
- teacher-filtered 20-question eval;
- expanded 25-question sentinel suite;
- `relational_aggregate`, context value;
- scale `.10`;
- output protection `256/10`;
- input protection `256/20`;
- ORCA r16 soft.

Results:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline relational `.10` | `6/20` | `11` | `0.578` | `0.183` | `3` | `4.688` |
| ORCA purified | `6/20` | `11` | `0.544` | `0.166` | `1` | `2.956` |
| ORCA kept-only | `1/20` | `3` | `-0.041` | `-0.006` | `1` | `1.011` |
| ORCA removed-only | `1/20` | `1` | `0.377` | `0.046` | `1` | `1.129` |
| ORCA residual-only | `9/20` | `13` | `0.603` | `0.157` | `2` | `1.287` |

Mean update decomposition:

| run | update Fro | basis residual Fro | projected Fro | kept projected Fro | removed projected Fro |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept-only | `0.420` | `1.532` | `1.410` | `0.420` | `1.087` |
| removed-only | `0.536` | `0.693` | `0.796` | `0.309` | `0.536` |
| residual-only | `1.679` | `1.679` | `1.582` | `0.488` | `1.228` |

Strong-protection residual-only:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORCA residual-only, output/input `1024/40` | `2/20` | `8` | `0.618` | `0.155` | `1` | `0.426` |

Sentinel accuracy improved overall on that run (`12/25 -> 15/25`), but one
previously correct sentinel still flipped: "Which animal is known for barking?"
changed from `Dog` to `Cloud`.

### Updated diagnosis for 5.5 Pro

The ablations falsify the simple "ORCA kept atoms are useful, removed atoms are
toxic" story.

Instead:

- ORCA-kept projected atoms are mostly inert.
- ORCA-removed projected atoms are also mostly inert.
- The acquisition-heavy, teacher-aligned component is the basis residual outside
  the current mixed ORCA atom subspace.
- Stronger protection preserves the centered teacher alignment of that residual
  component but crushes acquisition from `9/20` to `2/20`, while still leaving a
  single sentinel c2w.

So the new blocker is:

> How do we build a closed-form purifier for the acquisition-bearing basis
> residual, rather than classifying atoms inside a low-rank basis that mostly
> captures inert/generic projected movement?

Please propose the next mathematical tool under the original constraints:

- one lesson/context forward pass only;
- closed-form solve only;
- no generated probes, null prompts, quizzes, labels, next-token training,
  runtime router/RAG, or SAE;
- must be compatible with all-layer writes;
- target is a general surprise/consolidation rule, not translation-specific.

The result we want is not necessarily `20/20`. A useful next frontier would be:

- `6-10/20` acquisition;
- `0` sentinel correct-to-wrong flips;
- before-correct sentinel margin drop below about `1.0`;
- positive centered teacher alignment/projection;
- a path to continual learning where old learned transformations are preserved
  without stored sidecar state and without suppressing the next write into zero.

The sharp question:

> What is the right residual-coordinate geometry after quotienting the ORCA
> mixed basis, and how should the remaining single c2w failure be removed
> without deleting the residual component that produces acquisition?

## Postscript 6: Q-RICO Was Explored And Under-Acquires

We implemented your proposed Q-RICO direction and tested both low-rank
reconstruction and direct residual-filter versions.

Implementation:

- `--intrinsic-target-purifier qrico`
- Q-RICO around `relational_aggregate`, context value;
- quotient residual map `M_perp = M0 - Pi_U M0 Pi_V`;
- target-orthogonal option-scramble metric from same-pass top-logit contrast
  rows;
- two solve modes:
  - `sylvester`: rebuild a low-rank residual map from key/value bases;
  - `residual_filter`: keep the direct quotient residual map and shrink
    target-orthogonal value-side option-scramble directions;
- no full update-map SVD in the final implementation; it uses QR row bases plus
  cheap map-probe rows for runtime.

Verification:

```bash
.venv/bin/python -m pytest -q
# 69 passed
```

All runs:

- all 28 layers;
- 6 lessons, 8 examples;
- teacher-filtered 20-question eval;
- expanded 25-question sentinel suite;
- `relational_aggregate`, context value;
- output/input weak stack `256/10`, `256/20`;
- top-32 option contrasts.

Results:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q-RICO CCA, deflate `16/16`, trust on, scale `.10` | `2/20` | `7` | `0.537` | `0.121` | `3` | `2.542` |
| Q-RICO map-probe CCA, deflate `16/16`, trust on, scale `.10` | `0/20` | `0` | `0.149` | `0.006` | `0` | `0.105` |
| Q-RICO map-probe CCA, deflate `4/4`, no trust, scale `.10` | `1/20` | `1` | `0.592` | `0.096` | `0` | `0.569` |
| Q-RICO map-probe CCA, no deflate, no trust, scale `.10` | `1/20` | `3` | `0.568` | `0.127` | `1` | `2.327` |
| Q-RICO residual-filter, deflate `4/4`, no trust, scale `.10` | `1/20` | `4` | `0.638` | `0.125` | `0` | `0.219` |
| Q-RICO residual-filter, deflate `4/4`, no trust, scale `.20` | `1/20` | `12` | `0.606` | `0.174` | `0` | `1.978` |

Comparison target:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORCA residual-only, scale `.10` | `9/20` | `13` | `0.603` | `0.157` | `2` | `1.287` |

Interpretation:

Q-RICO's safety intuition is partly right, but the reconstruction/filtering
does not preserve the behaviorally useful threshold-crossing component.

Important observations:

- The low-rank CCA route is too lossy. It can report high row capture but still
  fail behaviorally.
- The direct residual-filter version is the best Q-RICO form: at scale `.10`,
  it gets `0` c2w and low sentinel drop (`0.219`) while retaining positive
  teacher alignment (`centered cos 0.638`, projection ratio `0.125`).
- But it only gets `1/20`; increasing scale to `.20` increases churn and margin
  damage without improving acquisition.
- No-deflation Q-RICO is not a rescue: it remains `1/20` and reintroduces a c2w.

So the updated blocker is sharper:

> ORCA residual-only contains the threshold-crossing acquisition component, but
> Q-RICO-style option-scramble filtering removes or fails to preserve that
> component. We need a full-rank/high-rank purification of the direct residual
> map, not a low-rank residual reconstruction and not a value-only option
> filter.

Please propose the next mathematical tool under the same constraints:

- one lesson/context forward pass only;
- closed-form solve only;
- no generated probes, null prompts, quizzes, labels, next-token training,
  runtime router/RAG, or SAE;
- all-layer compatible;
- general surprise/consolidation rule, not translation-specific.

The next method should specifically answer:

1. How do we keep the direct residual map's behavior-threshold component that
   gives `9/20`?
2. How do we remove the `2` c2w failures from ORCA residual-only without
   collapsing acquisition to `1/20`?
3. What diagnostic can tell whether a purifier is preserving threshold-crossing
   acquisition rather than only preserving small positive teacher-aligned score
   movement?

## Postscript 7: SPECTRA Was Implemented; Q-RICO Key16 Is The Current Safe Frontier

We implemented the SPECTRA direction you proposed:

- `--intrinsic-target-purifier spectra`;
- high-rank residual map kept directly;
- tail functionals preserved by closed-form rank-one constraints;
- generic option-contrast hazards clipped by closed-form rank-one constraints;
- tests added for tail preservation and hazard clipping.

Verification:

```bash
.venv/bin/python -m pytest -q
# 71 passed
```

But two things changed the diagnosis.

First, the earlier Q-RICO comparison had a confound. Q-RICO was run with
`key_feature_top_k=8`, while ORCA residual-only used `key_feature_top_k=16`.
Re-running the best Q-RICO residual-filter setup with key16 gave:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q-RICO key16, scale `.10` | `5/20` | `11` | `0.578` | `0.162` | `0` | `0.964` |

Scale pressure failed:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q-RICO key16, scale `.15` | `4/20` | `10` | `0.515` | `0.146` | `7` | `6.738` |
| Q-RICO key16, scale `.20` | `5/20` | `10` | `0.584` | `0.163` | `6` | `5.032` |

So the best safe single-task frontier is now:

\[
5/20,\quad 0\text{ c2w},\quad \text{before-correct drop}=0.964.
\]

Second, the first SPECTRA implementation was safe but too weak:

| run | edited | changed | centered cos | projection ratio | c2w | before-correct drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPECTRA fast, hazard rank `4`, budget `.25` | `2/20` | `2` | `0.292` | `0.032` | `0` | `0.262` |
| SPECTRA fast, no-hazard | `2/20` | `2` | `0.137` | `0.010` | `0` | `0.201` |
| SPECTRA fast, mild hazard rank `1`, budget `.50` | `1/20` | `2` | `-0.235` | `-0.019` | `0` | `0.130` |

The diagnostic was decisive: tail retention was ~`0.999` and correction norm
was tiny, but no-hazard was already weak. Therefore the problem was not the
hazard clipping. It was that the fast quotient basis did not preserve the ORCA
residual-only acquisition component.

We patched SPECTRA to use the exact ORCA mixed basis, but the exact-basis
all-layer jobs ran for more than 35 minutes without completing. That route needs
optimization before it is useful.

Current state:

| method | acquisition | safety |
| --- | ---: | --- |
| ORCA residual-only weak stack | `9/20` | unsafe: `2` c2w, drop `1.287` |
| Q-RICO key16 residual-filter `.10` | `5/20` | safe: `0` c2w, drop `0.964` |
| SPECTRA fast | `2/20` | safe but too weak |

Updated blocker:

> The best safe method is now Q-RICO key16, not SPECTRA. SPECTRA may still be
> conceptually right, but only if exact ORCA quotienting can be made cheap
> enough and shown to reproduce residual-only behavior in a no-hazard verifier.
> Scaling Q-RICO is not the answer because sentinel c2w returns immediately.

Near-term research questions:

1. How should Q-RICO key16 be extended to continual learning? The continual
   script now supports Q-RICO. Old-key negatives are too crude, and stored old
   transformation protection is now disallowed; we need a weight-only analogue.
2. Can Q-RICO's `5/20` safe acquisition be improved by key/row construction
   rather than scale?
3. Is there a cheap exact-ORCA quotient for SPECTRA, or should SPECTRA be
   dropped until we have a better basis construction?
4. What single-pass diagnostic predicts the sharp safety cliff between Q-RICO
   `.10` and `.15`?

## Postscript 8: The Primary Benchmark Is Now Two-Task Continual Learning

We agree that single-task acquisition is the wrong main hill. It was useful for
finding a safe one-context actuator, but the actual goal is sequential learning
without capability damage.

We updated the continual runner to support Q-RICO/SPECTRA purifier flags and
added a compact summarizer:

- `scripts/minilang_intrinsic_continual.py` now exposes the same purifier flags
  as `scripts/minilang_write.py`;
- `scripts/summarize_continual_run.py` reports task acquisition, retention, and
  sentinel shifts;
- changed continual default
  `--intrinsic-surprise-input-penalty-usage-power` from `1.0` to `0.0` so it
  matches the safe single-task Q-RICO setup unless overridden.

Verification:

```bash
.venv/bin/python -m py_compile scripts/summarize_continual_run.py
.venv/bin/python -m pytest -q
# 71 passed
```

Two-task benchmark settings:

- tasks: Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 8-question eval per task;
- all 28 layers;
- Q-RICO residual-filter;
- key feature top-k `16`;
- scale `.10`;
- output/input weak stack `256/10`, `256/20`;
- input penalty usage power `0.0`;
- expanded sentinel suite.

### No old-task protection

| step | eval task | edited | delta vs base | retention delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | Lyran | `3/8` | `+2/8` | `0` |
| 1 | Lyran | `2/8` | `+1/8` | `-1/8` |
| 1 | Vomar | `0/8` | `-1/8` | `0` |

Sentinel:

| step | sentinel acc | c2w | before-correct drop |
| ---: | ---: | ---: | ---: |
| 0 | `18/25` | `0` | `1.717` |
| 1 | `18/25` | `1` | `3.721` |

Interpretation: task 0 weakly acquires, then partially forgets; task 1 does not
acquire; sentinel safety breaks after the second write.

### Old selected keys as negatives

| step | eval task | edited | delta vs base | retention delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | Lyran | `2/8` | `+1/8` | `0` |
| 1 | Lyran | `2/8` | `+1/8` | `0` |
| 1 | Vomar | `1/8` | `0/8` | `0` |

Sentinel:

| step | sentinel acc | c2w | before-correct drop |
| ---: | ---: | ---: | ---: |
| 0 | `18/25` | `0` | `0.601` |
| 1 | `19/25` | `0` | `0.628` |

Interpretation: old-key negatives preserve task 0 and sentinel safety, but task
1 stays at baseline. Update diagnostics show why:

- task 0 mean update Frobenius: `2.76`;
- task 1 mean update Frobenius with old negatives: `1.37`.

So the old-key method mostly preserves by shrinking/suppressing the next write.

### Updated blocker

The method is not yet a continual learner.

Q-RICO key16 `.10` can safely acquire a little from one context, but in
sequence:

- no old protection causes forgetting and sentinel damage;
- old-key protection preserves but suppresses new acquisition.

The previous interpretation was "protect old key-to-value transformations, not
old keys." The new hard constraint refines this: the method cannot store old
transformations either. Any protection must be encoded in the updated weights
themselves, or be inferable later from the current weights plus the new single
context pass.

The two-task benchmark should now be the main objective:

\[
\text{task0 immediate} > \text{base},\quad
\text{task0 after task1} \ge \text{task0 immediate},\quad
\text{task1 immediate} > \text{base},\quad
\text{sentinel c2w}=0.
\]

Please focus the next proposal on this continual-learning objective, not
single-task `20/20` acquisition, and obey the no-sidecar-state rule.

## Postscript 9: New Hard Constraint - No Cross-Session Stored State

We are adding a stricter deployment constraint.

After the model reads a context and performs the closed-form write, the next
context/session must receive only the updated model weights. The system may not
carry forward any separate data structure from previous sessions:

- no stored selected keys;
- no old-task negative banks;
- no activation traces;
- no examples or summaries;
- no low-rank atoms;
- no old key-to-value transformation records;
- no Fisher/EWC/Laplace/sketch metrics;
- no runtime routers or memory slots.

This means old-key negatives and old-transformation metrics are now diagnostic
ablations only. They are not acceptable final answers.

The research question is now sharper:

> Can a one-pass closed-form surprise write make learned transformations
> self-preserving inside the weights, so future writes computed only from the
> current weights and a new single context pass do not overwrite them?

Please update any proposed continual-learning mechanism accordingly. If you
believe this is impossible, give the formal identifiability or information
argument, and state the weakest additional assumption that would make it
possible while staying close to the spirit of weight-only consolidation.

## Postscript 10: OCEP-Q Tested and Mostly Falsified

After your OCEP-Q proposal, I implemented it as a post-solve purifier:

- `ocep_residual`: applied to the direct relational aggregate context-value
  candidate;
- `ocep_qrico`: Q-RICO residual-filter first, then OCEP;
- no sidecar state, no old keys/transforms, no probes;
- object-key basis from selected weighted keys;
- generic key basis from current down/value geometry, high-upstream-norm
  one-hot channel anchors, low-surprise same-pass keys, and protected negative
  rows already used by the solve;
- option/readout basis from output basis, target rows, and same-pass top-logit
  contrasts.

Implementation tests passed:

```bash
.venv/bin/python -m pytest -q
# 78 passed
```

Single-task all-layer results on the same regenerated 20-item eval split:

| run | edited | sentinel c2w | before-correct drop | notes |
| --- | ---: | ---: | ---: | --- |
| OCEP residual, object rank 64, cap .35 | `1/20` | `6` | `5.586` | leakage reduced `0.704`, object disturbed `0.105` |
| OCEP residual, object rank 256, cap .10 | `4/20` | `5` | `5.636` | object delta `0.000`, leakage reduced `0.191` |
| OCEP-Q, object rank 256, cap .10 | `1/20` | `0` | `0.849` | safe but inert |
| Q-RICO residual-filter control | `1/20` | `0` | `0.742` | same split/settings as OCEP-Q |

Baseline was `2/20`, context was `15/20`, expanded sentinel before was `12/25`.

Interpretation:

- aggressive OCEP is unsafe and non-acquiring;
- object-preserving OCEP can recover some acquisition (`4/20`) but remains very
  unsafe (`5` c2w);
- OCEP-Q is safe but does not recover acquisition beyond Q-RICO on this split;
- preserving the selected object-key span exactly is not enough to prevent
  sentinel damage.

Current diagnosis update:

The unsafe acquisition component is not captured by this local
generic-key × option-basis leakage sketch. The dangerous part may be:

1. a more specific target-row-to-option interaction rather than a generic key
   interaction;
2. an update that is locally safe but becomes generic after propagation through
   later frozen layers;
3. additive false evidence in a downstream readout coordinate not visible in
   the local MLP-down option atlas;
4. or a broader issue: local post-solve output filters are the wrong class of
   tool, because they repeatedly trade acquisition for safety.

Please propose the next implementable mathematical tool under the same hard
constraints:

- one forward pass from the lesson/context;
- closed-form solve;
- surprise-driven;
- all-layer compatible;
- no probes/null prompts/quizzes/labels/SAE/RAG/router;
- no sidecar state across sessions after weights are written;
- primary success should be two-task learning plus sentinel preservation, not
  single-task frontier-chasing.

Given the OCEP result, I am especially interested in either:

- a safety metric based on predicted downstream propagation of the update
  through later frozen layers, still computed closed-form from the same pass and
  current weights; or
- a different weight-only consolidation mechanism that makes learned
  transformations self-preserving without gauge-scale distortion or stored
  metadata.

## Postscript 11: Benchmark Harness and SPECTRA Runtime Check

I added a standardized two-task benchmark launcher:

```text
scripts/continual_benchmark_grid.py
```

It prints or runs Modal/local commands for the current primary benchmark:

- two tasks, Lyran then Vomar;
- 6 lessons/task, 8 examples/lesson;
- teacher-filtered 8-question eval per task;
- all layers `0..27`;
- relational aggregate context-value target;
- final-aligned token selection;
- key feature top-k `16`;
- target scale `.10`;
- output/input protection `256/10`, `256/20`;
- input penalty usage power `0.0`;
- expanded sentinel suite.

Registered presets:

- `relational_raw`;
- `qrico_key16`;
- `spectra_noquotient`;
- `ocep_residual`;
- `ocep_qrico`;
- `seal_qrico_no_apply`.

Verification:

```bash
.venv/bin/python -m py_compile scripts/continual_benchmark_grid.py
.venv/bin/python -m pytest -q
# 78 passed
```

I also started a Modal run for direct-map SPECTRA without ORCA quotient:

```text
/modal-runs/spectra_noquotient_continual_s010_20260521
```

It loaded the model, teacher-filtered both tasks, scored both before-write
baselines, entered task-0 writing, and then stayed inside the write phase for
more than 10 minutes. I stopped the Modal app. So this is not a behavioral
score, but it is a practical falsifier for the current SPECTRA implementation:
even the no-quotient variant is too slow for the all-layer two-task research
loop.

Please treat the next proposal as needing to be cheap enough for the
`continual_benchmark_grid.py` loop. The current preferred direction remains:

1. a downstream-propagation safety metric that is cheaper than SPECTRA and
   computed from the same pass/current weights; or
2. a weight-only consolidation mark/protection mechanism that does not distort
   future surprise coordinates like gauge-scale sealing did.
