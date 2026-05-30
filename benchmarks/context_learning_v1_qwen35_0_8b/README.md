# Context Learning Benchmark v1

Model used for screening: `Qwen/Qwen3.5-0.8B-Base`.

Acceptance rule: an item is included only when the model is wrong without
the context and correct with the task context.

Accepted tasks: 10 / 10.

Saved eval split:

- total accepted items: `80`
- no-context baseline: `0/80`
- full-context teacher: `80/80`
- expanded sentinel baseline: `20/25`

The benchmark is model-screened, not model-agnostic. For another base model,
rerun `scripts/evaluate_context_learning_benchmark.py` and report its baseline
and full-context teacher scores before using write/edit results.

- `mini_language` (invented_language_translation), seed 1, 8 eval items.
- `user_profile` (personal_profile_memory), seed 1, 8 eval items.
- `symbolic_rules` (new_symbolic_rules), seed 1, 8 eval items.
- `taxonomy` (new_category_system), seed 1, 8 eval items.
- `map_legend` (map_symbol_rules), seed 1, 8 eval items.
- `api_protocol` (fictional_api_semantics), seed 1, 8 eval items.
- `game_rules` (new_game_mechanics), seed 1, 8 eval items.
- `causal_objects` (fictional_causal_world), seed 1, 8 eval items.
- `scheduling_policy` (custom_policy_learning), seed 1, 8 eval items.
- `social_protocol` (fictional_social_rules), seed 1, 8 eval items.

Important files:

- `manifest.json`: task list, seeds, screening settings, and sentinel path.
- `accepted_items.jsonl`: all 80 accepted no-context-wrong/context-correct items.
- `candidate_audit.jsonl`: every screened candidate with baseline/context
  prediction and margin.
- `screen_metrics.jsonl`: per-task seed screening metrics.
- `eval_metrics_qwen35_0_8b.json`: reproduced baseline/context metrics from
  the saved files.
- `safety/sentinels_expanded.jsonl`: expanded safety sentinel suite.
