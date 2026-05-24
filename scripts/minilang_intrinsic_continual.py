"""Sequential continual-learning diagnostic for intrinsic lesson writes.

This runner keeps the old mini-language triangle evaluation structure, but the
write itself is the current lesson-only intrinsic-surprise rule. There are no
answer traces, no generated quizzes for writing, and no prompt router: each
task is written from its lesson text, then all previously written tasks and the
sentinel suite are re-evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.modeling import clear_active_slot_weights, install_additive_memory, load_model_and_tokenizer
from scripts.minilang_continual_triangle import (
    TranslationQuestion,
    build_task_questions,
    evaluate_task_mc,
    release_device_cache,
    render_task_lesson,
    render_task_lesson_variant,
    task_profile,
)
from scripts.minilang_write import (
    add_metrics,
    add_sentinel_shift_metrics,
    append_jsonl,
    evenly_cap_rows,
    evaluate_generic_mc,
    run_intrinsic_surprise_writes,
    sentinel_questions,
)


def progress(message: str) -> None:
    print(f"[intrinsic-continual] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="runs/minilang_intrinsic_continual")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument(
        "--task-indices",
        default="",
        help="Optional comma-separated task profile indices, e.g. '1' for Vomar only or '1,0' for reverse order.",
    )
    parser.add_argument("--lessons-per-task", type=int, default=6)
    parser.add_argument("--lesson-examples", type=int, default=8)
    parser.add_argument(
        "--dice-diverse-contexts",
        type=int,
        default=0,
        help="Replace progressive same-format lessons with this many diverse renderings of the final task lesson.",
    )
    parser.add_argument("--eval-questions", type=int, default=8)
    parser.add_argument(
        "--eval-questions-jsonl",
        default="",
        help="Optional fixed eval question JSONL. Rows may include task_idx; if provided, teacher filtering is skipped.",
    )
    parser.add_argument("--teacher-filter-eval", action="store_true")
    parser.add_argument("--teacher-filter-candidates", type=int, default=120)
    parser.add_argument(
        "--teacher-filter-require-baseline-wrong",
        action="store_true",
        help="When teacher-filtering eval questions, require no-context baseline wrong and full-context teacher correct.",
    )
    parser.add_argument(
        "--early-stop-c2w-over",
        type=int,
        default=-1,
        help="Stop after a write/eval step if sentinel correct-to-wrong flips exceed this value. Negative disables.",
    )
    parser.add_argument(
        "--early-stop-task0-min-edited-correct",
        type=int,
        default=-1,
        help="Stop after task 0 if edited task-0 correct count is below this value. Negative disables.",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(28)))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sentinel-eval", action="store_true")
    parser.add_argument("--sentinel-suite", choices=["core", "expanded"], default="expanded")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="", choices=["", "eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")

    parser.add_argument(
        "--intrinsic-surprise-target-mode",
        choices=[
            "mlp_contribution",
            "associative_binding",
            "predictive_residual",
            "relational_aggregate",
            "relational_residual",
            "compatibility_residual",
            "conditional_relation_innovation",
            "feature_birth",
            "logit_error",
        ],
        default="relational_aggregate",
    )
    parser.add_argument("--intrinsic-surprise-token-mode", choices=["last", "top", "all", "final_aligned"], default="final_aligned")
    parser.add_argument("--intrinsic-surprise-top-tokens", type=int, default=16)
    parser.add_argument("--intrinsic-surprise-feature-top-k", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-target-feature-top-k", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-key-feature-top-k", type=int, default=8)
    parser.add_argument("--intrinsic-surprise-value-feature-top-k", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-pair-top-k", type=int, default=64)
    parser.add_argument("--wicr-compatibility-threshold", type=float, default=0.15)
    parser.add_argument("--wicr-compatibility-temperature", type=float, default=0.15)
    parser.add_argument("--wicr-posture-pcs", type=int, default=64)
    parser.add_argument("--wicr-target-vector-mode", choices=["normal", "value"], default="normal")
    parser.add_argument("--wicr-attention-edges", type=int, default=0)
    parser.add_argument("--wicr-attention-flow-mode", choices=["vo", "identity"], default="vo")
    parser.add_argument("--wicr-no-same-token-edges", action="store_true")
    parser.add_argument("--cori-feature-top-k", type=int, default=128)
    parser.add_argument("--cori-relation-rank", type=int, default=16)
    parser.add_argument("--cori-beta", type=float, default=3.0)
    parser.add_argument("--cori-edge-top-k", type=int, default=0)
    parser.add_argument("--cori-edge-attention-scale", type=float, default=0.5)
    parser.add_argument("--cori-sinkhorn-steps", type=int, default=0)
    parser.add_argument("--cori-target-mode", choices=["svd_value", "innovation_value"], default="svd_value")
    parser.add_argument("--intrinsic-surprise-bidirectional-pairs", action="store_true")
    parser.add_argument(
        "--intrinsic-surprise-relation-value-mode",
        choices=["residual", "full", "context"],
        default="context",
    )
    parser.add_argument("--intrinsic-surprise-value-source", choices=["base", "effective"], default="base")
    parser.add_argument("--intrinsic-surprise-effective-target-norm", choices=["raw", "base"], default="raw")
    parser.add_argument("--intrinsic-surprise-target-scale", type=float, default=0.10)
    parser.add_argument("--intrinsic-surprise-target-row-norm-cap", type=float, default=0.0)
    parser.add_argument("--intrinsic-surprise-center-targets", action="store_true")
    parser.add_argument("--intrinsic-surprise-weight-mode", choices=["linear", "exponential"], default="exponential")
    parser.add_argument("--intrinsic-surprise-exp-temperature", type=float, default=2.0)
    parser.add_argument("--intrinsic-surprise-exp-cap", type=float, default=20.0)
    parser.add_argument("--intrinsic-surprise-prediction-ridge", type=float, default=1.0)
    parser.add_argument("--intrinsic-span-readout-bridge", action="store_true")
    parser.add_argument("--intrinsic-span-readout-scale", type=float, default=1.0)
    parser.add_argument("--intrinsic-span-readout-max-items", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-persistence-power", type=float, default=1.0)
    parser.add_argument("--intrinsic-surprise-persistence-threshold", type=float, default=0.25)
    parser.add_argument("--intrinsic-surprise-persistence-min-tokens", type=int, default=2)
    parser.add_argument("--intrinsic-surprise-generic-rank", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-lm-head-generic-rank", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-output-penalty-rank", type=int, default=256)
    parser.add_argument("--intrinsic-surprise-output-penalty-weight", type=float, default=10.0)
    parser.add_argument("--intrinsic-surprise-input-penalty-features", type=int, default=1024)
    parser.add_argument("--intrinsic-surprise-input-penalty-weight", type=float, default=40.0)
    parser.add_argument("--intrinsic-surprise-input-penalty-usage-power", type=float, default=0.0)
    parser.add_argument(
        "--intrinsic-surprise-input-penalty-mode",
        choices=["onehot", "svd", "hybrid"],
        default="onehot",
    )
    parser.add_argument("--intrinsic-surprise-specificity-power", type=float, default=0.0)
    parser.add_argument("--intrinsic-surprise-readout-specificity-power", type=float, default=0.0)
    parser.add_argument("--intrinsic-surprise-project-generic", action="store_true")
    parser.add_argument("--intrinsic-surprise-lesson-format", choices=["raw", "chat_user"], default="raw")
    parser.add_argument("--intrinsic-surprise-birth-mode", choices=["state", "conjunction"], default="state")
    parser.add_argument("--intrinsic-surprise-birth-pairs", type=int, default=4)
    parser.add_argument("--intrinsic-surprise-birth-min-response", type=float, default=1e-4)
    parser.add_argument("--intrinsic-surprise-birth-trigger-scale", type=float, default=4.0)
    parser.add_argument("--intrinsic-surprise-birth-trigger-ridge", type=float, default=1e-3)
    parser.add_argument("--write-only-final", action="store_true")
    parser.add_argument("--dice-defer-apply", action="store_true")
    parser.add_argument(
        "--dice-support-space",
        choices=["coordinate", "column", "key_effect", "key_edge_effect", "target_group_effect", "svd"],
        default="coordinate",
    )
    parser.add_argument("--dice-subspace-rank", type=int, default=8)
    parser.add_argument("--dice-effect-rank", type=int, default=64)
    parser.add_argument("--dice-effect-key-cap", type=int, default=1024)
    parser.add_argument("--dice-effect-ridge", type=float, default=1e-3)
    parser.add_argument("--dice-support-threshold", type=float, default=0.75)
    parser.add_argument("--dice-support-temperature", type=float, default=16.0)
    parser.add_argument("--dice-support-strength", type=float, default=1.0)
    parser.add_argument("--dice-support-cap", type=float, default=2.0)
    parser.add_argument("--dice-support-floor", type=float, default=0.0)
    parser.add_argument("--dice-anti-contexts", type=int, default=0)
    parser.add_argument("--dice-anti-profile-offset", type=int, default=4)
    parser.add_argument("--dice-anti-threshold", type=float, default=0.50)
    parser.add_argument("--dice-anti-temperature", type=float, default=12.0)
    parser.add_argument("--dice-anti-strength", type=float, default=1.0)
    parser.add_argument("--memory-gate", action="store_true")
    parser.add_argument("--memory-gate-final-token-only", action="store_true")
    parser.add_argument("--memory-gate-threshold", type=float, default=0.95)
    parser.add_argument("--memory-gate-temperature", type=float, default=80.0)
    parser.add_argument(
        "--intrinsic-target-purifier",
        choices=[
            "none",
            "karp",
            "sharp_karp",
            "orca_karp",
            "qrico",
            "prism_q",
            "tdmi_q",
            "wm_coherence",
            "tag_ce",
            "cage_ce",
            "trace_q",
            "spectra",
            "seal_qrico",
            "ocep_residual",
            "ocep_qrico",
        ],
        default="none",
    )
    parser.add_argument("--karp-key-rank", type=int, default=64)
    parser.add_argument("--karp-value-rank", type=int, default=64)
    parser.add_argument("--karp-low-surprise-quantile", type=float, default=0.35)
    parser.add_argument("--karp-eta-cross", type=float, default=10.0)
    parser.add_argument("--karp-eta-key", type=float, default=0.15)
    parser.add_argument("--karp-eta-value", type=float, default=0.05)
    parser.add_argument("--karp-risk-ratio-cap", type=float, default=100.0)
    parser.add_argument("--karp-local-fisher-rank", type=int, default=0)
    parser.add_argument("--karp-local-fisher-top-k", type=int, default=32)
    parser.add_argument("--karp-local-fisher-max-positions", type=int, default=128)
    parser.add_argument("--karp-layer-risk-budget", type=float, default=0.0)
    parser.add_argument("--sharp-shadow-anchors", type=int, default=128)
    parser.add_argument("--sharp-key-rank", type=int, default=48)
    parser.add_argument("--sharp-value-rank", type=int, default=48)
    parser.add_argument("--sharp-signal-top-k", type=int, default=8)
    parser.add_argument("--sharp-low-surprise-quantile", type=float, default=0.25)
    parser.add_argument("--sharp-confidence-quantile", type=float, default=0.60)
    parser.add_argument("--sharp-eta", type=float, default=0.5)
    parser.add_argument("--sharp-shadow-weight", type=float, default=2.0)
    parser.add_argument("--sharp-karp-kappa", type=float, default=0.1)
    parser.add_argument("--sharp-shadow-temperature", type=float, default=0.05)
    parser.add_argument("--sharp-solve-mode", choices=["ridge", "shrink"], default="ridge")
    parser.add_argument("--orca-key-rank", type=int, default=48)
    parser.add_argument("--orca-value-rank", type=int, default=48)
    parser.add_argument("--orca-option-top-k", type=int, default=16)
    parser.add_argument("--orca-object-rank", type=int, default=128)
    parser.add_argument("--orca-off-object-rank", type=int, default=512)
    parser.add_argument("--orca-eta-orth", type=float, default=0.5)
    parser.add_argument("--orca-eta-posture", type=float, default=0.25)
    parser.add_argument("--orca-eta-off-object", type=float, default=0.5)
    parser.add_argument("--orca-eta-karp", type=float, default=0.25)
    parser.add_argument("--orca-signal-floor-quantile", type=float, default=0.0)
    parser.add_argument(
        "--orca-ablation-mode",
        choices=[
            "purified",
            "kept_only",
            "removed_only",
            "residual_only",
            "top_signal_kept",
            "top_risk_removed",
        ],
        default="purified",
    )
    parser.add_argument("--orca-ablation-fraction", type=float, default=0.25)
    parser.add_argument("--orca-nuisance-ridge", type=float, default=1e-3)
    parser.add_argument("--qrico-deflate-key-rank", type=int, default=16)
    parser.add_argument("--qrico-deflate-value-rank", type=int, default=16)
    parser.add_argument("--qrico-rank", type=int, default=64)
    parser.add_argument("--qrico-option-sketch-rank", type=int, default=256)
    parser.add_argument("--qrico-target-parallel-rank", type=int, default=4)
    parser.add_argument("--qrico-scramble-weight", type=float, default=0.35)
    parser.add_argument("--qrico-residual-row-weight-power", type=float, default=0.5)
    parser.add_argument("--qrico-quotient-mode", choices=["joint", "two_sided"], default="joint")
    parser.add_argument("--qrico-solve-mode", choices=["sylvester", "residual_filter"], default="sylvester")
    parser.add_argument("--qrico-cca-ridge", type=float, default=1e-3)
    parser.add_argument("--qrico-layer-evidence-min", type=float, default=0.03)
    parser.add_argument("--qrico-layer-evidence-target", type=float, default=0.20)
    parser.add_argument("--qrico-disable-layer-trust", action="store_true")
    parser.add_argument("--tdmi-object-endpoints", type=int, default=8)
    parser.add_argument("--tdmi-ambient-endpoints", type=int, default=16)
    parser.add_argument("--tdmi-object-rank", type=int, default=8)
    parser.add_argument("--tdmi-ambient-rank", type=int, default=16)
    parser.add_argument("--tdmi-horizon", type=int, default=4)
    parser.add_argument("--tdmi-trust-temperature", type=float, default=0.5)
    parser.add_argument("--tdmi-trust-threshold", type=float, default=0.0)
    parser.add_argument("--tdmi-trust-floor", type=float, default=0.15)
    parser.add_argument("--tdmi-disable-future", action="store_true")
    parser.add_argument("--wm-graph-top-k", type=int, default=8)
    parser.add_argument("--wm-trust-temperature", type=float, default=0.35)
    parser.add_argument("--wm-trust-threshold", type=float, default=0.0)
    parser.add_argument("--wm-trust-floor", type=float, default=0.25)
    parser.add_argument("--wm-future-weight", type=float, default=0.25)
    parser.add_argument("--wm-ambient-rank", type=int, default=16)
    parser.add_argument("--wm-ambient-weight", type=float, default=0.15)
    parser.add_argument("--wm-disable-future", action="store_true")
    parser.add_argument("--tagce-max-object-nodes", type=int, default=96)
    parser.add_argument("--tagce-max-ambient-nodes", type=int, default=96)
    parser.add_argument("--tagce-max-object-edges", type=int, default=192)
    parser.add_argument("--tagce-max-ambient-edges", type=int, default=192)
    parser.add_argument("--tagce-edge-smooth-alpha", type=float, default=0.35)
    parser.add_argument("--tagce-edge-sim-top-k", type=int, default=8)
    parser.add_argument("--tagce-posture-rank", type=int, default=64)
    parser.add_argument("--tagce-edge-nuisance-rank", type=int, default=32)
    parser.add_argument("--tagce-ambient-key-rank", type=int, default=16)
    parser.add_argument("--tagce-eta-ambient", type=float, default=0.25)
    parser.add_argument("--tagce-eta-posture-ambient", type=float, default=1.0)
    parser.add_argument("--tagce-anchor-weight", type=float, default=0.0)
    parser.add_argument("--tagce-potential-weight", type=float, default=0.0)
    parser.add_argument("--tagce-lowfreq-weight", type=float, default=0.0)
    parser.add_argument("--tagce-lowfreq-rank", type=int, default=4)
    parser.add_argument("--tagce-layer-veto-budget", type=float, default=0.75)
    parser.add_argument("--tagce-disable-layer-veto", action="store_true")
    parser.add_argument("--tagce-disable-schur", action="store_true")
    parser.add_argument("--tagce-disable-graph-settle", action="store_true")
    parser.add_argument("--tagce-shuffle-edge-targets", action="store_true")
    parser.add_argument("--tagce-shuffle-incidence", action="store_true")
    parser.add_argument("--cage-edge-max", type=int, default=192)
    parser.add_argument("--cage-ambient-edge-max", type=int, default=192)
    parser.add_argument("--cage-lowfreq-rank", type=int, default=4)
    parser.add_argument("--cage-ambient-rank", type=int, default=8)
    parser.add_argument("--cage-value-nuisance-rank", type=int, default=32)
    parser.add_argument("--cage-edge-weight", type=float, default=1.0)
    parser.add_argument("--cage-centroid-weight", type=float, default=0.35)
    parser.add_argument("--cage-lowfreq-weight", type=float, default=0.50)
    parser.add_argument("--cage-ambient-weight", type=float, default=2.0)
    parser.add_argument("--cage-schur-ridge", type=float, default=1e-3)
    parser.add_argument("--cage-prox-ridge", type=float, default=0.25)
    parser.add_argument("--cage-correction-cap", type=float, default=0.35)
    parser.add_argument("--cage-disable-schur", action="store_true")
    parser.add_argument("--cage-disable-graph-settle", action="store_true")
    parser.add_argument("--cage-disable-lowfreq", action="store_true")
    parser.add_argument("--cage-shuffle-graph", action="store_true")
    parser.add_argument("--prism-horizon", type=int, default=4)
    parser.add_argument("--prism-signal-rank", type=int, default=16)
    parser.add_argument("--prism-hazard-rank", type=int, default=16)
    parser.add_argument("--prism-option-top-k", type=int, default=8)
    parser.add_argument("--prism-generic-key-rank", type=int, default=128)
    parser.add_argument("--prism-low-surprise-rows", type=int, default=64)
    parser.add_argument("--prism-budget", type=float, default=0.25)
    parser.add_argument("--prism-correction-cap", type=float, default=0.35)
    parser.add_argument("--prism-signal-retention-min", type=float, default=0.90)
    parser.add_argument("--prism-no-residualize-hazard", action="store_true")
    parser.add_argument("--prism-disable-future", action="store_true")
    parser.add_argument(
        "--prism-ablation",
        choices=[
            "none",
            "no_residualize",
            "local_only",
            "shuffled_signal",
            "correction_only",
            "removed_hazard_only",
            "no_hazard",
        ],
        default="none",
    )
    parser.add_argument("--trace-object-endpoints", type=int, default=16)
    parser.add_argument("--trace-ambient-endpoints", type=int, default=32)
    parser.add_argument("--trace-option-top-k", type=int, default=8)
    parser.add_argument("--trace-option-contrasts", type=int, default=4)
    parser.add_argument("--trace-object-rank", type=int, default=16)
    parser.add_argument("--trace-ambient-rank", type=int, default=16)
    parser.add_argument("--trace-generic-key-rank", type=int, default=128)
    parser.add_argument("--trace-target-tau", type=float, default=1.0)
    parser.add_argument("--trace-target-floor", type=float, default=0.10)
    parser.add_argument("--trace-gamma", type=float, default=0.25)
    parser.add_argument("--trace-layer-trust-threshold", type=float, default=2.0)
    parser.add_argument("--trace-vjp-mode", choices=["local"], default="local")
    parser.add_argument("--seal-eta-erase", type=float, default=2.0)
    parser.add_argument("--seal-eta-seal", type=float, default=0.05)
    parser.add_argument("--seal-max-scale", type=float, default=1.10)
    parser.add_argument("--seal-salience-tau", type=float, default=1.0)
    parser.add_argument("--seal-disable-apply", action="store_true")
    parser.add_argument("--seal-canonicalize-surprise", action="store_true")
    parser.add_argument("--spectra-contrast-rank", type=int, default=128)
    parser.add_argument("--spectra-tail-anchors", type=int, default=32)
    parser.add_argument("--spectra-tail-quantile", type=float, default=0.80)
    parser.add_argument("--spectra-hazard-rank", type=int, default=4)
    parser.add_argument("--spectra-hazard-budget", type=float, default=0.25)
    parser.add_argument("--spectra-beta-tail", type=float, default=100.0)
    parser.add_argument("--spectra-beta-hazard", type=float, default=10.0)
    parser.add_argument("--spectra-generic-key-rank", type=int, default=256)
    parser.add_argument("--spectra-quotient-rank", type=int, default=16)
    parser.add_argument("--spectra-option-top-k", type=int, default=128)
    parser.add_argument("--spectra-no-orca-quotient", action="store_true")
    parser.add_argument(
        "--spectra-ablation",
        choices=["none", "no_tail", "no_hazard", "hazard_only", "shuffled_tail"],
        default="none",
    )
    parser.add_argument("--ocep-object-rank", type=int, default=64)
    parser.add_argument("--ocep-generic-rank", type=int, default=128)
    parser.add_argument("--ocep-option-rank", type=int, default=64)
    parser.add_argument("--ocep-option-output-rank", type=int, default=32)
    parser.add_argument("--ocep-option-local-rank", type=int, default=32)
    parser.add_argument("--ocep-low-surprise-rank", type=int, default=32)
    parser.add_argument("--ocep-weight-anchor-rank", type=int, default=96)
    parser.add_argument("--ocep-protected-rank", type=int, default=32)
    parser.add_argument("--ocep-ridge", type=float, default=1e-3)
    parser.add_argument("--ocep-correction-cap", type=float, default=0.35)
    parser.add_argument("--ocep-conflict-skip", type=float, default=1.1)
    parser.add_argument(
        "--old-task-negative-keys",
        action="store_true",
        help="Protect previous task write keys as negatives when solving later task writes.",
    )
    parser.add_argument("--old-task-negative-max-rows", type=int, default=256)
    parser.add_argument("--old-task-negative-scale", type=float, default=1.0)
    parser.add_argument(
        "--screen-before-write-only",
        action="store_true",
        help="Stop after teacher filtering and before-write baseline/context scoring.",
    )
    return parser.parse_args()


def load_eval_questions_jsonl(path: str, profiles) -> list[list[TranslationQuestion]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    profile_sets: list[list[TranslationQuestion]] = []
    multi_profile = len(profiles) > 1
    for profile in profiles:
        questions: list[TranslationQuestion] = []
        for row in rows:
            if "task_idx" in row and int(row["task_idx"]) != profile.idx:
                continue
            if "task_idx" not in row and multi_profile:
                continue
            answer_idx = row.get("answer_idx")
            if answer_idx is None:
                answer_letter = str(row["answer_letter"]).strip().upper()
                answer_idx = "ABCD".index(answer_letter)
            questions.append(
                TranslationQuestion(
                    sentence=str(row["sentence"]),
                    answer=str(row["answer"]),
                    options=[str(option) for option in row["options"]],
                    answer_idx=int(answer_idx),
                    category=str(row.get("category", "heldout_translation")),
                )
            )
        profile_sets.append(questions)
    if any(not questions for questions in profile_sets):
        counts = [len(questions) for questions in profile_sets]
        raise ValueError(f"Fixed eval question file {path} did not contain questions for all profiles: {counts}")
    return profile_sets


def write_config(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    paths = (
        output_dir / "metrics.jsonl",
        output_dir / "updates.jsonl",
        output_dir / "lessons.jsonl",
        output_dir / "eval_questions.jsonl",
        output_dir / "eval_details.jsonl",
    )
    for path in paths:
        if path.exists():
            path.unlink()
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    metrics_path, updates_path, lessons_path, questions_path, details_path = write_config(args, output_dir)

    progress("loading model")
    model, tokenizer, device = load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        attn_implementation=args.attn_implementation or None,
    )
    progress(f"loaded model on {device}; installing all-layer MLP memories")
    wrappers = install_additive_memory(model, args.layers, memory_dtype=torch.float32)
    progress(f"installed wrappers for {len(wrappers)} layers")

    if args.task_indices.strip():
        task_indices = [int(part.strip()) for part in args.task_indices.split(",") if part.strip()]
    else:
        task_indices = list(range(args.tasks))
    profiles = [task_profile(idx) for idx in task_indices]
    final_lesson_idx = args.lessons_per_task - 1
    lesson_texts: list[list[str]] = []
    dice_anti_lesson_texts: list[list[str]] = []
    contexts: list[str] = []
    eval_sets = []
    filter_stats: list[dict] = []
    fixed_eval_sets = load_eval_questions_jsonl(args.eval_questions_jsonl, profiles) if args.eval_questions_jsonl else None
    if fixed_eval_sets is not None and args.teacher_filter_eval:
        progress("fixed eval questions supplied; skipping teacher filtering")

    for profile_idx, profile in enumerate(profiles):
        if args.dice_diverse_contexts > 0:
            task_lessons = [
                render_task_lesson_variant(
                    profile,
                    final_lesson_idx,
                    args.lesson_examples,
                    args.seed,
                    variant_idx,
                )
                for variant_idx in range(args.dice_diverse_contexts)
            ]
        else:
            task_lessons = [
                render_task_lesson(profile, lesson_idx, args.lesson_examples, args.seed)
                for lesson_idx in range(args.lessons_per_task)
            ]
        lesson_texts.append(task_lessons)
        if args.dice_anti_contexts > 0:
            anti_profile = task_profile(profile.idx + args.dice_anti_profile_offset)
            dice_anti_lesson_texts.append(
                [
                    render_task_lesson_variant(
                        anti_profile,
                        final_lesson_idx,
                        args.lesson_examples,
                        args.seed + 777_001,
                        variant_idx,
                    )
                    for variant_idx in range(args.dice_anti_contexts)
                ]
            )
        else:
            dice_anti_lesson_texts.append([])
        contexts.append("\n\n".join(task_lessons))
        for idx, text in enumerate(task_lessons):
            append_jsonl(
                lessons_path,
                {
                    "task_idx": profile.idx,
                    "language": profile.name,
                    "lesson_idx": idx,
                    "render_mode": "dice_diverse" if args.dice_diverse_contexts > 0 else "standard",
                    "text": text,
                },
            )
        for idx, text in enumerate(dice_anti_lesson_texts[-1]):
            append_jsonl(
                lessons_path,
                {
                    "task_idx": profile.idx,
                    "language": profile.name,
                    "lesson_idx": idx,
                    "render_mode": "dice_anti",
                    "text": text,
                },
            )
        if fixed_eval_sets is not None:
            eval_sets.append(fixed_eval_sets[profile_idx])
        else:
            candidate_count = args.teacher_filter_candidates if args.teacher_filter_eval else args.eval_questions
            eval_sets.append(
                build_task_questions(
                    profile,
                    candidate_count,
                    args.seed + 91_000,
                    final_lesson_idx,
                    "heldout_translation",
                )
            )

    started = time.time()
    sentinels = sentinel_questions(args.sentinel_suite) if args.sentinel_eval else []
    sentinel_before = (
        evaluate_generic_mc(model, tokenizer, sentinels, device, args.max_length, args.chat_template)
        if sentinels
        else None
    )
    if sentinel_before is not None:
        row = {"stage": "sentinel_before", "step": -1, "seconds": time.time() - started}
        add_metrics(row, "sentinel", sentinel_before)
        append_jsonl(metrics_path, row)
        for idx, detail in enumerate(sentinel_before["details"]):
            append_jsonl(details_path, {"stage": "sentinel_before", "step": -1, "idx": idx, **detail})

    if args.teacher_filter_eval and fixed_eval_sets is None:
        for task_idx, profile in enumerate(profiles):
            candidates = eval_sets[task_idx]
            progress(f"teacher-filtering task={task_idx} language={profile.name} candidates={len(candidates)}")
            baseline_candidates = None
            if args.teacher_filter_require_baseline_wrong:
                baseline_candidates = evaluate_task_mc(
                    model,
                    tokenizer,
                    profile,
                    candidates,
                    device,
                    context=None,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                release_device_cache(device)
            context_candidates = evaluate_task_mc(
                model,
                tokenizer,
                profile,
                candidates,
                device,
                context=contexts[task_idx],
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            release_device_cache(device)
            filtered = [
                question
                for idx, (question, detail) in enumerate(zip(candidates, context_candidates["details"], strict=True))
                if detail["correct"]
                and (
                    baseline_candidates is None
                    or not bool(baseline_candidates["details"][idx]["correct"])
                )
            ]
            eval_sets[task_idx] = filtered[: args.eval_questions]
            stat = {
                "stage": "teacher_filter",
                "step": -1,
                "task_idx": task_idx,
                "language": profile.name,
                "teacher_filter_candidates": len(candidates),
                "teacher_filter_correct": len(filtered),
                "teacher_filter_selected": len(eval_sets[task_idx]),
                "teacher_filter_require_baseline_wrong": bool(args.teacher_filter_require_baseline_wrong),
                "seconds": time.time() - started,
            }
            filter_stats.append(stat)
            append_jsonl(metrics_path, stat)
    else:
        filter_stats = [
            {
                "teacher_filter_candidates": len(eval_set),
                "teacher_filter_correct": None,
                "teacher_filter_selected": len(eval_set),
                "teacher_filter_require_baseline_wrong": False,
                "eval_questions_jsonl": args.eval_questions_jsonl,
            }
            for eval_set in eval_sets
        ]

    for profile, eval_questions in zip(profiles, eval_sets, strict=True):
        for question in eval_questions:
            append_jsonl(
                questions_path,
                {
                    "task_idx": profile.idx,
                    "language": profile.name,
                    "sentence": question.sentence,
                    "answer": question.answer,
                    "options": question.options,
                    "answer_letter": question.answer_letter,
                    "category": question.category,
                },
            )

    baselines: list[dict] = []
    contexts_metrics: list[dict] = []
    for task_idx, profile in enumerate(profiles):
        progress(f"scoring before-write task={task_idx} language={profile.name}")
        baseline = evaluate_task_mc(
            model,
            tokenizer,
            profile,
            eval_sets[task_idx],
            device,
            context=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        release_device_cache(device)
        context = evaluate_task_mc(
            model,
            tokenizer,
            profile,
            eval_sets[task_idx],
            device,
            context=contexts[task_idx],
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        release_device_cache(device)
        baselines.append(baseline)
        contexts_metrics.append(context)
        row = {
            "stage": "before_write",
            "step": -1,
            "task_idx": task_idx,
            "language": profile.name,
            "seconds": time.time() - started,
            "teacher_filter_candidates": filter_stats[task_idx]["teacher_filter_candidates"],
            "teacher_filter_correct": filter_stats[task_idx]["teacher_filter_correct"],
            "teacher_filter_selected": filter_stats[task_idx]["teacher_filter_selected"],
        }
        add_metrics(row, "baseline", baseline)
        add_metrics(row, "context", context)
        append_jsonl(metrics_path, row)
        for stage, metrics in (("baseline", baseline), ("context", context)):
            for idx, detail in enumerate(metrics["details"]):
                append_jsonl(details_path, {"stage": stage, "step": -1, "task_idx": task_idx, "idx": idx, **detail})

    if args.screen_before_write_only:
        progress("screen-before-write-only requested; stopping before writes")
        append_jsonl(
            metrics_path,
            {
                "stage": "screen_complete",
                "step": -1,
                "seconds": time.time() - started,
            },
        )
        return

    acquisition_accuracy: list[float | None] = [None for _ in profiles]
    acquisition_margin: list[float | None] = [None for _ in profiles]
    old_negative_keys: dict[int, torch.Tensor] = {}

    for step, profile in enumerate(profiles):
        progress(f"writing task={step} language={profile.name} from lessons only")
        step_started = time.time()
        selected_keys: dict[int, list[torch.Tensor]] = {}
        run_intrinsic_surprise_writes(
            model,
            tokenizer,
            wrappers,
            lesson_texts[step],
            args,
            device,
            updates_path,
            slot_id=None,
            dice_anti_lesson_texts=dice_anti_lesson_texts[step],
            extra_negative_keys_by_layer=old_negative_keys if args.old_task_negative_keys else None,
            selected_keys_out_by_layer=selected_keys if args.old_task_negative_keys else None,
            max_extra_negative_rows=args.old_task_negative_max_rows,
            extra_negative_scale=args.old_task_negative_scale,
        )
        if args.old_task_negative_keys:
            for layer_idx, chunks in selected_keys.items():
                if not chunks:
                    continue
                new_rows = torch.cat(chunks, dim=0).contiguous()
                if layer_idx in old_negative_keys:
                    merged = torch.cat([old_negative_keys[layer_idx], new_rows], dim=0)
                else:
                    merged = new_rows
                if args.old_task_negative_max_rows > 0:
                    merged = evenly_cap_rows(merged, args.old_task_negative_max_rows)
                old_negative_keys[layer_idx] = merged.contiguous()
        release_device_cache(device)
        append_jsonl(
            metrics_path,
            {
                "stage": "write_complete",
                "step": step,
                "task_idx": step,
                "language": profile.name,
                "seconds": time.time() - started,
                "step_seconds": time.time() - step_started,
            },
        )

        if sentinel_before is not None:
            progress(f"evaluating sentinels after task={step}")
            sentinel_after = evaluate_generic_mc(
                model,
                tokenizer,
                sentinels,
                device,
                args.max_length,
                args.chat_template,
            )
            release_device_cache(device)
            row = {"stage": "sentinel_after_step", "step": step, "seconds": time.time() - started}
            add_metrics(row, "sentinel_before", sentinel_before)
            add_metrics(row, "sentinel_after", sentinel_after)
            row["sentinel_accuracy_delta"] = sentinel_after["accuracy"] - sentinel_before["accuracy"]
            row["sentinel_margin_delta"] = sentinel_after["mean_margin"] - sentinel_before["mean_margin"]
            add_sentinel_shift_metrics(row, sentinel_before, sentinel_after)
            sentinel_c2w = int(row.get("sentinel_correct_to_wrong", 0))
            append_jsonl(metrics_path, row)
            for idx, detail in enumerate(sentinel_after["details"]):
                append_jsonl(details_path, {"stage": "sentinel_after", "step": step, "idx": idx, **detail})
        else:
            sentinel_c2w = 0

        task0_edited_correct: int | None = None
        for eval_task_idx in range(step + 1):
            progress(f"after task={step}, evaluating task={eval_task_idx}")
            eval_profile = profiles[eval_task_idx]
            edited = evaluate_task_mc(
                model,
                tokenizer,
                eval_profile,
                eval_sets[eval_task_idx],
                device,
                context=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            release_device_cache(device)
            if eval_task_idx == step:
                acquisition_accuracy[eval_task_idx] = edited["accuracy"]
                acquisition_margin[eval_task_idx] = edited["mean_margin"]
            row = {
                "stage": "after_step",
                "step": step,
                "task_idx": eval_task_idx,
                "language": eval_profile.name,
                "seconds": time.time() - started,
            }
            add_metrics(row, "baseline", baselines[eval_task_idx])
            add_metrics(row, "context", contexts_metrics[eval_task_idx])
            add_metrics(row, "edited", edited)
            row["accuracy_delta"] = edited["accuracy"] - baselines[eval_task_idx]["accuracy"]
            row["internalization_ratio"] = (
                (edited["accuracy"] - baselines[eval_task_idx]["accuracy"])
                / (contexts_metrics[eval_task_idx]["accuracy"] - baselines[eval_task_idx]["accuracy"] + 1e-12)
            )
            row["closed_book_half_score_reached"] = edited["accuracy"] >= 0.5
            if acquisition_accuracy[eval_task_idx] is not None:
                row["acquisition_reference_accuracy"] = acquisition_accuracy[eval_task_idx]
                row["retention_accuracy_delta_from_acquisition"] = edited["accuracy"] - acquisition_accuracy[eval_task_idx]
                row["retention_preserved_from_acquisition"] = (
                    row["retention_accuracy_delta_from_acquisition"] >= -1e-12
                )
            if acquisition_margin[eval_task_idx] is not None:
                row["acquisition_reference_margin"] = acquisition_margin[eval_task_idx]
                row["retention_margin_delta_from_acquisition"] = edited["mean_margin"] - acquisition_margin[eval_task_idx]
            append_jsonl(metrics_path, row)
            for idx, detail in enumerate(edited["details"]):
                append_jsonl(details_path, {"stage": "edited", "step": step, "task_idx": eval_task_idx, "idx": idx, **detail})
            if step == 0 and eval_task_idx == 0:
                task0_edited_correct = int(edited["correct"])

        should_stop = False
        stop_reasons: list[str] = []
        if args.early_stop_c2w_over >= 0 and sentinel_c2w > args.early_stop_c2w_over:
            should_stop = True
            stop_reasons.append(f"sentinel_c2w={sentinel_c2w}>{args.early_stop_c2w_over}")
        if (
            step == 0
            and args.early_stop_task0_min_edited_correct >= 0
            and task0_edited_correct is not None
            and task0_edited_correct < args.early_stop_task0_min_edited_correct
        ):
            should_stop = True
            stop_reasons.append(
                f"task0_edited_correct={task0_edited_correct}<"
                f"{args.early_stop_task0_min_edited_correct}"
            )
        if should_stop:
            progress(f"early stopping after task={step}: {', '.join(stop_reasons)}")
            append_jsonl(
                metrics_path,
                {
                    "stage": "early_stop",
                    "step": step,
                    "seconds": time.time() - started,
                    "reasons": stop_reasons,
                    "sentinel_correct_to_wrong": sentinel_c2w,
                    "task0_edited_correct": task0_edited_correct,
                },
            )
            break

    clear_active_slot_weights(model)
    print(f"Wrote intrinsic continual metrics to {metrics_path}")
    print(f"Wrote intrinsic continual updates to {updates_path}")


if __name__ == "__main__":
    main()
