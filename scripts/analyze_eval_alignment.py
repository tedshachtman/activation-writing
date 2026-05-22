#!/usr/bin/env python3
"""Summarize score-space alignment between context and edited runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_details(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    by_stage: dict[str, dict[int, dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stage = row.get("stage")
            idx = row.get("idx")
            if stage is None or idx is None:
                continue
            by_stage.setdefault(str(stage), {})[int(idx)] = row
    return by_stage


def _center(xs: list[float]) -> list[float]:
    mean = sum(xs) / max(len(xs), 1)
    return [x - mean for x in xs]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(max(_dot(a, a), 0.0))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _score_delta(after: dict[str, Any], before: dict[str, Any], *, centered: bool) -> list[float]:
    a = [float(x) for x in after["scores"]]
    b = [float(x) for x in before["scores"]]
    if centered:
        a = _center(a)
        b = _center(b)
    return [x - y for x, y in zip(a, b)]


def _answer_score(row: dict[str, Any], *, centered: bool) -> float:
    scores = [float(x) for x in row["scores"]]
    if centered:
        scores = _center(scores)
    answer_letter = str(row["answer_letter"]).strip().upper()
    answer_idx = ord(answer_letter) - ord("A")
    return scores[answer_idx]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    stages = _read_details(run_dir / "eval_details.jsonl")
    baseline = stages.get("baseline", {})
    context = stages.get("context", {})
    edited = stages.get("edited", {})
    common = sorted(set(baseline) & set(context) & set(edited))

    raw_ctx: list[float] = []
    raw_edit: list[float] = []
    ctr_ctx: list[float] = []
    ctr_edit: list[float] = []
    per_item_cos: list[float] = []
    per_item_ctr_cos: list[float] = []
    answer_ctx_gain: list[float] = []
    answer_edit_gain: list[float] = []
    answer_ctx_gain_centered: list[float] = []
    answer_edit_gain_centered: list[float] = []
    opportunity_answer_edit_gain: list[float] = []
    opportunity_answer_edit_gain_centered: list[float] = []

    changed = gained = lost = 0
    context_only = captured = 0
    for idx in common:
        b = baseline[idx]
        c = context[idx]
        e = edited[idx]
        d_ctx = _score_delta(c, b, centered=False)
        d_edit = _score_delta(e, b, centered=False)
        d_ctx_c = _score_delta(c, b, centered=True)
        d_edit_c = _score_delta(e, b, centered=True)
        raw_ctx.extend(d_ctx)
        raw_edit.extend(d_edit)
        ctr_ctx.extend(d_ctx_c)
        ctr_edit.extend(d_edit_c)
        n_ctx = _norm(d_ctx)
        n_edit = _norm(d_edit)
        n_ctx_c = _norm(d_ctx_c)
        n_edit_c = _norm(d_edit_c)
        if n_ctx > 0 and n_edit > 0:
            per_item_cos.append(_dot(d_ctx, d_edit) / (n_ctx * n_edit))
        if n_ctx_c > 0 and n_edit_c > 0:
            per_item_ctr_cos.append(_dot(d_ctx_c, d_edit_c) / (n_ctx_c * n_edit_c))

        before_correct = bool(b.get("correct"))
        context_correct = bool(c.get("correct"))
        edited_correct = bool(e.get("correct"))
        if e.get("prediction") != b.get("prediction"):
            changed += 1
        if not before_correct and edited_correct:
            gained += 1
        if before_correct and not edited_correct:
            lost += 1
        if (not before_correct) and context_correct:
            context_only += 1
            if edited_correct:
                captured += 1
            opportunity_answer_edit_gain.append(_answer_score(e, centered=False) - _answer_score(b, centered=False))
            opportunity_answer_edit_gain_centered.append(
                _answer_score(e, centered=True) - _answer_score(b, centered=True)
            )
        answer_ctx_gain.append(_answer_score(c, centered=False) - _answer_score(b, centered=False))
        answer_edit_gain.append(_answer_score(e, centered=False) - _answer_score(b, centered=False))
        answer_ctx_gain_centered.append(_answer_score(c, centered=True) - _answer_score(b, centered=True))
        answer_edit_gain_centered.append(_answer_score(e, centered=True) - _answer_score(b, centered=True))

    raw_dot = _dot(raw_ctx, raw_edit)
    raw_ctx_norm2 = _dot(raw_ctx, raw_ctx)
    raw_edit_norm = _norm(raw_edit)
    raw_ctx_norm = _norm(raw_ctx)
    ctr_dot = _dot(ctr_ctx, ctr_edit)
    ctr_ctx_norm2 = _dot(ctr_ctx, ctr_ctx)
    ctr_edit_norm = _norm(ctr_edit)
    ctr_ctx_norm = _norm(ctr_ctx)

    sent_before = stages.get("sentinel_before", {})
    sent_after = stages.get("sentinel_after", {})
    sent_common = sorted(set(sent_before) & set(sent_after))
    sentinel_c2w = 0
    sentinel_w2c = 0
    sentinel_before_correct_drops: list[float] = []
    sentinel_abs_drifts: list[float] = []
    for idx in sent_common:
        b = sent_before[idx]
        a = sent_after[idx]
        if bool(b.get("correct")) and not bool(a.get("correct")):
            sentinel_c2w += 1
        if (not bool(b.get("correct"))) and bool(a.get("correct")):
            sentinel_w2c += 1
        delta = float(a.get("margin", 0.0)) - float(b.get("margin", 0.0))
        sentinel_abs_drifts.append(abs(delta))
        if bool(b.get("correct")):
            sentinel_before_correct_drops.append(max(0.0, -delta))

    return {
        "run": run_dir.name,
        "n": len(common),
        "baseline_correct": sum(1 for i in common if baseline[i].get("correct")),
        "context_correct": sum(1 for i in common if context[i].get("correct")),
        "edited_correct": sum(1 for i in common if edited[i].get("correct")),
        "changed_predictions": changed,
        "gained": gained,
        "lost": lost,
        "context_only": context_only,
        "captured_context_only": captured,
        "raw_global_cos": raw_dot / (raw_ctx_norm * raw_edit_norm) if raw_ctx_norm > 0 and raw_edit_norm > 0 else 0.0,
        "raw_projection_ratio": raw_dot / raw_ctx_norm2 if raw_ctx_norm2 > 0 else 0.0,
        "centered_global_cos": ctr_dot / (ctr_ctx_norm * ctr_edit_norm) if ctr_ctx_norm > 0 and ctr_edit_norm > 0 else 0.0,
        "centered_projection_ratio": ctr_dot / ctr_ctx_norm2 if ctr_ctx_norm2 > 0 else 0.0,
        "mean_item_cos": _mean(per_item_cos),
        "mean_item_centered_cos": _mean(per_item_ctr_cos),
        "mean_context_answer_gain": _mean(answer_ctx_gain),
        "mean_edit_answer_gain": _mean(answer_edit_gain),
        "mean_context_answer_gain_centered": _mean(answer_ctx_gain_centered),
        "mean_edit_answer_gain_centered": _mean(answer_edit_gain_centered),
        "mean_opportunity_edit_answer_gain": _mean(opportunity_answer_edit_gain),
        "mean_opportunity_edit_answer_gain_centered": _mean(opportunity_answer_edit_gain_centered),
        "sentinel_before_n": len(sent_common),
        "sentinel_c2w": sentinel_c2w,
        "sentinel_w2c": sentinel_w2c,
        "sentinel_mean_abs_margin_delta": _mean(sentinel_abs_drifts),
        "sentinel_before_correct_mean_drop": _mean(sentinel_before_correct_drops),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of a compact table.")
    args = parser.parse_args()

    summaries = [summarize_run(run) for run in args.runs]
    if args.json:
        for row in summaries:
            print(json.dumps(row, sort_keys=True))
        return

    headers = [
        "run",
        "base",
        "ctx",
        "edit",
        "chg",
        "gain",
        "lost",
        "cap",
        "ctr_cos",
        "ctr_proj",
        "ansΔ",
        "opp_ansΔ",
        "sent_c2w",
        "sent_drop",
    ]
    print("\t".join(headers))
    for row in summaries:
        print(
            "\t".join(
                [
                    str(row["run"]),
                    f"{row['baseline_correct']}/{row['n']}",
                    f"{row['context_correct']}/{row['n']}",
                    f"{row['edited_correct']}/{row['n']}",
                    str(row["changed_predictions"]),
                    str(row["gained"]),
                    str(row["lost"]),
                    f"{row['captured_context_only']}/{row['context_only']}",
                    f"{row['centered_global_cos']:.3f}",
                    f"{row['centered_projection_ratio']:.3f}",
                    f"{row['mean_edit_answer_gain_centered']:.3f}",
                    f"{row['mean_opportunity_edit_answer_gain_centered']:.3f}",
                    str(row["sentinel_c2w"]),
                    f"{row['sentinel_before_correct_mean_drop']:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
