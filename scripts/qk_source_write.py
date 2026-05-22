"""Closed-form Q/K attention projection writes.

This is a high-risk diagnostic: Q/K writes change routing, not just residual
values. The first pass writes final-token Q projections and suffix-window K
projections from context-induced teacher deltas, then evaluates whether this
helps synthetic rule internalization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from caic.evaluation import evaluate_yes_no, format_question_prompt
from caic.experiment import load_domain_rows
from caic.modeling import (
    capture_attention_projection_io,
    install_additive_attention_projection_memory,
    load_model_and_tokenizer,
)
from caic.synthetic import (
    DomainSpec,
    general_guard_questions,
    make_candidate_probes,
    make_gauntlet_questions,
    make_near_collision_questions,
    make_null_document,
)
from caic.tsoc import protected_ridge_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--domains-jsonl", required=True)
    parser.add_argument("--papers", type=int, default=1)
    parser.add_argument("--output", default="runs/qk_source_write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--projections", nargs="+", choices=["q", "k"], default=["q", "k"])
    parser.add_argument("--trace-probes", type=int, default=32)
    parser.add_argument("--q-trace-last-tokens", type=int, default=1)
    parser.add_argument("--k-trace-last-tokens", type=int, default=12)
    parser.add_argument("--q-eta", type=float, default=1.0)
    parser.add_argument("--k-eta", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    parser.add_argument("--negative-guards", type=int, default=8)
    parser.add_argument("--rival-negative-guards", type=int, default=8)
    parser.add_argument("--memory-gate", action="store_true")
    parser.add_argument("--q-gate-final-token-only", action="store_true", default=True)
    parser.add_argument("--k-gate-final-token-only", action="store_true")
    parser.add_argument("--memory-gate-threshold", type=float, default=0.95)
    parser.add_argument("--memory-gate-temperature", type=float, default=80.0)
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--retention-eval", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def prompts_for_questions(tokenizer, questions: list, paper: str | None, use_chat_template: bool) -> list[str]:
    return [
        format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        for record in questions
    ]


def negative_questions_for_domain(domain: DomainSpec, args: argparse.Namespace, paper_idx: int) -> list:
    questions = general_guard_questions()[: args.negative_guards]
    if args.rival_negative_guards > 0:
        rivals = make_near_collision_questions(
            domain,
            args.rival_negative_guards * 2,
            seed=args.seed * 300_000 + paper_idx,
            include_rival_prompts=True,
        )
        rival_only = [record for record in rivals if record.category == "near_collision_rival"]
        questions.extend(rival_only[: args.rival_negative_guards])
    return questions


def trace_last_tokens_for_projection(args: argparse.Namespace, projection: str) -> int:
    return args.q_trace_last_tokens if projection == "q" else args.k_trace_last_tokens


def eta_for_projection(args: argparse.Namespace, projection: str) -> float:
    return args.q_eta if projection == "q" else args.k_eta


def final_token_gate_for_projection(args: argparse.Namespace, projection: str) -> bool:
    return args.q_gate_final_token_only if projection == "q" else args.k_gate_final_token_only


def add_eval_metrics(row: dict, prefix: str, result) -> None:
    row.update(result.to_dict(prefix))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    updates_path = output_dir / "updates.jsonl"
    retention_path = output_dir / "retention.jsonl"
    for path in (metrics_path, updates_path, retention_path):
        if path.exists():
            path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    wrappers = {
        projection: install_additive_attention_projection_memory(
            model,
            args.layers,
            projection,
            memory_dtype=torch.float32,
        )
        for projection in args.projections
    }
    domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)

    for paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        probes = make_candidate_probes(domain, args.trace_probes, seed=args.seed * 500_000 + paper_idx)
        paper = domain.render_paper()
        full_prompts = prompts_for_questions(tokenizer, probes, paper, args.chat_template)
        key_prompts = prompts_for_questions(tokenizer, probes, None, args.chat_template)
        guard_questions = negative_questions_for_domain(domain, args, paper_idx)
        guard_prompts = prompts_for_questions(tokenizer, guard_questions, None, args.chat_template)

        full_captures = {
            projection: capture_attention_projection_io(
                model,
                tokenizer,
                full_prompts,
                args.layers,
                projection,
                device,
                args.batch_size,
                args.max_length,
                trace_last_tokens_for_projection(args, projection),
            )
            for projection in args.projections
        }
        guard_captures = {
            projection: capture_attention_projection_io(
                model,
                tokenizer,
                guard_prompts,
                args.layers,
                projection,
                device,
                args.batch_size,
                args.max_length,
                trace_last_tokens_for_projection(args, projection),
            )
            for projection in args.projections
        }

        updates: list[tuple[str, int, torch.Tensor]] = []
        for layer_idx in args.layers:
            for projection in args.projections:
                current = capture_attention_projection_io(
                    model,
                    tokenizer,
                    key_prompts,
                    [layer_idx],
                    projection,
                    device,
                    args.batch_size,
                    args.max_length,
                    trace_last_tokens_for_projection(args, projection),
                )
                trace_keys = current[layer_idx].keys
                targets = full_captures[projection][layer_idx].outputs.float() - current[layer_idx].outputs.float()
                update, stats = protected_ridge_update(
                    trace_keys,
                    targets,
                    negative_keys=guard_captures[projection][layer_idx].keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=eta_for_projection(args, projection),
                    max_update_norm=args.max_update_norm,
                )
                wrapper = wrappers[projection][layer_idx]
                if args.memory_gate:
                    wrapper.set_gate_last_token_only_(final_token_gate_for_projection(args, projection))
                    wrapper.set_gate_keys_(
                        trace_keys,
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                    )
                wrapper.add_memory_(update)
                updates.append((projection, layer_idx, update))
                row = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "projection": projection,
                    "layer": layer_idx,
                    "trace_rows": int(trace_keys.shape[0]),
                    "guard_rows": int(guard_captures[projection][layer_idx].keys.shape[0]),
                    "eta": eta_for_projection(args, projection),
                    "gate_final_token_only": final_token_gate_for_projection(args, projection),
                    "seconds": time.time() - started,
                }
                row.update(asdict(stats))
                append_jsonl(updates_path, row)

        gauntlets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": heldout_questions, **gauntlets}

        for projection, layer_idx, update in updates:
            wrappers[projection][layer_idx].add_memory_(-update)
        baseline_results = {
            group: evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for group, questions in eval_groups.items()
        }
        context_results = {
            group: evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=paper,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for group, questions in eval_groups.items()
        }
        for projection, layer_idx, update in updates:
            wrappers[projection][layer_idx].add_memory_(update)
        edited_results = {
            group: evaluate_yes_no(
                model,
                tokenizer,
                questions,
                device,
                paper=None,
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for group, questions in eval_groups.items()
        }
        for group, questions in eval_groups.items():
            row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "group": group,
                "question_count": len(questions),
                "projections": args.projections,
                "seconds": time.time() - started,
            }
            add_eval_metrics(row, "baseline", baseline_results[group])
            add_eval_metrics(row, "context", context_results[group])
            add_eval_metrics(row, "edited", edited_results[group])
            row["accuracy_delta"] = edited_results[group].accuracy - baseline_results[group].accuracy
            row["internalization_ratio"] = (
                (edited_results[group].accuracy - baseline_results[group].accuracy)
                / (context_results[group].accuracy - baseline_results[group].accuracy + 1e-12)
            )
            append_jsonl(metrics_path, row)

        if args.retention_eval:
            for seen_idx in range(paper_idx + 1):
                seen_domain = domains[seen_idx]
                seen_questions = eval_sets[seen_idx]
                retention_result = evaluate_yes_no(
                    model,
                    tokenizer,
                    seen_questions,
                    device,
                    paper=None,
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                context_result = evaluate_yes_no(
                    model,
                    tokenizer,
                    seen_questions,
                    device,
                    paper=seen_domain.render_paper(),
                    max_length=args.max_length,
                    use_chat_template=args.chat_template,
                )
                row = {
                    "after_paper_idx": paper_idx,
                    "seen_paper_idx": seen_idx,
                    "seen_domain_id": seen_domain.domain_id,
                    "seen_title": seen_domain.title,
                    "question_count": len(seen_questions),
                    "seconds": time.time() - started,
                }
                add_eval_metrics(row, "retention", retention_result)
                add_eval_metrics(row, "context", context_result)
                append_jsonl(retention_path, row)

    print(f"Wrote Q/K metrics to {metrics_path}")
    print(f"Wrote Q/K update stats to {updates_path}")
    if args.retention_eval:
        print(f"Wrote Q/K retention metrics to {retention_path}")


if __name__ == "__main__":
    main()
