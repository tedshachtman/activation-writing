"""Sequential attention-then-MLP closed-form writes.

For each selected layer, this runner first writes the context-induced attention
output-projection contribution, then writes the remaining block-output delta
through the MLP down-projection. This tests the idea that attention reconstructs
working-memory integration and MLPs transform that state into higher-level rule
features.
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
    capture_attention_io,
    capture_attention_projection_io,
    capture_block_io,
    capture_layer_io,
    install_additive_attention_projection_memory,
    install_additive_attention_memory,
    install_additive_memory,
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
    parser.add_argument("--output", default="runs/joint_attention_mlp_write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--qk-projections", nargs="*", choices=["q", "k"], default=[])
    parser.add_argument("--trace-probes", type=int, default=32)
    parser.add_argument("--trace-last-tokens", type=int, default=1)
    parser.add_argument("--q-trace-last-tokens", type=int, default=1)
    parser.add_argument("--k-trace-last-tokens", type=int, default=12)
    parser.add_argument("--eval-capture-last-tokens", type=int, default=1)
    parser.add_argument("--q-eta", type=float, default=1.0)
    parser.add_argument("--k-eta", type=float, default=1.0)
    parser.add_argument("--attention-eta", type=float, default=1.0)
    parser.add_argument("--mlp-eta", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    parser.add_argument("--negative-guards", type=int, default=8)
    parser.add_argument("--rival-negative-guards", type=int, default=8)
    parser.add_argument("--memory-gate", action="store_true")
    parser.add_argument("--memory-gate-final-token-only", action="store_true")
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


def trace_last_tokens_for_qk(args: argparse.Namespace, projection: str) -> int:
    return args.q_trace_last_tokens if projection == "q" else args.k_trace_last_tokens


def eta_for_qk(args: argparse.Namespace, projection: str) -> float:
    return args.q_eta if projection == "q" else args.k_eta


def final_token_gate_for_qk(args: argparse.Namespace, projection: str) -> bool:
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
    qk_wrappers = {
        projection: install_additive_attention_projection_memory(
            model,
            args.layers,
            projection,
            memory_dtype=torch.float32,
        )
        for projection in args.qk_projections
    }
    attention_wrappers = install_additive_attention_memory(model, args.layers, memory_dtype=torch.float32)
    mlp_wrappers = install_additive_memory(model, args.layers, memory_dtype=torch.float32)
    domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)

    for paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        probes = make_candidate_probes(domain, args.trace_probes, seed=args.seed * 500_000 + paper_idx)
        paper = domain.render_paper()
        key_prompts = prompts_for_questions(tokenizer, probes, None, args.chat_template)
        full_prompts = prompts_for_questions(tokenizer, probes, paper, args.chat_template)
        full_qk = {
            projection: capture_attention_projection_io(
                model,
                tokenizer,
                full_prompts,
                args.layers,
                projection,
                device,
                args.batch_size,
                args.max_length,
                trace_last_tokens_for_qk(args, projection),
            )
            for projection in args.qk_projections
        }

        full_attention = capture_attention_io(
            model,
            tokenizer,
            full_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            args.trace_last_tokens,
        )
        full_blocks = capture_block_io(
            model,
            tokenizer,
            full_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            args.trace_last_tokens,
        )

        guard_questions = negative_questions_for_domain(domain, args, paper_idx)
        guard_prompts = prompts_for_questions(tokenizer, guard_questions, None, args.chat_template)
        guard_qk = {
            projection: capture_attention_projection_io(
                model,
                tokenizer,
                guard_prompts,
                args.layers,
                projection,
                device,
                args.batch_size,
                args.max_length,
                trace_last_tokens_for_qk(args, projection),
            )
            for projection in args.qk_projections
        }
        guard_attention = capture_attention_io(
            model,
            tokenizer,
            guard_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            args.eval_capture_last_tokens,
        )
        guard_mlp = capture_layer_io(
            model,
            tokenizer,
            guard_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            args.eval_capture_last_tokens,
        )

        updates: list[tuple[str, int, torch.Tensor]] = []
        for layer_idx in args.layers:
            for projection in args.qk_projections:
                current_qk = capture_attention_projection_io(
                    model,
                    tokenizer,
                    key_prompts,
                    [layer_idx],
                    projection,
                    device,
                    args.batch_size,
                    args.max_length,
                    trace_last_tokens_for_qk(args, projection),
                )
                qk_targets = (
                    full_qk[projection][layer_idx].outputs.float()
                    - current_qk[layer_idx].outputs.float()
                )
                qk_update, qk_stats = protected_ridge_update(
                    current_qk[layer_idx].keys,
                    qk_targets,
                    negative_keys=guard_qk[projection][layer_idx].keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=eta_for_qk(args, projection),
                    max_update_norm=args.max_update_norm,
                )
                if args.memory_gate:
                    qk_wrappers[projection][layer_idx].set_gate_last_token_only_(
                        final_token_gate_for_qk(args, projection)
                    )
                    qk_wrappers[projection][layer_idx].set_gate_keys_(
                        current_qk[layer_idx].keys,
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                    )
                qk_wrappers[projection][layer_idx].add_memory_(qk_update)
                updates.append((projection, layer_idx, qk_update))
                qk_row = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "module": f"attention_{projection}_proj",
                    "layer": layer_idx,
                    "trace_rows": int(current_qk[layer_idx].keys.shape[0]),
                    "guard_rows": int(guard_qk[projection][layer_idx].keys.shape[0]),
                    "seconds": time.time() - started,
                }
                qk_row.update(asdict(qk_stats))
                append_jsonl(updates_path, qk_row)

            current_attention = capture_attention_io(
                model,
                tokenizer,
                key_prompts,
                [layer_idx],
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
            attention_targets = (
                full_attention[layer_idx].outputs.float()
                - current_attention[layer_idx].outputs.float()
            )
            attention_update, attention_stats = protected_ridge_update(
                current_attention[layer_idx].keys,
                attention_targets,
                negative_keys=guard_attention[layer_idx].keys,
                ridge=args.ridge,
                negative_weight=args.negative_weight,
                eta=args.attention_eta,
                max_update_norm=args.max_update_norm,
            )
            if args.memory_gate:
                attention_wrappers[layer_idx].set_gate_last_token_only_(args.memory_gate_final_token_only)
                attention_wrappers[layer_idx].set_gate_keys_(
                    current_attention[layer_idx].keys,
                    threshold=args.memory_gate_threshold,
                    temperature=args.memory_gate_temperature,
                )
            attention_wrappers[layer_idx].add_memory_(attention_update)
            updates.append(("attention", layer_idx, attention_update))
            attention_row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "module": "attention_o_proj",
                "layer": layer_idx,
                "trace_rows": int(current_attention[layer_idx].keys.shape[0]),
                "guard_rows": int(guard_attention[layer_idx].keys.shape[0]),
                "seconds": time.time() - started,
            }
            attention_row.update(asdict(attention_stats))
            append_jsonl(updates_path, attention_row)

            current_blocks = capture_block_io(
                model,
                tokenizer,
                key_prompts,
                [layer_idx],
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
            current_mlp = capture_layer_io(
                model,
                tokenizer,
                key_prompts,
                [layer_idx],
                device,
                args.batch_size,
                args.max_length,
                args.trace_last_tokens,
            )
            mlp_targets = full_blocks[layer_idx].outputs.float() - current_blocks[layer_idx].outputs.float()
            mlp_update, mlp_stats = protected_ridge_update(
                current_mlp[layer_idx].keys,
                mlp_targets,
                negative_keys=guard_mlp[layer_idx].keys,
                ridge=args.ridge,
                negative_weight=args.negative_weight,
                eta=args.mlp_eta,
                max_update_norm=args.max_update_norm,
            )
            if args.memory_gate:
                mlp_wrappers[layer_idx].set_gate_last_token_only_(args.memory_gate_final_token_only)
                mlp_wrappers[layer_idx].set_gate_keys_(
                    current_mlp[layer_idx].keys,
                    threshold=args.memory_gate_threshold,
                    temperature=args.memory_gate_temperature,
                )
            mlp_wrappers[layer_idx].add_memory_(mlp_update)
            updates.append(("mlp", layer_idx, mlp_update))
            mlp_row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "module": "mlp_down_proj",
                "layer": layer_idx,
                "trace_rows": int(current_mlp[layer_idx].keys.shape[0]),
                "guard_rows": int(guard_mlp[layer_idx].keys.shape[0]),
                "seconds": time.time() - started,
            }
            mlp_row.update(asdict(mlp_stats))
            append_jsonl(updates_path, mlp_row)

        gauntlets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": heldout_questions, **gauntlets}

        for module, layer_idx, update in updates:
            if module == "attention":
                wrapper = attention_wrappers[layer_idx]
            elif module == "mlp":
                wrapper = mlp_wrappers[layer_idx]
            else:
                wrapper = qk_wrappers[module][layer_idx]
            wrapper.add_memory_(-update)
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
        for module, layer_idx, update in updates:
            if module == "attention":
                wrapper = attention_wrappers[layer_idx]
            elif module == "mlp":
                wrapper = mlp_wrappers[layer_idx]
            else:
                wrapper = qk_wrappers[module][layer_idx]
            wrapper.add_memory_(update)
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
                "attention_eta": args.attention_eta,
                "mlp_eta": args.mlp_eta,
                "qk_projections": args.qk_projections,
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

    print(f"Wrote joint attention/MLP metrics to {metrics_path}")
    print(f"Wrote joint attention/MLP update stats to {updates_path}")
    if args.retention_eval:
        print(f"Wrote joint attention/MLP retention metrics to {retention_path}")


if __name__ == "__main__":
    main()
