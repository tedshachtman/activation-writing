"""Run a first Trajectory-Source Operator Consolidation diagnostic.

This script is deliberately narrower than the main CAIC runner. It tests the
new hypothesis from GPT-5.5 Pro: write a block-local context source term from
the paper trace, not raw answer-position teacher/student deltas.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from caic.evaluation import evaluate_yes_no, format_model_prompt, format_question_prompt
from caic.experiment import (
    answer_unembedding_direction,
    load_domain_rows,
    project_rows_away_from_direction,
)
from caic.modeling import (
    capture_block_io,
    capture_layer_io,
    install_additive_memory,
    load_model_and_tokenizer,
    memory_norms,
)
from caic.synthetic import (
    DomainSpec,
    generate_domains,
    general_guard_questions,
    make_candidate_probes,
    make_eval_questions,
    make_gauntlet_questions,
    make_near_collision_questions,
    make_null_document,
)
from caic.tsoc import (
    block_source_targets,
    mean_row_cosine,
    mean_row_l2,
    principal_components,
    project_rows_away_from_basis,
    projection_energy_fraction,
    protected_ridge_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--domains-jsonl", default=None)
    parser.add_argument("--papers", type=int, default=1)
    parser.add_argument("--output", default="runs/tsoc_source_write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain-difficulty", choices=["easy", "medium", "standard"], default="easy")
    parser.add_argument("--eval-questions", type=int, default=30)
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument(
        "--trace-source",
        choices=["paper_use", "question_probes"],
        default="paper_use",
        help="question_probes is a diagnostic trace source, not the final deployment method.",
    )
    parser.add_argument("--trace-probes", type=int, default=32)
    parser.add_argument("--trace-last-tokens", type=int, default=96)
    parser.add_argument("--eval-capture-last-tokens", type=int, default=12)
    parser.add_argument(
        "--target-mode",
        choices=["source_delta", "raw_block_delta"],
        default="source_delta",
    )
    parser.add_argument(
        "--sequential-replay",
        action="store_true",
        help="Write layers one at a time against the current edited no-context trajectory.",
    )
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    parser.add_argument(
        "--key-pcs-remove",
        type=int,
        default=0,
        help="Remove top common MLP-key PCs before the ridge solve, then project the update into the same subspace.",
    )
    parser.add_argument(
        "--key-pc-source",
        choices=["trace", "guards", "trace_and_guards"],
        default="trace_and_guards",
    )
    parser.add_argument("--project-answer-direction", action="store_true")
    parser.add_argument("--nuisance-pcs", type=int, default=0)
    parser.add_argument(
        "--nuisance-sources",
        nargs="*",
        choices=["answer_controls", "null_doc_delta", "target"],
        default=[],
    )
    parser.add_argument("--negative-guards", type=int, default=8)
    parser.add_argument("--rival-negative-guards", type=int, default=8)
    parser.add_argument("--memory-gate", action="store_true")
    parser.add_argument("--memory-gate-threshold", type=float, default=0.90)
    parser.add_argument("--memory-gate-temperature", type=float, default=80.0)
    parser.add_argument(
        "--memory-gate-final-token-only",
        action="store_true",
        help="Diagnostic: apply gated memory only to the final sequence token.",
    )
    parser.add_argument(
        "--alignment-diagnostics",
        action="store_true",
        help="Capture eval teacher/null block targets and compare them with the write effect.",
    )
    parser.add_argument(
        "--retention-eval",
        action="store_true",
        help="After each write, evaluate heldout accuracy for all papers seen so far.",
    )
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


def render_tsoc_support_and_use(domain: DomainSpec) -> tuple[str, str]:
    """Split a synthetic paper into support P and later use-sites U.

    U uses only the paper's rendered examples and generic prose. Hidden DSL
    labels are not consulted here beyond what is already in the paper text.
    """

    rule_lines = "\n".join(f"{idx + 1}. {rule.render()}" for idx, rule in enumerate(domain.rules))
    support = (
        f"{domain.title}: Chain Validity Notes\n\n"
        f"This note defines a synthetic rule system named {domain.title}. "
        f"A chain is a comma-separated sequence of marked operators. "
        f"The operators are {', '.join(domain.operators)}. "
        f"The marks are {', '.join(domain.marks)}.\n\n"
        f"Rules:\n{rule_lines}\n"
    )
    examples = "\n".join(
        f"- Under {domain.title}, the chain {domain.render_chain(ex.chain)} is "
        f"{'valid' if ex.answer else 'invalid'}."
        for ex in domain.examples
    )
    use = (
        "Worked applications:\n"
        f"{examples}\n\n"
        "Operational use: a later question may present a new chain in this same "
        "synthetic system. To answer it, track the operators, marks, exceptions, "
        "and rule interactions before deciding whether the chain is valid."
    )
    return support, use


def paper_use_trace_prompts_for_domain(
    tokenizer,
    domain: DomainSpec,
    use_chat_template: bool,
) -> tuple[list[str], list[str], list[str]]:
    support, use = render_tsoc_support_and_use(domain)
    null_support = make_null_document(
        seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id)),
        approx_words=len(support.split()),
    )
    instruction = "Read this note and continue tracking the rule system it defines.\n\n"
    full = format_model_prompt(tokenizer, instruction + support + "\n\n" + use, use_chat_template)
    null = format_model_prompt(tokenizer, instruction + null_support + "\n\n" + use, use_chat_template)
    return [full], [null], [null]


def question_probe_trace_prompts_for_domain(
    tokenizer,
    domain: DomainSpec,
    args: argparse.Namespace,
    paper_idx: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build question-shaped use-sites without putting answers in the trace."""

    probes = make_candidate_probes(
        domain,
        args.trace_probes,
        seed=args.seed * 500_000 + paper_idx,
    )
    paper = domain.render_paper()
    null_doc = make_null_document(
        seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id)),
        approx_words=len(paper.split()),
    )
    full_prompts = prompts_for_questions(tokenizer, probes, paper=paper, use_chat_template=args.chat_template)
    null_prompts = prompts_for_questions(tokenizer, probes, paper=null_doc, use_chat_template=args.chat_template)
    key_prompts = prompts_for_questions(tokenizer, probes, paper=None, use_chat_template=args.chat_template)
    return full_prompts, null_prompts, key_prompts


def trace_prompts_for_domain(
    tokenizer,
    domain: DomainSpec,
    args: argparse.Namespace,
    paper_idx: int,
) -> tuple[list[str], list[str], list[str]]:
    if args.trace_source == "paper_use":
        return paper_use_trace_prompts_for_domain(tokenizer, domain, args.chat_template)
    if args.trace_source == "question_probes":
        return question_probe_trace_prompts_for_domain(tokenizer, domain, args, paper_idx)
    raise ValueError(f"Unknown trace source: {args.trace_source}")


def prompts_for_questions(tokenizer, questions: list, paper: str | None, use_chat_template: bool) -> list[str]:
    return [
        format_question_prompt(tokenizer, record.question, paper=paper, use_chat_template=use_chat_template)
        for record in questions
    ]


def load_or_generate_domains(args: argparse.Namespace) -> tuple[list[DomainSpec], list[list]]:
    if args.domains_jsonl:
        return load_domain_rows(Path(args.domains_jsonl), args.papers)
    domains = generate_domains(args.papers, seed=args.seed, difficulty=args.domain_difficulty)
    eval_sets = [
        make_eval_questions(domain, args.eval_questions, seed=args.seed * 10_000 + idx)
        for idx, domain in enumerate(domains)
    ]
    return domains, eval_sets


def negative_questions_for_domain(
    domain: DomainSpec,
    args: argparse.Namespace,
    paper_idx: int,
) -> list:
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


def negative_prompts_for_domain(
    tokenizer,
    domain: DomainSpec,
    args: argparse.Namespace,
    paper_idx: int,
) -> list[str]:
    questions = negative_questions_for_domain(domain, args, paper_idx)
    return prompts_for_questions(tokenizer, questions, paper=None, use_chat_template=args.chat_template)


def capture_state_reentry(
    model,
    tokenizer,
    domain: DomainSpec,
    questions: list,
    layers: list[int],
    device: torch.device,
    args: argparse.Namespace,
    teacher_captures: dict[int, torch.Tensor] | None = None,
    student_captures: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    paper = domain.render_paper()
    teacher_prompts = prompts_for_questions(tokenizer, questions, paper=paper, use_chat_template=args.chat_template)
    student_prompts = prompts_for_questions(tokenizer, questions, paper=None, use_chat_template=args.chat_template)
    if teacher_captures is None:
        teacher = capture_layer_io(
            model,
            tokenizer,
            teacher_prompts,
            layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.eval_capture_last_tokens,
        )
        teacher_captures = {idx: cap.outputs for idx, cap in teacher.items()}
    if student_captures is None:
        student = capture_layer_io(
            model,
            tokenizer,
            student_prompts,
            layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.eval_capture_last_tokens,
        )
        student_captures = {idx: cap.outputs for idx, cap in student.items()}
    updated = capture_layer_io(
        model,
        tokenizer,
        student_prompts,
        layers,
        device,
        args.batch_size,
        args.max_length,
        capture_last_tokens=args.eval_capture_last_tokens,
    )
    updated_captures = {idx: cap.outputs for idx, cap in updated.items()}
    metrics: dict[int, dict[str, float]] = {}
    for layer_idx in layers:
        teacher_rows = teacher_captures[layer_idx]
        student_rows = student_captures[layer_idx]
        updated_rows = updated_captures[layer_idx]
        pre_l2 = mean_row_l2(student_rows, teacher_rows)
        post_l2 = mean_row_l2(updated_rows, teacher_rows)
        metrics[layer_idx] = {
            "pre_teacher_l2": pre_l2,
            "post_teacher_l2": post_l2,
            "teacher_l2_ratio": post_l2 / (pre_l2 + 1e-12),
            "pre_teacher_cosine": mean_row_cosine(student_rows, teacher_rows),
            "post_teacher_cosine": mean_row_cosine(updated_rows, teacher_rows),
        }
    return teacher_captures, student_captures, metrics


def add_eval_metrics(row: dict, prefix: str, result) -> None:
    row.update(result.to_dict(prefix))


def mean_max_cosine(rows: torch.Tensor, prototypes: torch.Tensor) -> float:
    if rows.numel() == 0 or prototypes.numel() == 0:
        return 0.0
    rows_n = torch.nn.functional.normalize(rows.float(), dim=1)
    proto_n = torch.nn.functional.normalize(prototypes.float(), dim=1)
    return float((rows_n @ proto_n.T).amax(dim=1).mean().item())


def repeated_answer_labels(questions: list, rows_per_question: int) -> torch.Tensor:
    labels = [1.0 if record.answer else -1.0 for record in questions]
    return torch.repeat_interleave(torch.tensor(labels, dtype=torch.float32), rows_per_question)


def add_labeled_alignment_metrics(
    row: dict,
    effect: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    answer_direction: torch.Tensor | None,
) -> None:
    labels = labels.to(dtype=torch.float32)
    for name, mask in (
        ("positive", labels > 0),
        ("negative", labels < 0),
    ):
        if int(mask.sum().item()) == 0:
            continue
        effect_rows = effect[mask]
        target_rows = target[mask]
        row[f"{name}_effect_target_cosine"] = mean_row_cosine(effect_rows, target_rows)
        row[f"{name}_effect_target_l2"] = mean_row_l2(effect_rows, target_rows)
        row[f"{name}_effect_target_norm_ratio"] = float(
            torch.linalg.vector_norm(effect_rows, dim=1).mean().item()
            / (torch.linalg.vector_norm(target_rows, dim=1).mean().item() + 1e-12)
        )
        if answer_direction is not None and answer_direction.numel() == effect.shape[1]:
            unit = torch.nn.functional.normalize(answer_direction.float(), dim=0)
            sign = 1.0 if name == "positive" else -1.0
            row[f"{name}_signed_effect_answer_projection"] = float(
                (sign * (effect_rows.float() @ unit)).mean().item()
            )
            row[f"{name}_signed_target_answer_projection"] = float(
                (sign * (target_rows.float() @ unit)).mean().item()
            )


def build_key_basis(
    trace_keys: torch.Tensor,
    guard_keys: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.key_pcs_remove <= 0:
        return torch.empty(0, trace_keys.shape[1], dtype=torch.float32)
    if args.key_pc_source == "trace":
        rows = trace_keys.float()
    elif args.key_pc_source == "guards":
        rows = guard_keys.float()
    elif args.key_pc_source == "trace_and_guards":
        rows = torch.cat([trace_keys.float(), guard_keys.float()], dim=0)
    else:
        raise ValueError(f"Unknown key PC source: {args.key_pc_source}")
    return principal_components(rows, args.key_pcs_remove)


def target_rows_for_mode(
    target_mode: str,
    full_inputs: torch.Tensor,
    full_outputs: torch.Tensor,
    null_inputs: torch.Tensor,
    null_outputs: torch.Tensor,
) -> torch.Tensor:
    if target_mode == "source_delta":
        return block_source_targets(full_inputs, full_outputs, null_inputs, null_outputs)
    if target_mode == "raw_block_delta":
        return full_outputs.float() - null_outputs.float()
    raise ValueError(f"Unknown target mode: {target_mode}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    updates_path = output_dir / "updates.jsonl"
    reentry_path = output_dir / "state_reentry.jsonl"
    trigger_path = output_dir / "trigger_overlap.jsonl"
    retention_path = output_dir / "retention.jsonl"
    for path in (metrics_path, updates_path, reentry_path, trigger_path, retention_path):
        if path.exists():
            path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    wrappers = install_additive_memory(model, args.layers, memory_dtype=torch.float32)
    domains, eval_sets = load_or_generate_domains(args)

    for paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        full_trace_prompts, null_trace_prompts, key_trace_prompts = trace_prompts_for_domain(
            tokenizer,
            domain,
            args,
            paper_idx,
        )
        full_blocks = capture_block_io(
            model,
            tokenizer,
            full_trace_prompts,
            args.layers,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        null_blocks = {}
        key_mlp = {}
        if not args.sequential_replay:
            null_blocks = capture_block_io(
                model,
                tokenizer,
                null_trace_prompts,
                args.layers,
                device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
            key_mlp = capture_layer_io(
                model,
                tokenizer,
                key_trace_prompts,
                args.layers,
                device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
        guard_questions = negative_questions_for_domain(domain, args, paper_idx)
        guard_prompts = prompts_for_questions(
            tokenizer,
            guard_questions,
            paper=None,
            use_chat_template=args.chat_template,
        )
        guard_captures = capture_layer_io(
            model,
            tokenizer,
            guard_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.eval_capture_last_tokens,
        )
        null_guard_captures = None
        if "null_doc_delta" in args.nuisance_sources:
            null_doc = make_null_document(
                seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id)) + 7919,
                approx_words=len(domain.render_paper().split()),
            )
            null_guard_prompts = prompts_for_questions(
                tokenizer,
                guard_questions,
                paper=null_doc,
                use_chat_template=args.chat_template,
            )
            null_guard_captures = capture_layer_io(
                model,
                tokenizer,
                null_guard_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.eval_capture_last_tokens,
            )

        answer_direction = None
        if args.project_answer_direction:
            answer_direction = answer_unembedding_direction(model, tokenizer)
        diagnostic_answer_direction = None
        if args.alignment_diagnostics:
            diagnostic_answer_direction = answer_unembedding_direction(model, tokenizer)

        updates: dict[int, torch.Tensor] = {}
        targets_by_layer: dict[int, torch.Tensor] = {}
        trace_keys_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx in args.layers:
            if args.sequential_replay:
                current_blocks = capture_block_io(
                    model,
                    tokenizer,
                    key_trace_prompts,
                    [layer_idx],
                    device,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
                current_mlp = capture_layer_io(
                    model,
                    tokenizer,
                    key_trace_prompts,
                    [layer_idx],
                    device,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
                targets = full_blocks[layer_idx].outputs.float() - current_blocks[layer_idx].outputs.float()
                trace_keys = current_mlp[layer_idx].keys
            else:
                targets = target_rows_for_mode(
                    args.target_mode,
                    full_blocks[layer_idx].inputs,
                    full_blocks[layer_idx].outputs,
                    null_blocks[layer_idx].inputs,
                    null_blocks[layer_idx].outputs,
                )
                trace_keys = key_mlp[layer_idx].keys
            target_fro_before_purification = float(torch.linalg.vector_norm(targets.float()).item())
            if answer_direction is not None:
                targets = project_rows_away_from_direction(targets, answer_direction)
            nuisance_basis = torch.empty(0, targets.shape[1], dtype=torch.float32)
            nuisance_energy_fraction = 0.0
            if args.nuisance_pcs > 0 and args.nuisance_sources:
                nuisance_rows: list[torch.Tensor] = []
                if "answer_controls" in args.nuisance_sources:
                    nuisance_rows.append(guard_captures[layer_idx].outputs.float())
                if "null_doc_delta" in args.nuisance_sources:
                    if null_guard_captures is None:
                        raise RuntimeError("null_doc_delta nuisance source requested but not captured.")
                    nuisance_rows.append(
                        null_guard_captures[layer_idx].outputs.float()
                        - guard_captures[layer_idx].outputs.float()
                    )
                if "target" in args.nuisance_sources:
                    nuisance_rows.append(targets.float())
                if nuisance_rows:
                    nuisance_basis = principal_components(
                        torch.cat(nuisance_rows, dim=0),
                        args.nuisance_pcs,
                    )
                    nuisance_energy_fraction = projection_energy_fraction(targets, nuisance_basis)
                    targets = project_rows_away_from_basis(targets, nuisance_basis)
            targets_by_layer[layer_idx] = targets
            trace_keys_by_layer[layer_idx] = trace_keys
            key_basis = build_key_basis(trace_keys, guard_captures[layer_idx].keys, args)
            solve_trace_keys = trace_keys
            solve_guard_keys = guard_captures[layer_idx].keys
            if key_basis.numel() > 0:
                solve_trace_keys = project_rows_away_from_basis(trace_keys, key_basis)
                solve_guard_keys = project_rows_away_from_basis(guard_captures[layer_idx].keys, key_basis)
            update, stats = protected_ridge_update(
                solve_trace_keys,
                targets,
                negative_keys=solve_guard_keys,
                ridge=args.ridge,
                negative_weight=args.negative_weight,
                eta=args.eta,
                max_update_norm=args.max_update_norm,
            )
            if key_basis.numel() > 0:
                update = project_rows_away_from_basis(update, key_basis)
            updates[layer_idx] = update
            if args.memory_gate:
                wrappers[layer_idx].set_gate_last_token_only_(args.memory_gate_final_token_only)
                wrappers[layer_idx].set_gate_keys_(
                    trace_keys,
                    threshold=args.memory_gate_threshold,
                    temperature=args.memory_gate_temperature,
                )
            wrappers[layer_idx].add_memory_(update)
            update_row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "layer": layer_idx,
                "target_mode": args.target_mode,
                "trace_source": args.trace_source,
                "sequential_replay": args.sequential_replay,
                "memory_gate_final_token_only": args.memory_gate_final_token_only,
                "project_answer_direction": args.project_answer_direction,
                "nuisance_pcs": args.nuisance_pcs,
                "nuisance_sources": args.nuisance_sources,
                "nuisance_basis_rows": int(nuisance_basis.shape[0]),
                "key_pcs_remove": args.key_pcs_remove,
                "key_pc_source": args.key_pc_source,
                "key_basis_rows": int(key_basis.shape[0]),
                "target_fro_before_purification": target_fro_before_purification,
                "target_fro_after_purification": float(torch.linalg.vector_norm(targets.float()).item()),
                "target_nuisance_energy_fraction": nuisance_energy_fraction,
                "trace_rows": int(trace_keys.shape[0]),
                "guard_rows": int(guard_captures[layer_idx].keys.shape[0]),
                "seconds": time.time() - started,
            }
            update_row.update(asdict(stats))
            append_jsonl(updates_path, update_row)

        gauntlets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": heldout_questions, **gauntlets}

        # Behavior after the write is measured on the edited model. For the
        # no-doc baseline, temporarily subtract the updates.
        for layer_idx, update in updates.items():
            wrappers[layer_idx].add_memory_(-update)
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
                paper=domain.render_paper(),
                max_length=args.max_length,
                use_chat_template=args.chat_template,
            )
            for group, questions in eval_groups.items()
        }
        pre_reentry: dict[str, tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = {}
        eval_key_captures: dict[str, dict[int, torch.Tensor]] = {}
        alignment_targets: dict[str, dict[int, torch.Tensor]] = {}
        for group, questions in eval_groups.items():
            student_prompts = prompts_for_questions(
                tokenizer,
                questions,
                paper=None,
                use_chat_template=args.chat_template,
            )
            if args.alignment_diagnostics:
                paper = domain.render_paper()
                null_doc = make_null_document(
                    seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id)) + 7919,
                    approx_words=len(paper.split()),
                )
                full_eval_blocks = capture_block_io(
                    model,
                    tokenizer,
                    prompts_for_questions(tokenizer, questions, paper=paper, use_chat_template=args.chat_template),
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.eval_capture_last_tokens,
                )
                null_eval_blocks = capture_block_io(
                    model,
                    tokenizer,
                    prompts_for_questions(tokenizer, questions, paper=null_doc, use_chat_template=args.chat_template),
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.eval_capture_last_tokens,
                )
                alignment_targets[group] = {
                    layer_idx: target_rows_for_mode(
                        args.target_mode,
                        full_eval_blocks[layer_idx].inputs,
                        full_eval_blocks[layer_idx].outputs,
                        null_eval_blocks[layer_idx].inputs,
                        null_eval_blocks[layer_idx].outputs,
                    )
                    for layer_idx in args.layers
                }
            student_keys = capture_layer_io(
                model,
                tokenizer,
                student_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.eval_capture_last_tokens,
            )
            eval_key_captures[group] = {idx: cap.keys for idx, cap in student_keys.items()}
            for layer_idx in args.layers:
                effect = eval_key_captures[group][layer_idx].float() @ updates[layer_idx].float().T
                trigger_row = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "group": group,
                    "layer": layer_idx,
                    "target_mode": args.target_mode,
                    "trace_source": args.trace_source,
                    "sequential_replay": args.sequential_replay,
                    "memory_gate_final_token_only": args.memory_gate_final_token_only,
                    "trace_rows": int(trace_keys_by_layer[layer_idx].shape[0]),
                    "eval_rows": int(eval_key_captures[group][layer_idx].shape[0]),
                    "mean_max_cosine_to_trace_keys": mean_max_cosine(
                        eval_key_captures[group][layer_idx],
                        trace_keys_by_layer[layer_idx],
                    ),
                    "mean_update_effect_norm": float(
                        torch.linalg.vector_norm(effect, dim=1).mean().item()
                    ),
                    "mean_trace_target_norm": float(
                        torch.linalg.vector_norm(targets_by_layer[layer_idx].float(), dim=1).mean().item()
                    ),
                    "seconds": time.time() - started,
                }
                if args.alignment_diagnostics:
                    target = alignment_targets[group][layer_idx].float()
                    trigger_row.update(
                        {
                            "mean_eval_target_norm": float(
                                torch.linalg.vector_norm(target, dim=1).mean().item()
                            ),
                            "effect_target_cosine": mean_row_cosine(effect, target),
                            "effect_target_l2": mean_row_l2(effect, target),
                            "effect_target_norm_ratio": float(
                                torch.linalg.vector_norm(effect, dim=1).mean().item()
                                / (torch.linalg.vector_norm(target, dim=1).mean().item() + 1e-12)
                            ),
                        }
                    )
                    labels = repeated_answer_labels(questions, args.eval_capture_last_tokens)
                    if labels.shape[0] == effect.shape[0]:
                        add_labeled_alignment_metrics(
                            trigger_row,
                            effect,
                            target,
                            labels,
                            diagnostic_answer_direction,
                        )
                append_jsonl(trigger_path, trigger_row)
            teacher_caps, student_caps, _metrics = capture_state_reentry(
                model,
                tokenizer,
                domain,
                questions,
                args.layers,
                device,
                args,
            )
            pre_reentry[group] = (teacher_caps, student_caps)
        for layer_idx, update in updates.items():
            wrappers[layer_idx].add_memory_(update)
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
                "target_mode": args.target_mode,
                "trace_source": args.trace_source,
                "sequential_replay": args.sequential_replay,
                "memory_gate_final_token_only": args.memory_gate_final_token_only,
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
            row.update(memory_norms(wrappers))
            append_jsonl(metrics_path, row)

            teacher_caps, student_caps = pre_reentry[group]
            _teacher, _student, reentry = capture_state_reentry(
                model,
                tokenizer,
                domain,
                questions,
                args.layers,
                device,
                args,
                teacher_captures=teacher_caps,
                student_captures=student_caps,
            )
            for layer_idx, layer_metrics in reentry.items():
                reentry_row = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "group": group,
                    "layer": layer_idx,
                    "target_mode": args.target_mode,
                    "trace_source": args.trace_source,
                    "sequential_replay": args.sequential_replay,
                    "memory_gate_final_token_only": args.memory_gate_final_token_only,
                    "seconds": time.time() - started,
                }
                reentry_row.update(layer_metrics)
                append_jsonl(reentry_path, reentry_row)

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

    print(f"Wrote TSOC metrics to {metrics_path}")
    print(f"Wrote TSOC update stats to {updates_path}")
    print(f"Wrote TSOC state reentry metrics to {reentry_path}")
    print(f"Wrote TSOC trigger-overlap metrics to {trigger_path}")
    if args.retention_eval:
        print(f"Wrote TSOC retention metrics to {retention_path}")


if __name__ == "__main__":
    main()
