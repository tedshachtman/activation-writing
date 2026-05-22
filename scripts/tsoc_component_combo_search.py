"""Search constrained mixtures of TSOC source-field components.

The component sweep asks whether one target principal component is useful.
This script asks the next question: can a small multi-component, multi-layer
mixture improve held-out behavior while preserving the anti-false-positive
gauntlets?

This is still a causal-patch diagnostic, not a persistent write. A mixture
must pass this search before it is worth turning into a merged weight update.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time
from typing import Iterator

import torch

from caic.evaluation import answer_margin, format_question_prompt, yes_no_logprobs
from caic.experiment import (
    answer_unembedding_direction,
    fit_linear_probe_direction,
    load_domain_rows,
    project_rows_away_from_direction,
)
from caic.modeling import (
    capture_block_io,
    capture_layer_io,
    get_decoder_layers,
    get_mlp_down_module,
    load_model_and_tokenizer,
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
    parser.add_argument("--domain-difficulty", choices=["easy", "medium", "standard"], default="easy")
    parser.add_argument("--eval-questions", type=int, default=30)
    parser.add_argument("--output", default="runs/tsoc_component_combo_search")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[12])
    parser.add_argument("--trace-probes", type=int, default=32)
    parser.add_argument("--capture-last-tokens", type=int, default=12)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument(
        "--basis-mode",
        choices=["source_pcs", "rule_probe", "source_pcs_plus_rule_probe"],
        default="source_pcs",
        help="rule_probe uses hidden DSL labels as a diagnostic target basis.",
    )
    parser.add_argument("--rule-target-margin", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
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
    parser.add_argument("--gauntlet-questions", type=int, default=20)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--max-eval-per-group", type=int, default=20)
    parser.add_argument("--search-split", choices=["even", "odd"], default="even")
    parser.add_argument("--coefficients", nargs="+", type=float, default=[-8.0, -4.0, 0.0, 4.0, 8.0])
    parser.add_argument("--random-trials", type=int, default=64)
    parser.add_argument("--max-active-components", type=int, default=3)
    parser.add_argument("--include-individuals", action="store_true", default=True)
    parser.add_argument("--include-full-component", action="store_true")
    parser.add_argument("--min-group-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--min-positive-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--min-positive-margin-delta", type=float, default=-0.50)
    parser.add_argument("--score-margin-weight", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=10)
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


def load_or_generate_domains(args: argparse.Namespace) -> tuple[list[DomainSpec], list[list]]:
    if args.domains_jsonl:
        return load_domain_rows(Path(args.domains_jsonl), args.papers)
    domains = generate_domains(args.papers, seed=args.seed, difficulty=args.domain_difficulty)
    eval_sets = [
        make_eval_questions(domain, args.eval_questions, seed=args.seed * 10_000 + idx)
        for idx, domain in enumerate(domains)
    ]
    return domains, eval_sets


def question_probe_trace_prompts(
    tokenizer,
    domain: DomainSpec,
    args: argparse.Namespace,
    paper_idx: int,
) -> tuple[list[str], list[str], list[str], list]:
    probes = make_candidate_probes(domain, args.trace_probes, seed=args.seed * 500_000 + paper_idx)
    paper = domain.render_paper()
    null_doc = make_null_document(
        seed=sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain.domain_id)),
        approx_words=len(paper.split()),
    )
    return (
        prompts_for_questions(tokenizer, probes, paper=paper, use_chat_template=args.chat_template),
        prompts_for_questions(tokenizer, probes, paper=null_doc, use_chat_template=args.chat_template),
        prompts_for_questions(tokenizer, probes, paper=None, use_chat_template=args.chat_template),
        probes,
    )


def negative_questions_for_domain(domain: DomainSpec, args: argparse.Namespace, paper_idx: int) -> list:
    questions = general_guard_questions()[: args.negative_guards]
    if args.rival_negative_guards > 0:
        rivals = make_near_collision_questions(
            domain,
            args.rival_negative_guards * 2,
            seed=args.seed * 300_000 + paper_idx,
            include_rival_prompts=True,
        )
        questions.extend(
            [record for record in rivals if record.category == "near_collision_rival"][
                : args.rival_negative_guards
            ]
        )
    return questions


@contextmanager
def patched_suffix_outputs(
    model,
    replacements_by_layer: dict[int, torch.Tensor],
    device: torch.device,
) -> Iterator[None]:
    layers = get_decoder_layers(model)
    handles = []

    def make_hook(replacement: torch.Tensor):
        def hook(_module, _inputs, output):
            patched = output.clone()
            repl = replacement.to(device=device, dtype=patched.dtype)
            if repl.ndim == 2:
                repl = repl.unsqueeze(0)
            token_count = min(repl.shape[1], patched.shape[1])
            patched[:, -token_count:, :] = repl[:, -token_count:, :]
            return patched

        return hook

    try:
        for raw_idx, replacement in replacements_by_layer.items():
            resolved = raw_idx if raw_idx >= 0 else len(layers) + raw_idx
            handles.append(get_mlp_down_module(layers[resolved]).register_forward_hook(make_hook(replacement)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def eval_with_replacements(
    model,
    tokenizer,
    questions: list,
    prompts: list[str],
    indices: list[int],
    device: torch.device,
    max_length: int,
    replacements_by_layer: dict[int, torch.Tensor] | None = None,
) -> dict[str, float]:
    correct = 0
    pos_correct = 0
    neg_correct = 0
    margins = []
    pos_margins = []
    neg_margins = []
    for idx in indices:
        record = questions[idx]
        prompt = prompts[idx]
        if replacements_by_layer:
            one_replacement = {layer: value[idx] for layer, value in replacements_by_layer.items()}
            with patched_suffix_outputs(model, one_replacement, device):
                yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        else:
            yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        pred = yes_lp >= no_lp
        correct += int(pred == record.answer)
        margin = answer_margin(yes_lp, no_lp, record.answer)
        margins.append(margin)
        if record.answer:
            pos_correct += int(pred == record.answer)
            pos_margins.append(margin)
        else:
            neg_correct += int(pred == record.answer)
            neg_margins.append(margin)

    n = len(indices)
    pos_n = len(pos_margins)
    neg_n = len(neg_margins)
    return {
        "accuracy": correct / n if n else 0.0,
        "positive_accuracy": pos_correct / pos_n if pos_n else 0.0,
        "negative_accuracy": neg_correct / neg_n if neg_n else 0.0,
        "mean_margin": sum(margins) / n if n else 0.0,
        "positive_mean_margin": sum(pos_margins) / pos_n if pos_n else 0.0,
        "negative_mean_margin": sum(neg_margins) / neg_n if neg_n else 0.0,
        "n": n,
        "positive_n": pos_n,
        "negative_n": neg_n,
    }


def split_indices(count: int, search_split: str) -> tuple[list[int], list[int]]:
    search = [idx for idx in range(count) if (idx % 2 == 0) == (search_split == "even")]
    test = [idx for idx in range(count) if idx not in set(search)]
    if not test and search:
        test = search[-1:]
        search = search[:-1]
    return search, test


def rows_for_questions(rows: torch.Tensor, question_count: int, capture_last_tokens: int) -> torch.Tensor:
    if rows.shape[0] != question_count * capture_last_tokens:
        raise ValueError(
            f"Expected {question_count * capture_last_tokens} rows, got {rows.shape[0]}; "
            "check capture_last_tokens."
        )
    return rows.reshape(question_count, capture_last_tokens, rows.shape[-1])


def final_token_rows(question_count: int, capture_last_tokens: int) -> torch.Tensor:
    return torch.arange(capture_last_tokens - 1, question_count * capture_last_tokens, capture_last_tokens)


def validity_probe_targets(
    outputs: torch.Tensor,
    questions: list,
    domain: DomainSpec,
    capture_last_tokens: int,
    ridge: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    labels = []
    for record in questions:
        valid, _failures = domain.validate(record.chain)
        labels.append(1.0 if valid else -1.0)
    question_labels = torch.tensor(labels, dtype=torch.float32)
    row_labels = torch.repeat_interleave(question_labels, capture_last_tokens)
    final_rows = final_token_rows(len(questions), capture_last_tokens)

    probe, bias = fit_linear_probe_direction(outputs[final_rows], question_labels, ridge)
    norm_sq = float(torch.dot(probe, probe).item())
    if norm_sq <= 1e-12:
        return torch.zeros_like(outputs), probe, bias
    scores = outputs.float() @ probe + bias
    signed_scores = scores * row_labels
    gaps = torch.clamp(margin - signed_scores, min=0.0)
    targets = (gaps * row_labels / (norm_sq + 1e-12)).unsqueeze(1) * probe.unsqueeze(0)
    return targets.to(dtype=outputs.dtype), probe, bias


def add_prefixed(row: dict, prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def candidate_label(coefficients: dict[str, float]) -> str:
    if not coefficients:
        return "zero"
    return ";".join(f"{name}={coef:g}" for name, coef in sorted(coefficients.items()) if coef != 0.0)


def build_trial_coefficients(
    component_names: list[str],
    coefficients: list[float],
    random_trials: int,
    max_active_components: int,
    seed: int,
    include_individuals: bool,
) -> list[dict[str, float]]:
    nonzero = [coef for coef in coefficients if coef != 0.0]
    trials: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    def add(row: dict[str, float]) -> None:
        compact = {name: float(coef) for name, coef in row.items() if coef != 0.0}
        key = tuple(sorted(compact.items()))
        if key in seen:
            return
        seen.add(key)
        trials.append(compact)

    add({})
    if include_individuals:
        for name in component_names:
            for coef in nonzero:
                add({name: coef})

    rng = random.Random(seed)
    if component_names and nonzero:
        for _ in range(random_trials):
            active_count = rng.randint(1, min(max_active_components, len(component_names)))
            active = rng.sample(component_names, active_count)
            add({name: rng.choice(nonzero) for name in active})
    return trials


def summarize_candidate(
    per_group: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    accuracy_deltas = []
    positive_accuracy_deltas = []
    positive_margin_deltas = []
    mean_margin_deltas = []
    for metrics in per_group.values():
        baseline = metrics["baseline"]
        patched = metrics["patched"]
        accuracy_deltas.append(patched["accuracy"] - baseline["accuracy"])
        positive_accuracy_deltas.append(patched["positive_accuracy"] - baseline["positive_accuracy"])
        positive_margin_deltas.append(patched["positive_mean_margin"] - baseline["positive_mean_margin"])
        mean_margin_deltas.append(patched["mean_margin"] - baseline["mean_margin"])

    min_acc = min(accuracy_deltas) if accuracy_deltas else 0.0
    mean_acc = sum(accuracy_deltas) / len(accuracy_deltas) if accuracy_deltas else 0.0
    min_pos_acc = min(positive_accuracy_deltas) if positive_accuracy_deltas else 0.0
    min_pos_margin = min(positive_margin_deltas) if positive_margin_deltas else 0.0
    mean_margin = sum(mean_margin_deltas) / len(mean_margin_deltas) if mean_margin_deltas else 0.0
    accepted = (
        min_acc >= args.min_group_accuracy_delta
        and min_pos_acc >= args.min_positive_accuracy_delta
        and min_pos_margin >= args.min_positive_margin_delta
    )
    score = min_acc + mean_acc + args.score_margin_weight * mean_margin + 0.05 * min_pos_margin
    return {
        "accepted": accepted,
        "score": score,
        "min_accuracy_delta": min_acc,
        "mean_accuracy_delta": mean_acc,
        "min_positive_accuracy_delta": min_pos_acc,
        "min_positive_margin_delta": min_pos_margin,
        "mean_margin_delta": mean_margin,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    candidates_path = output_dir / "combo_candidates.jsonl"
    groups_path = output_dir / "combo_groups.jsonl"
    components_path = output_dir / "components.jsonl"
    for path in (candidates_path, groups_path, components_path):
        if path.exists():
            path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    domains, eval_sets = load_or_generate_domains(args)

    for paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        full_prompts, null_prompts, key_prompts, trace_questions = question_probe_trace_prompts(
            tokenizer,
            domain,
            args,
            paper_idx,
        )
        full_blocks = capture_block_io(
            model,
            tokenizer,
            full_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.capture_last_tokens,
        )
        null_blocks = capture_block_io(
            model,
            tokenizer,
            null_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.capture_last_tokens,
        )
        key_captures = capture_layer_io(
            model,
            tokenizer,
            key_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.capture_last_tokens,
        )

        guard_questions = negative_questions_for_domain(domain, args, paper_idx)
        guard_prompts = prompts_for_questions(tokenizer, guard_questions, paper=None, use_chat_template=args.chat_template)
        guard_captures = capture_layer_io(
            model,
            tokenizer,
            guard_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.capture_last_tokens,
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
                capture_last_tokens=args.capture_last_tokens,
            )

        answer_direction = answer_unembedding_direction(model, tokenizer) if args.project_answer_direction else None

        gauntlets = make_gauntlet_questions(
            domain,
            args.gauntlet_questions,
            seed=args.seed * 200_000 + paper_idx,
            include_near_collision=args.near_collision_gauntlet,
        )
        eval_groups = {"heldout": heldout_questions, **gauntlets}
        eval_groups = {name: rows[: args.max_eval_per_group] for name, rows in eval_groups.items() if rows}
        group_prompts = {
            name: prompts_for_questions(tokenizer, rows, paper=None, use_chat_template=args.chat_template)
            for name, rows in eval_groups.items()
        }
        group_splits = {name: split_indices(len(rows), args.search_split) for name, rows in eval_groups.items()}

        eval_captures: dict[str, dict[int, torch.Tensor]] = {}
        base_outputs: dict[str, dict[int, torch.Tensor]] = {}
        for group_name, prompts in group_prompts.items():
            captures = capture_layer_io(
                model,
                tokenizer,
                prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.capture_last_tokens,
            )
            eval_captures[group_name] = {
                layer: rows_for_questions(captures[layer].keys, len(prompts), args.capture_last_tokens)
                for layer in args.layers
            }
            base_outputs[group_name] = {
                layer: rows_for_questions(captures[layer].outputs, len(prompts), args.capture_last_tokens)
                for layer in args.layers
            }

        baseline_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for group_name, questions in eval_groups.items():
            search_indices, test_indices = group_splits[group_name]
            baseline_metrics[group_name] = {
                "search": eval_with_replacements(
                    model,
                    tokenizer,
                    questions,
                    group_prompts[group_name],
                    search_indices,
                    device,
                    args.max_length,
                ),
                "test": eval_with_replacements(
                    model,
                    tokenizer,
                    questions,
                    group_prompts[group_name],
                    test_indices,
                    device,
                    args.max_length,
                ),
            }

        component_effects: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
        component_meta: dict[str, dict] = {}
        for layer_idx in args.layers:
            targets = block_source_targets(
                full_blocks[layer_idx].inputs,
                full_blocks[layer_idx].outputs,
                null_blocks[layer_idx].inputs,
                null_blocks[layer_idx].outputs,
            )
            if answer_direction is not None:
                targets = project_rows_away_from_direction(targets, answer_direction)
            nuisance_basis = torch.empty(0, targets.shape[1])
            nuisance_energy = 0.0
            if args.nuisance_pcs > 0 and args.nuisance_sources:
                nuisance_rows = []
                if "answer_controls" in args.nuisance_sources:
                    nuisance_rows.append(guard_captures[layer_idx].outputs.float())
                if "null_doc_delta" in args.nuisance_sources:
                    assert null_guard_captures is not None
                    nuisance_rows.append(
                        null_guard_captures[layer_idx].outputs.float()
                        - guard_captures[layer_idx].outputs.float()
                    )
                if "target" in args.nuisance_sources:
                    nuisance_rows.append(targets.float())
                nuisance_basis = principal_components(torch.cat(nuisance_rows, dim=0), args.nuisance_pcs)
                nuisance_energy = projection_energy_fraction(targets, nuisance_basis)
                targets = project_rows_away_from_basis(targets, nuisance_basis)

            component_basis = principal_components(targets, args.components)
            layer_components: list[tuple[str, torch.Tensor | None, float]] = []
            if args.include_full_component and args.basis_mode in {"source_pcs", "source_pcs_plus_rule_probe"}:
                layer_components.append((f"l{layer_idx}_full", None, 1.0))
            if args.basis_mode in {"source_pcs", "source_pcs_plus_rule_probe"}:
                for pc_idx in range(component_basis.shape[0]):
                    component = component_basis[pc_idx]
                    projected = (targets.float() @ component).unsqueeze(1) * component.unsqueeze(0)
                    energy = float(
                        torch.linalg.vector_norm(projected).square()
                        / torch.linalg.vector_norm(targets.float()).square().clamp_min(1e-12)
                    )
                    layer_components.append((f"l{layer_idx}_pc{pc_idx}", component, energy))

            rule_probe_target = None
            rule_probe_bias = None
            if args.basis_mode in {"rule_probe", "source_pcs_plus_rule_probe"}:
                rule_probe_target, rule_probe_direction, rule_probe_bias = validity_probe_targets(
                    key_captures[layer_idx].outputs,
                    trace_questions,
                    domain,
                    args.capture_last_tokens,
                    args.ridge,
                    args.rule_target_margin,
                )
                if answer_direction is not None:
                    rule_probe_target = project_rows_away_from_direction(rule_probe_target, answer_direction)
                if nuisance_basis.numel() > 0:
                    rule_probe_target = project_rows_away_from_basis(rule_probe_target, nuisance_basis)
                rule_energy = float(
                    torch.linalg.vector_norm(rule_probe_target.float()).square()
                    / torch.linalg.vector_norm(targets.float()).square().clamp_min(1e-12)
                )
                layer_components.append((f"l{layer_idx}_rule_probe", rule_probe_direction, rule_energy))

            for component_name, component, component_energy in layer_components:
                if component_name.endswith("_rule_probe"):
                    assert rule_probe_target is not None
                    component_targets = rule_probe_target
                elif component is None:
                    component_targets = targets
                else:
                    component_targets = (targets.float() @ component).unsqueeze(1) * component.unsqueeze(0)
                update, stats = protected_ridge_update(
                    key_captures[layer_idx].keys,
                    component_targets,
                    negative_keys=guard_captures[layer_idx].keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=1.0,
                    max_update_norm=None,
                )
                component_effects[component_name] = {}
                for group_name, questions in eval_groups.items():
                    effect = eval_captures[group_name][layer_idx].reshape(-1, eval_captures[group_name][layer_idx].shape[-1])
                    effect = effect.float() @ update.float().T
                    component_effects[component_name][group_name] = {
                        layer_idx: rows_for_questions(effect, len(questions), args.capture_last_tokens)
                    }
                component_meta[component_name] = {
                    "paper_idx": paper_idx,
                    "domain_id": domain.domain_id,
                    "title": domain.title,
                    "component": component_name,
                    "layer": layer_idx,
                    "component_energy_fraction": component_energy,
                    "basis_mode": args.basis_mode,
                    "rule_target_margin": args.rule_target_margin,
                    "rule_probe_bias": rule_probe_bias,
                    "nuisance_pcs": args.nuisance_pcs,
                    "nuisance_sources": args.nuisance_sources,
                    "nuisance_basis_rows": int(nuisance_basis.shape[0]),
                    "target_nuisance_energy_fraction": nuisance_energy,
                    "update_fro": stats.update_fro,
                    "target_fro": stats.target_fro,
                    "fit_rmse": stats.fit_rmse,
                    "negative_rmse": stats.negative_rmse,
                }
                append_jsonl(components_path, component_meta[component_name])

        component_names = list(component_effects)
        trials = build_trial_coefficients(
            component_names,
            args.coefficients,
            args.random_trials,
            args.max_active_components,
            seed=args.seed * 900_000 + paper_idx,
            include_individuals=args.include_individuals,
        )

        candidate_summaries = []
        for trial_idx, coefficients in enumerate(trials):
            split_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {"search": {}, "test": {}}
            for split_name in ("search", "test"):
                for group_name, questions in eval_groups.items():
                    indices = group_splits[group_name][0 if split_name == "search" else 1]
                    replacements: dict[int, torch.Tensor] = {}
                    for component_name, coef in coefficients.items():
                        if coef == 0.0:
                            continue
                        for layer_idx, effect in component_effects[component_name][group_name].items():
                            if layer_idx not in replacements:
                                replacements[layer_idx] = base_outputs[group_name][layer_idx].clone()
                            replacements[layer_idx] = replacements[layer_idx] + coef * effect
                    patched = eval_with_replacements(
                        model,
                        tokenizer,
                        questions,
                        group_prompts[group_name],
                        indices,
                        device,
                        args.max_length,
                        replacements_by_layer=replacements if replacements else None,
                    )
                    split_results[split_name][group_name] = {
                        "baseline": baseline_metrics[group_name][split_name],
                        "patched": patched,
                    }

            search_summary = summarize_candidate(split_results["search"], args)
            test_summary = summarize_candidate(split_results["test"], args)
            row = {
                "paper_idx": paper_idx,
                "domain_id": domain.domain_id,
                "title": domain.title,
                "trial_idx": trial_idx,
                "candidate": candidate_label(coefficients),
                "coefficients": coefficients,
                "active_components": len(coefficients),
                "seconds": time.time() - started,
            }
            for key, value in search_summary.items():
                row[f"search_{key}"] = value
            for key, value in test_summary.items():
                row[f"test_{key}"] = value
            append_jsonl(candidates_path, row)
            candidate_summaries.append(row)

            for split_name in ("search", "test"):
                for group_name, metrics in split_results[split_name].items():
                    group_row = {
                        "paper_idx": paper_idx,
                        "domain_id": domain.domain_id,
                        "trial_idx": trial_idx,
                        "candidate": row["candidate"],
                        "split": split_name,
                        "group": group_name,
                        "search_indices": group_splits[group_name][0],
                        "test_indices": group_splits[group_name][1],
                    }
                    add_prefixed(group_row, "baseline", metrics["baseline"])
                    add_prefixed(group_row, "patched", metrics["patched"])
                    group_row["accuracy_delta"] = metrics["patched"]["accuracy"] - metrics["baseline"]["accuracy"]
                    group_row["positive_accuracy_delta"] = (
                        metrics["patched"]["positive_accuracy"] - metrics["baseline"]["positive_accuracy"]
                    )
                    group_row["negative_accuracy_delta"] = (
                        metrics["patched"]["negative_accuracy"] - metrics["baseline"]["negative_accuracy"]
                    )
                    group_row["margin_delta"] = metrics["patched"]["mean_margin"] - metrics["baseline"]["mean_margin"]
                    group_row["positive_margin_delta"] = (
                        metrics["patched"]["positive_mean_margin"] - metrics["baseline"]["positive_mean_margin"]
                    )
                    append_jsonl(groups_path, group_row)

        accepted = [row for row in candidate_summaries if row["search_accepted"]]
        ranked = sorted(
            accepted if accepted else candidate_summaries,
            key=lambda row: (float(row["search_score"]), float(row["test_score"])),
            reverse=True,
        )
        summary_path = output_dir / f"paper_{paper_idx}_top_candidates.json"
        summary_path.write_text(json.dumps(ranked[: args.top_k], indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"paper {paper_idx}: wrote {len(candidate_summaries)} candidates "
            f"({len(accepted)} accepted on search) to {candidates_path}"
        )


if __name__ == "__main__":
    main()
