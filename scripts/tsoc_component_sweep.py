"""Causal patch sweep over TSOC source-field components.

This diagnostic decomposes TSOC source targets into principal components, fits
a closed-form down-projection update for each component, and directly patches
the predicted effect into evaluation prompts. It asks which parts of the source
field are behaviorally useful before we write them into weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Iterator

import torch

from caic.evaluation import answer_margin, format_model_prompt, format_question_prompt, yes_no_logprobs
from caic.experiment import answer_unembedding_direction, load_domain_rows, project_rows_away_from_direction
from caic.modeling import (
    capture_block_io,
    capture_layer_io,
    get_decoder_layers,
    get_mlp_down_module,
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
    parser.add_argument("--domains-jsonl", required=True)
    parser.add_argument("--papers", type=int, default=1)
    parser.add_argument("--output", default="runs/tsoc_component_sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[8])
    parser.add_argument("--trace-probes", type=int, default=32)
    parser.add_argument("--capture-last-tokens", type=int, default=12)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
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
    parser.add_argument("--gauntlet-questions", type=int, default=12)
    parser.add_argument("--near-collision-gauntlet", action="store_true")
    parser.add_argument("--max-eval-per-group", type=int, default=12)
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


def question_probe_trace_prompts(
    tokenizer,
    domain: DomainSpec,
    args: argparse.Namespace,
    paper_idx: int,
) -> tuple[list[str], list[str], list[str]]:
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
def patched_suffix_output(model, layer_idx: int, replacement: torch.Tensor, device: torch.device) -> Iterator[None]:
    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    module = get_mlp_down_module(layers[resolved])

    def hook(_module, _inputs, output):
        patched = output.clone()
        repl = replacement.to(device=device, dtype=patched.dtype)
        if repl.ndim == 2:
            repl = repl.unsqueeze(0)
        token_count = min(repl.shape[1], patched.shape[1])
        patched[:, -token_count:, :] = repl[:, -token_count:, :]
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def eval_with_optional_replacements(
    model,
    tokenizer,
    questions: list,
    prompts: list[str],
    device: torch.device,
    max_length: int,
    layer_idx: int | None = None,
    replacements: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    correct = 0
    pos_correct = 0
    neg_correct = 0
    margins = []
    pos_margins = []
    neg_margins = []
    for idx, (record, prompt) in enumerate(zip(questions, prompts)):
        if layer_idx is None:
            yes_lp, no_lp = yes_no_logprobs(model, tokenizer, prompt, device, max_length=max_length)
        else:
            assert replacements is not None
            with patched_suffix_output(model, layer_idx, replacements[idx], device):
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
    n = len(questions)
    pos_n = len(pos_margins)
    neg_n = len(neg_margins)
    return {
        "accuracy": correct / n if n else 0.0,
        "positive_accuracy": pos_correct / pos_n if pos_n else 0.0,
        "negative_accuracy": neg_correct / neg_n if neg_n else 0.0,
        "mean_margin": sum(margins) / n if n else 0.0,
        "positive_mean_margin": sum(pos_margins) / pos_n if pos_n else 0.0,
        "negative_mean_margin": sum(neg_margins) / neg_n if neg_n else 0.0,
    }


def add_prefixed(row: dict, prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def rows_for_question(rows: torch.Tensor, question_idx: int, capture_last_tokens: int) -> torch.Tensor:
    start = question_idx * capture_last_tokens
    return rows[start : start + capture_last_tokens]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    results_path = output_dir / "component_sweep.jsonl"
    if results_path.exists():
        results_path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    domains, eval_sets = load_domain_rows(Path(args.domains_jsonl), args.papers)

    for paper_idx, (domain, heldout_questions) in enumerate(zip(domains, eval_sets)):
        started = time.time()
        full_prompts, null_prompts, key_prompts = question_probe_trace_prompts(tokenizer, domain, args, paper_idx)
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
            all_components = [("full", None)]
            all_components.extend((f"pc_{idx}", component_basis[idx]) for idx in range(component_basis.shape[0]))

            for component_name, component in all_components:
                if component is None:
                    component_targets = targets
                    component_energy = 1.0
                else:
                    component_targets = (targets.float() @ component).unsqueeze(1) * component.unsqueeze(0)
                    component_energy = float(
                        torch.linalg.vector_norm(component_targets).square()
                        / torch.linalg.vector_norm(targets.float()).square().clamp_min(1e-12)
                    )
                update, stats = protected_ridge_update(
                    key_captures[layer_idx].keys,
                    component_targets,
                    negative_keys=guard_captures[layer_idx].keys,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=1.0,
                    max_update_norm=None,
                )

                for group_name, questions in eval_groups.items():
                    prompts = prompts_for_questions(
                        tokenizer,
                        questions,
                        paper=None,
                        use_chat_template=args.chat_template,
                    )
                    baseline = eval_with_optional_replacements(
                        model,
                        tokenizer,
                        questions,
                        prompts,
                        device,
                        args.max_length,
                    )
                    eval_capture = capture_layer_io(
                        model,
                        tokenizer,
                        prompts,
                        [layer_idx],
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.capture_last_tokens,
                    )[layer_idx]
                    effect = eval_capture.keys.float() @ update.float().T
                    for scale in args.scales:
                        replacements = [
                            rows_for_question(eval_capture.outputs, idx, args.capture_last_tokens)
                            + scale * rows_for_question(effect, idx, args.capture_last_tokens)
                            for idx in range(len(questions))
                        ]
                        patched = eval_with_optional_replacements(
                            model,
                            tokenizer,
                            questions,
                            prompts,
                            device,
                            args.max_length,
                            layer_idx=layer_idx,
                            replacements=replacements,
                        )
                        row = {
                            "paper_idx": paper_idx,
                            "domain_id": domain.domain_id,
                            "title": domain.title,
                            "layer": layer_idx,
                            "group": group_name,
                            "component": component_name,
                            "component_energy_fraction": component_energy,
                            "scale": scale,
                            "trace_rows": int(key_captures[layer_idx].keys.shape[0]),
                            "eval_rows": int(eval_capture.keys.shape[0]),
                            "nuisance_pcs": args.nuisance_pcs,
                            "nuisance_sources": args.nuisance_sources,
                            "nuisance_basis_rows": int(nuisance_basis.shape[0]),
                            "target_nuisance_energy_fraction": nuisance_energy,
                            "update_fro": stats.update_fro,
                            "mean_effect_norm": float(torch.linalg.vector_norm(effect, dim=1).mean().item()),
                            "seconds": time.time() - started,
                        }
                        add_prefixed(row, "baseline", baseline)
                        add_prefixed(row, "patched", patched)
                        row["accuracy_delta"] = patched["accuracy"] - baseline["accuracy"]
                        row["positive_accuracy_delta"] = patched["positive_accuracy"] - baseline["positive_accuracy"]
                        row["negative_accuracy_delta"] = patched["negative_accuracy"] - baseline["negative_accuracy"]
                        row["margin_delta"] = patched["mean_margin"] - baseline["mean_margin"]
                        append_jsonl(results_path, row)
    print(f"Wrote TSOC component sweep metrics to {results_path}")


if __name__ == "__main__":
    main()
