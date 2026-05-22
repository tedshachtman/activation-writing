"""Sequential multi-task mini-language consolidation triangle.

This runner tests whether the layer-20 mini-language write behaves like
continual learning. It creates several independent invented translation tasks,
applies one closed-form layer-20 write per task, and evaluates every learned
task after each new write.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import random
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.evaluation import format_model_prompt
from caic.modeling import (
    capture_attention_io,
    capture_attention_io_at_token_indices,
    capture_block_io,
    capture_layer_io,
    capture_layer_io_at_token_indices,
    clear_active_slot_weights,
    install_additive_attention_memory,
    install_additive_memory,
    load_model_and_tokenizer,
    set_active_slot_weights_for_prompts,
)
from caic.tsoc import protected_ridge_update
from scripts.minilang_write import (
    ADJECTIVES,
    LETTERS,
    NOUNS,
    TENSES,
    VERBS,
    TranslationQuestion,
    Word,
    add_metrics,
    append_jsonl,
    answer_prefixes,
    evaluate_generic_mc,
    guard_prompts,
    introduced_items,
    make_question_from_slots,
    make_translation_question,
    option_logprobs,
    sentinel_questions,
)


def progress(message: str) -> None:
    print(f"[triangle] {message}", flush=True)


def release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class TaskProfile:
    idx: int
    name: str
    prefix: str
    nouns: list[Word]
    verbs: list[Word]
    adjectives: list[Word]
    tenses: list[Word]


LANGUAGE_NAMES = ["Lyran", "Vomar", "Seldic", "Nareth", "Orvian", "Caldrin", "Mireth", "Pavric"]
TOKEN_PREFIXES = ["ly", "vo", "se", "na", "or", "ka", "mi", "pa"]


def rotated(items: list[Word], amount: int) -> list[Word]:
    if not items:
        return []
    amount %= len(items)
    return items[amount:] + items[:amount]


def task_profile(task_idx: int) -> TaskProfile:
    """Build one independent language with unique source tokens.

    Source tokens are namespaced by task, while English meanings are permuted
    within each category. That keeps the evaluation unambiguous but makes each
    task an actual new mapping rather than the same language with new examples.
    """

    prefix = TOKEN_PREFIXES[task_idx % len(TOKEN_PREFIXES)]
    name = LANGUAGE_NAMES[task_idx % len(LANGUAGE_NAMES)]
    noun_meanings = rotated(NOUNS, task_idx)
    verb_meanings = rotated(VERBS, task_idx * 2)
    adjective_meanings = rotated(ADJECTIVES, task_idx * 3)
    nouns = [
        Word(f"{prefix}{src.src}", meaning.en)
        for src, meaning in zip(NOUNS, noun_meanings, strict=True)
    ]
    verbs = [
        Word(f"{prefix}{src.src}", meaning.en, meaning.past)
        for src, meaning in zip(VERBS, verb_meanings, strict=True)
    ]
    adjectives = [
        Word(f"{prefix}{src.src}", meaning.en)
        for src, meaning in zip(ADJECTIVES, adjective_meanings, strict=True)
    ]
    # Keep tense markers shared so the task focuses on lexical/role binding.
    return TaskProfile(task_idx, name, prefix, nouns, verbs, adjectives, list(TENSES))


def profile_items(profile: TaskProfile, lesson_idx: int) -> tuple[list[Word], list[Word], list[Word], list[Word]]:
    base_nouns, base_verbs, base_adjectives, base_tenses = introduced_items(lesson_idx)
    return (
        profile.nouns[: len(base_nouns)],
        profile.verbs[: len(base_verbs)],
        profile.adjectives[: len(base_adjectives)],
        profile.tenses[: len(base_tenses)],
    )


def render_task_question(profile: TaskProfile, question: TranslationQuestion) -> str:
    return (
        f"Translate this {profile.name} sentence into English. "
        "Write only the English sentence.\n\n"
        f"{profile.name}: {question.sentence}\n"
        "English:"
    )


def format_task_prompt(tokenizer, profile: TaskProfile, question: TranslationQuestion, context: str | None, use_chat_template: bool) -> str:
    body = ""
    if context:
        body = context.strip() + "\n\n"
    body += render_task_question(profile, question)
    return format_model_prompt(tokenizer, body, use_chat_template)


def task_object_gate_prompts(
    tokenizer,
    profile: TaskProfile,
    questions: list[TranslationQuestion],
    use_chat_template: bool,
    max_prompts: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    selected = list(questions)
    if max_prompts > 0 and len(selected) > max_prompts:
        selected = random.Random(seed).sample(selected, max_prompts)
    prompts: list[str] = []
    indices: list[int] = []
    for question in selected:
        prompt = format_task_prompt(tokenizer, profile, question, None, use_chat_template)
        marker = "\nEnglish:"
        marker_index = prompt.rfind(marker)
        prefix = prompt[:marker_index] if marker_index >= 0 else prompt
        token_index = len(tokenizer.encode(prefix, add_special_tokens=False)) - 1
        prompts.append(prompt)
        indices.append(max(0, token_index))
    return prompts, indices


def trace_task_prompts(
    tokenizer,
    profile: TaskProfile,
    question: TranslationQuestion,
    answer: str,
    context: str | None,
    use_chat_template: bool,
    token_teacher_forcing: bool,
) -> list[str]:
    base = format_task_prompt(tokenizer, profile, question, context, use_chat_template)
    return [base + prefix for prefix in answer_prefixes(tokenizer, answer, token_teacher_forcing)]


def render_task_lesson(profile: TaskProfile, lesson_idx: int, example_count: int, seed: int) -> str:
    rng = random.Random(seed + profile.idx * 100_003 + lesson_idx * 997)
    nouns, verbs, adjectives, tenses = profile_items(profile, lesson_idx)
    examples = [
        make_translation_question(
            rng,
            nouns,
            verbs,
            adjectives,
            tenses,
            category="lesson_example",
            force_modifier=lesson_idx >= 3,
        )
        for _ in range(example_count)
    ]
    noun_lines = ", ".join(f"{word.src}={word.en}" for word in nouns)
    verb_lines = ", ".join(
        f"{word.src}={word.en}" + (f"/{word.past}" if word.past else "")
        for word in verbs
    )
    adjective_lines = ", ".join(f"{word.src}={word.en}" for word in adjectives) or "(none yet)"
    tense_lines = ", ".join(
        {
            "na": "na=present",
            "pa": "pa=past",
            "fu": "fu=future",
        }[word.src]
        for word in tenses
    )
    example_lines = "\n".join(f"- {item.sentence} -> {item.answer}" for item in examples)
    return (
        f"{profile.name} lesson {lesson_idx + 1}.\n"
        f"{profile.name} is an invented language. Translate it using these rules.\n"
        "Sentence order is: TENSE VERB SUBJECT OBJECT.\n"
        "Adjectives come after the noun they modify.\n"
        "English output should use ordinary English word order: subject, verb, object.\n\n"
        f"Tense words: {tense_lines}.\n"
        f"Nouns: {noun_lines}.\n"
        f"Verbs: {verb_lines}.\n"
        f"Adjectives: {adjective_lines}.\n\n"
        f"Examples:\n{example_lines}\n"
    )


def build_task_questions(
    profile: TaskProfile,
    count: int,
    seed: int,
    lesson_idx: int,
    category: str,
    balanced: bool = False,
) -> list[TranslationQuestion]:
    rng = random.Random(seed + profile.idx * 100_003)
    nouns, verbs, adjectives, tenses = profile_items(profile, lesson_idx)
    if not balanced:
        return [
            make_translation_question(
                rng,
                nouns,
                verbs,
                adjectives,
                tenses,
                category=category,
                force_modifier=True,
            )
            for _ in range(count)
        ]
    questions: list[TranslationQuestion] = []
    for idx in range(count):
        tense = tenses[idx % len(tenses)]
        verb = verbs[(idx // len(tenses)) % len(verbs)]
        subject = nouns[(idx // max(1, len(tenses) * len(verbs))) % len(nouns)]
        obj = nouns[(idx * 2 + 1) % len(nouns)]
        if obj.src == subject.src:
            obj = nouns[(nouns.index(subject) + 1) % len(nouns)]
        subj_adj = adjectives[idx % len(adjectives)] if adjectives else None
        obj_adj = adjectives[(idx + 1) % len(adjectives)] if adjectives else None
        questions.append(
            make_question_from_slots(
                rng,
                tense,
                verb,
                subject,
                obj,
                subj_adj,
                obj_adj,
                nouns,
                verbs,
                adjectives,
                tenses,
                category,
            )
        )
    return questions


def evaluate_task_mc(
    model,
    tokenizer,
    profile: TaskProfile,
    questions: list[TranslationQuestion],
    device: torch.device,
    context: str | None,
    max_length: int,
    use_chat_template: bool,
) -> dict:
    correct = 0
    margins = []
    details = []
    predictions = []
    for question in questions:
        prompt = format_task_prompt(tokenizer, profile, question, context, use_chat_template)
        scores = option_logprobs(model, tokenizer, prompt, question.options, device, max_length)
        pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        correct += int(pred_idx == question.answer_idx)
        predictions.append(LETTERS[pred_idx])
        sorted_wrong = max(score for idx, score in enumerate(scores) if idx != question.answer_idx)
        margin = scores[question.answer_idx] - sorted_wrong
        margins.append(margin)
        details.append(
            {
                "sentence": question.sentence,
                "answer": question.answer,
                "options": question.options,
                "answer_letter": question.answer_letter,
                "prediction": LETTERS[pred_idx],
                "prediction_text": question.options[pred_idx],
                "scores": scores,
                "margin": margin,
                "correct": pred_idx == question.answer_idx,
                "category": question.category,
            }
        )
    return {
        "accuracy": correct / len(questions) if questions else 0.0,
        "correct": correct,
        "n": len(questions),
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "predictions": predictions,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="runs/minilang_continual_triangle")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--lessons-per-task", type=int, default=4)
    parser.add_argument("--lesson-examples", type=int, default=8)
    parser.add_argument("--trace-probes", type=int, default=4)
    parser.add_argument("--balanced-trace", action="store_true")
    parser.add_argument("--eval-questions", type=int, default=8)
    parser.add_argument("--teacher-filter-eval", action="store_true")
    parser.add_argument("--teacher-filter-candidates", type=int, default=80)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    parser.add_argument("--trace-last-tokens", type=int, default=1)
    parser.add_argument("--target-mode", choices=["output_delta"], default="output_delta")
    parser.add_argument("--write-attention-o", action="store_true", default=True)
    parser.add_argument("--no-write-attention-o", dest="write_attention_o", action="store_false")
    parser.add_argument("--write-mlp", action="store_true", default=True)
    parser.add_argument("--no-write-mlp", dest="write_mlp", action="store_false")
    parser.add_argument("--merge-updates", action="store_true")
    parser.add_argument("--memory-gate", action="store_true")
    parser.add_argument("--memory-gate-final-token-only", action="store_true")
    parser.add_argument("--memory-gate-threshold", type=float, default=0.95)
    parser.add_argument("--memory-gate-temperature", type=float, default=80.0)
    parser.add_argument("--activation-object-gate", action="store_true")
    parser.add_argument("--object-gate-token-window", type=int, default=8)
    parser.add_argument("--object-gate-threshold", type=float, default=0.90)
    parser.add_argument("--object-gate-temperature", type=float, default=40.0)
    parser.add_argument("--object-gate-floor", type=float, default=0.0)
    parser.add_argument("--object-gate-max-prompts", type=int, default=0)
    parser.add_argument("--sentinel-eval", action="store_true")
    parser.add_argument("--token-teacher-forcing-trace", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


@torch.no_grad()
def apply_update(wrapper, update: torch.Tensor, merge: bool) -> None:
    if merge:
        wrapper.base.weight.add_(update.to(device=wrapper.base.weight.device, dtype=wrapper.base.weight.dtype))
        return
    wrapper.add_memory_(update)


def main() -> None:
    args = parse_args()
    if args.merge_updates and (args.memory_gate or args.activation_object_gate):
        raise ValueError("--merge-updates cannot be combined with activation gates; merged updates are always on.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    updates_path = output_dir / "updates.jsonl"
    details_path = output_dir / "eval_details.jsonl"
    lessons_path = output_dir / "lessons.jsonl"
    questions_path = output_dir / "eval_questions.jsonl"
    for path in (metrics_path, updates_path, details_path, lessons_path, questions_path):
        if path.exists():
            path.unlink()

    progress("loading model")
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    progress(f"loaded model on {device}")
    layers = [args.layer]
    memory_dtype = torch.float16 if device.type == "mps" and args.dtype == "float16" else torch.float32
    progress(f"installing wrappers memory_dtype={memory_dtype}")
    mlp_wrappers = install_additive_memory(model, layers, memory_dtype=memory_dtype) if args.write_mlp else {}
    attn_wrappers = (
        install_additive_attention_memory(model, layers, memory_dtype=memory_dtype)
        if args.write_attention_o
        else {}
    )
    progress("installed wrappers")

    profiles = [task_profile(idx) for idx in range(args.tasks)]
    progress("built task profiles")
    final_lesson_idx = args.lessons_per_task - 1
    lesson_texts: list[list[str]] = []
    contexts: list[str] = []
    eval_sets: list[list[TranslationQuestion]] = []
    filter_stats: list[dict] = []
    for profile in profiles:
        task_lessons = [
            render_task_lesson(profile, lesson_idx, args.lesson_examples, args.seed)
            for lesson_idx in range(args.lessons_per_task)
        ]
        lesson_texts.append(task_lessons)
        contexts.append("\n\n".join(task_lessons))
        candidate_count = args.teacher_filter_candidates if args.teacher_filter_eval else args.eval_questions
        eval_questions = build_task_questions(
            profile,
            candidate_count,
            args.seed + 91_000,
            final_lesson_idx,
            "heldout_translation",
        )
        eval_sets.append(eval_questions)
        for idx, text in enumerate(task_lessons):
            append_jsonl(lessons_path, {"task_idx": profile.idx, "language": profile.name, "lesson_idx": idx, "text": text})
    progress("built lessons and eval sets")

    started = time.time()
    baselines: list[dict] = []
    contexts_metrics: list[dict] = []
    sentinels = sentinel_questions() if args.sentinel_eval else []
    sentinel_before = (
        evaluate_generic_mc(model, tokenizer, sentinels, device, args.max_length, args.chat_template)
        if sentinels
        else None
    )
    if sentinel_before is not None:
        row = {"stage": "sentinel_before", "step": -1, "seconds": time.time() - started}
        add_metrics(row, "sentinel", sentinel_before)
        append_jsonl(metrics_path, row)
    if args.teacher_filter_eval:
        for task_idx, profile in enumerate(profiles):
            candidates = eval_sets[task_idx]
            progress(
                f"teacher-filtering task={task_idx} language={profile.name} "
                f"candidates={len(candidates)}"
            )
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
            filtered = [
                question
                for question, detail in zip(candidates, context_candidates["details"], strict=True)
                if detail["correct"]
            ]
            if len(filtered) < args.eval_questions:
                progress(
                    f"task={task_idx} only {len(filtered)} context-correct candidates; "
                    "using all available filtered questions"
                )
            eval_sets[task_idx] = filtered[: args.eval_questions]
            stat = {
                "stage": "teacher_filter",
                "step": -1,
                "task_idx": task_idx,
                "language": profile.name,
                "teacher_filter_candidates": len(candidates),
                "teacher_filter_correct": len(filtered),
                "teacher_filter_selected": len(eval_sets[task_idx]),
                "seconds": time.time() - started,
            }
            filter_stats.append(stat)
            append_jsonl(metrics_path, stat)
            release_device_cache(device)
    else:
        filter_stats = [
            {
                "teacher_filter_candidates": len(eval_set),
                "teacher_filter_correct": None,
                "teacher_filter_selected": len(eval_set),
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
    for task_idx, profile in enumerate(profiles):
        progress(f"scoring baseline/context task={task_idx} language={profile.name}")
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
        progress(f"wrote before_write task={task_idx}")

    progress("capturing guard keys")
    guard = guard_prompts(tokenizer, args.chat_template)
    guard_mlp = (
        capture_layer_io(
            model,
            tokenizer,
            guard,
            layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        if args.write_mlp
        else None
    )
    release_device_cache(device)
    guard_attn = (
        capture_attention_io(
            model,
            tokenizer,
            guard,
            layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        if args.write_attention_o
        else None
    )
    release_device_cache(device)

    acquisition_accuracy: list[float | None] = [None for _ in profiles]
    acquisition_margin: list[float | None] = [None for _ in profiles]

    for step, profile in enumerate(profiles):
        progress(f"writing step={step} language={profile.name}")
        step_started = time.time()
        probes = build_task_questions(
            profile,
            args.trace_probes,
            args.seed + step * 10_000,
            final_lesson_idx,
            "trace_translation",
            balanced=args.balanced_trace,
        )
        full_prompts = [
            prompt
            for question in probes
            for prompt in trace_task_prompts(
                tokenizer,
                profile,
                question,
                question.answer,
                contexts[step],
                args.chat_template,
                args.token_teacher_forcing_trace,
            )
        ]
        key_prompts = [
            prompt
            for question in probes
            for prompt in trace_task_prompts(
                tokenizer,
                profile,
                question,
                question.answer,
                None,
                args.chat_template,
                args.token_teacher_forcing_trace,
            )
        ]
        object_gate_mlp = None
        object_gate_attn = None
        if args.activation_object_gate:
            object_prompts, object_indices = task_object_gate_prompts(
                tokenizer,
                profile,
                probes,
                args.chat_template,
                args.object_gate_max_prompts,
                args.seed + step * 10_000 + 997,
            )
            if args.write_mlp:
                progress(f"step={step} capturing mlp object gate keys")
                object_gate_mlp = capture_layer_io_at_token_indices(
                    model,
                    tokenizer,
                    object_prompts,
                    object_indices,
                    layers,
                    device,
                    args.max_length,
                    capture_window=args.object_gate_token_window,
                )
                release_device_cache(device)
            if args.write_attention_o:
                progress(f"step={step} capturing attention object gate keys")
                object_gate_attn = capture_attention_io_at_token_indices(
                    model,
                    tokenizer,
                    object_prompts,
                    object_indices,
                    layers,
                    device,
                    args.max_length,
                    capture_window=args.object_gate_token_window,
                )
                release_device_cache(device)
        progress(f"step={step} capturing full block states rows={len(full_prompts)}")
        full_blocks = capture_block_io(
            model,
            tokenizer,
            full_prompts,
            layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        release_device_cache(device)
        progress(f"step={step} capturing current block states rows={len(key_prompts)}")
        current_blocks = capture_block_io(
            model,
            tokenizer,
            key_prompts,
            layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        release_device_cache(device)
        targets = full_blocks[args.layer].outputs.float() - current_blocks[args.layer].outputs.float()
        progress(f"step={step} built targets rows={targets.shape[0]}")

        if args.write_attention_o:
            assert guard_attn is not None
            progress(f"step={step} capturing attention keys")
            current_attn = capture_attention_io(
                model,
                tokenizer,
                key_prompts,
                layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
            release_device_cache(device)
            progress(f"step={step} solving attention update")
            attn_update, attn_stats = protected_ridge_update(
                current_attn[args.layer].keys,
                targets,
                negative_keys=guard_attn[args.layer].keys,
                ridge=args.ridge,
                negative_weight=args.negative_weight,
                eta=args.eta,
                max_update_norm=args.max_update_norm,
            )
            if args.memory_gate:
                attn_wrappers[args.layer].set_gate_last_token_only_(args.memory_gate_final_token_only)
                attn_wrappers[args.layer].set_gate_keys_(
                    current_attn[args.layer].keys,
                    threshold=args.memory_gate_threshold,
                    temperature=args.memory_gate_temperature,
                    append=True,
                )
            object_gate_rows = 0
            if args.activation_object_gate:
                assert object_gate_attn is not None
                attn_wrappers[args.layer].set_object_gate_keys_(
                    object_gate_attn[args.layer].keys,
                    threshold=args.object_gate_threshold,
                    temperature=args.object_gate_temperature,
                    floor=args.object_gate_floor,
                    append=True,
                )
                object_gate_rows = int(object_gate_attn[args.layer].keys.shape[0])
            apply_update(attn_wrappers[args.layer], attn_update, args.merge_updates)
            progress(f"step={step} applied attention update")
            row = {
                "step": step,
                "task_idx": step,
                "language": profile.name,
                "module": "attention_o",
                "layer": args.layer,
                "trace_rows": int(current_attn[args.layer].keys.shape[0]),
                "guard_rows": int(guard_attn[args.layer].keys.shape[0]),
                "object_gate_rows": object_gate_rows,
                "seconds": time.time() - step_started,
            }
            row.update(attn_stats.__dict__)
            append_jsonl(updates_path, row)

        if args.write_mlp:
            assert guard_mlp is not None
            progress(f"step={step} capturing mlp keys")
            current_mlp = capture_layer_io(
                model,
                tokenizer,
                key_prompts,
                layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
            release_device_cache(device)
            progress(f"step={step} solving mlp update")
            mlp_update, mlp_stats = protected_ridge_update(
                current_mlp[args.layer].keys,
                targets,
                negative_keys=guard_mlp[args.layer].keys,
                ridge=args.ridge,
                negative_weight=args.negative_weight,
                eta=args.eta,
                max_update_norm=args.max_update_norm,
            )
            if args.memory_gate:
                mlp_wrappers[args.layer].set_gate_last_token_only_(args.memory_gate_final_token_only)
                mlp_wrappers[args.layer].set_gate_keys_(
                    current_mlp[args.layer].keys,
                    threshold=args.memory_gate_threshold,
                    temperature=args.memory_gate_temperature,
                    append=True,
                )
            object_gate_rows = 0
            if args.activation_object_gate:
                assert object_gate_mlp is not None
                mlp_wrappers[args.layer].set_object_gate_keys_(
                    object_gate_mlp[args.layer].keys,
                    threshold=args.object_gate_threshold,
                    temperature=args.object_gate_temperature,
                    floor=args.object_gate_floor,
                    append=True,
                )
                object_gate_rows = int(object_gate_mlp[args.layer].keys.shape[0])
            apply_update(mlp_wrappers[args.layer], mlp_update, args.merge_updates)
            progress(f"step={step} applied mlp update")
            row = {
                "step": step,
                "task_idx": step,
                "language": profile.name,
                "module": "mlp_down",
                "layer": args.layer,
                "trace_rows": int(current_mlp[args.layer].keys.shape[0]),
                "guard_rows": int(guard_mlp[args.layer].keys.shape[0]),
                "object_gate_rows": object_gate_rows,
                "seconds": time.time() - step_started,
            }
            row.update(mlp_stats.__dict__)
            append_jsonl(updates_path, row)

        if sentinel_before is not None:
            progress(f"step={step} evaluating sentinels")
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
            append_jsonl(metrics_path, row)

        for eval_task_idx in range(step + 1):
            progress(f"step={step} evaluating task={eval_task_idx}")
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
                row["retention_accuracy_delta_from_acquisition"] = (
                    edited["accuracy"] - acquisition_accuracy[eval_task_idx]
                )
                row["retention_preserved_from_acquisition"] = (
                    row["retention_accuracy_delta_from_acquisition"] >= -1e-12
                )
            if acquisition_margin[eval_task_idx] is not None:
                row["acquisition_reference_margin"] = acquisition_margin[eval_task_idx]
                row["retention_margin_delta_from_acquisition"] = (
                    edited["mean_margin"] - acquisition_margin[eval_task_idx]
                )
            append_jsonl(metrics_path, row)
            for idx, detail in enumerate(edited["details"]):
                append_jsonl(details_path, {"step": step, "task_idx": eval_task_idx, "idx": idx, **detail})

    clear_active_slot_weights(model)
    print(f"Wrote continual triangle metrics to {metrics_path}")
    print(f"Wrote continual triangle updates to {updates_path}")


if __name__ == "__main__":
    main()
