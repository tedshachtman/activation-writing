"""Sequential closed-form writes for a synthetic mini-language.

The task is intentionally different from the yes/no mini-paper benchmark:
teach one coherent invented translation system across many lessons, consolidate
each lesson, then test no-context translation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import torch

from caic.contrastive_gate import DensityRatioGateParams, fit_contrastive_density_gate, score_tokens, sequence_gate
from caic.evaluation import format_model_prompt
from caic.intrinsic_surprise import (
    IntrinsicSurpriseSelection,
    apply_mlp_gauge_seal_,
    apply_intrinsic_feature_birth_update_,
    base_down_weight,
    down_output_basis_specificity,
    down_value_specificity,
    effective_down_weight,
    gauge_canonical_key_scale,
    karp_purify_update,
    orca_karp_purify_update,
    ocep_purify_update,
    ocep_qrico_purify_update,
    prism_q_purify_update,
    project_rows_away_from_basis,
    qrico_purify_update,
    seal_qrico_purify_update,
    tdmi_q_transport_scores,
    trace_q_purify_update,
    select_intrinsic_compatibility_residual_write,
    select_intrinsic_conditional_relation_innovation_write,
    select_intrinsic_conjunctive_feature_birth_update,
    select_intrinsic_feature_birth_update,
    select_intrinsic_schur_transport_actuator_write,
    select_intrinsic_associative_binding_write,
    select_intrinsic_predictive_residual_write,
    select_intrinsic_relational_aggregate_write,
    select_intrinsic_relational_residual_write,
    select_intrinsic_surprise_write,
    sharp_karp_purify_update,
    spectra_purify_update,
)
from caic.modeling import (
    capture_attention_io,
    capture_attention_io_at_token_indices,
    capture_attention_projection_io,
    capture_attention_projection_io_at_token_indices,
    capture_block_io,
    capture_layer_io,
    capture_layer_io_at_token_indices,
    clear_active_slot_weights,
    get_decoder_layers,
    LayerCapture,
    get_mlp_down_module,
    install_additive_attention_projection_memory,
    install_additive_attention_memory,
    install_additive_memory,
    load_model_and_tokenizer,
    set_active_slot_weights_for_prompts,
)
from caic.tsoc import block_source_targets, protected_metric_update, protected_ridge_update


LETTERS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class Word:
    src: str
    en: str
    past: str | None = None


@dataclass
class TranslationQuestion:
    sentence: str
    answer: str
    options: list[str]
    answer_idx: int
    category: str

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer_idx]


@dataclass(frozen=True)
class LexicalTrace:
    src: str
    answer: str
    category: str


@dataclass(frozen=True)
class ContextSpanTrace:
    prefix: str
    answer: str
    category: str


@dataclass
class IntrinsicLayerCapture:
    keys: torch.Tensor
    outputs: torch.Tensor
    mlp_inputs: torch.Tensor
    attentions: torch.Tensor | None = None


@dataclass(frozen=True)
class PackedUseItem:
    prompt: str
    answer: str
    category: str


NOUNS = [
    Word("niv", "cat"),
    Word("drem", "dog"),
    Word("palo", "bird"),
    Word("soka", "child"),
    Word("vesh", "teacher"),
    Word("tul", "robot"),
    Word("zani", "artist"),
    Word("melo", "farmer"),
]

VERBS = [
    Word("lum", "sees", "saw"),
    Word("vek", "likes", "liked"),
    Word("narp", "helps", "helped"),
    Word("shon", "chases", "chased"),
    Word("bir", "finds", "found"),
    Word("kel", "holds", "held"),
]

ADJECTIVES = [
    Word("ro", "big"),
    Word("mi", "small"),
    Word("sen", "red"),
    Word("tal", "blue"),
    Word("mok", "happy"),
    Word("gev", "quiet"),
]

TENSES = [
    Word("na", "present"),
    Word("pa", "past"),
    Word("fu", "future"),
]


def introduced_items(lesson_idx: int) -> tuple[list[Word], list[Word], list[Word], list[Word]]:
    # Keep the language small at first, then add words gradually. By lesson 12
    # all lexical items are available; later lessons reinforce compositions.
    noun_count = min(len(NOUNS), 3 + lesson_idx // 2)
    verb_count = min(len(VERBS), 2 + lesson_idx // 3)
    adjective_count = min(len(ADJECTIVES), max(0, lesson_idx - 1) // 2)
    tense_count = 1
    if lesson_idx >= 4:
        tense_count = 2
    if lesson_idx >= 8:
        tense_count = 3
    return (
        NOUNS[:noun_count],
        VERBS[:verb_count],
        ADJECTIVES[:adjective_count],
        TENSES[:tense_count],
    )


def lexical_traces_for_lesson(lesson_idx: int, language_idx: int | None = None) -> list[LexicalTrace]:
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx if language_idx is None else language_idx)
    traces: list[LexicalTrace] = []
    for tense in tenses:
        traces.append(LexicalTrace(tense.src, tense.en, "tense"))
    for noun in nouns:
        traces.append(LexicalTrace(noun.src, noun.en, "noun"))
    for verb in verbs:
        traces.append(LexicalTrace(verb.src, verb.en, "verb"))
    for adjective in adjectives:
        traces.append(LexicalTrace(adjective.src, adjective.en, "adjective"))
    return traces


def language_terms_for_lesson(lesson_idx: int) -> list[str]:
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx)
    terms = ["lyran"]
    for word in [*nouns, *verbs, *adjectives, *tenses]:
        terms.append(word.src)
    return terms


def english_np(noun: Word, adjective: Word | None) -> str:
    if adjective is None:
        return f"the {noun.en}"
    return f"the {adjective.en} {noun.en}"


def english_verb(verb: Word, tense: Word) -> str:
    if tense.src == "pa":
        assert verb.past is not None
        return verb.past
    if tense.src == "fu":
        base = {
            "sees": "see",
            "likes": "like",
            "helps": "help",
            "chases": "chase",
            "finds": "find",
            "holds": "hold",
        }[verb.en]
        return f"will {base}"
    return verb.en


def make_sentence(
    rng: random.Random,
    nouns: list[Word],
    verbs: list[Word],
    adjectives: list[Word],
    tenses: list[Word],
    force_modifier: bool = False,
) -> tuple[str, str, dict[str, Word | None]]:
    tense = rng.choice(tenses)
    verb = rng.choice(verbs)
    subject, obj = rng.sample(nouns, 2)
    use_subj_adj = bool(adjectives) and (force_modifier or rng.random() < 0.65)
    use_obj_adj = bool(adjectives) and (force_modifier or rng.random() < 0.65)
    subj_adj = rng.choice(adjectives) if use_subj_adj else None
    obj_adj = rng.choice(adjectives) if use_obj_adj else None

    parts = [tense.src, verb.src, subject.src]
    if subj_adj is not None:
        parts.append(subj_adj.src)
    parts.append(obj.src)
    if obj_adj is not None:
        parts.append(obj_adj.src)
    source = " ".join(parts)
    english = f"{english_np(subject, subj_adj)} {english_verb(verb, tense)} {english_np(obj, obj_adj)}."
    slots: dict[str, Word | None] = {
        "tense": tense,
        "verb": verb,
        "subject": subject,
        "object": obj,
        "subj_adj": subj_adj,
        "obj_adj": obj_adj,
    }
    return source, english, slots


def add_unique(items: list[str], value: str, answer: str) -> None:
    if value != answer and value not in items:
        items.append(value)


def first_other(items: list[Word], excluded: set[str]) -> Word | None:
    for item in items:
        if item.src not in excluded:
            return item
    return None


def distractors(answer: str, slots: dict[str, Word | None], nouns: list[Word], verbs: list[Word], adjectives: list[Word], tenses: list[Word]) -> list[str]:
    tense = slots["tense"]
    verb = slots["verb"]
    subject = slots["subject"]
    obj = slots["object"]
    subj_adj = slots["subj_adj"]
    obj_adj = slots["obj_adj"]
    assert isinstance(tense, Word)
    assert isinstance(verb, Word)
    assert isinstance(subject, Word)
    assert isinstance(obj, Word)
    out: list[str] = []
    add_unique(
        out,
        f"{english_np(obj, obj_adj)} {english_verb(verb, tense)} {english_np(subject, subj_adj)}.",
        answer,
    )
    alt_verb = first_other(verbs + VERBS, {verb.src})
    if alt_verb is not None:
        add_unique(
            out,
            f"{english_np(subject, subj_adj)} {english_verb(alt_verb, tense)} {english_np(obj, obj_adj)}.",
            answer,
        )
    if len(tenses) > 1:
        alt_tense = first_other(tenses, {tense.src})
        if alt_tense is not None:
            add_unique(
                out,
                f"{english_np(subject, subj_adj)} {english_verb(verb, alt_tense)} {english_np(obj, obj_adj)}.",
                answer,
            )
    else:
        alt_tense = first_other(TENSES, {tense.src})
        if alt_tense is not None:
            add_unique(
                out,
                f"{english_np(subject, subj_adj)} {english_verb(verb, alt_tense)} {english_np(obj, obj_adj)}.",
                answer,
            )
    if adjectives and (subj_adj is not None or obj_adj is not None):
        add_unique(
            out,
            f"{english_np(subject, obj_adj)} {english_verb(verb, tense)} {english_np(obj, subj_adj)}.",
            answer,
        )
    alt_obj = first_other(nouns + NOUNS, {subject.src, obj.src})
    if alt_obj is not None:
        add_unique(
            out,
            f"{english_np(subject, subj_adj)} {english_verb(verb, tense)} {english_np(alt_obj, obj_adj)}.",
            answer,
        )
    return out


def make_translation_question(
    rng: random.Random,
    nouns: list[Word],
    verbs: list[Word],
    adjectives: list[Word],
    tenses: list[Word],
    category: str,
    force_modifier: bool = False,
) -> TranslationQuestion:
    source, answer, slots = make_sentence(rng, nouns, verbs, adjectives, tenses, force_modifier=force_modifier)
    wrong = distractors(answer, slots, nouns, verbs, adjectives, tenses)
    while len(wrong) < 3:
        _src, candidate, _slots = make_sentence(rng, NOUNS, VERBS, ADJECTIVES, TENSES)
        if candidate != answer and candidate not in wrong:
            wrong.append(candidate)
    options = [answer, *wrong[:3]]
    rng.shuffle(options)
    return TranslationQuestion(
        sentence=source,
        answer=answer,
        options=options,
        answer_idx=options.index(answer),
        category=category,
    )


def make_question_from_slots(
    rng: random.Random,
    tense: Word,
    verb: Word,
    subject: Word,
    obj: Word,
    subj_adj: Word | None,
    obj_adj: Word | None,
    nouns: list[Word],
    verbs: list[Word],
    adjectives: list[Word],
    tenses: list[Word],
    category: str,
) -> TranslationQuestion:
    parts = [tense.src, verb.src, subject.src]
    if subj_adj is not None:
        parts.append(subj_adj.src)
    parts.append(obj.src)
    if obj_adj is not None:
        parts.append(obj_adj.src)
    source = " ".join(parts)
    answer = f"{english_np(subject, subj_adj)} {english_verb(verb, tense)} {english_np(obj, obj_adj)}."
    slots: dict[str, Word | None] = {
        "tense": tense,
        "verb": verb,
        "subject": subject,
        "object": obj,
        "subj_adj": subj_adj,
        "obj_adj": obj_adj,
    }
    wrong = distractors(answer, slots, nouns, verbs, adjectives, tenses)
    while len(wrong) < 3:
        _src, candidate, _slots = make_sentence(rng, NOUNS, VERBS, ADJECTIVES, TENSES)
        if candidate != answer and candidate not in wrong:
            wrong.append(candidate)
    options = [answer, *wrong[:3]]
    rng.shuffle(options)
    return TranslationQuestion(
        sentence=source,
        answer=answer,
        options=options,
        answer_idx=options.index(answer),
        category=category,
    )


PERSPECTIVE_INSTRUCTIONS = {
    "direct": "Translate this Lyran sentence into English. Write only the English sentence.",
    "grammar": (
        "Use the Lyran grammar rules from the lesson. Convert from Lyran word order "
        "to ordinary English word order. Write only the English sentence."
    ),
    "lexicon": (
        "Recall the meanings of the Lyran words from the lesson, then translate the "
        "sentence. Write only the English sentence."
    ),
    "roles": (
        "Identify the tense, action, subject, object, and modifiers internally, then "
        "translate. Write only the English sentence."
    ),
}


def render_question(question: TranslationQuestion, perspective: str = "direct") -> str:
    instruction = PERSPECTIVE_INSTRUCTIONS[perspective]
    return (
        f"{instruction}\n\n"
        f"Lyran: {question.sentence}\n"
        "English:"
    )


def render_object_gate_question(question: TranslationQuestion) -> str:
    """Prompt whose suffix is the source object, not the answer boundary."""

    return (
        f"{PERSPECTIVE_INSTRUCTIONS['direct']}\n\n"
        f"Lyran: {question.sentence}"
    )


def render_lexical_question(trace: LexicalTrace) -> str:
    return (
        "Translate this Lyran word or marker into English. "
        "Write only the English meaning.\n\n"
        f"Lyran: {trace.src}\n"
        "English:"
    )


def render_context_span_question(trace: ContextSpanTrace) -> str:
    return (
        "Complete this missing continuation from a document. "
        "Write only the missing text.\n\n"
        f"Known text: {trace.prefix}\n"
        "Missing text:"
    )


def render_lesson(lesson_idx: int, example_count: int, seed: int, language_idx: int | None = None) -> str:
    rng = random.Random(seed + lesson_idx * 997)
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx if language_idx is None else language_idx)
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
    example_lines = "\n".join(
        f"- {item.sentence} -> {item.answer}"
        for item in examples
    )
    return (
        f"Lyran lesson {lesson_idx + 1}.\n"
        "Lyran is an invented language. Translate it using these rules.\n"
        "Sentence order is: TENSE VERB SUBJECT OBJECT.\n"
        "Adjectives come after the noun they modify.\n"
        "English output should use ordinary English word order: subject, verb, object.\n\n"
        f"Tense words: {tense_lines}.\n"
        f"Nouns: {noun_lines}.\n"
        f"Verbs: {verb_lines}.\n"
        f"Adjectives: {adjective_lines}.\n\n"
        f"Examples:\n{example_lines}\n"
    )


def add_context_span_trace(
    traces: list[ContextSpanTrace],
    seen: set[tuple[str, str]],
    prefix: str,
    answer: str,
    category: str,
) -> None:
    prefix = " ".join(prefix.strip().split())
    answer = " ".join(answer.strip().split())
    if not prefix or not answer:
        return
    if len(prefix) < 2 or len(answer) < 1 or len(answer) > 160:
        return
    key = (prefix, answer)
    if key in seen:
        return
    seen.add(key)
    traces.append(ContextSpanTrace(prefix, answer, category))


def context_span_traces(context: str, seed: int, max_items: int = 0) -> list[ContextSpanTrace]:
    """Extract generic self-supervised content traces from context text.

    This intentionally reads only the rendered context text. It does not use
    hidden DSL objects or mini-language vocabulary lists. Structured spans such
    as `x=y` and `x -> y` are treated as ordinary surface evidence, and a small
    line-continuation fallback covers less structured prose.
    """

    traces: list[ContextSpanTrace] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.removeprefix("- ").strip()
        if " -> " in line:
            left, right = line.split(" -> ", 1)
            add_context_span_trace(traces, seen, f"{left.strip()} ->", right.strip(), "arrow_span")
        if "=" in line:
            after_colon = line.split(":", 1)[1] if ":" in line else line
            for item in after_colon.split(","):
                if "=" not in item:
                    continue
                left, right = item.split("=", 1)
                add_context_span_trace(
                    traces,
                    seen,
                    f"{left.strip()}=",
                    right.strip().strip(".;"),
                    "equals_span",
                )
        words = line.split()
        if 8 <= len(words) <= 80:
            split_at = min(max(4, len(words) // 2), 12)
            answer_words = words[split_at : split_at + 12]
            add_context_span_trace(
                traces,
                seen,
                " ".join(words[:split_at]),
                " ".join(answer_words),
                "line_continuation",
            )
    if max_items > 0 and len(traces) > max_items:
        rng = random.Random(seed)
        traces = rng.sample(traces, max_items)
    return traces


def build_questions(
    count: int,
    seed: int,
    lesson_idx: int,
    category: str,
    language_idx: int | None = None,
) -> list[TranslationQuestion]:
    rng = random.Random(seed)
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx if language_idx is None else language_idx)
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


def build_balanced_questions(
    count: int,
    seed: int,
    lesson_idx: int,
    category: str,
    language_idx: int | None = None,
) -> list[TranslationQuestion]:
    rng = random.Random(seed)
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx if language_idx is None else language_idx)
    questions = []
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


def question_key(question: TranslationQuestion) -> tuple[str, str]:
    return question.sentence, question.answer


def lesson_example_keys(lesson_texts: list[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for text in lesson_texts:
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("- ") or " -> " not in line:
                continue
            source, answer = line[2:].split(" -> ", 1)
            keys.add((source.strip(), answer.strip()))
    return keys


def trace_probe_keys(args: argparse.Namespace) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    build_trace_questions = build_balanced_questions if args.balanced_trace else build_questions
    trace_seed = args.seed + getattr(args, "trace_seed_offset", 0)
    final_lesson_idx = args.lessons - 1
    final_language_idx = (
        min(final_lesson_idx, args.freeze_language_after)
        if args.freeze_language_after is not None
        else final_lesson_idx
    )
    if args.ensemble_corpora > 1:
        lesson_indices = range(args.lessons) if args.ensemble_per_lesson else [final_lesson_idx]
        for corpus_idx in range(args.ensemble_corpora):
            corpus_seed = args.seed + (corpus_idx + 1) * args.ensemble_seed_stride
            for lesson_idx in lesson_indices:
                trace_language_idx = (
                    min(lesson_idx, args.freeze_language_after)
                    if args.freeze_language_after is not None
                    else lesson_idx
                )
                if args.ensemble_per_lesson:
                    shared_seed = trace_seed + 17_003 + lesson_idx * 10_000
                    corpus_probe_seed = corpus_seed + 17_003 + lesson_idx * 10_000
                else:
                    shared_seed = trace_seed + 17_003
                    corpus_probe_seed = corpus_seed + 17_003
                probe_seed = shared_seed if args.ensemble_shared_probes else corpus_probe_seed
                for question in build_trace_questions(
                    args.trace_probes,
                    probe_seed,
                    lesson_idx,
                    "trace_translation",
                    language_idx=trace_language_idx,
                ):
                    keys.add(question_key(question))
        if args.ensemble_include_anchor:
            for lesson_idx in lesson_indices:
                trace_language_idx = (
                    min(lesson_idx, args.freeze_language_after)
                    if args.freeze_language_after is not None
                    else lesson_idx
                )
                probe_seed = trace_seed + 17_003 + lesson_idx * 10_000 if args.ensemble_per_lesson else trace_seed + 17_003
                for question in build_trace_questions(
                    args.trace_probes,
                    probe_seed,
                    lesson_idx,
                    "trace_translation",
                    language_idx=trace_language_idx,
                ):
                    keys.add(question_key(question))
        return keys

    for lesson_idx in range(args.lessons):
        trace_language_idx = (
            min(lesson_idx, args.freeze_language_after)
            if args.freeze_language_after is not None
            else lesson_idx
        )
        for question in build_trace_questions(
            args.trace_probes,
            trace_seed + lesson_idx * 10_000,
            lesson_idx,
            "trace_translation",
            language_idx=trace_language_idx,
        ):
            keys.add(question_key(question))
    return keys


def build_unique_random_questions(
    count: int,
    seed: int,
    lesson_idx: int,
    category: str,
    language_idx: int | None = None,
    max_attempts: int = 10_000,
) -> list[TranslationQuestion]:
    rng = random.Random(seed)
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx if language_idx is None else language_idx)
    questions: list[TranslationQuestion] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(questions) < count and attempts < max_attempts:
        attempts += 1
        question = make_translation_question(
            rng,
            nouns,
            verbs,
            adjectives,
            tenses,
            category=category,
            force_modifier=True,
        )
        key = question_key(question)
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)
    if len(questions) < count:
        raise ValueError(
            f"Only generated {len(questions)} unique questions after {max_attempts} attempts; requested {count}."
        )
    return questions


def build_exhaustive_modified_questions(
    seed: int,
    lesson_idx: int,
    category: str,
    language_idx: int | None = None,
) -> list[TranslationQuestion]:
    rng = random.Random(seed)
    nouns, verbs, adjectives, tenses = introduced_items(lesson_idx if language_idx is None else language_idx)
    subj_adjectives = adjectives if adjectives else [None]
    obj_adjectives = adjectives if adjectives else [None]
    questions: list[TranslationQuestion] = []
    for tense in tenses:
        for verb in verbs:
            for subject in nouns:
                for obj in nouns:
                    if obj.src == subject.src:
                        continue
                    for subj_adj in subj_adjectives:
                        for obj_adj in obj_adjectives:
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


def build_eval_questions(args: argparse.Namespace, lesson_texts: list[str]) -> tuple[list[TranslationQuestion], dict[str, int | str]]:
    if args.eval_questions_jsonl:
        rows = [
            json.loads(line)
            for line in Path(args.eval_questions_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        questions = []
        for row in rows:
            options = list(row["options"])
            answer = row["answer"]
            answer_idx = options.index(answer)
            questions.append(
                TranslationQuestion(
                    sentence=row["sentence"],
                    answer=answer,
                    options=options,
                    answer_idx=answer_idx,
                    category=row.get("category", "heldout_translation_jsonl"),
                )
            )
        if not questions:
            raise ValueError(f"No eval questions found in {args.eval_questions_jsonl}")
        metadata: dict[str, int | str] = {
            "eval_mode": "jsonl",
            "eval_original_count": len(questions),
            "eval_deduped_count": len(questions),
            "eval_final_count": len(questions),
            "eval_duplicate_removed": 0,
            "eval_lesson_overlap_count": 0,
            "eval_trace_overlap_count": 0,
            "eval_exclude_lesson_overlaps": int(args.exclude_eval_lesson_overlaps),
            "eval_exclude_trace_overlaps": int(args.exclude_eval_trace_overlaps),
            "eval_questions_jsonl": args.eval_questions_jsonl,
        }
        return questions, metadata

    final_lesson_idx = args.lessons - 1
    final_language_idx = (
        min(final_lesson_idx, args.freeze_language_after)
        if args.freeze_language_after is not None
        else final_lesson_idx
    )
    seed = args.seed + 91_000
    if args.eval_mode == "random":
        questions = build_questions(
            args.eval_questions,
            seed,
            final_lesson_idx,
            "heldout_translation",
            language_idx=final_language_idx,
        )
    elif args.eval_mode == "unique_random":
        questions = build_unique_random_questions(
            args.eval_questions,
            seed,
            final_lesson_idx,
            "heldout_translation",
            language_idx=final_language_idx,
            max_attempts=args.eval_max_attempts,
        )
    else:
        questions = build_exhaustive_modified_questions(
            seed,
            final_lesson_idx,
            "heldout_translation_exhaustive",
            language_idx=final_language_idx,
        )

    original_count = len(questions)
    seen: set[tuple[str, str]] = set()
    deduped: list[TranslationQuestion] = []
    for question in questions:
        key = question_key(question)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
    questions = deduped

    lesson_keys = lesson_example_keys(lesson_texts)
    trace_keys = trace_probe_keys(args)
    lesson_overlap_count = sum(1 for question in questions if question_key(question) in lesson_keys)
    trace_overlap_count = sum(1 for question in questions if question_key(question) in trace_keys)
    if args.exclude_eval_lesson_overlaps:
        questions = [question for question in questions if question_key(question) not in lesson_keys]
    if args.exclude_eval_trace_overlaps:
        questions = [question for question in questions if question_key(question) not in trace_keys]
    if not questions:
        raise ValueError("Strict eval filtering removed every question.")

    metadata: dict[str, int | str] = {
        "eval_mode": args.eval_mode,
        "eval_original_count": original_count,
        "eval_deduped_count": len(deduped),
        "eval_final_count": len(questions),
        "eval_duplicate_removed": original_count - len(deduped),
        "eval_lesson_overlap_count": lesson_overlap_count,
        "eval_trace_overlap_count": trace_overlap_count,
        "eval_exclude_lesson_overlaps": int(args.exclude_eval_lesson_overlaps),
        "eval_exclude_trace_overlaps": int(args.exclude_eval_trace_overlaps),
    }
    return questions, metadata


def format_prompt(
    tokenizer,
    question: TranslationQuestion,
    context: str | None,
    use_chat_template: bool,
    perspective: str = "direct",
) -> str:
    body = ""
    if context:
        body = context.strip() + "\n\n"
    body += render_question(question, perspective=perspective)
    return format_model_prompt(tokenizer, body, use_chat_template)


def object_gate_prompts_for_questions(
    tokenizer,
    questions: list[TranslationQuestion],
    use_chat_template: bool,
    max_prompts: int = 0,
    seed: int = 0,
) -> list[str]:
    selected = list(questions)
    if max_prompts > 0 and len(selected) > max_prompts:
        selected = random.Random(seed).sample(selected, max_prompts)
    return [
        format_model_prompt(tokenizer, render_object_gate_question(question), use_chat_template)
        for question in selected
    ]


def object_gate_prompts_and_indices(
    tokenizer,
    questions: list[TranslationQuestion],
    use_chat_template: bool,
    max_prompts: int = 0,
    seed: int = 0,
) -> tuple[list[str], list[int]]:
    selected = list(questions)
    if max_prompts > 0 and len(selected) > max_prompts:
        selected = random.Random(seed).sample(selected, max_prompts)
    prompts: list[str] = []
    indices: list[int] = []
    for question in selected:
        prompt = format_prompt(tokenizer, question, None, use_chat_template)
        marker = "\nEnglish:"
        marker_index = prompt.rfind(marker)
        prefix = prompt[:marker_index] if marker_index >= 0 else prompt
        token_index = len(encode_no_special(tokenizer, prefix)) - 1
        prompts.append(prompt)
        indices.append(max(0, token_index))
    return prompts, indices


def language_object_gate_prompts_and_indices(
    tokenizer,
    questions: list[TranslationQuestion],
    use_chat_template: bool,
    *,
    language_name: str = "Lyran",
    token_prefix: str = "",
    max_prompts: int = 0,
    seed: int = 0,
) -> tuple[list[str], list[int]]:
    selected = list(questions)
    if max_prompts > 0 and len(selected) > max_prompts:
        selected = random.Random(seed).sample(selected, max_prompts)
    prompts: list[str] = []
    indices: list[int] = []
    for question in selected:
        sentence = question.sentence
        if token_prefix:
            sentence = " ".join(f"{token_prefix}{token}" for token in sentence.split())
        body = (
            f"Translate this {language_name} sentence into English. "
            "Write only the English sentence.\n\n"
            f"{language_name}: {sentence}\n"
            "English:"
        )
        prompt = format_model_prompt(tokenizer, body, use_chat_template)
        marker = "\nEnglish:"
        marker_index = prompt.rfind(marker)
        prefix = prompt[:marker_index] if marker_index >= 0 else prompt
        token_index = len(encode_no_special(tokenizer, prefix)) - 1
        prompts.append(prompt)
        indices.append(max(0, token_index))
    return prompts, indices


def format_lexical_prompt(tokenizer, trace: LexicalTrace, context: str | None, use_chat_template: bool) -> str:
    body = ""
    if context:
        body = context.strip() + "\n\n"
    body += render_lexical_question(trace)
    return format_model_prompt(tokenizer, body, use_chat_template)


def format_context_span_prompt(
    tokenizer,
    trace: ContextSpanTrace,
    context: str | None,
    use_chat_template: bool,
) -> str:
    body = ""
    if context:
        body = context.strip() + "\n\n"
    body += render_context_span_question(trace)
    return format_model_prompt(tokenizer, body, use_chat_template)


def answer_prefixes(tokenizer, answer: str, token_level: bool = False) -> list[str]:
    """Prefixes used to expose translation as a sequence of next-token states."""

    if token_level:
        token_ids = encode_no_special(tokenizer, " " + answer)
        return [
            tokenizer.decode(token_ids[:idx], clean_up_tokenization_spaces=False)
            for idx in range(len(token_ids))
        ]
    words = answer.strip().split()
    prefixes = [""]
    running: list[str] = []
    for word in words:
        running.append(word)
        prefixes.append(" " + " ".join(running))
    return prefixes[:-1]


def teacher_forced_prompts(
    tokenizer,
    base: str,
    answer: str,
    teacher_forcing: bool,
    token_teacher_forcing: bool = False,
) -> list[str]:
    if not teacher_forcing:
        return [base]
    return [base + prefix for prefix in answer_prefixes(tokenizer, answer, token_teacher_forcing)]


def trace_prompts_for_answer(
    tokenizer,
    question: TranslationQuestion,
    answer: str,
    context: str | None,
    use_chat_template: bool,
    teacher_forcing: bool,
    token_teacher_forcing: bool = False,
    perspective: str = "direct",
) -> list[str]:
    base = format_prompt(tokenizer, question, context, use_chat_template, perspective=perspective)
    return teacher_forced_prompts(tokenizer, base, answer, teacher_forcing, token_teacher_forcing)


def trace_prompts_for_question(
    tokenizer,
    question: TranslationQuestion,
    context: str | None,
    use_chat_template: bool,
    teacher_forcing: bool,
    token_teacher_forcing: bool = False,
    perspective: str = "direct",
) -> list[str]:
    return trace_prompts_for_answer(
        tokenizer,
        question,
        question.answer,
        context,
        use_chat_template,
        teacher_forcing,
        token_teacher_forcing,
        perspective=perspective,
    )


def trace_prompts_for_lexical_trace(
    tokenizer,
    trace: LexicalTrace,
    context: str | None,
    use_chat_template: bool,
    teacher_forcing: bool,
    token_teacher_forcing: bool = False,
) -> list[str]:
    base = format_lexical_prompt(tokenizer, trace, context, use_chat_template)
    return teacher_forced_prompts(tokenizer, base, trace.answer, teacher_forcing, token_teacher_forcing)


def trace_prompts_for_context_span_trace(
    tokenizer,
    trace: ContextSpanTrace,
    context: str | None,
    use_chat_template: bool,
    teacher_forcing: bool,
    token_teacher_forcing: bool = False,
) -> list[str]:
    base = format_context_span_prompt(tokenizer, trace, context, use_chat_template)
    return teacher_forced_prompts(tokenizer, base, trace.answer, teacher_forcing, token_teacher_forcing)


def packed_items_from_questions(questions: list[TranslationQuestion]) -> list[PackedUseItem]:
    return [
        PackedUseItem(
            prompt=f'Translate the Lyran sentence into English: "{question.sentence}"',
            answer=question.answer,
            category=question.category,
        )
        for question in questions
    ]


def packed_items_from_context_spans(spans: list[ContextSpanTrace]) -> list[PackedUseItem]:
    return [
        PackedUseItem(
            prompt=f'Complete the missing text after this context fragment: "{span.prefix}"',
            answer=span.answer,
            category=f"context_span:{span.category}",
        )
        for span in spans
    ]


def packed_use_prompts(
    tokenizer,
    items: list[PackedUseItem],
    context: str | None,
    use_chat_template: bool,
    teacher_forcing: bool,
    token_teacher_forcing: bool,
    mode: str,
) -> list[str]:
    """Build pre-answer prefixes from a packed use episode.

    `clean` captures each use-site with the lesson and current question only.
    `curriculum` captures a single quiz-like episode where earlier Q/A pairs
    remain in context before later answers.
    """

    if mode not in {"clean", "curriculum"}:
        raise ValueError(f"Unknown packed use mode: {mode}")
    prompts: list[str] = []
    answered_lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        current = f"Q{idx}: {item.prompt}\nA{idx}:"
        quiz_lines = [*answered_lines, current] if mode == "curriculum" else [current]
        body_parts: list[str] = []
        if context:
            body_parts.append(context.strip())
        body_parts.append(
            "Use the information above to answer the quiz. "
            "Each answer should contain only the requested text.\n\n"
            + "\n\n".join(quiz_lines)
        )
        base = format_model_prompt(tokenizer, "\n\n".join(body_parts), use_chat_template)
        prompts.extend(
            teacher_forced_prompts(
                tokenizer,
                base,
                item.answer,
                teacher_forcing,
                token_teacher_forcing,
            )
        )
        answered_lines.append(f"Q{idx}: {item.prompt}\nA{idx}: {item.answer}")
    return prompts


def encode_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@torch.no_grad()
def option_logprobs(model, tokenizer, prompt: str, options: list[str], device: torch.device, max_length: int) -> list[float]:
    prompt_ids = encode_no_special(tokenizer, prompt)
    option_ids = [encode_no_special(tokenizer, " " + option) for option in options]
    max_option_len = max(len(ids) for ids in option_ids)
    prompt_budget = max(8, max_length - max_option_len)
    prompt_ids = prompt_ids[-prompt_budget:]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    set_active_slot_weights_for_prompts(model, [prompt] * len(options))
    try:
        prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prefix_mask = torch.ones_like(prefix)
        set_active_slot_weights_for_prompts(model, [prompt])
        outputs = model(input_ids=prefix, attention_mask=prefix_mask, use_cache=True)
        first_log_probs = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
        scores = [float(first_log_probs[ids[0]].item()) if ids else float("-inf") for ids in option_ids]
        counts = [1 if ids else 0 for ids in option_ids]

        if max_option_len > 1:
            cache = outputs.past_key_values
            repeated = cache.batch_repeat_interleave(len(options))
            if repeated is not None:
                cache = repeated
            prev_tokens = torch.tensor(
                [[ids[0] if ids else pad_id] for ids in option_ids],
                dtype=torch.long,
                device=device,
            )
            for step in range(1, max_option_len):
                set_active_slot_weights_for_prompts(model, [prompt] * len(options))
                attention_mask = torch.ones(
                    len(options),
                    len(prompt_ids) + step,
                    dtype=torch.long,
                    device=device,
                )
                outputs = model(
                    input_ids=prev_tokens,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = outputs.past_key_values
                log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
                next_tokens = []
                for idx, ids in enumerate(option_ids):
                    if step < len(ids):
                        token_id = ids[step]
                        scores[idx] += float(log_probs[idx, token_id].item())
                        counts[idx] += 1
                        next_tokens.append(token_id)
                    else:
                        next_tokens.append(pad_id)
                prev_tokens = torch.tensor([[token_id] for token_id in next_tokens], dtype=torch.long, device=device)

        return [score / max(count, 1) for score, count in zip(scores, counts)]
    finally:
        clear_active_slot_weights(model)


@torch.no_grad()
def answer_letter_logprobs(model, tokenizer, prompt: str, device: torch.device, max_length: int) -> list[float]:
    """Legacy diagnostic for answer-letter bias, not used as the main metric."""

    set_active_slot_weights_for_prompts(model, [prompt])
    try:
        tokens = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        outputs = model(**tokens, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
        values = []
        for letter in LETTERS:
            token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if not token_ids:
                token_ids = tokenizer.encode(letter, add_special_tokens=False)
            values.append(float(log_probs[token_ids[0]].item()))
        return values
    finally:
        clear_active_slot_weights(model)


def evaluate_mc(model, tokenizer, questions: list[TranslationQuestion], device: torch.device, context: str | None, max_length: int, use_chat_template: bool) -> dict:
    correct = 0
    margins = []
    predictions = []
    details = []
    for question in questions:
        prompt = format_prompt(tokenizer, question, context, use_chat_template)
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


def sentinel_questions(suite: str = "core") -> list[dict]:
    core = [
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat color is a clear daytime sky usually described as?\nA. Blue\nB. Seven\nC. Quickly\nD. Table\n\nAnswer:",
            "options": ["Blue", "Seven", "Quickly", "Table"],
            "answer_idx": 0,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat is 7 plus 5?\nA. 10\nB. 11\nC. 12\nD. 13\n\nAnswer:",
            "options": ["10", "11", "12", "13"],
            "answer_idx": 2,
            "category": "arithmetic",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word is a verb?\nA. Marble\nB. Run\nC. Green\nD. North\n\nAnswer:",
            "options": ["Marble", "Run", "Green", "North"],
            "answer_idx": 1,
            "category": "grammar",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich animal is known for barking?\nA. Dog\nB. Spoon\nC. Cloud\nD. Triangle\n\nAnswer:",
            "options": ["Dog", "Spoon", "Cloud", "Triangle"],
            "answer_idx": 0,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich city is the capital of France?\nA. Madrid\nB. Berlin\nC. Paris\nD. Rome\n\nAnswer:",
            "options": ["Madrid", "Berlin", "Paris", "Rome"],
            "answer_idx": 2,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word means the opposite of hot?\nA. Cold\nB. Tall\nC. Loud\nD. Smooth\n\nAnswer:",
            "options": ["Cold", "Tall", "Loud", "Smooth"],
            "answer_idx": 0,
            "category": "semantics",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich item is used for writing?\nA. Pencil\nB. Ocean\nC. Yesterday\nD. Running\n\nAnswer:",
            "options": ["Pencil", "Ocean", "Yesterday", "Running"],
            "answer_idx": 0,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat is the plural of child?\nA. Childs\nB. Children\nC. Childes\nD. Childing\n\nAnswer:",
            "options": ["Childs", "Children", "Childes", "Childing"],
            "answer_idx": 1,
            "category": "grammar",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich number is larger?\nA. 14\nB. 9\nC. 3\nD. 0\n\nAnswer:",
            "options": ["14", "9", "3", "0"],
            "answer_idx": 0,
            "category": "arithmetic",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich sentence is ordinary English?\nA. na lum niv drem\nB. The cat sees the dog.\nC. pa vek soka palo\nD. fu narp vesh tul\n\nAnswer:",
            "options": ["na lum niv drem", "The cat sees the dog.", "pa vek soka palo", "fu narp vesh tul"],
            "answer_idx": 1,
            "category": "lyran_contamination",
        },
    ]
    if suite == "core":
        return core
    if suite != "expanded":
        raise ValueError(f"Unknown sentinel suite {suite!r}")
    expanded = [
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat is 9 minus 4?\nA. 3\nB. 4\nC. 5\nD. 6\n\nAnswer:",
            "options": ["3", "4", "5", "6"],
            "answer_idx": 2,
            "category": "arithmetic",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word is a noun?\nA. Table\nB. Quickly\nC. Blue\nD. Under\n\nAnswer:",
            "options": ["Table", "Quickly", "Blue", "Under"],
            "answer_idx": 0,
            "category": "grammar",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich month comes after March?\nA. January\nB. April\nC. August\nD. November\n\nAnswer:",
            "options": ["January", "April", "August", "November"],
            "answer_idx": 1,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word means the opposite of empty?\nA. Full\nB. Late\nC. Thin\nD. Quiet\n\nAnswer:",
            "options": ["Full", "Late", "Thin", "Quiet"],
            "answer_idx": 0,
            "category": "semantics",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat is 3 times 4?\nA. 7\nB. 9\nC. 12\nD. 14\n\nAnswer:",
            "options": ["7", "9", "12", "14"],
            "answer_idx": 2,
            "category": "arithmetic",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich sentence is ordinary English?\nA. The teacher opens the book.\nB. mi vek soka ro\nC. pa narp vesh tul\nD. fu lum niv drem\n\nAnswer:",
            "options": ["The teacher opens the book.", "mi vek soka ro", "pa narp vesh tul", "fu lum niv drem"],
            "answer_idx": 0,
            "category": "lyran_contamination",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich item is used for drinking water?\nA. Cup\nB. Mountain\nC. Tuesday\nD. Running\n\nAnswer:",
            "options": ["Cup", "Mountain", "Tuesday", "Running"],
            "answer_idx": 0,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat is the plural of mouse?\nA. Mouses\nB. Mice\nC. Mousing\nD. Mices\n\nAnswer:",
            "options": ["Mouses", "Mice", "Mousing", "Mices"],
            "answer_idx": 1,
            "category": "grammar",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word means the opposite of early?\nA. Late\nB. Soft\nC. Wide\nD. Bright\n\nAnswer:",
            "options": ["Late", "Soft", "Wide", "Bright"],
            "answer_idx": 0,
            "category": "semantics",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich number is smallest?\nA. 18\nB. 2\nC. 7\nD. 11\n\nAnswer:",
            "options": ["18", "2", "7", "11"],
            "answer_idx": 1,
            "category": "arithmetic",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich city is in Japan?\nA. Tokyo\nB. Cairo\nC. Lima\nD. Oslo\n\nAnswer:",
            "options": ["Tokyo", "Cairo", "Lima", "Oslo"],
            "answer_idx": 0,
            "category": "common_knowledge",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word is an adjective?\nA. Running\nB. Chair\nC. Green\nD. Below\n\nAnswer:",
            "options": ["Running", "Chair", "Green", "Below"],
            "answer_idx": 2,
            "category": "grammar",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich sentence is ordinary English?\nA. na vek drem mi\nB. The child reads the note.\nC. ro soka tul pa\nD. mi narp palo fu\n\nAnswer:",
            "options": ["na vek drem mi", "The child reads the note.", "ro soka tul pa", "mi narp palo fu"],
            "answer_idx": 1,
            "category": "lyran_contamination",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhat is 20 divided by 5?\nA. 2\nB. 3\nC. 4\nD. 5\n\nAnswer:",
            "options": ["2", "3", "4", "5"],
            "answer_idx": 2,
            "category": "arithmetic",
        },
        {
            "prompt": "Choose the correct answer. Write only the answer text.\n\nWhich word means the opposite of noisy?\nA. Quiet\nB. Heavy\nC. Short\nD. Sour\n\nAnswer:",
            "options": ["Quiet", "Heavy", "Short", "Sour"],
            "answer_idx": 0,
            "category": "semantics",
        },
    ]
    return core + expanded


def evaluate_generic_mc(model, tokenizer, rows: list[dict], device: torch.device, max_length: int, use_chat_template: bool) -> dict:
    correct = 0
    margins = []
    predictions = []
    details = []
    for row in rows:
        prompt = format_model_prompt(tokenizer, row["prompt"], use_chat_template)
        scores = option_logprobs(model, tokenizer, prompt, row["options"], device, max_length)
        pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        answer_idx = int(row["answer_idx"])
        correct += int(pred_idx == answer_idx)
        sorted_wrong = max(score for idx, score in enumerate(scores) if idx != answer_idx)
        margin = scores[answer_idx] - sorted_wrong
        margins.append(margin)
        predictions.append(LETTERS[pred_idx])
        details.append(
            {
                "prompt": row["prompt"],
                "options": row["options"],
                "answer_letter": LETTERS[answer_idx],
                "answer": row["options"][answer_idx],
                "prediction": LETTERS[pred_idx],
                "prediction_text": row["options"][pred_idx],
                "scores": scores,
                "margin": margin,
                "correct": pred_idx == answer_idx,
                "category": row.get("category", "sentinel"),
            }
        )
    return {
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "n": len(rows),
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "predictions": predictions,
        "details": details,
    }


def add_metrics(row: dict, prefix: str, metrics: dict) -> None:
    row[f"{prefix}_accuracy"] = metrics["accuracy"]
    row[f"{prefix}_correct"] = metrics["correct"]
    row[f"{prefix}_n"] = metrics["n"]
    row[f"{prefix}_mean_margin"] = metrics["mean_margin"]


def add_sentinel_shift_metrics(row: dict, before: dict, after: dict, prefix: str = "sentinel") -> None:
    before_details = before.get("details", [])
    after_details = after.get("details", [])
    count = min(len(before_details), len(after_details))
    if count == 0:
        return
    margin_deltas = [
        float(after_details[idx]["margin"]) - float(before_details[idx]["margin"])
        for idx in range(count)
    ]
    margin_drops = [max(0.0, -delta) for delta in margin_deltas]
    before_correct = [bool(before_details[idx]["correct"]) for idx in range(count)]
    after_correct = [bool(after_details[idx]["correct"]) for idx in range(count)]
    correct_to_wrong = sum(1 for b, a in zip(before_correct, after_correct) if b and not a)
    wrong_to_correct = sum(1 for b, a in zip(before_correct, after_correct) if (not b) and a)
    before_correct_count = sum(before_correct)
    correct_deltas = [
        delta for delta, was_correct in zip(margin_deltas, before_correct) if was_correct
    ]
    correct_drops = [max(0.0, -delta) for delta in correct_deltas]
    row[f"{prefix}_correct_to_wrong"] = correct_to_wrong
    row[f"{prefix}_wrong_to_correct"] = wrong_to_correct
    row[f"{prefix}_net_flip_delta"] = wrong_to_correct - correct_to_wrong
    row[f"{prefix}_preserved_correct"] = before_correct_count - correct_to_wrong
    row[f"{prefix}_preservation_rate"] = (
        (before_correct_count - correct_to_wrong) / before_correct_count if before_correct_count else 1.0
    )
    row[f"{prefix}_mean_abs_margin_delta"] = sum(abs(delta) for delta in margin_deltas) / count
    row[f"{prefix}_mean_margin_drop"] = sum(margin_drops) / count
    row[f"{prefix}_max_margin_drop"] = max(margin_drops)
    row[f"{prefix}_severe_margin_drop_count"] = sum(1 for drop in margin_drops if drop >= 1.0)
    row[f"{prefix}_before_correct_mean_margin_delta"] = (
        sum(correct_deltas) / len(correct_deltas) if correct_deltas else 0.0
    )
    row[f"{prefix}_before_correct_mean_margin_drop"] = (
        sum(correct_drops) / len(correct_drops) if correct_drops else 0.0
    )


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def directional_consensus_update(
    updates: list[torch.Tensor],
    *,
    min_agreement: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep the update direction shared across independently rendered lessons."""
    if not updates:
        raise ValueError("directional_consensus_update requires at least one update")
    stack = torch.stack(updates, dim=0)
    flat = stack.flatten(1)
    norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1e-8)
    unit = flat / norms[:, None]
    mean_unit = unit.mean(dim=0)
    agreement = torch.linalg.vector_norm(mean_unit).clamp_min(1e-8)
    direction = mean_unit / agreement
    projections = torch.matmul(flat, direction)
    positive = projections > 0
    mean_positive_projection = projections.clamp_min(0).mean()
    if float(agreement.item()) < min_agreement:
        final_flat = torch.zeros_like(direction)
    else:
        final_flat = direction * mean_positive_projection * agreement
    stats = {
        "directional_agreement": float(agreement.item()),
        "directional_min_agreement": float(min_agreement),
        "directional_positive_fraction": float(positive.float().mean().item()),
        "directional_mean_update_fro": float(norms.mean().item()),
        "directional_projection_mean": float(projections.mean().item()),
        "directional_projection_min": float(projections.min().item()),
        "directional_projection_max": float(projections.max().item()),
    }
    return final_flat.reshape_as(updates[0]), stats


def anchored_directional_update(
    anchor_update: torch.Tensor,
    support_updates: list[torch.Tensor],
    *,
    residual_scale: float,
    min_agreement: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shared direction plus a downscaled anchor-only residual.

    The support updates estimate what is stable across independently rendered
    lessons. The anchor update is the actual lesson trace we ultimately care
    about. We keep the shared update at full strength, then add only the part of
    the anchor that is orthogonal to the shared direction, scaled down.
    """

    if not support_updates:
        raise ValueError("anchored_directional_update requires at least one support update")
    shared, stats = directional_consensus_update(support_updates, min_agreement=min_agreement)
    anchor_flat = anchor_update.flatten()
    shared_flat = shared.flatten()
    shared_norm = torch.linalg.vector_norm(shared_flat).clamp_min(1e-8)
    shared_unit = shared_flat / shared_norm
    anchor_projection = torch.dot(anchor_flat, shared_unit)
    anchor_parallel_flat = anchor_projection * shared_unit
    anchor_residual_flat = anchor_flat - anchor_parallel_flat
    final_flat = shared_flat + residual_scale * anchor_residual_flat
    anchor_norm = torch.linalg.vector_norm(anchor_flat).clamp_min(1e-8)
    residual_norm = torch.linalg.vector_norm(anchor_residual_flat)
    stats.update(
        {
            "anchor_residual_scale": float(residual_scale),
            "anchor_update_fro": float(anchor_norm.item()),
            "anchor_projection_fro": float(anchor_projection.abs().item()),
            "anchor_residual_fro": float(residual_norm.item()),
            "anchor_shared_cosine": float((anchor_projection / anchor_norm).item()),
            "shared_update_fro": float(shared_norm.item()),
        }
    )
    return final_flat.reshape_as(anchor_update), stats


def dice_support_consensus_update(
    updates: list[torch.Tensor],
    *,
    support_threshold: float = 0.75,
    support_temperature: float = 16.0,
    support_strength: float = 1.0,
    support_cap: float = 2.0,
    support_floor: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Coordinate-level high-support consensus across diverse contexts.

    DICE is intentionally stricter than the older mean/directional ensembles:
    each update coordinate is kept only when its sign recurs across many
    independently rendered contexts. The goal is to preserve invariants shared
    across diverse contexts while suppressing context-local surface posture.
    """

    if not updates:
        raise ValueError("dice_support_consensus_update requires at least one update")
    if len(updates) == 1:
        return updates[0], {
            "dice_context_count": 1.0,
            "dice_support_threshold": float(support_threshold),
            "dice_gate_mean": 1.0,
            "dice_high_support_fraction": 1.0,
            "dice_mean_support": 1.0,
            "dice_mean_update_fro": float(torch.linalg.vector_norm(updates[0]).item()),
            "dice_final_update_fro": float(torch.linalg.vector_norm(updates[0]).item()),
        }

    stack = torch.stack([update.detach().float().cpu() for update in updates], dim=0)
    positive = (stack > 0).float().mean(dim=0)
    negative = (stack < 0).float().mean(dim=0)
    support = torch.maximum(positive, negative)
    mean_update = stack.mean(dim=0)
    threshold = min(max(float(support_threshold), 0.0), 1.0)
    temperature = max(float(support_temperature), 1e-6)
    gate = torch.sigmoid((support - threshold) * temperature)
    floor = min(max(float(support_floor), 0.0), 1.0)
    gate = floor + (1.0 - floor) * gate
    if support_strength != 0:
        gain = torch.exp((support - threshold) * float(support_strength))
        gain = gain.clamp(max=max(float(support_cap), 1.0))
        gate = gate * gain
    final_update = mean_update * gate
    norms = torch.linalg.vector_norm(stack.flatten(1), dim=1)
    stats = {
        "dice_context_count": float(len(updates)),
        "dice_support_threshold": threshold,
        "dice_support_temperature": float(support_temperature),
        "dice_support_strength": float(support_strength),
        "dice_support_cap": float(support_cap),
        "dice_support_floor": floor,
        "dice_mean_support": float(support.mean().item()),
        "dice_support_p90": float(torch.quantile(support.flatten(), 0.90).item()),
        "dice_support_p99": float(torch.quantile(support.flatten(), 0.99).item()),
        "dice_gate_mean": float(gate.mean().item()),
        "dice_gate_p90": float(torch.quantile(gate.flatten(), 0.90).item()),
        "dice_gate_p99": float(torch.quantile(gate.flatten(), 0.99).item()),
        "dice_high_support_fraction": float((support >= threshold).float().mean().item()),
        "dice_mean_update_fro": float(norms.mean().item()),
        "dice_mean_update_fro_std": float(norms.std(unbiased=False).item()),
        "dice_mean_map_fro": float(torch.linalg.vector_norm(mean_update).item()),
        "dice_final_update_fro": float(torch.linalg.vector_norm(final_update).item()),
    }
    stats.update(proposal_alignment_stats([update.detach().float().cpu() for update in updates]))
    return final_update.contiguous(), stats


def subspace_consensus_update(
    updates: list[torch.Tensor],
    *,
    rank: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep the low-rank update subspace shared across rendered corpora."""

    if not updates:
        raise ValueError("subspace_consensus_update requires at least one update")
    flat = torch.stack([update.flatten().float() for update in updates], dim=0)
    rank = max(1, min(rank, flat.shape[0], flat.shape[1]))
    _u, singular_values, vh = torch.linalg.svd(flat, full_matrices=False)
    basis = vh[:rank]
    projected = flat @ basis.T @ basis
    final_flat = projected.mean(dim=0)
    total_energy = torch.linalg.vector_norm(flat).square().clamp_min(1e-12)
    kept_energy = torch.linalg.vector_norm(projected).square()
    stats = {
        "subspace_rank": float(rank),
        "subspace_supports": float(len(updates)),
        "subspace_energy_fraction": float((kept_energy / total_energy).item()),
        "subspace_singular_value_0": float(singular_values[0].item()) if singular_values.numel() else 0.0,
        "subspace_update_fro": float(torch.linalg.vector_norm(final_flat).item()),
        "subspace_mean_input_fro": float(torch.linalg.vector_norm(flat, dim=1).mean().item()),
    }
    return final_flat.reshape_as(updates[0]).contiguous(), stats


def anchored_subspace_update(
    anchor_update: torch.Tensor,
    support_updates: list[torch.Tensor],
    *,
    rank: int,
    residual_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project the anchor write into a multi-direction support subspace."""

    if not support_updates:
        raise ValueError("anchored_subspace_update requires at least one support update")
    support_flat = torch.stack([update.flatten().float() for update in support_updates], dim=0)
    rank = max(1, min(rank, support_flat.shape[0], support_flat.shape[1]))
    _u, singular_values, vh = torch.linalg.svd(support_flat, full_matrices=False)
    basis = vh[:rank]
    anchor_flat = anchor_update.flatten().float()
    anchor_projection_flat = anchor_flat @ basis.T @ basis
    anchor_residual_flat = anchor_flat - anchor_projection_flat
    final_flat = anchor_projection_flat + residual_scale * anchor_residual_flat
    anchor_norm = torch.linalg.vector_norm(anchor_flat).clamp_min(1e-8)
    projection_norm = torch.linalg.vector_norm(anchor_projection_flat)
    residual_norm = torch.linalg.vector_norm(anchor_residual_flat)
    stats = {
        "subspace_rank": float(rank),
        "subspace_supports": float(len(support_updates)),
        "subspace_singular_value_0": float(singular_values[0].item()) if singular_values.numel() else 0.0,
        "anchor_residual_scale": float(residual_scale),
        "anchor_update_fro": float(anchor_norm.item()),
        "anchor_projection_fro": float(projection_norm.item()),
        "anchor_residual_fro": float(residual_norm.item()),
        "anchor_subspace_energy_fraction": float((projection_norm.square() / anchor_norm.square()).item()),
        "subspace_mean_support_fro": float(torch.linalg.vector_norm(support_flat, dim=1).mean().item()),
    }
    return final_flat.reshape_as(anchor_update).contiguous(), stats


def proposal_alignment_stats(updates: list[torch.Tensor]) -> dict[str, float]:
    if len(updates) < 2:
        return {}
    flat = torch.stack([update.flatten() for update in updates], dim=0)
    norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1e-8)
    unit = flat / norms[:, None]
    cosine = torch.matmul(unit, unit.T)
    offdiag = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool)]
    return {
        "proposal_cosine_mean": float(offdiag.mean().item()),
        "proposal_cosine_min": float(offdiag.min().item()),
        "proposal_cosine_max": float(offdiag.max().item()),
        "proposal_update_fro_mean": float(norms.mean().item()),
        "proposal_update_fro_min": float(norms.min().item()),
        "proposal_update_fro_max": float(norms.max().item()),
    }


def update_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.flatten().float()
    right_flat = right.flatten().float()
    denom = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(left_flat, right_flat) / denom).item())


def project_update_onto_reference(
    update: torch.Tensor,
    reference: torch.Tensor,
    *,
    positive_only: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    update_flat = update.flatten().float()
    reference_flat = reference.flatten().float()
    reference_norm = torch.linalg.vector_norm(reference_flat).clamp_min(1e-8)
    reference_unit = reference_flat / reference_norm
    projection = torch.dot(update_flat, reference_unit)
    if positive_only and float(projection.item()) <= 0:
        projected_flat = torch.zeros_like(update_flat)
    else:
        projected_flat = projection * reference_unit
    projected = projected_flat.reshape_as(update).contiguous()
    return projected, {
        "projection_reference_fro": float(reference_norm.item()),
        "projection_component_fro": float(torch.linalg.vector_norm(projected_flat).item()),
        "projection_input_fro": float(torch.linalg.vector_norm(update_flat).item()),
        "projection_signed_fro": float(projection.item()),
    }


def perspective_filtered_update(
    direct_update: torch.Tensor,
    support_updates: list[torch.Tensor],
    *,
    mode: str,
    residual_scale: float,
    min_agreement: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use extra perspectives as a filter over the deployment-shaped update.

    The direct update is the only update whose keys and target posture match
    deployment. Perspective updates are treated as witnesses: if they point in
    a compatible direction, we keep that component of the direct write; if not,
    we shrink it rather than averaging in another noisy target.
    """

    direct_norm = torch.linalg.vector_norm(direct_update.flatten().float()).clamp_min(1e-8)
    if not support_updates:
        return direct_update, {
            "perspective_filter_supports": 0,
            "perspective_filter_mode": 0.0,
            "perspective_filter_residual_scale": float(residual_scale),
            "perspective_filter_direct_fro": float(direct_norm.item()),
            "perspective_filter_final_fro": float(direct_norm.item()),
        }

    support_update, stats = directional_consensus_update(
        support_updates,
        min_agreement=min_agreement,
    )
    support_flat = support_update.flatten().float()
    support_norm = torch.linalg.vector_norm(support_flat)
    direct_flat = direct_update.flatten().float()
    final_flat: torch.Tensor
    projection = torch.tensor(0.0)
    cosine = torch.tensor(0.0)
    if float(support_norm.item()) <= 1e-8:
        final_flat = residual_scale * direct_flat
    else:
        support_unit = support_flat / support_norm.clamp_min(1e-8)
        projection = torch.dot(direct_flat, support_unit)
        cosine = projection / direct_norm
        if mode == "project":
            aligned_flat = projection.clamp_min(0) * support_unit
            residual_flat = direct_flat - aligned_flat
            final_flat = aligned_flat + residual_scale * residual_flat
        elif mode == "cosine_scale":
            scale = residual_scale + (1.0 - residual_scale) * cosine.clamp(0, 1)
            final_flat = scale * direct_flat
        else:
            raise ValueError(f"Unknown perspective filter mode: {mode}")
    final = final_flat.reshape_as(direct_update).contiguous()
    final_norm = torch.linalg.vector_norm(final_flat)
    stats.update(
        {
            "perspective_filter_supports": float(len(support_updates)),
            "perspective_filter_mode_project": float(mode == "project"),
            "perspective_filter_mode_cosine_scale": float(mode == "cosine_scale"),
            "perspective_filter_residual_scale": float(residual_scale),
            "perspective_filter_min_agreement": float(min_agreement),
            "perspective_filter_direct_fro": float(direct_norm.item()),
            "perspective_filter_support_fro": float(support_norm.item()),
            "perspective_filter_projection_fro": float(projection.item()),
            "perspective_filter_cosine": float(cosine.item()),
            "perspective_filter_final_fro": float(final_norm.item()),
        }
    )
    return final, stats


def perspective_filtered_targets(
    direct_targets: torch.Tensor,
    support_targets: list[torch.Tensor],
    *,
    threshold: float,
    temperature: float,
    floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gate trace rows by target agreement across teacher perspectives."""

    if not support_targets:
        return direct_targets, {
            "perspective_target_filter_supports": 0,
            "perspective_target_filter_mean_gate": 1.0,
        }
    for target in support_targets:
        if target.shape != direct_targets.shape:
            raise ValueError(
                f"Perspective target shape {tuple(target.shape)} != direct target shape {tuple(direct_targets.shape)}"
            )
    direct = direct_targets.float()
    support = torch.stack([target.float() for target in support_targets], dim=0)
    direct_unit = torch.nn.functional.normalize(direct, dim=1)
    support_unit = torch.nn.functional.normalize(support, dim=2)
    cosine = (support_unit * direct_unit.unsqueeze(0)).sum(dim=2)
    agreement = cosine.mean(dim=0)
    gate = floor + (1.0 - floor) * torch.sigmoid((agreement - threshold) * temperature)
    filtered = direct * gate[:, None]
    return filtered.contiguous(), {
        "perspective_target_filter_supports": float(len(support_targets)),
        "perspective_target_filter_threshold": float(threshold),
        "perspective_target_filter_temperature": float(temperature),
        "perspective_target_filter_floor": float(floor),
        "perspective_target_filter_mean_cosine": float(agreement.mean().item()),
        "perspective_target_filter_min_cosine": float(agreement.min().item()),
        "perspective_target_filter_max_cosine": float(agreement.max().item()),
        "perspective_target_filter_mean_gate": float(gate.mean().item()),
        "perspective_target_filter_min_gate": float(gate.min().item()),
        "perspective_target_filter_max_gate": float(gate.max().item()),
        "perspective_target_filter_direct_fro": float(torch.linalg.vector_norm(direct).item()),
        "perspective_target_filter_final_fro": float(torch.linalg.vector_norm(filtered).item()),
    }


def suppress_update_on_keys(
    update: torch.Tensor,
    keys: torch.Tensor | None,
    *,
    strength: float,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Subtract the component of an update that fires on protected keys."""

    if keys is None or keys.numel() == 0 or strength <= 0:
        return update, {
            "option_projection_strength": float(strength),
            "option_projection_rows": 0,
            "option_projection_removed_fro": 0.0,
        }
    b = keys.detach().float()
    update_f = update.float()
    system = b @ b.T + ridge * torch.eye(b.shape[0], dtype=b.dtype)
    removable = update_f @ b.T @ torch.linalg.pinv(system) @ b
    adjusted = update_f - strength * removable
    return adjusted.contiguous(), {
        "option_projection_strength": float(strength),
        "option_projection_rows": int(b.shape[0]),
        "option_projection_removed_fro": float(torch.linalg.vector_norm(strength * removable).item()),
    }


def normalized_row_weights(
    score: torch.Tensor,
    *,
    floor: float,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    score_f = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if score_f.numel() == 0:
        return score_f
    if score_f.numel() > 1:
        z = (score_f - score_f.mean()) / score_f.std(unbiased=False).clamp_min(1e-6)
    else:
        z = score_f * 0
    weights = torch.sigmoid(temperature * z).clamp_min(floor)
    if top_k > 0 and top_k < weights.numel():
        keep = torch.topk(score_f, k=top_k).indices
        masked = torch.full_like(weights, floor)
        masked[keep] = weights[keep]
        weights = masked
    return weights / weights.mean().clamp_min(1e-6)


def activation_energy_weights(
    mode: str,
    *,
    full_inputs: torch.Tensor,
    full_outputs: torch.Tensor,
    current_inputs: torch.Tensor,
    current_outputs: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor | None,
    floor: float,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor | None, dict[str, float | str]]:
    if mode == "none":
        return None, {}
    scores: list[torch.Tensor] = []
    names: list[str] = []
    if mode in {"block_action", "combined"}:
        scores.append(torch.linalg.vector_norm((full_outputs - full_inputs).float(), dim=1))
        names.append("block_action")
    if mode in {"source_norm", "combined"}:
        scores.append(torch.linalg.vector_norm(targets.float(), dim=1))
        names.append("source_norm")
    if mode in {"action_excess", "combined"}:
        full_action = torch.linalg.vector_norm((full_outputs - full_inputs).float(), dim=1)
        current_action = torch.linalg.vector_norm((current_outputs - current_inputs).float(), dim=1)
        scores.append(full_action - current_action)
        names.append("action_excess")
    if mode in {"mlp_key", "combined"}:
        if keys is None:
            return None, {"energy_skipped_no_keys": 1.0}
        scores.append(torch.linalg.vector_norm(keys.float(), dim=1))
        names.append("mlp_key")
    if not scores:
        raise ValueError(f"Unknown activation energy weighting mode: {mode}")
    standardized = []
    for score in scores:
        standardized.append((score - score.mean()) / score.std(unbiased=False).clamp_min(1e-6))
    combined = torch.stack(standardized, dim=0).mean(dim=0)
    weights = normalized_row_weights(
        combined,
        floor=floor,
        temperature=temperature,
        top_k=top_k,
    )
    stats: dict[str, float | str] = {
        "energy_components": "+".join(names),
        "energy_weight_mean": float(weights.mean().item()),
        "energy_weight_min": float(weights.min().item()),
        "energy_weight_max": float(weights.max().item()),
        "energy_score_mean": float(combined.mean().item()),
        "energy_score_std": float(combined.std(unbiased=False).item()),
    }
    return weights.to(targets.device), stats


def key_separation_stats(keys: torch.Tensor, negative_keys: torch.Tensor | None) -> dict[str, float]:
    if negative_keys is None or negative_keys.numel() == 0:
        return {
            "key_sep_negative_rows": 0.0,
            "key_sep_neg_to_pos_meanmax": 0.0,
            "key_sep_neg_to_pos_max": 0.0,
        }
    pos = torch.nn.functional.normalize(keys.detach().float(), dim=1)
    neg = torch.nn.functional.normalize(negative_keys.detach().float(), dim=1)
    sim = neg @ pos.T
    neg_max = sim.max(dim=1).values
    return {
        "key_sep_negative_rows": float(negative_keys.shape[0]),
        "key_sep_neg_to_pos_meanmax": float(neg_max.mean().item()),
        "key_sep_neg_to_pos_max": float(neg_max.max().item()),
    }


def key_separation_scale(
    stats: dict[str, float],
    *,
    threshold: float,
    temperature: float,
    floor: float,
) -> float:
    meanmax = stats["key_sep_neg_to_pos_meanmax"]
    scale = floor + (1.0 - floor) / (1.0 + torch.exp(torch.tensor((meanmax - threshold) * temperature))).item()
    return float(scale)


def configure_object_gate(
    wrapper,
    keys: torch.Tensor,
    args: argparse.Namespace,
    *,
    negative_groups: dict[str, torch.Tensor] | None = None,
    calibration_keys: torch.Tensor | None = None,
) -> dict[str, float]:
    if args.object_gate_mode == "cosine":
        wrapper.set_object_gate_keys_(
            keys,
            threshold=args.object_gate_threshold,
            temperature=args.object_gate_temperature,
            floor=args.object_gate_floor,
            append=True,
        )
        return {}
    if negative_groups is None:
        raise ValueError("density_ratio object gate requires negative_groups")
    params, stats = fit_contrastive_density_gate(
        keys,
        negative_groups,
        calibration_keys,
        rank_q=args.object_gate_density_rank_q,
        rank_k=args.object_gate_density_rank_k,
        beta=args.object_gate_density_beta,
        shrink=args.object_gate_density_shrink,
        gaussian_ridge=args.object_gate_density_ridge,
        target_neg_fpr=args.object_gate_density_target_neg_fpr,
        kappa=args.object_gate_density_kappa,
        pool_top_k=args.object_gate_density_pool_top_k,
    )
    wrapper.add_density_object_gate_(params, append=True)
    wrapper.object_gate_floor = float(args.object_gate_floor)
    return stats


def density_gate_group_stats(
    params: DensityRatioGateParams,
    keys: torch.Tensor,
    *,
    group: str,
    floor: float,
) -> dict[str, float]:
    if keys.numel() == 0:
        return {
            f"density_diag_{group}_rows": 0.0,
        }
    scores = score_tokens(keys.detach().float(), params)
    raw_gate = torch.sigmoid((scores - params.tau) / max(params.temperature, 1e-6))
    if floor >= 1.0:
        gate = torch.ones_like(raw_gate)
    elif floor > 0.0:
        gate = floor + (1.0 - floor) * raw_gate
    else:
        gate = raw_gate
    quantiles = torch.quantile(scores, torch.tensor([0.05, 0.50, 0.95], device=scores.device))
    gate_quantiles = torch.quantile(gate, torch.tensor([0.05, 0.50, 0.95], device=gate.device))
    return {
        f"density_diag_{group}_rows": float(keys.shape[0]),
        f"density_diag_{group}_score_mean": float(scores.mean().item()),
        f"density_diag_{group}_score_q05": float(quantiles[0].item()),
        f"density_diag_{group}_score_q50": float(quantiles[1].item()),
        f"density_diag_{group}_score_q95": float(quantiles[2].item()),
        f"density_diag_{group}_gate_mean": float(gate.mean().item()),
        f"density_diag_{group}_gate_q05": float(gate_quantiles[0].item()),
        f"density_diag_{group}_gate_q50": float(gate_quantiles[1].item()),
        f"density_diag_{group}_gate_q95": float(gate_quantiles[2].item()),
    }


def density_gate_value(keys: torch.Tensor, params: DensityRatioGateParams, *, floor: float) -> float:
    raw = sequence_gate(keys.detach().float(), params).reshape(()).item()
    if floor >= 1.0:
        return 1.0
    if floor > 0.0:
        return float(floor + (1.0 - floor) * raw)
    return float(raw)


def _prompt_gate_summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_count": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.05, 0.50, 0.95]))
    return {
        f"{prefix}_count": float(tensor.numel()),
        f"{prefix}_mean": float(tensor.mean().item()),
        f"{prefix}_q05": float(quantiles[0].item()),
        f"{prefix}_q50": float(quantiles[1].item()),
        f"{prefix}_q95": float(quantiles[2].item()),
        f"{prefix}_active_050": float((tensor >= 0.50).float().mean().item()),
        f"{prefix}_active_090": float((tensor >= 0.90).float().mean().item()),
    }


@torch.no_grad()
def install_prompt_object_gate_router(
    model,
    tokenizer,
    wrappers: dict[int, object],
    eval_questions: list[TranslationQuestion],
    sentinels: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Cache prompt-level object gates from content/source activations.

    The prompt string is only used as an eval-harness cache key. The stored
    values are computed from activation density-ratio scores at source/content
    token positions, then routed into the memory wrapper so readout tokens do
    not have to re-prove object presence.
    """

    if not wrappers:
        return {"prompt_object_router_enabled": 0.0}
    gated_wrappers = {
        layer_idx: wrapper
        for layer_idx, wrapper in wrappers.items()
        if getattr(wrapper, "object_density_gates", None)
    }
    if not gated_wrappers:
        return {"prompt_object_router_enabled": 0.0}

    old_router = getattr(model, "_caic_object_gate_router", None)
    if old_router is not None:
        delattr(model, "_caic_object_gate_router")
    clear_active_slot_weights(model)
    try:
        eval_prompts, eval_indices = object_gate_prompts_and_indices(
            tokenizer,
            eval_questions,
            args.chat_template,
            max_prompts=0,
            seed=args.seed + 137_001,
        )
        sentinel_prompts = [
            format_model_prompt(tokenizer, row["prompt"], args.chat_template)
            for row in sentinels
        ]
        sentinel_indices = [
            max(0, len(encode_no_special(tokenizer, prompt)) - 1)
            for prompt in sentinel_prompts
        ]
        all_prompts = [*eval_prompts, *sentinel_prompts]
        all_indices = [*eval_indices, *sentinel_indices]

        wrapper_scores: dict[int, dict[str, float]] = {
            id(wrapper): {} for wrapper in gated_wrappers.values()
        }
        stats: dict[str, float] = {"prompt_object_router_enabled": 1.0}
        for layer_idx, wrapper in gated_wrappers.items():
            eval_values: list[float] = []
            sentinel_values: list[float] = []
            if args.object_gate_prompt_router_mode == "translation_oracle":
                for prompt in eval_prompts:
                    wrapper_scores[id(wrapper)][prompt] = 1.0
                    eval_values.append(1.0)
                for prompt in sentinel_prompts:
                    wrapper_scores[id(wrapper)][prompt] = 0.0
                    sentinel_values.append(0.0)
            else:
                prompts_to_score = all_prompts
                indices_to_score = all_indices
                if args.object_gate_prompt_router_mode == "translation_source":
                    prompts_to_score = eval_prompts
                    indices_to_score = eval_indices
                    for prompt in sentinel_prompts:
                        wrapper_scores[id(wrapper)][prompt] = 0.0
                        sentinel_values.append(0.0)
                for prompt_idx, (prompt, token_index) in enumerate(zip(prompts_to_score, indices_to_score, strict=True)):
                    captures = capture_layer_io_at_token_indices(
                        model,
                        tokenizer,
                        [prompt],
                        [token_index],
                        [layer_idx],
                        device,
                        args.max_length,
                        capture_window=args.object_gate_token_window,
                    )
                    keys = captures[layer_idx].keys
                    value = max(
                        density_gate_value(keys, params, floor=args.object_gate_floor)
                        for params in wrapper.object_density_gates
                    )
                    wrapper_scores[id(wrapper)][prompt] = value
                    if prompt_idx < len(eval_prompts):
                        eval_values.append(value)
                    else:
                        sentinel_values.append(value)
            stats.update(_prompt_gate_summary(eval_values, f"prompt_router_l{layer_idx}_eval"))
            stats.update(_prompt_gate_summary(sentinel_values, f"prompt_router_l{layer_idx}_sentinel"))

        def router(prompts: list[str]) -> dict[int, torch.Tensor]:
            routed: dict[int, torch.Tensor] = {}
            for wrapper_id, scores in wrapper_scores.items():
                routed[wrapper_id] = torch.tensor(
                    [scores.get(prompt, 0.0) for prompt in prompts],
                    dtype=torch.float32,
                )
            return routed

        model._caic_object_gate_router = router  # noqa: SLF001
        return stats
    except Exception:
        if old_router is not None:
            model._caic_object_gate_router = old_router  # noqa: SLF001
        raise


def enabled_attention_projections(args: argparse.Namespace) -> list[str]:
    projections: list[str] = []
    if args.write_attention_q or args.write_attention_qk:
        projections.append("q")
    if args.write_attention_k or args.write_attention_qk:
        projections.append("k")
    if args.write_attention_v:
        projections.append("v")
    return projections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="runs/minilang_write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-seed-offset", type=int, default=0)
    parser.add_argument("--lessons", type=int, default=20)
    parser.add_argument("--lesson-examples", type=int, default=5)
    parser.add_argument("--freeze-language-after", type=int, default=None)
    parser.add_argument("--trace-probes", type=int, default=16)
    parser.add_argument("--balanced-trace", action="store_true")
    parser.add_argument(
        "--trace-perspectives",
        nargs="+",
        choices=sorted(PERSPECTIVE_INSTRUCTIONS),
        default=["direct"],
    )
    parser.add_argument("--teacher-perspectives-only", action="store_true")
    parser.add_argument("--perspective-filter", action="store_true")
    parser.add_argument("--perspective-filter-granularity", choices=["update", "target"], default="update")
    parser.add_argument("--perspective-filter-mode", choices=["project", "cosine_scale"], default="project")
    parser.add_argument("--perspective-filter-residual-scale", type=float, default=0.25)
    parser.add_argument("--perspective-filter-min-agreement", type=float, default=0.0)
    parser.add_argument("--perspective-target-filter-threshold", type=float, default=0.25)
    parser.add_argument("--perspective-target-filter-temperature", type=float, default=8.0)
    parser.add_argument("--perspective-target-filter-floor", type=float, default=0.0)
    parser.add_argument("--eval-questions", type=int, default=40)
    parser.add_argument("--eval-questions-jsonl", default="")
    parser.add_argument("--eval-mode", choices=["random", "unique_random", "exhaustive_modified"], default="random")
    parser.add_argument("--eval-max-attempts", type=int, default=10_000)
    parser.add_argument("--exclude-eval-lesson-overlaps", action="store_true")
    parser.add_argument("--exclude-eval-trace-overlaps", action="store_true")
    parser.add_argument("--layers", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--trace-last-tokens", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--sentinel-negative-keys", action="store_true")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=50.0)
    parser.add_argument("--memory-gate", action="store_true")
    parser.add_argument("--memory-gate-final-token-only", action="store_true")
    parser.add_argument("--memory-gate-threshold", type=float, default=0.95)
    parser.add_argument("--memory-gate-temperature", type=float, default=80.0)
    parser.add_argument("--activation-object-gate", action="store_true")
    parser.add_argument("--object-gate-mode", choices=["cosine", "density_ratio"], default="cosine")
    parser.add_argument("--object-gate-probes", type=int, default=0)
    parser.add_argument("--object-gate-token-window", type=int, default=8)
    parser.add_argument("--object-gate-threshold", type=float, default=0.90)
    parser.add_argument("--object-gate-temperature", type=float, default=40.0)
    parser.add_argument("--object-gate-floor", type=float, default=0.0)
    parser.add_argument("--object-gate-max-prompts", type=int, default=0)
    parser.add_argument("--object-gate-rival-negatives", type=int, default=0)
    parser.add_argument("--object-gate-diagnostics", action="store_true")
    parser.add_argument("--object-gate-prompt-router", action="store_true")
    parser.add_argument(
        "--object-gate-prompt-router-mode",
        choices=["density", "translation_source", "translation_oracle"],
        default="density",
    )
    parser.add_argument("--object-gate-density-rank-q", type=int, default=64)
    parser.add_argument("--object-gate-density-rank-k", type=int, default=8)
    parser.add_argument("--object-gate-density-beta", type=float, default=1e-3)
    parser.add_argument("--object-gate-density-shrink", type=float, default=0.2)
    parser.add_argument("--object-gate-density-ridge", type=float, default=1e-3)
    parser.add_argument("--object-gate-density-target-neg-fpr", type=float, default=1e-3)
    parser.add_argument("--object-gate-density-kappa", type=float, default=1.0)
    parser.add_argument("--object-gate-density-pool-top-k", type=int, default=8)
    parser.add_argument("--term-slot-gate", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--teacher-forcing-trace", action="store_true")
    parser.add_argument("--token-teacher-forcing-trace", action="store_true")
    parser.add_argument("--packed-use-trace", action="store_true")
    parser.add_argument("--packed-use-mode", choices=["clean", "curriculum"], default="curriculum")
    parser.add_argument("--packed-use-span-items", type=int, default=0)
    parser.add_argument("--trace-context", choices=["lesson", "cumulative", "full"], default="lesson")
    parser.add_argument("--write-only-final", action="store_true")
    parser.add_argument("--shuffle-targets", action="store_true")
    parser.add_argument("--option-negative-keys", action="store_true")
    parser.add_argument("--option-negative-mode", choices=["ridge", "project"], default="ridge")
    parser.add_argument("--option-negative-attention", action="store_true")
    parser.add_argument("--option-negative-project-strength", type=float, default=0.5)
    parser.add_argument("--option-negative-project-ridge", type=float, default=1.0)
    parser.add_argument("--max-option-negative-prompts", type=int, default=64)
    parser.add_argument("--target-mode", choices=["output_delta", "source"], default="source")
    parser.add_argument(
        "--activation-energy-weighting",
        choices=["none", "block_action", "source_norm", "action_excess", "mlp_key", "combined"],
        default="none",
    )
    parser.add_argument("--activation-energy-weight-floor", type=float, default=0.05)
    parser.add_argument("--activation-energy-weight-temperature", type=float, default=2.0)
    parser.add_argument("--activation-energy-top-k", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-write", action="store_true")
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
            "schur_transport_actuator",
            "feature_birth",
            "logit_error",
        ],
        default="mlp_contribution",
    )
    parser.add_argument("--intrinsic-surprise-token-mode", choices=["last", "top", "all", "final_aligned"], default="last")
    parser.add_argument("--intrinsic-surprise-top-tokens", type=int, default=16)
    parser.add_argument("--intrinsic-surprise-feature-top-k", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-target-feature-top-k", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-key-feature-top-k", type=int, default=8)
    parser.add_argument("--intrinsic-surprise-value-feature-top-k", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-pair-top-k", type=int, default=16)
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
    parser.add_argument("--star-object-summary-gain", type=float, default=0.5)
    parser.add_argument("--star-future-layer-horizon", type=int, default=4)
    parser.add_argument("--star-future-token-top-k", type=int, default=8)
    parser.add_argument("--star-future-layer-decay", type=float, default=2.0)
    parser.add_argument("--star-future-token-decay", type=float, default=64.0)
    parser.add_argument("--star-future-relation-power", type=float, default=1.0)
    parser.add_argument("--star-ordinary-key-rank", type=int, default=32)
    parser.add_argument("--star-value-projection-features", type=int, default=128)
    parser.add_argument("--star-value-projection-ridge", type=float, default=1e-2)
    parser.add_argument("--star-schur-ridge", type=float, default=1e-3)
    parser.add_argument("--star-map-ridge", type=float, default=1e-3)
    parser.add_argument("--star-posture-negative-scale", type=float, default=1.0)
    parser.add_argument("--star-min-coherence", type=float, default=0.0)
    parser.add_argument("--star-shuffle-future-targets", action="store_true")
    parser.add_argument("--star-shuffle-keys", action="store_true")
    parser.add_argument("--intrinsic-surprise-bidirectional-pairs", action="store_true")
    parser.add_argument(
        "--intrinsic-surprise-relation-value-mode",
        choices=["residual", "full", "context"],
        default="residual",
    )
    parser.add_argument(
        "--intrinsic-surprise-value-source",
        choices=["base", "effective"],
        default="base",
    )
    parser.add_argument(
        "--intrinsic-surprise-effective-target-norm",
        choices=["raw", "base"],
        default="raw",
    )
    parser.add_argument("--intrinsic-surprise-target-scale", type=float, default=1.0)
    parser.add_argument("--intrinsic-surprise-target-row-norm-cap", type=float, default=0.0)
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
    parser.add_argument("--intrinsic-surprise-center-targets", action="store_true")
    parser.add_argument("--intrinsic-surprise-weight-mode", choices=["linear", "exponential"], default="linear")
    parser.add_argument("--intrinsic-surprise-exp-temperature", type=float, default=1.0)
    parser.add_argument("--intrinsic-surprise-exp-cap", type=float, default=100.0)
    parser.add_argument("--intrinsic-surprise-prediction-ridge", type=float, default=1.0)
    parser.add_argument("--intrinsic-span-readout-bridge", action="store_true")
    parser.add_argument("--intrinsic-span-readout-scale", type=float, default=1.0)
    parser.add_argument("--intrinsic-span-readout-max-items", type=int, default=32)
    parser.add_argument("--intrinsic-surprise-persistence-power", type=float, default=0.0)
    parser.add_argument("--intrinsic-surprise-persistence-threshold", type=float, default=0.25)
    parser.add_argument("--intrinsic-surprise-persistence-min-tokens", type=int, default=2)
    parser.add_argument("--intrinsic-surprise-generic-rank", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-lm-head-generic-rank", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-output-penalty-rank", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-output-penalty-weight", type=float, default=0.0)
    parser.add_argument("--intrinsic-surprise-input-penalty-features", type=int, default=0)
    parser.add_argument("--intrinsic-surprise-input-penalty-weight", type=float, default=0.0)
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
    parser.add_argument("--key-separation-filter", action="store_true")
    parser.add_argument("--key-separation-threshold", type=float, default=0.9)
    parser.add_argument("--key-separation-temperature", type=float, default=40.0)
    parser.add_argument("--key-separation-floor", type=float, default=0.0)
    parser.add_argument("--write-attention-o", action="store_true")
    parser.add_argument("--write-attention-q", action="store_true")
    parser.add_argument("--write-attention-k", action="store_true")
    parser.add_argument("--write-attention-qk", action="store_true")
    parser.add_argument("--write-attention-v", action="store_true")
    parser.add_argument("--attention-projection-eta-scale", type=float, default=1.0)
    parser.add_argument("--attention-projection-max-update-norm", type=float, default=None)
    parser.add_argument("--no-write-mlp", dest="write_mlp", action="store_false", default=True)
    parser.add_argument("--cache-current-captures", action="store_true")
    parser.add_argument("--ensemble-corpora", type=int, default=1)
    parser.add_argument(
        "--ensemble-reduction",
        choices=[
            "mean",
            "sum",
            "snr",
            "directional",
            "anchored_directional",
            "subspace",
            "anchored_subspace",
        ],
        default="mean",
    )
    parser.add_argument("--ensemble-per-lesson", action="store_true")
    parser.add_argument("--ensemble-include-anchor", action="store_true")
    parser.add_argument("--ensemble-anchor-residual-scale", type=float, default=0.25)
    parser.add_argument("--ensemble-seed-stride", type=int, default=100_000)
    parser.add_argument("--ensemble-shared-probes", action="store_true")
    parser.add_argument("--ensemble-snr-threshold", type=float, default=1.0)
    parser.add_argument("--ensemble-snr-temperature", type=float, default=4.0)
    parser.add_argument("--ensemble-directional-min-agreement", type=float, default=0.0)
    parser.add_argument("--ensemble-subspace-rank", type=int, default=2)
    parser.add_argument("--dice-defer-apply", action="store_true")
    parser.add_argument("--dice-support-threshold", type=float, default=0.75)
    parser.add_argument("--dice-support-temperature", type=float, default=16.0)
    parser.add_argument("--dice-support-strength", type=float, default=1.0)
    parser.add_argument("--dice-support-cap", type=float, default=2.0)
    parser.add_argument("--dice-support-floor", type=float, default=0.0)
    parser.add_argument("--lexical-channel", action="store_true")
    parser.add_argument("--lexical-channel-attention", action="store_true")
    parser.add_argument("--lexical-channel-scale", type=float, default=0.25)
    parser.add_argument("--lexical-channel-max-items", type=int, default=0)
    parser.add_argument("--context-span-channel", action="store_true")
    parser.add_argument("--context-span-channel-attention", action="store_true")
    parser.add_argument("--context-span-channel-mode", choices=["raw", "aligned"], default="raw")
    parser.add_argument("--context-span-channel-scale", type=float, default=0.1)
    parser.add_argument("--context-span-channel-max-items", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--sentinel-eval", action="store_true")
    parser.add_argument("--sentinel-suite", choices=["core", "expanded"], default="core")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="", choices=["", "eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--chat-template", dest="chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    return parser.parse_args()


def guard_prompts(tokenizer, use_chat_template: bool) -> list[str]:
    prompts = [
        "Choose the correct answer. Answer only with A, B, C, or D.\n\nWhich option names a color?\nA. Tuesday\nB. Blue\nC. Quickly\nD. Table\n\nAnswer:",
        "Choose the correct answer. Answer only with A, B, C, or D.\n\nWhich option is a number?\nA. Seven\nB. Cloud\nC. Window\nD. Softly\n\nAnswer:",
        "Choose the correct answer. Answer only with A, B, C, or D.\n\nWhich option is an animal?\nA. Pencil\nB. River\nC. Horse\nD. Yesterday\n\nAnswer:",
        "Choose the correct answer. Answer only with A, B, C, or D.\n\nWhich option is a verb?\nA. Run\nB. Marble\nC. Green\nD. North\n\nAnswer:",
    ]
    return [format_model_prompt(tokenizer, prompt, use_chat_template) for prompt in prompts]


def sentinel_guard_prompts(tokenizer, use_chat_template: bool) -> list[str]:
    return [
        format_model_prompt(tokenizer, row["prompt"], use_chat_template)
        for row in sentinel_questions("core")
    ]


def format_intrinsic_lesson_prompt(tokenizer, lesson_text: str, args: argparse.Namespace) -> str:
    if args.intrinsic_surprise_lesson_format == "raw":
        return lesson_text
    if not args.chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return lesson_text
    messages = [{"role": "user", "content": lesson_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


@torch.no_grad()
def capture_intrinsic_lesson_forward(
    model,
    tokenizer,
    prompt: str,
    layer_indices: list[int],
    device: torch.device,
    max_length: int,
    capture_attentions: bool = False,
) -> tuple[dict[int, object], torch.Tensor, torch.Tensor]:
    layers = get_decoder_layers(model)
    resolved_layers = [idx if idx >= 0 else len(layers) + idx for idx in layer_indices]
    stores: dict[int, dict[str, list[torch.Tensor]]] = {
        idx: {"keys": [], "outputs": [], "mlp_inputs": []} for idx in resolved_layers
    }

    def make_down_hook(layer_idx: int):
        def hook(_module, module_inputs: tuple[torch.Tensor, ...], module_output: torch.Tensor):
            key = module_inputs[0].detach().float().cpu()
            out = module_output.detach().float().cpu()
            stores[layer_idx]["keys"].append(key.reshape(-1, key.shape[-1]))
            stores[layer_idx]["outputs"].append(out.reshape(-1, out.shape[-1]))

        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(_module, module_inputs: tuple[torch.Tensor, ...], _module_output: torch.Tensor):
            mlp_input = module_inputs[0].detach().float().cpu()
            stores[layer_idx]["mlp_inputs"].append(mlp_input.reshape(-1, mlp_input.shape[-1]))

        return hook

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)
    trimmed_ids = full_ids[-max_length:]
    input_ids = torch.tensor([trimmed_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    for idx in resolved_layers:
        handles.append(getattr(layers[idx], "mlp").register_forward_hook(make_mlp_hook(idx)))
        handles.append(get_mlp_down_module(layers[idx]).register_forward_hook(make_down_hook(idx)))
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=capture_attentions,
        )
    finally:
        for handle in handles:
            handle.remove()
        clear_active_slot_weights(model)
    attentions_by_layer: dict[int, torch.Tensor] = {}
    output_attentions = getattr(outputs, "attentions", None)
    if capture_attentions and output_attentions is not None:
        for idx in resolved_layers:
            if idx < len(output_attentions) and output_attentions[idx] is not None:
                attn = output_attentions[idx]
                if attn.ndim == 4:
                    attn = attn[0]
                attentions_by_layer[idx] = attn.detach().float().cpu()
    captures = {}
    for idx, store in stores.items():
        captures[idx] = IntrinsicLayerCapture(
            keys=torch.cat(store["keys"], dim=0),
            outputs=torch.cat(store["outputs"], dim=0),
            mlp_inputs=torch.cat(store["mlp_inputs"], dim=0),
            attentions=attentions_by_layer.get(idx),
        )
    return captures, input_ids[0].detach().cpu(), outputs.logits[0].detach().float().cpu()


def lesson_logit_error_targets(model, input_ids: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.numel() < 2:
        return torch.empty(0, logits.shape[-1]), torch.empty(0)
    labels = input_ids[1:].long()
    pred_logits = logits[:-1].float()
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    probs = torch.softmax(pred_logits, dim=-1)
    lm_weight = model.lm_head.weight.detach().float().cpu()
    expected = probs.cpu() @ lm_weight
    target = lm_weight[labels] - expected
    losses = -log_probs[torch.arange(labels.shape[0]), labels.cpu()].cpu()
    return target.contiguous(), losses.contiguous()


def prompt_offsets_for_input_ids(
    tokenizer,
    prompt: str,
    input_ids: torch.Tensor,
) -> list[tuple[int, int]] | None:
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    offsets = encoded.get("offset_mapping")
    token_ids = encoded.get("input_ids")
    if offsets is None or token_ids is None:
        return None
    keep = int(input_ids.numel())
    if keep <= 0 or len(offsets) < keep:
        return None
    return [(int(start), int(end)) for start, end in offsets[-keep:]]


def token_index_before_char(offsets: list[tuple[int, int]], char_pos: int) -> int | None:
    best_idx = None
    best_end = -1
    for idx, (start, end) in enumerate(offsets):
        if start < char_pos <= end:
            return idx
        if end <= char_pos and end >= best_end:
            best_idx = idx
            best_end = end
    return best_idx


def intrinsic_span_readout_selection(
    model,
    tokenizer,
    prompt: str,
    lesson_text: str,
    input_ids: torch.Tensor,
    keys: torch.Tensor,
    *,
    seed: int,
    max_items: int,
    target_scale: float,
) -> IntrinsicSurpriseSelection | None:
    """Build a cheap lesson-derived readout bridge from surface spans.

    This uses only the lesson text and the same forward-pass keys. Structured
    spans such as ``x=y`` and ``x -> y`` provide source/prefix positions. The
    target is the mean LM-head vector of the answer text, giving the safe
    relational write a small path into existing answer/readout machinery without
    future questions or next-token loss.
    """

    offsets = prompt_offsets_for_input_ids(tokenizer, prompt, input_ids)
    if offsets is None:
        return None
    traces = context_span_traces(lesson_text, seed=seed, max_items=max_items)
    if not traces:
        return None
    lm_head = getattr(model, "lm_head", None)
    lm_weight = getattr(lm_head, "weight", None)
    if not isinstance(lm_weight, torch.Tensor):
        return None
    lm = lm_weight.detach().float().cpu()
    keys_f = keys.detach().float().cpu()
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    token_indices: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    target_keys: list[torch.Tensor] = []
    seen: set[tuple[int, str]] = set()
    for trace in traces:
        char_start = prompt.find(trace.prefix)
        if char_start < 0:
            continue
        char_pos = char_start + len(trace.prefix)
        token_idx = token_index_before_char(offsets, char_pos)
        if token_idx is None or token_idx < 0 or token_idx >= keys_f.shape[0]:
            continue
        answer_ids = encode_no_special(tokenizer, " " + trace.answer)
        answer_ids = [idx for idx in answer_ids if 0 <= idx < lm.shape[0]]
        if not answer_ids:
            continue
        key = (token_idx, trace.answer)
        if key in seen:
            continue
        seen.add(key)
        target = lm[answer_ids].mean(dim=0)
        if float(torch.linalg.vector_norm(target).item()) <= 1e-12:
            continue
        rows.append(keys_f[token_idx])
        targets.append(float(target_scale) * target)
        token_indices.append(torch.tensor(token_idx, dtype=torch.long))
        scores.append(torch.tensor(float(len(answer_ids)), dtype=torch.float32))
        target_keys.append(torch.zeros_like(keys_f[token_idx]))
    if not rows:
        return None
    weights = shape_readout_weights(torch.stack(scores, dim=0))
    return IntrinsicSurpriseSelection(
        keys=torch.stack(rows, dim=0).contiguous(),
        targets=torch.stack(targets, dim=0).contiguous(),
        weights=weights.contiguous(),
        token_indices=torch.stack(token_indices, dim=0).contiguous(),
        row_scores=torch.stack(scores, dim=0).contiguous(),
        feature_scores=torch.empty(len(rows), 0),
        target_keys=torch.stack(target_keys, dim=0).contiguous(),
        feature_indices=None,
    )


def shape_readout_weights(scores: torch.Tensor) -> torch.Tensor:
    scores_f = scores.detach().float().clamp_min(1e-12)
    return (scores_f / scores_f.mean().clamp_min(1e-12)).contiguous()


def cap_row_norms(rows: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0 or rows.numel() == 0:
        return rows
    rows_f = rows.float()
    norms = torch.linalg.vector_norm(rows_f, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (float(max_norm) / norms).clamp(max=1.0)
    return (rows_f * scale).contiguous()


def center_targets(targets: torch.Tensor, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if targets.numel() == 0:
        return targets, torch.zeros(targets.shape[-1] if targets.ndim else 0)
    targets_f = targets.float()
    if weights is not None and weights.numel() == targets_f.shape[0]:
        w = weights.detach().float().clamp_min(0).unsqueeze(1)
        denom = w.sum().clamp_min(1e-12)
        mean = (targets_f * w).sum(dim=0, keepdim=True) / denom
    else:
        mean = targets_f.mean(dim=0, keepdim=True)
    return (targets_f - mean).contiguous(), mean.squeeze(0).contiguous()


def orthonormal_row_basis(bases: list[torch.Tensor | None]) -> torch.Tensor | None:
    rows = [basis.detach().float().cpu() for basis in bases if basis is not None and basis.numel() > 0]
    if not rows:
        return None
    matrix = torch.cat(rows, dim=0)
    q, _r = torch.linalg.qr(matrix.T, mode="reduced")
    return q.T.contiguous()


def lm_head_generic_basis(model, rank: int) -> torch.Tensor | None:
    if rank <= 0:
        return None
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    lm_weight = weight.detach().float().cpu()
    gram = lm_weight.T @ lm_weight / max(float(lm_weight.shape[0]), 1.0)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    keep = max(1, min(int(rank), eigvecs.shape[1]))
    _ = eigvals  # kept for easy debugger inspection if this function is expanded.
    return eigvecs[:, -keep:].T.contiguous()


def local_logit_fisher_basis(
    model,
    logits: torch.Tensor,
    *,
    rank: int,
    top_k: int = 32,
    max_positions: int = 128,
) -> torch.Tensor | None:
    """Same-pass residual output-risk basis from the model's own distribution.

    This is a cheap local Fisher approximation. For selected lesson positions,
    it builds factors ``sqrt(p_i) * (W_U[i] - E_p[W_U])`` over the top-k next
    tokens under the model's current distribution. It uses no labels, sentinels,
    probes, or next-token optimization; it is only a local observability metric
    for KARP's value-side risk decomposition.
    """

    if rank <= 0 or top_k <= 0:
        return None
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    logits_f = logits.detach().float().cpu()
    if logits_f.ndim != 2 or logits_f.shape[0] == 0:
        return None
    lm_weight = weight.detach().float().cpu()
    if logits_f.shape[1] != lm_weight.shape[0]:
        return None

    token_count = logits_f.shape[0]
    if max_positions > 0 and token_count > max_positions:
        positions = torch.linspace(0, token_count - 1, steps=max_positions).round().long().unique()
        logits_f = logits_f[positions]

    keep = max(1, min(int(top_k), logits_f.shape[1]))
    top_vals, top_idx = torch.topk(logits_f, k=keep, dim=1)
    probs = torch.softmax(top_vals, dim=1)
    selected = lm_weight[top_idx.reshape(-1)].reshape(top_idx.shape[0], keep, lm_weight.shape[1])
    expected = (probs.unsqueeze(-1) * selected).sum(dim=1, keepdim=True)
    rows = (selected - expected) * probs.clamp_min(1e-12).sqrt().unsqueeze(-1)
    rows = rows.reshape(-1, lm_weight.shape[1])
    norms = torch.linalg.vector_norm(rows, dim=1)
    rows = rows[norms > 1e-8]
    if rows.shape[0] == 0:
        return None
    rows = rows - rows.mean(dim=0, keepdim=True)
    if torch.linalg.vector_norm(rows) <= 1e-8:
        return None
    _u, _s, vh = torch.linalg.svd(rows, full_matrices=False)
    keep_rank = max(1, min(int(rank), vh.shape[0]))
    return vh[:keep_rank].contiguous()


def normalized_nonzero_median(rows: torch.Tensor) -> float:
    values = rows.detach().float().abs()
    values = values[values > 0]
    if values.numel() == 0:
        return 1.0
    return float(values.median().clamp_min(1e-6).item())


def intrinsic_input_penalty_keys(
    selection_keys: torch.Tensor,
    down_weight: torch.Tensor,
    output_basis: torch.Tensor | None,
    feature_count: int,
    usage_power: float = 0.0,
    mode: str = "onehot",
) -> torch.Tensor | None:
    """Weight-derived generic feature keys for closed-form input protection."""

    if feature_count <= 0:
        return None
    down = down_weight.detach().float().cpu()
    if down.ndim != 2:
        return None
    keep = max(1, min(int(feature_count), down.shape[1]))
    amplitude = normalized_nonzero_median(selection_keys)
    if mode in {"svd", "hybrid"}:
        rank = keep if mode == "svd" else max(1, keep // 2)
        rank = min(rank, down.shape[0], down.shape[1])
        _u, _s, vh = torch.linalg.svd(down, full_matrices=False)
        svd_keys = amplitude * vh[:rank].contiguous()
        if mode == "svd":
            return svd_keys.contiguous()
        keep = max(1, keep - rank)
    elif mode != "onehot":
        raise ValueError(f"Unknown intrinsic input penalty mode {mode!r}")
    column_norm = torch.linalg.vector_norm(down, dim=0)
    score = column_norm / column_norm.median().clamp_min(1e-12)
    if output_basis is not None and output_basis.numel() > 0:
        basis = output_basis.detach().float().cpu()
        projected_norm = torch.linalg.vector_norm(basis @ down, dim=0)
        score = score + projected_norm / projected_norm.median().clamp_min(1e-12)
    if usage_power > 0 and selection_keys.numel() > 0:
        usage = selection_keys.detach().float().cpu().abs().amax(dim=0)
        usage_scale = usage / usage[usage > 0].median().clamp_min(1e-12) if torch.any(usage > 0) else usage
        score = score / (1.0 + usage_scale).pow(float(usage_power))
    feature_idx = torch.topk(score, k=keep, largest=True).indices
    keys = torch.zeros(keep, down.shape[1], dtype=torch.float32)
    keys[torch.arange(keep), feature_idx] = amplitude
    if mode == "hybrid":
        keys = torch.cat([svd_keys, keys], dim=0)
    return keys.contiguous()


def evenly_cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
    if max_rows <= 0 or rows.shape[0] <= max_rows:
        return rows
    indices = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long()
    return rows[indices].contiguous()


def merge_negative_keys(
    primary: torch.Tensor | None,
    extra: torch.Tensor | None,
    *,
    max_extra_rows: int = 0,
    extra_scale: float = 1.0,
) -> torch.Tensor | None:
    rows: list[torch.Tensor] = []
    if primary is not None and primary.numel() > 0:
        rows.append(primary.detach().float().cpu())
    if extra is not None and extra.numel() > 0:
        extra_f = extra.detach().float().cpu()
        if max_extra_rows > 0:
            extra_f = evenly_cap_rows(extra_f, max_extra_rows)
        if extra_scale != 1.0:
            extra_f = extra_f * float(extra_scale)
        rows.append(extra_f)
    if not rows:
        return None
    return torch.cat(rows, dim=0).contiguous()


def run_intrinsic_surprise_writes(
    model,
    tokenizer,
    wrappers: dict[int, object],
    lesson_texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
    updates_path: Path,
    *,
    slot_id: int | None,
    extra_negative_keys_by_layer: dict[int, torch.Tensor] | None = None,
    selected_keys_out_by_layer: dict[int, list[torch.Tensor]] | None = None,
    max_extra_negative_rows: int = 0,
    extra_negative_scale: float = 1.0,
) -> None:
    layers = get_decoder_layers(model)
    generic_geometry: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    dice_deferred_updates: dict[int, list[torch.Tensor]] = {}
    lm_basis = lm_head_generic_basis(model, args.intrinsic_surprise_lm_head_generic_rank)
    output_penalty_basis = lm_head_generic_basis(model, args.intrinsic_surprise_output_penalty_rank)
    if (
        args.intrinsic_surprise_generic_rank > 0
        or lm_basis is not None
        or (output_penalty_basis is not None and args.intrinsic_surprise_readout_specificity_power != 0)
    ):
        for layer_idx, wrapper in wrappers.items():
            down_weight = (
                effective_down_weight(wrapper)
                if args.intrinsic_surprise_value_source == "effective"
                else base_down_weight(wrapper)
            )
            if args.intrinsic_surprise_generic_rank > 0:
                specificity, basis = down_value_specificity(down_weight, args.intrinsic_surprise_generic_rank)
            else:
                specificity = torch.ones(down_weight.shape[1])
                basis = None
            feature_weights = specificity.pow(float(args.intrinsic_surprise_specificity_power))
            if output_penalty_basis is not None and args.intrinsic_surprise_readout_specificity_power != 0:
                readout_specificity = down_output_basis_specificity(down_weight, output_penalty_basis)
                feature_weights = feature_weights * readout_specificity.pow(
                    float(args.intrinsic_surprise_readout_specificity_power)
                )
            target_basis = orthonormal_row_basis([
                basis if args.intrinsic_surprise_project_generic else None,
                lm_basis,
            ])
            generic_geometry[layer_idx] = (
                feature_weights,
                target_basis,
            )
    used_birth_neurons: dict[int, list[int]] = {layer_idx: [] for layer_idx in wrappers}
    capture_wicr_attentions = (
        (
            args.intrinsic_surprise_target_mode == "compatibility_residual"
            and args.wicr_attention_edges > 0
        )
        or (
            args.intrinsic_surprise_target_mode == "conditional_relation_innovation"
            and args.cori_edge_top_k > 0
        )
        or (
            args.intrinsic_surprise_target_mode == "schur_transport_actuator"
            and args.cori_edge_top_k > 0
        )
    )
    for lesson_idx, lesson_text in enumerate(lesson_texts):
        if args.write_only_final and lesson_idx != len(lesson_texts) - 1:
            continue
        prompt = format_intrinsic_lesson_prompt(tokenizer, lesson_text, args)
        captures, input_ids, logits = capture_intrinsic_lesson_forward(
            model,
            tokenizer,
            prompt,
            args.layers,
            device,
            args.max_length,
            capture_attentions=capture_wicr_attentions,
        )
        karp_local_output_basis = local_logit_fisher_basis(
            model,
            logits,
            rank=args.karp_local_fisher_rank,
            top_k=args.karp_local_fisher_top_k,
            max_positions=args.karp_local_fisher_max_positions,
        )
        karp_output_basis = orthonormal_row_basis([output_penalty_basis, karp_local_output_basis])
        sharp_top_values = torch.empty(0)
        sharp_top_indices = torch.empty(0, dtype=torch.long)
        sharp_lm_indices = torch.empty(0, dtype=torch.long)
        sharp_lm_rows = torch.empty(0)
        if args.intrinsic_target_purifier in {
            "sharp_karp",
            "orca_karp",
            "qrico",
            "prism_q",
            "tdmi_q",
            "trace_q",
            "spectra",
            "seal_qrico",
            "ocep_residual",
            "ocep_qrico",
        }:
            requested_top_k = (
                int(args.sharp_signal_top_k)
                if args.intrinsic_target_purifier == "sharp_karp"
                else int(args.qrico_option_sketch_rank)
                if args.intrinsic_target_purifier in {"qrico", "seal_qrico", "tdmi_q"}
                else int(args.prism_option_top_k)
                if args.intrinsic_target_purifier == "prism_q"
                else int(args.trace_option_top_k)
                if args.intrinsic_target_purifier == "trace_q"
                else int(args.spectra_option_top_k)
                if args.intrinsic_target_purifier == "spectra"
                else max(2, int(args.ocep_option_local_rank) * 2)
                if args.intrinsic_target_purifier in {"ocep_residual", "ocep_qrico"}
                else int(args.orca_option_top_k)
            )
            sharp_top_k = max(2, min(requested_top_k, logits.shape[1]))
            sharp_top_values, sharp_top_indices = torch.topk(logits.detach().float().cpu(), k=sharp_top_k, dim=1)
            sharp_lm_indices = sharp_top_indices.reshape(-1).unique(sorted=True)
            lm_head = getattr(model, "lm_head", None)
            lm_weight = getattr(lm_head, "weight", None)
            if not isinstance(lm_weight, torch.Tensor):
                raise ValueError(f"{args.intrinsic_target_purifier} requires a tensor lm_head.weight")
            sharp_lm_rows = lm_weight.detach().index_select(
                0, sharp_lm_indices.to(lm_weight.device)
            ).float().cpu()
        if args.intrinsic_surprise_target_mode == "logit_error":
            logit_error_targets, token_losses = lesson_logit_error_targets(model, input_ids, logits)
        else:
            logit_error_targets = torch.empty(0)
            token_losses = torch.empty(0)
        for layer_idx, wrapper in wrappers.items():
            layer = layers[layer_idx]
            keys = captures[layer_idx].keys
            down_weight = (
                effective_down_weight(wrapper)
                if args.intrinsic_surprise_value_source == "effective"
                else base_down_weight(wrapper)
            )
            feature_weights, target_projection_basis = generic_geometry.get(layer_idx, (None, None))
            usable_keys = keys
            if args.intrinsic_surprise_target_mode == "logit_error":
                usable_rows = min(keys.shape[0] - 1, logit_error_targets.shape[0])
                if usable_rows <= 0:
                    continue
                usable_keys = keys[:usable_rows]
            scoring_usable_keys = None
            if args.seal_canonicalize_surprise:
                canonical_scale = gauge_canonical_key_scale(down_weight, karp_output_basis)
                scoring_usable_keys = usable_keys.detach().float().cpu() * canonical_scale.unsqueeze(0)
            if args.intrinsic_surprise_target_mode == "feature_birth":
                avoid_neurons = torch.tensor(used_birth_neurons.get(layer_idx, []), dtype=torch.long)
                if args.intrinsic_surprise_birth_mode == "conjunction":
                    update = select_intrinsic_conjunctive_feature_birth_update(
                        captures[layer_idx].mlp_inputs[: usable_keys.shape[0]],
                        usable_keys,
                        layer,
                        down_weight,
                        token_mode=args.intrinsic_surprise_token_mode,
                        top_tokens=args.intrinsic_surprise_top_tokens,
                        feature_top_k=args.intrinsic_surprise_feature_top_k,
                        key_feature_top_k=args.intrinsic_surprise_key_feature_top_k,
                        value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                        pair_count=args.intrinsic_surprise_birth_pairs,
                        target_scale=args.intrinsic_surprise_target_scale,
                        min_response=args.intrinsic_surprise_birth_min_response,
                        persistence_power=args.intrinsic_surprise_persistence_power,
                        persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                        persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                        feature_weights=feature_weights,
                        target_projection_basis=target_projection_basis,
                        current_down_weight=effective_down_weight(wrapper),
                        avoid_neurons=avoid_neurons,
                    )
                else:
                    update = select_intrinsic_feature_birth_update(
                        captures[layer_idx].mlp_inputs[: usable_keys.shape[0]],
                        usable_keys,
                        layer,
                        down_weight,
                        token_mode=args.intrinsic_surprise_token_mode,
                        top_tokens=args.intrinsic_surprise_top_tokens,
                        feature_top_k=args.intrinsic_surprise_feature_top_k,
                        value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                        target_scale=args.intrinsic_surprise_target_scale,
                        trigger_scale=args.intrinsic_surprise_birth_trigger_scale,
                        trigger_ridge=args.intrinsic_surprise_birth_trigger_ridge,
                        persistence_power=args.intrinsic_surprise_persistence_power,
                        persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                        persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                        feature_weights=feature_weights,
                        target_projection_basis=target_projection_basis,
                        current_down_weight=effective_down_weight(wrapper),
                        avoid_neurons=avoid_neurons,
                    )
                used_birth_neurons.setdefault(layer_idx, []).extend(int(idx) for idx in update.neuron_indices.tolist())
                stats = apply_intrinsic_feature_birth_update_(
                    layer,
                    wrapper,
                    update,
                    eta=args.eta,
                    max_down_update_norm=args.max_update_norm,
                )
                selected_token_texts = [
                    tokenizer.decode([int(input_ids[int(idx.item())])])
                    for idx in update.token_indices[:32]
                    if int(idx.item()) < input_ids.numel()
                ]
                append_jsonl(
                    updates_path,
                    {
                        "lesson_idx": lesson_idx,
                        "layer": layer_idx,
                        "module": "mlp_feature_birth",
                        "write_mode": "intrinsic_surprise",
                        "lesson_token_count": int(keys.shape[0]),
                        "selected_rows": int(update.neuron_indices.numel()),
                        "selected_token_min": int(update.token_indices.min().item()),
                        "selected_token_max": int(update.token_indices.max().item()),
                        "selected_neuron_min": int(update.neuron_indices.min().item()),
                        "selected_neuron_max": int(update.neuron_indices.max().item()),
                        "selected_token_texts": selected_token_texts,
                        "row_score_mean": float(update.row_scores.mean().item()),
                        "row_score_max": float(update.row_scores.max().item()),
                        "target_key_fro": float(torch.linalg.vector_norm(update.target_keys).item()),
                        "intrinsic_target_mode": args.intrinsic_surprise_target_mode,
                        "intrinsic_birth_mode": args.intrinsic_surprise_birth_mode,
                        "intrinsic_value_source": args.intrinsic_surprise_value_source,
                        "intrinsic_effective_target_norm": args.intrinsic_surprise_effective_target_norm,
                        "intrinsic_persistence_power": args.intrinsic_surprise_persistence_power,
                        "intrinsic_generic_rank": args.intrinsic_surprise_generic_rank,
                        "intrinsic_lm_head_generic_rank": args.intrinsic_surprise_lm_head_generic_rank,
                        "intrinsic_specificity_power": args.intrinsic_surprise_specificity_power,
                        "intrinsic_readout_specificity_power": args.intrinsic_surprise_readout_specificity_power,
                        "intrinsic_project_generic": bool(args.intrinsic_surprise_project_generic),
                        "selected_feature_count": int(update.feature_indices.unique().numel()),
                        **stats,
                    },
                )
                continue
            if args.intrinsic_surprise_target_mode == "associative_binding":
                selection = select_intrinsic_associative_binding_write(
                    usable_keys,
                    layer,
                    down_weight,
                    token_mode=args.intrinsic_surprise_token_mode,
                    top_tokens=args.intrinsic_surprise_top_tokens,
                    feature_top_k=args.intrinsic_surprise_feature_top_k,
                    key_feature_top_k=args.intrinsic_surprise_key_feature_top_k,
                    value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                        target_scale=args.intrinsic_surprise_target_scale,
                        persistence_power=args.intrinsic_surprise_persistence_power,
                        persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                        persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                        feature_weights=feature_weights,
                        target_projection_basis=target_projection_basis,
                        surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            elif args.intrinsic_surprise_target_mode == "predictive_residual":
                selection = select_intrinsic_predictive_residual_write(
                    usable_keys,
                    layer,
                    down_weight,
                    token_mode=args.intrinsic_surprise_token_mode,
                    top_tokens=args.intrinsic_surprise_top_tokens,
                    feature_top_k=args.intrinsic_surprise_feature_top_k,
                    key_feature_top_k=args.intrinsic_surprise_key_feature_top_k,
                    value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                    target_scale=args.intrinsic_surprise_target_scale,
                    prediction_ridge=args.intrinsic_surprise_prediction_ridge,
                    persistence_power=args.intrinsic_surprise_persistence_power,
                    persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                    persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                    feature_weights=feature_weights,
                    target_projection_basis=target_projection_basis,
                    surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            elif args.intrinsic_surprise_target_mode == "relational_residual":
                selection = select_intrinsic_relational_residual_write(
                    usable_keys,
                    layer,
                    down_weight,
                    token_mode=args.intrinsic_surprise_token_mode,
                    top_tokens=args.intrinsic_surprise_top_tokens,
                    feature_top_k=args.intrinsic_surprise_feature_top_k,
                    key_feature_top_k=args.intrinsic_surprise_key_feature_top_k,
                    value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                    pair_top_k=args.intrinsic_surprise_pair_top_k,
                    bidirectional_pairs=args.intrinsic_surprise_bidirectional_pairs,
                    relation_value_mode=args.intrinsic_surprise_relation_value_mode,
                    target_scale=args.intrinsic_surprise_target_scale,
                    prediction_ridge=args.intrinsic_surprise_prediction_ridge,
                    persistence_power=args.intrinsic_surprise_persistence_power,
                    persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                    persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                    feature_weights=feature_weights,
                    target_projection_basis=target_projection_basis,
                    surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            elif args.intrinsic_surprise_target_mode == "relational_aggregate":
                try:
                    selection = select_intrinsic_relational_aggregate_write(
                        usable_keys,
                        layer,
                        down_weight,
                        scoring_keys=scoring_usable_keys,
                        token_mode=args.intrinsic_surprise_token_mode,
                        top_tokens=args.intrinsic_surprise_top_tokens,
                        feature_top_k=args.intrinsic_surprise_feature_top_k,
                        key_feature_top_k=args.intrinsic_surprise_key_feature_top_k,
                        value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                        pair_top_k=args.intrinsic_surprise_pair_top_k,
                        bidirectional_pairs=args.intrinsic_surprise_bidirectional_pairs,
                        relation_value_mode=args.intrinsic_surprise_relation_value_mode,
                        target_scale=args.intrinsic_surprise_target_scale,
                        prediction_ridge=args.intrinsic_surprise_prediction_ridge,
                        persistence_power=args.intrinsic_surprise_persistence_power,
                        persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                        persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                        feature_weights=feature_weights,
                        target_projection_basis=target_projection_basis,
                        surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                        surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                        surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                    )
                except ValueError as exc:
                    if "No nonzero relational aggregate examples" in str(exc):
                        continue
                    raise
            elif args.intrinsic_surprise_target_mode == "compatibility_residual":
                wicr_target_basis = orthonormal_row_basis([target_projection_basis, output_penalty_basis])
                selection = select_intrinsic_compatibility_residual_write(
                    captures[layer_idx].mlp_inputs[: usable_keys.shape[0]],
                    usable_keys,
                    layer,
                    down_weight,
                    token_mode=args.intrinsic_surprise_token_mode,
                    top_tokens=args.intrinsic_surprise_top_tokens,
                    feature_top_k=args.intrinsic_surprise_feature_top_k,
                    key_feature_top_k=args.intrinsic_surprise_key_feature_top_k,
                    value_feature_top_k=args.intrinsic_surprise_value_feature_top_k,
                    pair_top_k=args.intrinsic_surprise_pair_top_k,
                    compatibility_threshold=args.wicr_compatibility_threshold,
                    compatibility_temperature=args.wicr_compatibility_temperature,
                    posture_pcs=args.wicr_posture_pcs,
                    target_vector_mode=args.wicr_target_vector_mode,
                    attention_probs=captures[layer_idx].attentions,
                    attention_edge_top_k=args.wicr_attention_edges,
                    attention_flow_mode=args.wicr_attention_flow_mode,
                    include_same_token_edges=not args.wicr_no_same_token_edges,
                    target_scale=args.intrinsic_surprise_target_scale,
                    persistence_power=args.intrinsic_surprise_persistence_power,
                    persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                    persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                    feature_weights=feature_weights,
                    target_projection_basis=wicr_target_basis,
                    surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            elif args.intrinsic_surprise_target_mode == "conditional_relation_innovation":
                cori_target_basis = orthonormal_row_basis([target_projection_basis, output_penalty_basis])
                selection = select_intrinsic_conditional_relation_innovation_write(
                    captures[layer_idx].mlp_inputs[: usable_keys.shape[0]],
                    usable_keys,
                    layer,
                    down_weight,
                    feature_top_k=args.cori_feature_top_k,
                    relation_rank=args.cori_relation_rank,
                    beta=args.cori_beta,
                    edge_top_k=args.cori_edge_top_k,
                    edge_attention_scale=args.cori_edge_attention_scale,
                    sinkhorn_steps=args.cori_sinkhorn_steps,
                    target_mode=args.cori_target_mode,
                    target_scale=args.intrinsic_surprise_target_scale,
                    persistence_power=args.intrinsic_surprise_persistence_power,
                    persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                    persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                    feature_weights=feature_weights,
                    target_projection_basis=cori_target_basis,
                    attention_probs=captures[layer_idx].attentions,
                    surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            elif args.intrinsic_surprise_target_mode == "schur_transport_actuator":
                star_target_basis = orthonormal_row_basis([target_projection_basis, output_penalty_basis])
                future_mlp_inputs_by_layer = {
                    future_idx: future_capture.mlp_inputs[: usable_keys.shape[0]]
                    for future_idx, future_capture in captures.items()
                    if future_capture.mlp_inputs.shape[0] >= usable_keys.shape[0]
                }
                selection = select_intrinsic_schur_transport_actuator_write(
                    captures[layer_idx].mlp_inputs[: usable_keys.shape[0]],
                    usable_keys,
                    layer,
                    down_weight,
                    layer_idx=layer_idx,
                    future_mlp_inputs_by_layer=future_mlp_inputs_by_layer,
                    feature_top_k=args.cori_feature_top_k,
                    relation_rank=args.cori_relation_rank,
                    beta=args.cori_beta,
                    edge_top_k=args.cori_edge_top_k,
                    edge_attention_scale=args.cori_edge_attention_scale,
                    sinkhorn_steps=args.cori_sinkhorn_steps,
                    target_scale=args.intrinsic_surprise_target_scale,
                    object_summary_gain=args.star_object_summary_gain,
                    future_layer_horizon=args.star_future_layer_horizon,
                    future_token_top_k=args.star_future_token_top_k,
                    future_layer_decay=args.star_future_layer_decay,
                    future_token_decay=args.star_future_token_decay,
                    future_relation_power=args.star_future_relation_power,
                    ordinary_key_rank=args.star_ordinary_key_rank,
                    value_projection_features=args.star_value_projection_features,
                    value_projection_ridge=args.star_value_projection_ridge,
                    schur_ridge=args.star_schur_ridge,
                    map_ridge=args.star_map_ridge,
                    posture_negative_scale=args.star_posture_negative_scale,
                    min_coherence=args.star_min_coherence,
                    shuffle_future_targets=args.star_shuffle_future_targets,
                    shuffle_keys=args.star_shuffle_keys,
                    persistence_power=args.intrinsic_surprise_persistence_power,
                    persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                    persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                    feature_weights=feature_weights,
                    target_projection_basis=star_target_basis,
                    attention_probs=captures[layer_idx].attentions,
                    surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            else:
                selection = select_intrinsic_surprise_write(
                    usable_keys,
                    layer,
                    down_weight,
                    token_mode=args.intrinsic_surprise_token_mode,
                    top_tokens=args.intrinsic_surprise_top_tokens,
                    feature_top_k=args.intrinsic_surprise_feature_top_k,
                    target_feature_top_k=args.intrinsic_surprise_target_feature_top_k,
                    target_scale=args.intrinsic_surprise_target_scale,
                    persistence_power=args.intrinsic_surprise_persistence_power,
                    persistence_threshold_fraction=args.intrinsic_surprise_persistence_threshold,
                    persistence_min_tokens=args.intrinsic_surprise_persistence_min_tokens,
                    feature_weights=feature_weights,
                    target_projection_basis=target_projection_basis,
                    surprise_weight_mode=args.intrinsic_surprise_weight_mode,
                    surprise_weight_temperature=args.intrinsic_surprise_exp_temperature,
                    surprise_weight_cap=args.intrinsic_surprise_exp_cap,
                )
            targets = selection.targets
            positive_weights = selection.weights
            if (
                args.intrinsic_surprise_value_source == "effective"
                and args.intrinsic_surprise_effective_target_norm == "base"
            ):
                reference_targets = args.intrinsic_surprise_target_scale * (
                    selection.target_keys @ base_down_weight(wrapper).T
                )
                if target_projection_basis is not None and target_projection_basis.numel() > 0:
                    reference_targets = project_rows_away_from_basis(reference_targets, target_projection_basis)
                targets = (
                    torch.nn.functional.normalize(targets.float(), dim=1)
                    * torch.linalg.vector_norm(reference_targets.float(), dim=1, keepdim=True)
                )
            if args.intrinsic_surprise_target_mode == "logit_error":
                targets = args.intrinsic_surprise_target_scale * logit_error_targets[selection.token_indices]
                loss_weights = token_losses[selection.token_indices]
                positive_weights = positive_weights * (loss_weights / loss_weights.mean().clamp_min(1e-12))
            uncapped_target_fro = float(torch.linalg.vector_norm(targets.float()).item())
            target_center_norm = 0.0
            if args.intrinsic_surprise_center_targets:
                targets, target_center = center_targets(targets, positive_weights)
                target_center_norm = float(torch.linalg.vector_norm(target_center).item())
            targets = cap_row_norms(targets, args.intrinsic_surprise_target_row_norm_cap)
            input_penalty_keys = intrinsic_input_penalty_keys(
                selection.keys,
                effective_down_weight(wrapper),
                output_penalty_basis,
                args.intrinsic_surprise_input_penalty_features,
                usage_power=args.intrinsic_surprise_input_penalty_usage_power,
                mode=args.intrinsic_surprise_input_penalty_mode,
            )
            selection_negative_keys = getattr(selection, "negative_keys", None)
            input_and_selection_negative_keys = merge_negative_keys(
                input_penalty_keys,
                selection_negative_keys,
            )
            extra_negative = (
                extra_negative_keys_by_layer.get(layer_idx)
                if extra_negative_keys_by_layer is not None
                else None
            )
            negative_keys = merge_negative_keys(
                input_and_selection_negative_keys,
                extra_negative,
                max_extra_rows=max_extra_negative_rows,
                extra_scale=extra_negative_scale,
            )
            use_metric_solve = (
                output_penalty_basis is not None and args.intrinsic_surprise_output_penalty_weight > 0
            )
            if use_metric_solve:
                update, stats = protected_metric_update(
                    selection.keys,
                    targets,
                    negative_keys=negative_keys,
                    output_penalty_basis=output_penalty_basis,
                    positive_weights=positive_weights,
                    ridge=args.ridge,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_penalty_weight=args.intrinsic_surprise_output_penalty_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
            else:
                update, stats = protected_ridge_update(
                    selection.keys,
                    targets,
                    negative_keys=negative_keys,
                    positive_weights=positive_weights,
                    ridge=args.ridge,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    eta=args.eta,
                    max_update_norm=args.max_update_norm,
                )
            seal_scales = None
            if args.intrinsic_target_purifier == "karp":
                karp = karp_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    key_rank=args.karp_key_rank,
                    value_rank=args.karp_value_rank,
                    eta_cross=args.karp_eta_cross,
                    eta_key=args.karp_eta_key,
                    eta_value=args.karp_eta_value,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = karp.update
                if args.karp_layer_risk_budget > 0:
                    risk_after = float(karp.diagnostics.get("karp_cross_risk_after", 0.0))
                    trust_scale = min(
                        1.0,
                        (float(args.karp_layer_risk_budget) / max(risk_after, 1e-12)) ** 0.5,
                    )
                    if trust_scale < 1.0:
                        update = update * trust_scale
                    karp.diagnostics["karp_layer_risk_budget"] = float(args.karp_layer_risk_budget)
                    karp.diagnostics["karp_layer_trust_scale"] = float(trust_scale)
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(karp.diagnostics)
            elif args.intrinsic_target_purifier == "sharp_karp":
                sharp = sharp_karp_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    key_rank=args.sharp_key_rank,
                    value_rank=args.sharp_value_rank,
                    low_surprise_quantile=args.sharp_low_surprise_quantile,
                    confidence_quantile=args.sharp_confidence_quantile,
                    max_anchors=args.sharp_shadow_anchors,
                    signal_top_k=args.sharp_signal_top_k,
                    eta_sharp=args.sharp_eta,
                    shadow_weight=args.sharp_shadow_weight,
                    karp_eta_cross=args.karp_eta_cross,
                    karp_eta_key=args.karp_eta_key,
                    karp_eta_value=args.karp_eta_value,
                    karp_kappa=args.sharp_karp_kappa,
                    ridge=args.ridge,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    shadow_temperature=args.sharp_shadow_temperature,
                    solve_mode=args.sharp_solve_mode,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = sharp.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(sharp.diagnostics)
            elif args.intrinsic_target_purifier == "orca_karp":
                orca = orca_karp_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    down_weight=effective_down_weight(wrapper),
                    feature_indices=selection.feature_indices,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    key_rank=args.orca_key_rank,
                    value_rank=args.orca_value_rank,
                    option_top_k=args.orca_option_top_k,
                    object_rank=args.orca_object_rank,
                    off_object_rank=args.orca_off_object_rank,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    eta_orth=args.orca_eta_orth,
                    eta_posture=args.orca_eta_posture,
                    eta_off_object=args.orca_eta_off_object,
                    eta_karp=args.orca_eta_karp,
                    eta_key=args.karp_eta_key,
                    eta_value=args.karp_eta_value,
                    signal_floor_quantile=args.orca_signal_floor_quantile,
                    ablation_mode=args.orca_ablation_mode,
                    ablation_fraction=args.orca_ablation_fraction,
                    nuisance_ridge=args.orca_nuisance_ridge,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = orca.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(orca.diagnostics)
            elif args.intrinsic_target_purifier == "qrico":
                qrico = qrico_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    deflate_key_rank=args.qrico_deflate_key_rank,
                    deflate_value_rank=args.qrico_deflate_value_rank,
                    rank=args.qrico_rank,
                    option_sketch_rank=args.qrico_option_sketch_rank,
                    target_parallel_rank=args.qrico_target_parallel_rank,
                    scramble_weight=args.qrico_scramble_weight,
                    residual_row_weight_power=args.qrico_residual_row_weight_power,
                    quotient_mode=args.qrico_quotient_mode,
                    solve_mode=args.qrico_solve_mode,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    cca_ridge=args.qrico_cca_ridge,
                    layer_evidence_min=args.qrico_layer_evidence_min,
                    layer_evidence_target=args.qrico_layer_evidence_target,
                    apply_layer_trust=not args.qrico_disable_layer_trust,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = qrico.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(qrico.diagnostics)
            elif args.intrinsic_target_purifier == "tdmi_q":
                preliminary_qrico = qrico_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    deflate_key_rank=args.qrico_deflate_key_rank,
                    deflate_value_rank=args.qrico_deflate_value_rank,
                    rank=args.qrico_rank,
                    option_sketch_rank=args.qrico_option_sketch_rank,
                    target_parallel_rank=args.qrico_target_parallel_rank,
                    scramble_weight=args.qrico_scramble_weight,
                    residual_row_weight_power=args.qrico_residual_row_weight_power,
                    quotient_mode=args.qrico_quotient_mode,
                    solve_mode=args.qrico_solve_mode,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    cca_ridge=args.qrico_cca_ridge,
                    layer_evidence_min=args.qrico_layer_evidence_min,
                    layer_evidence_target=args.qrico_layer_evidence_target,
                    apply_layer_trust=not args.qrico_disable_layer_trust,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                future_outputs_by_layer = {
                    future_idx: future_capture.outputs[: usable_keys.shape[0]]
                    for future_idx, future_capture in captures.items()
                    if future_capture.outputs.shape[0] >= usable_keys.shape[0]
                }
                tdmi = tdmi_q_transport_scores(
                    preliminary_qrico.update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    layer=layer,
                    future_outputs_by_layer=future_outputs_by_layer,
                    layer_idx=layer_idx,
                    object_endpoints=args.tdmi_object_endpoints,
                    ambient_endpoints=args.tdmi_ambient_endpoints,
                    object_rank=args.tdmi_object_rank,
                    ambient_rank=args.tdmi_ambient_rank,
                    horizon=args.tdmi_horizon,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    trust_temperature=args.tdmi_trust_temperature,
                    trust_threshold=args.tdmi_trust_threshold,
                    trust_floor=args.tdmi_trust_floor,
                    use_future_outputs=not args.tdmi_disable_future,
                )
                tdmi_weights = positive_weights.detach().float().cpu() * tdmi.row_trust
                if use_metric_solve:
                    tdmi_base_update, tdmi_stats = protected_metric_update(
                        selection.keys,
                        targets,
                        negative_keys=negative_keys,
                        output_penalty_basis=output_penalty_basis,
                        positive_weights=tdmi_weights,
                        ridge=args.ridge,
                        negative_weight=args.intrinsic_surprise_input_penalty_weight,
                        output_penalty_weight=args.intrinsic_surprise_output_penalty_weight,
                        eta=args.eta,
                        max_update_norm=args.max_update_norm,
                    )
                else:
                    tdmi_base_update, tdmi_stats = protected_ridge_update(
                        selection.keys,
                        targets,
                        negative_keys=negative_keys,
                        positive_weights=tdmi_weights,
                        ridge=args.ridge,
                        negative_weight=args.intrinsic_surprise_input_penalty_weight,
                        eta=args.eta,
                        max_update_norm=args.max_update_norm,
                    )
                qrico = qrico_purify_update(
                    tdmi_base_update,
                    keys=selection.keys,
                    targets=targets,
                    weights=tdmi_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    deflate_key_rank=args.qrico_deflate_key_rank,
                    deflate_value_rank=args.qrico_deflate_value_rank,
                    rank=args.qrico_rank,
                    option_sketch_rank=args.qrico_option_sketch_rank,
                    target_parallel_rank=args.qrico_target_parallel_rank,
                    scramble_weight=args.qrico_scramble_weight,
                    residual_row_weight_power=args.qrico_residual_row_weight_power,
                    quotient_mode=args.qrico_quotient_mode,
                    solve_mode=args.qrico_solve_mode,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    cca_ridge=args.qrico_cca_ridge,
                    layer_evidence_min=args.qrico_layer_evidence_min,
                    layer_evidence_target=args.qrico_layer_evidence_target,
                    apply_layer_trust=not args.qrico_disable_layer_trust,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = qrico.update
                stats = tdmi_stats
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(
                    {f"tdmi_pre_{key}": value for key, value in preliminary_qrico.diagnostics.items()}
                )
                selection.diagnostics.update(tdmi.diagnostics)
                selection.diagnostics.update(qrico.diagnostics)
            elif args.intrinsic_target_purifier == "prism_q":
                future_outputs_by_layer = {
                    future_idx: future_capture.outputs[: usable_keys.shape[0]]
                    for future_idx, future_capture in captures.items()
                    if future_capture.outputs.shape[0] >= usable_keys.shape[0]
                }
                prism = prism_q_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer_idx=layer_idx,
                    layer=layer,
                    future_outputs_by_layer=future_outputs_by_layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    horizon=args.prism_horizon,
                    signal_rank=args.prism_signal_rank,
                    hazard_rank=args.prism_hazard_rank,
                    option_top_k=args.prism_option_top_k,
                    generic_key_rank=args.prism_generic_key_rank,
                    low_surprise_rows=args.prism_low_surprise_rows,
                    budget=args.prism_budget,
                    correction_cap=args.prism_correction_cap,
                    signal_retention_min=args.prism_signal_retention_min,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    residualize_hazard=not args.prism_no_residualize_hazard,
                    use_future_outputs=not args.prism_disable_future,
                    ablation_mode=args.prism_ablation,
                    ridge=args.qrico_cca_ridge,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = prism.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(prism.diagnostics)
            elif args.intrinsic_target_purifier == "trace_q":
                qrico = qrico_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    deflate_key_rank=args.qrico_deflate_key_rank,
                    deflate_value_rank=args.qrico_deflate_value_rank,
                    rank=args.qrico_rank,
                    option_sketch_rank=args.qrico_option_sketch_rank,
                    target_parallel_rank=args.qrico_target_parallel_rank,
                    scramble_weight=args.qrico_scramble_weight,
                    residual_row_weight_power=args.qrico_residual_row_weight_power,
                    quotient_mode=args.qrico_quotient_mode,
                    solve_mode=args.qrico_solve_mode,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    cca_ridge=args.qrico_cca_ridge,
                    layer_evidence_min=args.qrico_layer_evidence_min,
                    layer_evidence_target=args.qrico_layer_evidence_target,
                    apply_layer_trust=not args.qrico_disable_layer_trust,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                trace = trace_q_purify_update(
                    qrico.update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    object_endpoints=args.trace_object_endpoints,
                    ambient_endpoints=args.trace_ambient_endpoints,
                    option_top_k=args.trace_option_top_k,
                    option_contrasts=args.trace_option_contrasts,
                    object_rank=args.trace_object_rank,
                    ambient_rank=args.trace_ambient_rank,
                    generic_key_rank=args.trace_generic_key_rank,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    target_tau=args.trace_target_tau,
                    target_floor=args.trace_target_floor,
                    collateral_weight=args.trace_gamma,
                    layer_trust_threshold=args.trace_layer_trust_threshold,
                )
                update = trace.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(qrico.diagnostics)
                selection.diagnostics.update(trace.diagnostics)
            elif args.intrinsic_target_purifier == "ocep_residual":
                up_module = getattr(getattr(layer, "mlp", None), "up_proj", None)
                if up_module is None:
                    up_module = getattr(getattr(layer, "mlp", None), "fc1", None)
                if up_module is None:
                    up_module = getattr(getattr(layer, "mlp", None), "c_fc", None)
                up_weight = getattr(up_module, "weight", None)
                ocep = ocep_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    down_weight=effective_down_weight(wrapper),
                    up_weight=up_weight.detach() if isinstance(up_weight, torch.Tensor) else None,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    object_rank=args.ocep_object_rank,
                    generic_rank=args.ocep_generic_rank,
                    option_rank=args.ocep_option_rank,
                    option_output_rank=args.ocep_option_output_rank,
                    option_local_rank=args.ocep_option_local_rank,
                    generic_low_surprise_rank=args.ocep_low_surprise_rank,
                    generic_weight_anchor_rank=args.ocep_weight_anchor_rank,
                    generic_protected_rank=args.ocep_protected_rank,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    ridge=args.ocep_ridge,
                    correction_cap=args.ocep_correction_cap,
                    conflict_skip=args.ocep_conflict_skip,
                )
                update = ocep.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(ocep.diagnostics)
            elif args.intrinsic_target_purifier == "ocep_qrico":
                up_module = getattr(getattr(layer, "mlp", None), "up_proj", None)
                if up_module is None:
                    up_module = getattr(getattr(layer, "mlp", None), "fc1", None)
                if up_module is None:
                    up_module = getattr(getattr(layer, "mlp", None), "c_fc", None)
                up_weight = getattr(up_module, "weight", None)
                ocep = ocep_qrico_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    down_weight=effective_down_weight(wrapper),
                    up_weight=up_weight.detach() if isinstance(up_weight, torch.Tensor) else None,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    deflate_key_rank=args.qrico_deflate_key_rank,
                    deflate_value_rank=args.qrico_deflate_value_rank,
                    rank=args.qrico_rank,
                    option_sketch_rank=args.qrico_option_sketch_rank,
                    target_parallel_rank=args.qrico_target_parallel_rank,
                    scramble_weight=args.qrico_scramble_weight,
                    residual_row_weight_power=args.qrico_residual_row_weight_power,
                    quotient_mode=args.qrico_quotient_mode,
                    solve_mode=args.qrico_solve_mode,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    cca_ridge=args.qrico_cca_ridge,
                    layer_evidence_min=args.qrico_layer_evidence_min,
                    layer_evidence_target=args.qrico_layer_evidence_target,
                    apply_layer_trust=not args.qrico_disable_layer_trust,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                    object_rank=args.ocep_object_rank,
                    generic_rank=args.ocep_generic_rank,
                    ocep_option_rank=args.ocep_option_rank,
                    option_output_rank=args.ocep_option_output_rank,
                    option_local_rank=args.ocep_option_local_rank,
                    generic_low_surprise_rank=args.ocep_low_surprise_rank,
                    generic_weight_anchor_rank=args.ocep_weight_anchor_rank,
                    generic_protected_rank=args.ocep_protected_rank,
                    ocep_ridge=args.ocep_ridge,
                    correction_cap=args.ocep_correction_cap,
                    conflict_skip=args.ocep_conflict_skip,
                )
                update = ocep.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(ocep.diagnostics)
            elif args.intrinsic_target_purifier == "seal_qrico":
                up_module = getattr(getattr(layer, "mlp", None), "up_proj", None)
                if up_module is None:
                    up_module = getattr(getattr(layer, "mlp", None), "fc1", None)
                if up_module is None:
                    up_module = getattr(getattr(layer, "mlp", None), "c_fc", None)
                up_weight = getattr(up_module, "weight", None)
                if not isinstance(up_weight, torch.Tensor):
                    raise ValueError("seal_qrico requires an MLP up projection weight")
                seal = seal_qrico_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    up_weight=up_weight.detach(),
                    current_down_weight=effective_down_weight(wrapper),
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    deflate_key_rank=args.qrico_deflate_key_rank,
                    deflate_value_rank=args.qrico_deflate_value_rank,
                    rank=args.qrico_rank,
                    option_sketch_rank=args.qrico_option_sketch_rank,
                    target_parallel_rank=args.qrico_target_parallel_rank,
                    scramble_weight=args.qrico_scramble_weight,
                    residual_row_weight_power=args.qrico_residual_row_weight_power,
                    quotient_mode=args.qrico_quotient_mode,
                    solve_mode=args.qrico_solve_mode,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    negative_weight=args.intrinsic_surprise_input_penalty_weight,
                    output_weight=args.intrinsic_surprise_output_penalty_weight,
                    cca_ridge=args.qrico_cca_ridge,
                    layer_evidence_min=args.qrico_layer_evidence_min,
                    layer_evidence_target=args.qrico_layer_evidence_target,
                    apply_layer_trust=not args.qrico_disable_layer_trust,
                    salience_tau=args.seal_salience_tau,
                    eta_erase=args.seal_eta_erase,
                    eta_seal=0.0 if args.seal_disable_apply else args.seal_eta_seal,
                    max_seal_scale=args.seal_max_scale,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = seal.update
                seal_scales = seal.seal_scales
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(seal.diagnostics)
            elif args.intrinsic_target_purifier == "spectra":
                spectra = spectra_purify_update(
                    update,
                    keys=selection.keys,
                    targets=targets,
                    weights=positive_weights,
                    all_keys=usable_keys,
                    all_outputs=captures[layer_idx].outputs[: usable_keys.shape[0]],
                    token_indices=selection.token_indices,
                    logit_top_values=sharp_top_values,
                    logit_top_indices=sharp_top_indices,
                    lm_head_indices=sharp_lm_indices,
                    lm_head_rows=sharp_lm_rows,
                    layer=layer,
                    negative_keys=negative_keys,
                    output_basis=karp_output_basis,
                    quotient_rank=args.spectra_quotient_rank,
                    contrast_rank=args.spectra_contrast_rank,
                    tail_anchors=args.spectra_tail_anchors,
                    tail_quantile=args.spectra_tail_quantile,
                    hazard_rank=args.spectra_hazard_rank,
                    hazard_budget=args.spectra_hazard_budget,
                    beta_tail=args.spectra_beta_tail,
                    beta_hazard=args.spectra_beta_hazard,
                    generic_key_rank=args.spectra_generic_key_rank,
                    option_top_k=args.spectra_option_top_k,
                    low_surprise_quantile=args.karp_low_surprise_quantile,
                    input_metric_weight=args.intrinsic_surprise_input_penalty_weight,
                    quotient_mode=args.qrico_quotient_mode,
                    use_orca_quotient=not args.spectra_no_orca_quotient,
                    ablation_mode=args.spectra_ablation,
                    ridge=args.qrico_cca_ridge,
                    risk_ratio_cap=args.karp_risk_ratio_cap,
                )
                update = spectra.update
                fit = selection.keys.detach().float() @ update.T
                stats.fit_rmse = float(torch.sqrt(torch.mean((fit - targets.detach().float()).square())).item())
                stats.update_fro = float(torch.linalg.vector_norm(update).item())
                if negative_keys is not None and negative_keys.numel() > 0:
                    neg_fit = negative_keys.detach().float() @ update.T
                    stats.negative_rmse = float(torch.sqrt(torch.mean(neg_fit.square())).item())
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(spectra.diagnostics)
            if args.dice_defer_apply:
                dice_deferred_updates.setdefault(layer_idx, []).append(update.detach().float().cpu())
                append_jsonl(
                    updates_path,
                    {
                        "lesson_idx": lesson_idx,
                        "layer": layer_idx,
                        "module": "mlp_down",
                        "write_mode": "intrinsic_surprise",
                        "ensemble_phase": "dice_proposal",
                        "lesson_token_count": int(keys.shape[0]),
                        "selected_rows": int(selection.keys.shape[0]),
                        "selected_token_min": int(selection.token_indices.min().item()),
                        "selected_token_max": int(selection.token_indices.max().item()),
                        "row_score_mean": float(selection.row_scores.mean().item()),
                        "row_score_max": float(selection.row_scores.max().item()),
                        "intrinsic_target_purifier": args.intrinsic_target_purifier,
                        "intrinsic_target_mode": args.intrinsic_surprise_target_mode,
                        "intrinsic_relation_value_mode": args.intrinsic_surprise_relation_value_mode,
                        "dice_defer_apply": True,
                        "update_fro": float(torch.linalg.vector_norm(update).item()),
                        **(selection.diagnostics or {}),
                        **stats.__dict__,
                    },
                )
                continue
            if args.memory_gate:
                wrapper.set_gate_last_token_only_(args.memory_gate_final_token_only)
                wrapper.set_gate_keys_(
                    selection.keys,
                    threshold=args.memory_gate_threshold,
                    temperature=args.memory_gate_temperature,
                    append=True,
                )
            wrapper.add_memory_(update, slot_id=slot_id)
            if seal_scales is not None and not args.seal_disable_apply:
                seal_apply_stats = apply_mlp_gauge_seal_(layer, wrapper, seal_scales)
                if selection.diagnostics is None:
                    selection.diagnostics = {}
                selection.diagnostics.update(seal_apply_stats)
            if selected_keys_out_by_layer is not None:
                selected_keys_out_by_layer.setdefault(layer_idx, []).append(selection.keys.detach().float().cpu())
            append_jsonl(
                updates_path,
                {
                    "lesson_idx": lesson_idx,
                    "layer": layer_idx,
                    "module": "mlp_down",
                    "write_mode": "intrinsic_surprise",
                    "lesson_token_count": int(keys.shape[0]),
                    "selected_rows": int(selection.keys.shape[0]),
                    "selected_token_min": int(selection.token_indices.min().item()),
                    "selected_token_max": int(selection.token_indices.max().item()),
                    "row_score_mean": float(selection.row_scores.mean().item()),
                    "row_score_max": float(selection.row_scores.max().item()),
                    "target_key_fro": float(torch.linalg.vector_norm(selection.target_keys).item()),
                    "uncapped_target_fro": uncapped_target_fro,
                    "target_center_norm": target_center_norm,
                    "intrinsic_center_targets": bool(args.intrinsic_surprise_center_targets),
                    "intrinsic_target_row_norm_cap": args.intrinsic_surprise_target_row_norm_cap,
                    "intrinsic_target_purifier": args.intrinsic_target_purifier,
                    "karp_key_rank_arg": args.karp_key_rank,
                    "karp_value_rank_arg": args.karp_value_rank,
                    "karp_low_surprise_quantile": args.karp_low_surprise_quantile,
                    "karp_eta_cross_arg": args.karp_eta_cross,
                    "karp_eta_key_arg": args.karp_eta_key,
                    "karp_eta_value_arg": args.karp_eta_value,
                    "karp_risk_ratio_cap": args.karp_risk_ratio_cap,
                    "karp_local_fisher_rank_arg": args.karp_local_fisher_rank,
                    "karp_local_fisher_top_k": args.karp_local_fisher_top_k,
                    "karp_local_fisher_max_positions": args.karp_local_fisher_max_positions,
                    "karp_local_fisher_basis_rank": (
                        int(karp_local_output_basis.shape[0])
                        if karp_local_output_basis is not None
                        else 0
                    ),
                    "karp_output_basis_rank": (
                        int(karp_output_basis.shape[0]) if karp_output_basis is not None else 0
                    ),
                    "karp_layer_risk_budget": args.karp_layer_risk_budget,
                    "sharp_shadow_anchors_arg": args.sharp_shadow_anchors,
                    "sharp_key_rank_arg": args.sharp_key_rank,
                    "sharp_value_rank_arg": args.sharp_value_rank,
                    "sharp_signal_top_k_arg": args.sharp_signal_top_k,
                    "sharp_low_surprise_quantile": args.sharp_low_surprise_quantile,
                    "sharp_confidence_quantile": args.sharp_confidence_quantile,
                    "sharp_eta_arg": args.sharp_eta,
                    "sharp_shadow_weight_arg": args.sharp_shadow_weight,
                    "sharp_karp_kappa_arg": args.sharp_karp_kappa,
                    "sharp_shadow_temperature": args.sharp_shadow_temperature,
                    "sharp_solve_mode": args.sharp_solve_mode,
                    "orca_key_rank_arg": args.orca_key_rank,
                    "orca_value_rank_arg": args.orca_value_rank,
                    "orca_option_top_k": args.orca_option_top_k,
                    "orca_object_rank": args.orca_object_rank,
                    "orca_off_object_rank": args.orca_off_object_rank,
                    "orca_eta_orth": args.orca_eta_orth,
                    "orca_eta_posture": args.orca_eta_posture,
                    "orca_eta_off_object": args.orca_eta_off_object,
                    "orca_eta_karp": args.orca_eta_karp,
                    "orca_signal_floor_quantile": args.orca_signal_floor_quantile,
                    "orca_ablation_mode": args.orca_ablation_mode,
                    "orca_ablation_fraction": args.orca_ablation_fraction,
                    "orca_nuisance_ridge": args.orca_nuisance_ridge,
                    "qrico_deflate_key_rank_arg": args.qrico_deflate_key_rank,
                    "qrico_deflate_value_rank_arg": args.qrico_deflate_value_rank,
                    "qrico_rank_arg": args.qrico_rank,
                    "qrico_option_sketch_rank": args.qrico_option_sketch_rank,
                    "qrico_target_parallel_rank": args.qrico_target_parallel_rank,
                    "qrico_scramble_weight_arg": args.qrico_scramble_weight,
                    "qrico_residual_row_weight_power": args.qrico_residual_row_weight_power,
                    "qrico_quotient_mode": args.qrico_quotient_mode,
                    "qrico_solve_mode": args.qrico_solve_mode,
                    "qrico_cca_ridge": args.qrico_cca_ridge,
                    "qrico_layer_evidence_min": args.qrico_layer_evidence_min,
                    "qrico_layer_evidence_target": args.qrico_layer_evidence_target,
                    "qrico_disable_layer_trust": bool(args.qrico_disable_layer_trust),
                    "prism_horizon": args.prism_horizon,
                    "prism_signal_rank": args.prism_signal_rank,
                    "prism_hazard_rank": args.prism_hazard_rank,
                    "prism_option_top_k": args.prism_option_top_k,
                    "prism_generic_key_rank": args.prism_generic_key_rank,
                    "prism_low_surprise_rows": args.prism_low_surprise_rows,
                    "prism_budget": args.prism_budget,
                    "prism_correction_cap": args.prism_correction_cap,
                    "prism_signal_retention_min": args.prism_signal_retention_min,
                    "prism_no_residualize_hazard": bool(args.prism_no_residualize_hazard),
                    "prism_disable_future": bool(args.prism_disable_future),
                    "prism_ablation": args.prism_ablation,
                    "trace_object_endpoints": args.trace_object_endpoints,
                    "trace_ambient_endpoints": args.trace_ambient_endpoints,
                    "trace_option_top_k": args.trace_option_top_k,
                    "trace_option_contrasts": args.trace_option_contrasts,
                    "trace_object_rank": args.trace_object_rank,
                    "trace_ambient_rank": args.trace_ambient_rank,
                    "trace_generic_key_rank": args.trace_generic_key_rank,
                    "trace_target_tau": args.trace_target_tau,
                    "trace_target_floor": args.trace_target_floor,
                    "trace_gamma": args.trace_gamma,
                    "trace_layer_trust_threshold": args.trace_layer_trust_threshold,
                    "trace_vjp_mode": args.trace_vjp_mode,
                    "seal_eta_erase": args.seal_eta_erase,
                    "seal_eta_seal": args.seal_eta_seal,
                    "seal_max_scale": args.seal_max_scale,
                    "seal_salience_tau": args.seal_salience_tau,
                    "seal_disable_apply": bool(args.seal_disable_apply),
                    "seal_canonicalize_surprise": bool(args.seal_canonicalize_surprise),
                    "spectra_contrast_rank": args.spectra_contrast_rank,
                    "spectra_tail_anchors": args.spectra_tail_anchors,
                    "spectra_tail_quantile": args.spectra_tail_quantile,
                    "spectra_hazard_rank": args.spectra_hazard_rank,
                    "spectra_hazard_budget": args.spectra_hazard_budget,
                    "spectra_beta_tail": args.spectra_beta_tail,
                    "spectra_beta_hazard": args.spectra_beta_hazard,
                    "spectra_generic_key_rank": args.spectra_generic_key_rank,
                    "spectra_quotient_rank": args.spectra_quotient_rank,
                    "spectra_option_top_k": args.spectra_option_top_k,
                    "spectra_no_orca_quotient": bool(args.spectra_no_orca_quotient),
                    "spectra_ablation": args.spectra_ablation,
                    "ocep_object_rank": args.ocep_object_rank,
                    "ocep_generic_rank": args.ocep_generic_rank,
                    "ocep_option_rank": args.ocep_option_rank,
                    "ocep_option_output_rank": args.ocep_option_output_rank,
                    "ocep_option_local_rank": args.ocep_option_local_rank,
                    "ocep_low_surprise_rank": args.ocep_low_surprise_rank,
                    "ocep_weight_anchor_rank": args.ocep_weight_anchor_rank,
                    "ocep_protected_rank": args.ocep_protected_rank,
                    "ocep_ridge": args.ocep_ridge,
                    "ocep_correction_cap": args.ocep_correction_cap,
                    "ocep_conflict_skip": args.ocep_conflict_skip,
                    "intrinsic_target_mode": args.intrinsic_surprise_target_mode,
                    "intrinsic_value_source": args.intrinsic_surprise_value_source,
                    "intrinsic_effective_target_norm": args.intrinsic_surprise_effective_target_norm,
                    "intrinsic_weight_mode": args.intrinsic_surprise_weight_mode,
                    "intrinsic_exp_temperature": args.intrinsic_surprise_exp_temperature,
                    "intrinsic_exp_cap": args.intrinsic_surprise_exp_cap,
                    "intrinsic_prediction_ridge": args.intrinsic_surprise_prediction_ridge,
                    "intrinsic_pair_top_k": args.intrinsic_surprise_pair_top_k,
                    "intrinsic_bidirectional_pairs": bool(args.intrinsic_surprise_bidirectional_pairs),
                    "intrinsic_relation_value_mode": args.intrinsic_surprise_relation_value_mode,
                    "wicr_compatibility_threshold": getattr(args, "wicr_compatibility_threshold", 0.0),
                    "wicr_compatibility_temperature": getattr(args, "wicr_compatibility_temperature", 0.0),
                    "wicr_posture_pcs": getattr(args, "wicr_posture_pcs", 0),
                    "wicr_target_vector_mode": getattr(args, "wicr_target_vector_mode", ""),
                    "wicr_attention_edges": getattr(args, "wicr_attention_edges", 0),
                    "wicr_attention_flow_mode": getattr(args, "wicr_attention_flow_mode", ""),
                    "wicr_include_same_token_edges": not bool(getattr(args, "wicr_no_same_token_edges", False)),
                    "cori_feature_top_k": getattr(args, "cori_feature_top_k", 0),
                    "cori_relation_rank": getattr(args, "cori_relation_rank", 0),
                    "cori_beta": getattr(args, "cori_beta", 0.0),
                    "cori_edge_top_k": getattr(args, "cori_edge_top_k", 0),
                    "cori_edge_attention_scale": getattr(args, "cori_edge_attention_scale", 0.0),
                    "cori_sinkhorn_steps": getattr(args, "cori_sinkhorn_steps", 0),
                    "cori_target_mode": getattr(args, "cori_target_mode", ""),
                    "star_object_summary_gain": getattr(args, "star_object_summary_gain", 0.0),
                    "star_future_layer_horizon": getattr(args, "star_future_layer_horizon", 0),
                    "star_future_token_top_k": getattr(args, "star_future_token_top_k", 0),
                    "star_future_layer_decay": getattr(args, "star_future_layer_decay", 0.0),
                    "star_future_token_decay": getattr(args, "star_future_token_decay", 0.0),
                    "star_future_relation_power": getattr(args, "star_future_relation_power", 0.0),
                    "star_ordinary_key_rank": getattr(args, "star_ordinary_key_rank", 0),
                    "star_value_projection_features": getattr(args, "star_value_projection_features", 0),
                    "star_value_projection_ridge": getattr(args, "star_value_projection_ridge", 0.0),
                    "star_schur_ridge": getattr(args, "star_schur_ridge", 0.0),
                    "star_map_ridge": getattr(args, "star_map_ridge", 0.0),
                    "star_posture_negative_scale": getattr(args, "star_posture_negative_scale", 0.0),
                    "star_min_coherence": getattr(args, "star_min_coherence", 0.0),
                    "star_shuffle_future_targets": bool(getattr(args, "star_shuffle_future_targets", False)),
                    "star_shuffle_keys": bool(getattr(args, "star_shuffle_keys", False)),
                    "intrinsic_persistence_power": args.intrinsic_surprise_persistence_power,
                    "intrinsic_generic_rank": args.intrinsic_surprise_generic_rank,
                    "intrinsic_lm_head_generic_rank": args.intrinsic_surprise_lm_head_generic_rank,
                    "intrinsic_output_penalty_rank": args.intrinsic_surprise_output_penalty_rank,
                    "intrinsic_output_penalty_weight": args.intrinsic_surprise_output_penalty_weight,
                    "intrinsic_input_penalty_features": args.intrinsic_surprise_input_penalty_features,
                    "intrinsic_input_penalty_weight": args.intrinsic_surprise_input_penalty_weight,
                    "intrinsic_input_penalty_usage_power": args.intrinsic_surprise_input_penalty_usage_power,
                    "intrinsic_input_penalty_mode": args.intrinsic_surprise_input_penalty_mode,
                    "intrinsic_input_penalty_rows": (
                        int(input_penalty_keys.shape[0]) if input_penalty_keys is not None else 0
                    ),
                    "intrinsic_selection_negative_rows": (
                        int(selection_negative_keys.shape[0])
                        if selection_negative_keys is not None and selection_negative_keys.numel() > 0
                        else 0
                    ),
                    "intrinsic_extra_negative_rows": (
                        int(evenly_cap_rows(extra_negative, max_extra_negative_rows).shape[0])
                        if extra_negative is not None and extra_negative.numel() > 0
                        else 0
                    ),
                    "intrinsic_negative_rows_total": (
                        int(negative_keys.shape[0]) if negative_keys is not None else 0
                    ),
                    "intrinsic_specificity_power": args.intrinsic_surprise_specificity_power,
                    "intrinsic_readout_specificity_power": args.intrinsic_surprise_readout_specificity_power,
                    "intrinsic_project_generic": bool(args.intrinsic_surprise_project_generic),
                    "selected_feature_count": (
                        int(selection.feature_indices.unique().numel())
                        if selection.feature_indices is not None
                        else 0
                    ),
                    "token_loss_mean": (
                        float(token_losses.mean().item())
                        if args.intrinsic_surprise_target_mode == "logit_error" and token_losses.numel()
                        else 0.0
                    ),
                    "selected_token_loss_mean": (
                        float(token_losses[selection.token_indices.clamp_max(token_losses.shape[0] - 1)].mean().item())
                        if args.intrinsic_surprise_target_mode == "logit_error" and token_losses.numel()
                        else 0.0
                    ),
                    **(selection.diagnostics or {}),
                    **stats.__dict__,
                },
            )
            if args.intrinsic_span_readout_bridge:
                readout_selection = intrinsic_span_readout_selection(
                    model,
                    tokenizer,
                    prompt,
                    lesson_text,
                    input_ids,
                    usable_keys,
                    seed=args.seed + lesson_idx * 9173 + layer_idx,
                    max_items=args.intrinsic_span_readout_max_items,
                    target_scale=args.intrinsic_span_readout_scale,
                )
                if (
                    readout_selection is not None
                    and readout_selection.keys.numel() > 0
                    and readout_selection.targets.shape[1] == down_weight.shape[0]
                ):
                    readout_input_penalty_keys = intrinsic_input_penalty_keys(
                        readout_selection.keys,
                        effective_down_weight(wrapper),
                        output_penalty_basis,
                        args.intrinsic_surprise_input_penalty_features,
                        usage_power=args.intrinsic_surprise_input_penalty_usage_power,
                        mode=args.intrinsic_surprise_input_penalty_mode,
                    )
                    readout_negative_keys = merge_negative_keys(
                        readout_input_penalty_keys,
                        extra_negative,
                        max_extra_rows=max_extra_negative_rows,
                        extra_scale=extra_negative_scale,
                    )
                    if use_metric_solve:
                        readout_update, readout_stats = protected_metric_update(
                            readout_selection.keys,
                            readout_selection.targets,
                            negative_keys=readout_negative_keys,
                            output_penalty_basis=output_penalty_basis,
                            positive_weights=readout_selection.weights,
                            ridge=args.ridge,
                            negative_weight=args.intrinsic_surprise_input_penalty_weight,
                            output_penalty_weight=args.intrinsic_surprise_output_penalty_weight,
                            eta=args.eta,
                            max_update_norm=args.max_update_norm,
                        )
                    else:
                        readout_update, readout_stats = protected_ridge_update(
                            readout_selection.keys,
                            readout_selection.targets,
                            negative_keys=readout_negative_keys,
                            positive_weights=readout_selection.weights,
                            ridge=args.ridge,
                            negative_weight=args.intrinsic_surprise_input_penalty_weight,
                            eta=args.eta,
                            max_update_norm=args.max_update_norm,
                    )
                    wrapper.add_memory_(readout_update, slot_id=slot_id)
                    if selected_keys_out_by_layer is not None:
                        selected_keys_out_by_layer.setdefault(layer_idx, []).append(
                            readout_selection.keys.detach().float().cpu()
                        )
                    append_jsonl(
                        updates_path,
                        {
                            "lesson_idx": lesson_idx,
                            "layer": layer_idx,
                            "module": "mlp_down",
                            "write_mode": "intrinsic_surprise",
                            "channel": "span_readout_bridge",
                            "lesson_token_count": int(keys.shape[0]),
                            "selected_rows": int(readout_selection.keys.shape[0]),
                            "selected_token_min": int(readout_selection.token_indices.min().item()),
                            "selected_token_max": int(readout_selection.token_indices.max().item()),
                            "row_score_mean": float(readout_selection.row_scores.mean().item()),
                            "row_score_max": float(readout_selection.row_scores.max().item()),
                            "target_key_fro": float(torch.linalg.vector_norm(readout_selection.target_keys).item()),
                            "readout_target_fro": float(torch.linalg.vector_norm(readout_selection.targets).item()),
                            "intrinsic_target_mode": args.intrinsic_surprise_target_mode,
                            "intrinsic_span_readout_scale": args.intrinsic_span_readout_scale,
                            "intrinsic_span_readout_max_items": args.intrinsic_span_readout_max_items,
                            "intrinsic_output_penalty_rank": args.intrinsic_surprise_output_penalty_rank,
                            "intrinsic_output_penalty_weight": args.intrinsic_surprise_output_penalty_weight,
                            "intrinsic_input_penalty_features": args.intrinsic_surprise_input_penalty_features,
                            "intrinsic_input_penalty_weight": args.intrinsic_surprise_input_penalty_weight,
                            "intrinsic_input_penalty_usage_power": args.intrinsic_surprise_input_penalty_usage_power,
                            "intrinsic_input_penalty_mode": args.intrinsic_surprise_input_penalty_mode,
                            "intrinsic_input_penalty_rows": (
                                int(readout_input_penalty_keys.shape[0])
                                if readout_input_penalty_keys is not None
                                else 0
                            ),
                            "intrinsic_extra_negative_rows": (
                                int(evenly_cap_rows(extra_negative, max_extra_negative_rows).shape[0])
                                if extra_negative is not None and extra_negative.numel() > 0
                                else 0
                            ),
                            "intrinsic_negative_rows_total": (
                                int(readout_negative_keys.shape[0]) if readout_negative_keys is not None else 0
                            ),
                            **readout_stats.__dict__,
                        },
                    )

    if args.dice_defer_apply:
        for layer_idx, updates in dice_deferred_updates.items():
            if layer_idx not in wrappers or not updates:
                continue
            final_update, dice_stats = dice_support_consensus_update(
                updates,
                support_threshold=args.dice_support_threshold,
                support_temperature=args.dice_support_temperature,
                support_strength=args.dice_support_strength,
                support_cap=args.dice_support_cap,
                support_floor=args.dice_support_floor,
            )
            wrappers[layer_idx].add_memory_(final_update, slot_id=slot_id)
            append_jsonl(
                updates_path,
                {
                    "lesson_idx": -1,
                    "layer": layer_idx,
                    "module": "mlp_down",
                    "write_mode": "intrinsic_surprise",
                    "ensemble_phase": "dice_applied",
                    "count": len(updates),
                    "reduction": "dice_support_consensus",
                    "update_fro": float(torch.linalg.vector_norm(final_update).item()),
                    **dice_stats,
                },
            )


def main() -> None:
    args = parse_args()
    attention_projection_names = enabled_attention_projections(args)
    if (
        args.activation_object_gate
        and args.object_gate_mode == "density_ratio"
        and (args.write_attention_o or attention_projection_names)
    ):
        raise ValueError("density_ratio object gates are currently implemented for MLP writes only.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    updates_path = output_dir / "updates.jsonl"
    lessons_path = output_dir / "lessons.jsonl"
    questions_path = output_dir / "eval_questions.jsonl"
    eval_details_path = output_dir / "eval_details.jsonl"
    for path in (metrics_path, updates_path, lessons_path, questions_path, eval_details_path):
        if path.exists():
            path.unlink()

    model, tokenizer, device = load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        attn_implementation=args.attn_implementation or None,
    )
    wrappers = install_additive_memory(model, args.layers, memory_dtype=torch.float32) if args.write_mlp else {}
    attention_wrappers = (
        install_additive_attention_memory(model, args.layers, memory_dtype=torch.float32)
        if args.write_attention_o
        else {}
    )
    attention_projection_wrappers = {
        projection: install_additive_attention_projection_memory(
            model,
            args.layers,
            projection,
            memory_dtype=torch.float32,
        )
        for projection in attention_projection_names
    }

    lesson_texts = [
        render_lesson(
            idx,
            args.lesson_examples,
            args.seed,
            language_idx=min(idx, args.freeze_language_after)
            if args.freeze_language_after is not None
            else None,
        )
        for idx in range(args.lessons)
    ]
    for idx, text in enumerate(lesson_texts):
        append_jsonl(lessons_path, {"lesson_idx": idx, "text": text})

    final_lesson_idx = args.lessons - 1
    final_language_idx = (
        min(final_lesson_idx, args.freeze_language_after)
        if args.freeze_language_after is not None
        else final_lesson_idx
    )
    slot_id: int | None = None
    if args.term_slot_gate:
        slot_terms = language_terms_for_lesson(final_language_idx)
        slot_ids: list[int] = []
        for wrapper_group in (wrappers, attention_wrappers, *attention_projection_wrappers.values()):
            for wrapper in wrapper_group.values():
                slot_ids.append(wrapper.add_slot_(slot_terms))
        if slot_ids:
            if len(set(slot_ids)) != 1:
                raise RuntimeError(f"Term slot ids diverged across wrappers: {slot_ids}")
            slot_id = slot_ids[0]
    eval_questions, eval_metadata = build_eval_questions(args, lesson_texts)
    for question in eval_questions:
        append_jsonl(questions_path, {
            "sentence": question.sentence,
            "answer": question.answer,
            "options": question.options,
            "answer_letter": question.answer_letter,
            "category": question.category,
        })

    full_context = "\n\n".join(lesson_texts)
    started = time.time()
    baseline = evaluate_mc(
        model,
        tokenizer,
        eval_questions,
        device,
        context=None,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    context = evaluate_mc(
        model,
        tokenizer,
        eval_questions,
        device,
        context=full_context,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    sentinels = sentinel_questions(args.sentinel_suite) if args.sentinel_eval else []
    sentinel_before = (
        evaluate_generic_mc(model, tokenizer, sentinels, device, args.max_length, args.chat_template)
        if sentinels
        else None
    )
    row = {"stage": "before_write", "seconds": time.time() - started}
    row.update(eval_metadata)
    add_metrics(row, "baseline", baseline)
    add_metrics(row, "context", context)
    if sentinel_before is not None:
        add_metrics(row, "sentinel_before", sentinel_before)
    append_jsonl(metrics_path, row)
    for stage, metrics in (("baseline", baseline), ("context", context)):
        for idx, detail in enumerate(metrics["details"]):
            append_jsonl(eval_details_path, {"stage": stage, "idx": idx, **detail})
    if sentinel_before is not None:
        for idx, detail in enumerate(sentinel_before["details"]):
            append_jsonl(eval_details_path, {"stage": "sentinel_before", "idx": idx, **detail})
    if args.skip_write:
        print(f"Wrote mini-language metrics to {metrics_path}")
        return

    if args.intrinsic_surprise_write:
        if not args.write_mlp:
            raise ValueError("--intrinsic-surprise-write currently requires MLP writes.")
        run_intrinsic_surprise_writes(
            model,
            tokenizer,
            wrappers,
            lesson_texts,
            args,
            device,
            updates_path,
            slot_id=slot_id,
        )
        edited = evaluate_mc(
            model,
            tokenizer,
            eval_questions,
            device,
            context=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        sentinel_after = (
            evaluate_generic_mc(model, tokenizer, sentinels, device, args.max_length, args.chat_template)
            if sentinels
            else None
        )
        final_row = {"stage": "after_write", "seconds": time.time() - started, "write_mode": "intrinsic_surprise"}
        final_row.update(eval_metadata)
        add_metrics(final_row, "baseline", baseline)
        add_metrics(final_row, "context", context)
        add_metrics(final_row, "edited", edited)
        if sentinel_before is not None and sentinel_after is not None:
            add_metrics(final_row, "sentinel_before", sentinel_before)
            add_metrics(final_row, "sentinel_after", sentinel_after)
            final_row["sentinel_accuracy_delta"] = sentinel_after["accuracy"] - sentinel_before["accuracy"]
            final_row["sentinel_margin_delta"] = sentinel_after["mean_margin"] - sentinel_before["mean_margin"]
            add_sentinel_shift_metrics(final_row, sentinel_before, sentinel_after)
        final_row["accuracy_delta"] = edited["accuracy"] - baseline["accuracy"]
        final_row["internalization_ratio"] = (
            (edited["accuracy"] - baseline["accuracy"])
            / (context["accuracy"] - baseline["accuracy"] + 1e-12)
        )
        append_jsonl(metrics_path, final_row)
        for idx, detail in enumerate(edited["details"]):
            append_jsonl(eval_details_path, {"stage": "edited", "idx": idx, **detail})
        if sentinel_after is not None:
            for idx, detail in enumerate(sentinel_after["details"]):
                append_jsonl(eval_details_path, {"stage": "sentinel_after", "idx": idx, **detail})
        print(f"Wrote mini-language metrics to {metrics_path}")
        print(f"Wrote mini-language updates to {updates_path}")
        return

    guard = guard_prompts(tokenizer, args.chat_template)
    if args.sentinel_negative_keys:
        guard = [*guard, *sentinel_guard_prompts(tokenizer, args.chat_template)]
    guard_captures = capture_layer_io(
        model,
        tokenizer,
        guard,
        args.layers,
        device,
        args.batch_size,
        args.max_length,
        capture_last_tokens=args.trace_last_tokens,
    )
    guard_attn_captures = None
    if args.write_attention_o and args.cache_current_captures:
        guard_attn_captures = capture_attention_io(
            model,
            tokenizer,
            guard,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
    guard_projection_captures = {
        projection: capture_attention_projection_io(
            model,
            tokenizer,
            guard,
            args.layers,
            projection,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        for projection in attention_projection_names
    }
    object_gate_density_neg_mlp = None
    object_gate_diagnostic_mlp = None
    if args.activation_object_gate and args.object_gate_mode == "density_ratio":
        sentinel_gate_captures = capture_layer_io(
            model,
            tokenizer,
            sentinel_guard_prompts(tokenizer, args.chat_template),
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        rival_gate_captures = None
        if args.object_gate_rival_negatives > 0:
            build_trace_questions = build_balanced_questions if args.balanced_trace else build_questions
            rival_questions = build_trace_questions(
                args.object_gate_rival_negatives,
                args.seed + args.trace_seed_offset + 80_001,
                final_lesson_idx,
                "object_gate_rival",
                language_idx=final_language_idx,
            )
            rival_prompts, rival_indices = language_object_gate_prompts_and_indices(
                tokenizer,
                rival_questions,
                args.chat_template,
                language_name="Vomar",
                token_prefix="vo_",
                max_prompts=0,
                seed=args.seed + 80_977,
            )
            rival_gate_captures = capture_layer_io_at_token_indices(
                model,
                tokenizer,
                rival_prompts,
                rival_indices,
                args.layers,
                device,
                args.max_length,
                capture_window=args.object_gate_token_window,
            )
        object_gate_density_neg_mlp = {
            layer_idx: {
                "default_answer": guard_captures[layer_idx].keys,
                "sentinel": sentinel_gate_captures[layer_idx].keys,
                **(
                    {"rival": rival_gate_captures[layer_idx].keys}
                    if rival_gate_captures is not None
                    else {}
                ),
            }
            for layer_idx in args.layers
        }
        if args.object_gate_diagnostics:
            heldout_prompts, heldout_indices = object_gate_prompts_and_indices(
                tokenizer,
                eval_questions,
                args.chat_template,
                max_prompts=0,
                seed=args.seed + 91_977,
            )
            heldout_captures = capture_layer_io_at_token_indices(
                model,
                tokenizer,
                heldout_prompts,
                heldout_indices,
                args.layers,
                device,
                args.max_length,
                capture_window=args.object_gate_token_window,
            )
            object_gate_diagnostic_mlp = {"heldout_eval": heldout_captures}
            if rival_gate_captures is not None:
                object_gate_diagnostic_mlp["rival"] = rival_gate_captures

    if args.ensemble_corpora > 1:
        if args.ensemble_reduction in {"anchored_directional", "anchored_subspace"} and not args.ensemble_include_anchor:
            raise ValueError(f"--ensemble-reduction {args.ensemble_reduction} requires --ensemble-include-anchor")
        build_trace_questions = build_balanced_questions if args.balanced_trace else build_questions
        corpus_bank: list[tuple[int, int, list[str]]] = []
        if args.ensemble_include_anchor:
            corpus_bank.append((-1, args.seed, lesson_texts))
        for corpus_idx in range(args.ensemble_corpora):
            corpus_seed = args.seed + (corpus_idx + 1) * args.ensemble_seed_stride
            corpus_texts = [
                render_lesson(
                    idx,
                    args.lesson_examples,
                    corpus_seed,
                    language_idx=min(idx, args.freeze_language_after)
                    if args.freeze_language_after is not None
                    else None,
                )
                for idx in range(args.lessons)
            ]
            corpus_bank.append((corpus_idx, corpus_seed, corpus_texts))

        write_lesson_indices = range(args.lessons) if args.ensemble_per_lesson else [final_lesson_idx]
        for write_lesson_idx in write_lesson_indices:
            aggregate_updates: dict[tuple[str, int], torch.Tensor] = {}
            aggregate_counts: dict[tuple[str, int], int] = {}
            proposal_updates: dict[tuple[str, int], list[torch.Tensor]] = {}
            anchor_updates: dict[tuple[str, int], torch.Tensor] = {}
            lexical_updates: dict[tuple[str, int], torch.Tensor] = {}
            lexical_trace_rows: dict[tuple[str, int], int] = {}
            context_span_updates: dict[tuple[str, int], torch.Tensor] = {}
            context_span_trace_rows: dict[tuple[str, int], int] = {}
            trace_language_idx = (
                min(write_lesson_idx, args.freeze_language_after)
                if args.freeze_language_after is not None
                else write_lesson_idx
            )
            object_gate_mlp = None
            object_gate_attn = None
            object_gate_projections: dict[str, dict[int, object]] = {}
            if args.activation_object_gate:
                gate_probe_lesson_idx = write_lesson_idx if args.ensemble_per_lesson else final_lesson_idx
                gate_probe_language_idx = trace_language_idx if args.ensemble_per_lesson else final_language_idx
                gate_probe_seed = (
                    args.seed
                    + args.trace_seed_offset
                    + 17_003
                    + (write_lesson_idx * 10_000 if args.ensemble_per_lesson else 0)
                )
                gate_probe_count = args.object_gate_probes if args.object_gate_probes > 0 else args.trace_probes
                gate_questions = build_trace_questions(
                    gate_probe_count,
                    gate_probe_seed,
                    gate_probe_lesson_idx,
                    "object_gate",
                    language_idx=gate_probe_language_idx,
                )
                gate_prompts, gate_token_indices = object_gate_prompts_and_indices(
                    tokenizer,
                    gate_questions,
                    args.chat_template,
                    max_prompts=args.object_gate_max_prompts,
                    seed=gate_probe_seed + 977,
                )
                if args.write_mlp:
                    object_gate_mlp = capture_layer_io_at_token_indices(
                        model,
                        tokenizer,
                        gate_prompts,
                        gate_token_indices,
                        args.layers,
                        device,
                        args.max_length,
                        capture_window=args.object_gate_token_window,
                    )
                if args.write_attention_o:
                    object_gate_attn = capture_attention_io_at_token_indices(
                        model,
                        tokenizer,
                        gate_prompts,
                        gate_token_indices,
                        args.layers,
                        device,
                        args.max_length,
                        capture_window=args.object_gate_token_window,
                    )
                object_gate_projections = {
                    projection: capture_attention_projection_io_at_token_indices(
                        model,
                        tokenizer,
                        gate_prompts,
                        gate_token_indices,
                        args.layers,
                        projection,
                        device,
                        args.max_length,
                        capture_window=args.object_gate_token_window,
                    )
                    for projection in attention_projection_names
                }
            for corpus_idx, corpus_seed, corpus_texts in corpus_bank:
                corpus_started = time.time()
                if args.ensemble_per_lesson:
                    if args.trace_context == "lesson":
                        teacher_context = corpus_texts[write_lesson_idx]
                    elif args.trace_context == "cumulative":
                        teacher_context = "\n\n".join(corpus_texts[: write_lesson_idx + 1])
                    else:
                        teacher_context = "\n\n".join(corpus_texts)
                    probe_lesson_idx = write_lesson_idx
                    probe_language_idx = trace_language_idx
                    shared_probe_seed = args.seed + args.trace_seed_offset + 17_003 + write_lesson_idx * 10_000
                    corpus_probe_seed = corpus_seed + 17_003 + write_lesson_idx * 10_000
                else:
                    teacher_context = "\n\n".join(corpus_texts)
                    probe_lesson_idx = final_lesson_idx
                    probe_language_idx = final_language_idx
                    shared_probe_seed = args.seed + args.trace_seed_offset + 17_003
                    corpus_probe_seed = corpus_seed + 17_003
                probes = build_trace_questions(
                    args.trace_probes,
                    shared_probe_seed if args.ensemble_shared_probes else corpus_probe_seed,
                    probe_lesson_idx,
                    "trace_translation",
                    language_idx=probe_language_idx,
                )
                perspective_filter_prompts: dict[str, list[str]] = {}
                if args.packed_use_trace:
                    packed_items = packed_items_from_questions(probes)
                    if args.packed_use_span_items > 0:
                        packed_items.extend(
                            packed_items_from_context_spans(
                                context_span_traces(
                                    teacher_context,
                                    seed=corpus_seed + write_lesson_idx * 53_921 + 17,
                                    max_items=args.packed_use_span_items,
                                )
                            )
                        )
                    full_prompts = packed_use_prompts(
                        tokenizer,
                        packed_items,
                        teacher_context,
                        args.chat_template,
                        args.teacher_forcing_trace,
                        args.token_teacher_forcing_trace,
                        args.packed_use_mode,
                    )
                    key_prompts = packed_use_prompts(
                        tokenizer,
                        packed_items,
                        None,
                        args.chat_template,
                        args.teacher_forcing_trace,
                        args.token_teacher_forcing_trace,
                        args.packed_use_mode,
                    )
                else:
                    if args.perspective_filter:
                        full_prompts = [
                            prompt
                            for question in probes
                            for prompt in trace_prompts_for_question(
                                tokenizer,
                                question,
                                teacher_context,
                                args.chat_template,
                                args.teacher_forcing_trace,
                                args.token_teacher_forcing_trace,
                                perspective="direct",
                            )
                        ]
                        key_prompts = [
                            prompt
                            for question in probes
                            for prompt in trace_prompts_for_question(
                                tokenizer,
                                question,
                                None,
                                args.chat_template,
                                args.teacher_forcing_trace,
                                args.token_teacher_forcing_trace,
                                perspective="direct",
                            )
                        ]
                        perspective_filter_prompts = {
                            perspective: [
                                prompt
                                for question in probes
                                for prompt in trace_prompts_for_question(
                                    tokenizer,
                                    question,
                                    teacher_context,
                                    args.chat_template,
                                    args.teacher_forcing_trace,
                                    args.token_teacher_forcing_trace,
                                    perspective=perspective,
                                )
                            ]
                            for perspective in args.trace_perspectives
                            if perspective != "direct"
                        }
                    else:
                        full_prompts = [
                            prompt
                            for question in probes
                            for perspective in args.trace_perspectives
                            for prompt in trace_prompts_for_question(
                                tokenizer,
                                question,
                                teacher_context,
                                args.chat_template,
                                args.teacher_forcing_trace,
                                args.token_teacher_forcing_trace,
                                perspective=perspective,
                            )
                        ]
                        key_prompts = [
                            prompt
                            for question in probes
                            for perspective in args.trace_perspectives
                            for prompt in trace_prompts_for_question(
                                tokenizer,
                                question,
                                None,
                                args.chat_template,
                                args.teacher_forcing_trace,
                                args.token_teacher_forcing_trace,
                                perspective="direct" if args.teacher_perspectives_only else perspective,
                            )
                        ]
                option_negative_prompts: list[str] = []
                if args.option_negative_keys:
                    for question in probes:
                        for option in question.options:
                            if option == question.answer:
                                continue
                            option_negative_prompts.extend(
                                trace_prompts_for_answer(
                                    tokenizer,
                                    question,
                                    option,
                                    None,
                                    args.chat_template,
                                    args.teacher_forcing_trace,
                                    args.token_teacher_forcing_trace,
                                )
                            )
                    if len(option_negative_prompts) > args.max_option_negative_prompts:
                        rng = random.Random(corpus_seed + write_lesson_idx * 101)
                        option_negative_prompts = rng.sample(
                            option_negative_prompts,
                            args.max_option_negative_prompts,
                        )
                full_blocks = capture_block_io(
                    model,
                    tokenizer,
                    full_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
                current_blocks = capture_block_io(
                    model,
                    tokenizer,
                    key_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
                current_mlp = (
                    capture_layer_io(
                        model,
                        tokenizer,
                        key_prompts,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    if args.write_mlp
                    else None
                )
                current_attn = (
                    capture_attention_io(
                        model,
                        tokenizer,
                        key_prompts,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    if args.write_attention_o
                    else None
                )
                option_negative_mlp = (
                    capture_layer_io(
                        model,
                        tokenizer,
                        option_negative_prompts,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    if option_negative_prompts and args.write_mlp
                    else None
                )
                option_negative_attn = (
                    capture_attention_io(
                        model,
                        tokenizer,
                        option_negative_prompts,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    if option_negative_prompts and args.write_attention_o and args.option_negative_attention
                    else None
                )
                full_projection_captures = {
                    projection: capture_attention_projection_io(
                        model,
                        tokenizer,
                        full_prompts,
                        args.layers,
                        projection,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    for projection in attention_projection_names
                }
                current_projection_captures = {
                    projection: capture_attention_projection_io(
                        model,
                        tokenizer,
                        key_prompts,
                        args.layers,
                        projection,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    for projection in attention_projection_names
                }
                perspective_filter_blocks: dict[str, dict[int, object]] = {}
                perspective_filter_projection_captures: dict[str, dict[str, dict[int, object]]] = {}
                if perspective_filter_prompts:
                    perspective_filter_blocks = {
                        perspective: capture_block_io(
                            model,
                            tokenizer,
                            prompts,
                            args.layers,
                            device,
                            args.batch_size,
                            args.max_length,
                            capture_last_tokens=args.trace_last_tokens,
                        )
                        for perspective, prompts in perspective_filter_prompts.items()
                    }
                    perspective_filter_projection_captures = {
                        perspective: {
                            projection: capture_attention_projection_io(
                                model,
                                tokenizer,
                                prompts,
                                args.layers,
                                projection,
                                device,
                                args.batch_size,
                                args.max_length,
                                capture_last_tokens=args.trace_last_tokens,
                            )
                            for projection in attention_projection_names
                        }
                        for perspective, prompts in perspective_filter_prompts.items()
                    }
                guard_attn = guard_attn_captures
                if args.write_attention_o and guard_attn is None:
                    guard_attn = capture_attention_io(
                        model,
                        tokenizer,
                        guard,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                lexical_full_blocks = None
                lexical_current_blocks = None
                lexical_current_mlp = None
                lexical_current_attn = None
                lexical_prompts_count = 0
                span_full_blocks = None
                span_current_blocks = None
                span_current_mlp = None
                span_current_attn = None
                span_prompts_count = 0
                span_traces: list[ContextSpanTrace] = []
                if args.lexical_channel and corpus_idx == -1:
                    lexical_traces = lexical_traces_for_lesson(write_lesson_idx, language_idx=trace_language_idx)
                    if args.lexical_channel_max_items > 0 and len(lexical_traces) > args.lexical_channel_max_items:
                        rng = random.Random(args.seed + write_lesson_idx * 9973 + 41)
                        lexical_traces = rng.sample(lexical_traces, args.lexical_channel_max_items)
                    lexical_full_prompts = [
                        prompt
                        for trace in lexical_traces
                        for prompt in trace_prompts_for_lexical_trace(
                            tokenizer,
                            trace,
                            teacher_context,
                            args.chat_template,
                            args.teacher_forcing_trace,
                            args.token_teacher_forcing_trace,
                        )
                    ]
                    lexical_key_prompts = [
                        prompt
                        for trace in lexical_traces
                        for prompt in trace_prompts_for_lexical_trace(
                            tokenizer,
                            trace,
                            None,
                            args.chat_template,
                            args.teacher_forcing_trace,
                            args.token_teacher_forcing_trace,
                        )
                    ]
                    lexical_prompts_count = len(lexical_key_prompts)
                    lexical_full_blocks = capture_block_io(
                        model,
                        tokenizer,
                        lexical_full_prompts,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    lexical_current_blocks = capture_block_io(
                        model,
                        tokenizer,
                        lexical_key_prompts,
                        args.layers,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                    if args.write_mlp:
                        lexical_current_mlp = capture_layer_io(
                            model,
                            tokenizer,
                            lexical_key_prompts,
                            args.layers,
                            device,
                            args.batch_size,
                            args.max_length,
                            capture_last_tokens=args.trace_last_tokens,
                        )
                    if args.write_attention_o and args.lexical_channel_attention:
                        lexical_current_attn = capture_attention_io(
                            model,
                            tokenizer,
                            lexical_key_prompts,
                            args.layers,
                            device,
                            args.batch_size,
                            args.max_length,
                            capture_last_tokens=args.trace_last_tokens,
                        )
                if args.context_span_channel and corpus_idx == -1:
                    span_traces = context_span_traces(
                        teacher_context,
                        seed=args.seed + write_lesson_idx * 31337 + 7,
                        max_items=args.context_span_channel_max_items,
                    )
                    span_full_prompts = [
                        prompt
                        for trace in span_traces
                        for prompt in trace_prompts_for_context_span_trace(
                            tokenizer,
                            trace,
                            teacher_context,
                            args.chat_template,
                            args.teacher_forcing_trace,
                            args.token_teacher_forcing_trace,
                        )
                    ]
                    span_key_prompts = [
                        prompt
                        for trace in span_traces
                        for prompt in trace_prompts_for_context_span_trace(
                            tokenizer,
                            trace,
                            None,
                            args.chat_template,
                            args.teacher_forcing_trace,
                            args.token_teacher_forcing_trace,
                        )
                    ]
                    span_prompts_count = len(span_key_prompts)
                    if span_prompts_count:
                        span_full_blocks = capture_block_io(
                            model,
                            tokenizer,
                            span_full_prompts,
                            args.layers,
                            device,
                            args.batch_size,
                            args.max_length,
                            capture_last_tokens=args.trace_last_tokens,
                        )
                        span_current_blocks = capture_block_io(
                            model,
                            tokenizer,
                            span_key_prompts,
                            args.layers,
                            device,
                            args.batch_size,
                            args.max_length,
                            capture_last_tokens=args.trace_last_tokens,
                        )
                        if args.write_mlp:
                            span_current_mlp = capture_layer_io(
                                model,
                                tokenizer,
                                span_key_prompts,
                                args.layers,
                                device,
                                args.batch_size,
                                args.max_length,
                                capture_last_tokens=args.trace_last_tokens,
                            )
                        if args.write_attention_o and args.context_span_channel_attention:
                            span_current_attn = capture_attention_io(
                                model,
                                tokenizer,
                                span_key_prompts,
                                args.layers,
                                device,
                                args.batch_size,
                                args.max_length,
                                capture_last_tokens=args.trace_last_tokens,
                            )
                for layer_idx in args.layers:
                    if args.target_mode == "source":
                        targets = block_source_targets(
                            full_blocks[layer_idx].inputs,
                            full_blocks[layer_idx].outputs,
                            current_blocks[layer_idx].inputs,
                            current_blocks[layer_idx].outputs,
                        )
                    else:
                        targets = full_blocks[layer_idx].outputs.float() - current_blocks[layer_idx].outputs.float()
                    perspective_target_filter_stats = {}
                    if perspective_filter_blocks and args.perspective_filter_granularity == "target":
                        support_targets_for_filter = []
                        for support_blocks in perspective_filter_blocks.values():
                            if args.target_mode == "source":
                                support_targets = block_source_targets(
                                    support_blocks[layer_idx].inputs,
                                    support_blocks[layer_idx].outputs,
                                    current_blocks[layer_idx].inputs,
                                    current_blocks[layer_idx].outputs,
                                )
                            else:
                                support_targets = (
                                    support_blocks[layer_idx].outputs.float()
                                    - current_blocks[layer_idx].outputs.float()
                                )
                            support_targets_for_filter.append(support_targets)
                        targets, perspective_target_filter_stats = perspective_filtered_targets(
                            targets,
                            support_targets_for_filter,
                            threshold=args.perspective_target_filter_threshold,
                            temperature=args.perspective_target_filter_temperature,
                            floor=args.perspective_target_filter_floor,
                        )
                    if args.shuffle_targets:
                        generator = torch.Generator().manual_seed(corpus_seed + write_lesson_idx * 1000 + layer_idx)
                        targets = targets[torch.randperm(targets.shape[0], generator=generator)]
                    if args.write_attention_o:
                        assert current_attn is not None and guard_attn is not None
                        attn_negative_keys = guard_attn[layer_idx].keys
                        if option_negative_attn is not None and args.option_negative_mode == "ridge":
                            attn_negative_keys = torch.cat(
                                [attn_negative_keys, option_negative_attn[layer_idx].keys],
                                dim=0,
                            )
                        attn_update, attn_stats = protected_ridge_update(
                            current_attn[layer_idx].keys,
                            targets,
                            negative_keys=attn_negative_keys,
                            ridge=args.ridge,
                            negative_weight=args.negative_weight,
                            eta=args.eta,
                            max_update_norm=args.max_update_norm,
                        )
                        option_projection_stats = {}
                        if option_negative_attn is not None and args.option_negative_mode == "project":
                            attn_update, option_projection_stats = suppress_update_on_keys(
                                attn_update,
                                option_negative_attn[layer_idx].keys,
                                strength=args.option_negative_project_strength,
                                ridge=args.option_negative_project_ridge,
                            )
                        perspective_filter_stats = {}
                        if perspective_filter_blocks and args.perspective_filter_granularity == "update":
                            perspective_support_updates = []
                            for support_blocks in perspective_filter_blocks.values():
                                if args.target_mode == "source":
                                    support_targets = block_source_targets(
                                        support_blocks[layer_idx].inputs,
                                        support_blocks[layer_idx].outputs,
                                        current_blocks[layer_idx].inputs,
                                        current_blocks[layer_idx].outputs,
                                    )
                                else:
                                    support_targets = (
                                        support_blocks[layer_idx].outputs.float()
                                        - current_blocks[layer_idx].outputs.float()
                                    )
                                support_update, _support_stats = protected_ridge_update(
                                    current_attn[layer_idx].keys,
                                    support_targets,
                                    negative_keys=attn_negative_keys,
                                    ridge=args.ridge,
                                    negative_weight=args.negative_weight,
                                    eta=args.eta,
                                    max_update_norm=args.max_update_norm,
                                )
                                perspective_support_updates.append(support_update)
                            attn_update, perspective_filter_stats = perspective_filtered_update(
                                attn_update,
                                perspective_support_updates,
                                mode=args.perspective_filter_mode,
                                residual_scale=args.perspective_filter_residual_scale,
                                min_agreement=args.perspective_filter_min_agreement,
                            )
                        key = ("attention_o", layer_idx)
                        aggregate_updates[key] = aggregate_updates.get(key, torch.zeros_like(attn_update)) + attn_update
                        aggregate_counts[key] = aggregate_counts.get(key, 0) + 1
                        if args.ensemble_reduction in {"anchored_directional", "anchored_subspace"} and corpus_idx == -1:
                            anchor_updates[key] = attn_update
                        elif args.ensemble_reduction in {
                            "snr",
                            "directional",
                            "anchored_directional",
                            "subspace",
                            "anchored_subspace",
                        }:
                            proposal_updates.setdefault(key, []).append(attn_update)
                        row = {
                            "corpus_idx": corpus_idx,
                            "lesson_idx": write_lesson_idx,
                            "layer": layer_idx,
                            "module": "attention_o",
                            "trace_rows": int(current_attn[layer_idx].keys.shape[0]),
                            "guard_rows": int(guard_attn[layer_idx].keys.shape[0]),
                            "option_negative_rows": int(
                                option_negative_attn[layer_idx].keys.shape[0]
                                if option_negative_attn is not None
                                else 0
                            ),
                            "seconds": time.time() - corpus_started,
                            "ensemble_phase": "proposal",
                        }
                        row.update(attn_stats.__dict__)
                        row.update(option_projection_stats)
                        row.update(perspective_filter_stats)
                        row.update(perspective_target_filter_stats)
                        append_jsonl(updates_path, row)
                    for projection in attention_projection_names:
                        current_projection = current_projection_captures[projection][layer_idx]
                        full_projection = full_projection_captures[projection][layer_idx]
                        projection_targets = full_projection.outputs.float() - current_projection.outputs.float()
                        if args.shuffle_targets:
                            generator = torch.Generator().manual_seed(
                                args.seed
                                + corpus_seed
                                + write_lesson_idx * 1000
                                + layer_idx * 10
                                + ord(projection)
                            )
                            projection_targets = projection_targets[
                                torch.randperm(projection_targets.shape[0], generator=generator)
                            ]
                        projection_update, projection_stats = protected_ridge_update(
                            current_projection.keys,
                            projection_targets,
                            negative_keys=guard_projection_captures[projection][layer_idx].keys,
                            ridge=args.ridge,
                            negative_weight=args.negative_weight,
                            eta=args.eta * args.attention_projection_eta_scale,
                            max_update_norm=(
                                args.attention_projection_max_update_norm
                                if args.attention_projection_max_update_norm is not None
                                else args.max_update_norm
                            ),
                        )
                        perspective_filter_stats = {}
                        if (
                            perspective_filter_projection_captures
                            and args.perspective_filter_granularity == "update"
                        ):
                            perspective_support_updates = []
                            for support_projection_captures in perspective_filter_projection_captures.values():
                                support_projection = support_projection_captures[projection][layer_idx]
                                support_targets = (
                                    support_projection.outputs.float()
                                    - current_projection.outputs.float()
                                )
                                support_update, _support_stats = protected_ridge_update(
                                    current_projection.keys,
                                    support_targets,
                                    negative_keys=guard_projection_captures[projection][layer_idx].keys,
                                    ridge=args.ridge,
                                    negative_weight=args.negative_weight,
                                    eta=args.eta * args.attention_projection_eta_scale,
                                    max_update_norm=(
                                        args.attention_projection_max_update_norm
                                        if args.attention_projection_max_update_norm is not None
                                        else args.max_update_norm
                                    ),
                                )
                                perspective_support_updates.append(support_update)
                            projection_update, perspective_filter_stats = perspective_filtered_update(
                                projection_update,
                                perspective_support_updates,
                                mode=args.perspective_filter_mode,
                                residual_scale=args.perspective_filter_residual_scale,
                                min_agreement=args.perspective_filter_min_agreement,
                            )
                        key = (f"attention_{projection}", layer_idx)
                        aggregate_updates[key] = (
                            aggregate_updates.get(key, torch.zeros_like(projection_update))
                            + projection_update
                        )
                        aggregate_counts[key] = aggregate_counts.get(key, 0) + 1
                        if args.ensemble_reduction in {"anchored_directional", "anchored_subspace"} and corpus_idx == -1:
                            anchor_updates[key] = projection_update
                        elif args.ensemble_reduction in {
                            "snr",
                            "directional",
                            "anchored_directional",
                            "subspace",
                            "anchored_subspace",
                        }:
                            proposal_updates.setdefault(key, []).append(projection_update)
                        row = {
                            "corpus_idx": corpus_idx,
                            "lesson_idx": write_lesson_idx,
                            "layer": layer_idx,
                            "module": f"attention_{projection}",
                            "trace_rows": int(current_projection.keys.shape[0]),
                            "guard_rows": int(guard_projection_captures[projection][layer_idx].keys.shape[0]),
                            "seconds": time.time() - corpus_started,
                            "ensemble_phase": "proposal",
                            "projection_target_mode": "projection_output_delta",
                            "attention_projection_eta_scale": args.attention_projection_eta_scale,
                        }
                        row.update(projection_stats.__dict__)
                        row.update(perspective_filter_stats)
                        append_jsonl(updates_path, row)
                    if args.write_mlp:
                        assert current_mlp is not None
                        mlp_negative_keys = guard_captures[layer_idx].keys
                        if option_negative_mlp is not None and args.option_negative_mode == "ridge":
                            mlp_negative_keys = torch.cat(
                                [mlp_negative_keys, option_negative_mlp[layer_idx].keys],
                                dim=0,
                            )
                        update, stats = protected_ridge_update(
                            current_mlp[layer_idx].keys,
                            targets,
                            negative_keys=mlp_negative_keys,
                            ridge=args.ridge,
                            negative_weight=args.negative_weight,
                            eta=args.eta,
                            max_update_norm=args.max_update_norm,
                        )
                        option_projection_stats = {}
                        if option_negative_mlp is not None and args.option_negative_mode == "project":
                            update, option_projection_stats = suppress_update_on_keys(
                                update,
                                option_negative_mlp[layer_idx].keys,
                                strength=args.option_negative_project_strength,
                                ridge=args.option_negative_project_ridge,
                            )
                        perspective_filter_stats = {}
                        if perspective_filter_blocks and args.perspective_filter_granularity == "update":
                            perspective_support_updates = []
                            for support_blocks in perspective_filter_blocks.values():
                                if args.target_mode == "source":
                                    support_targets = block_source_targets(
                                        support_blocks[layer_idx].inputs,
                                        support_blocks[layer_idx].outputs,
                                        current_blocks[layer_idx].inputs,
                                        current_blocks[layer_idx].outputs,
                                    )
                                else:
                                    support_targets = (
                                        support_blocks[layer_idx].outputs.float()
                                        - current_blocks[layer_idx].outputs.float()
                                    )
                                support_update, _support_stats = protected_ridge_update(
                                    current_mlp[layer_idx].keys,
                                    support_targets,
                                    negative_keys=mlp_negative_keys,
                                    ridge=args.ridge,
                                    negative_weight=args.negative_weight,
                                    eta=args.eta,
                                    max_update_norm=args.max_update_norm,
                                )
                                perspective_support_updates.append(support_update)
                            update, perspective_filter_stats = perspective_filtered_update(
                                update,
                                perspective_support_updates,
                                mode=args.perspective_filter_mode,
                                residual_scale=args.perspective_filter_residual_scale,
                                min_agreement=args.perspective_filter_min_agreement,
                            )
                        key = ("mlp_down", layer_idx)
                        aggregate_updates[key] = aggregate_updates.get(key, torch.zeros_like(update)) + update
                        aggregate_counts[key] = aggregate_counts.get(key, 0) + 1
                        if args.ensemble_reduction in {"anchored_directional", "anchored_subspace"} and corpus_idx == -1:
                            anchor_updates[key] = update
                        elif args.ensemble_reduction in {
                            "snr",
                            "directional",
                            "anchored_directional",
                            "subspace",
                            "anchored_subspace",
                        }:
                            proposal_updates.setdefault(key, []).append(update)
                        row = {
                            "corpus_idx": corpus_idx,
                            "lesson_idx": write_lesson_idx,
                            "layer": layer_idx,
                            "module": "mlp_down",
                            "trace_rows": int(current_mlp[layer_idx].keys.shape[0]),
                            "guard_rows": int(guard_captures[layer_idx].keys.shape[0]),
                            "option_negative_rows": int(
                                option_negative_mlp[layer_idx].keys.shape[0]
                                if option_negative_mlp is not None
                                else 0
                            ),
                            "seconds": time.time() - corpus_started,
                            "ensemble_phase": "proposal",
                        }
                        row.update(stats.__dict__)
                        row.update(option_projection_stats)
                        row.update(perspective_filter_stats)
                        row.update(perspective_target_filter_stats)
                        append_jsonl(updates_path, row)
                    if args.lexical_channel and corpus_idx == -1:
                        assert lexical_full_blocks is not None and lexical_current_blocks is not None
                        if args.target_mode == "source":
                            lexical_targets = block_source_targets(
                                lexical_full_blocks[layer_idx].inputs,
                                lexical_full_blocks[layer_idx].outputs,
                                lexical_current_blocks[layer_idx].inputs,
                                lexical_current_blocks[layer_idx].outputs,
                            )
                        else:
                            lexical_targets = (
                                lexical_full_blocks[layer_idx].outputs.float()
                                - lexical_current_blocks[layer_idx].outputs.float()
                            )
                        if args.write_attention_o and args.lexical_channel_attention:
                            assert lexical_current_attn is not None and guard_attn is not None
                            lexical_attn_update, lexical_attn_stats = protected_ridge_update(
                                lexical_current_attn[layer_idx].keys,
                                lexical_targets,
                                negative_keys=guard_attn[layer_idx].keys,
                                ridge=args.ridge,
                                negative_weight=args.negative_weight,
                                eta=args.eta,
                                max_update_norm=args.max_update_norm,
                            )
                            key = ("attention_o", layer_idx)
                            lexical_updates[key] = lexical_attn_update
                            lexical_trace_rows[key] = int(lexical_current_attn[layer_idx].keys.shape[0])
                            row = {
                                "corpus_idx": corpus_idx,
                                "lesson_idx": write_lesson_idx,
                                "layer": layer_idx,
                                "module": "attention_o",
                                "channel": "lexical",
                                "trace_rows": int(lexical_current_attn[layer_idx].keys.shape[0]),
                                "lexical_items": len(lexical_traces),
                                "lexical_prompts": lexical_prompts_count,
                                "guard_rows": int(guard_attn[layer_idx].keys.shape[0]),
                                "seconds": time.time() - corpus_started,
                                "ensemble_phase": "lexical_proposal",
                            }
                            row.update(lexical_attn_stats.__dict__)
                            append_jsonl(updates_path, row)
                        if args.write_mlp:
                            assert lexical_current_mlp is not None
                            lexical_mlp_update, lexical_mlp_stats = protected_ridge_update(
                                lexical_current_mlp[layer_idx].keys,
                                lexical_targets,
                                negative_keys=guard_captures[layer_idx].keys,
                                ridge=args.ridge,
                                negative_weight=args.negative_weight,
                                eta=args.eta,
                                max_update_norm=args.max_update_norm,
                            )
                            key = ("mlp_down", layer_idx)
                            lexical_updates[key] = lexical_mlp_update
                            lexical_trace_rows[key] = int(lexical_current_mlp[layer_idx].keys.shape[0])
                            row = {
                                "corpus_idx": corpus_idx,
                                "lesson_idx": write_lesson_idx,
                                "layer": layer_idx,
                                "module": "mlp_down",
                                "channel": "lexical",
                                "trace_rows": int(lexical_current_mlp[layer_idx].keys.shape[0]),
                                "lexical_items": len(lexical_traces),
                                "lexical_prompts": lexical_prompts_count,
                                "guard_rows": int(guard_captures[layer_idx].keys.shape[0]),
                                "seconds": time.time() - corpus_started,
                                "ensemble_phase": "lexical_proposal",
                            }
                            row.update(lexical_mlp_stats.__dict__)
                            append_jsonl(updates_path, row)
                    if args.context_span_channel and corpus_idx == -1 and span_prompts_count:
                        assert span_full_blocks is not None and span_current_blocks is not None
                        if args.target_mode == "source":
                            span_targets = block_source_targets(
                                span_full_blocks[layer_idx].inputs,
                                span_full_blocks[layer_idx].outputs,
                                span_current_blocks[layer_idx].inputs,
                                span_current_blocks[layer_idx].outputs,
                            )
                        else:
                            span_targets = (
                                span_full_blocks[layer_idx].outputs.float()
                                - span_current_blocks[layer_idx].outputs.float()
                            )
                        if args.write_attention_o and args.context_span_channel_attention:
                            assert span_current_attn is not None and guard_attn is not None
                            span_attn_update, span_attn_stats = protected_ridge_update(
                                span_current_attn[layer_idx].keys,
                                span_targets,
                                negative_keys=guard_attn[layer_idx].keys,
                                ridge=args.ridge,
                                negative_weight=args.negative_weight,
                                eta=args.eta,
                                max_update_norm=args.max_update_norm,
                            )
                            key = ("attention_o", layer_idx)
                            context_span_updates[key] = span_attn_update
                            context_span_trace_rows[key] = int(span_current_attn[layer_idx].keys.shape[0])
                            row = {
                                "corpus_idx": corpus_idx,
                                "lesson_idx": write_lesson_idx,
                                "layer": layer_idx,
                                "module": "attention_o",
                                "channel": "context_span",
                                "trace_rows": int(span_current_attn[layer_idx].keys.shape[0]),
                                "context_span_items": len(span_traces),
                                "context_span_prompts": span_prompts_count,
                                "guard_rows": int(guard_attn[layer_idx].keys.shape[0]),
                                "seconds": time.time() - corpus_started,
                                "ensemble_phase": "context_span_proposal",
                            }
                            row.update(span_attn_stats.__dict__)
                            append_jsonl(updates_path, row)
                        if args.write_mlp:
                            assert span_current_mlp is not None
                            span_mlp_update, span_mlp_stats = protected_ridge_update(
                                span_current_mlp[layer_idx].keys,
                                span_targets,
                                negative_keys=guard_captures[layer_idx].keys,
                                ridge=args.ridge,
                                negative_weight=args.negative_weight,
                                eta=args.eta,
                                max_update_norm=args.max_update_norm,
                            )
                            key = ("mlp_down", layer_idx)
                            context_span_updates[key] = span_mlp_update
                            context_span_trace_rows[key] = int(span_current_mlp[layer_idx].keys.shape[0])
                            row = {
                                "corpus_idx": corpus_idx,
                                "lesson_idx": write_lesson_idx,
                                "layer": layer_idx,
                                "module": "mlp_down",
                                "channel": "context_span",
                                "trace_rows": int(span_current_mlp[layer_idx].keys.shape[0]),
                                "context_span_items": len(span_traces),
                                "context_span_prompts": span_prompts_count,
                                "guard_rows": int(guard_captures[layer_idx].keys.shape[0]),
                                "seconds": time.time() - corpus_started,
                                "ensemble_phase": "context_span_proposal",
                            }
                            row.update(span_mlp_stats.__dict__)
                            append_jsonl(updates_path, row)

            for (module, layer_idx), update_sum in aggregate_updates.items():
                count = aggregate_counts[(module, layer_idx)]
                applied_extra: dict[str, float] = {}
                if args.ensemble_reduction == "mean":
                    final_update = update_sum / count
                elif args.ensemble_reduction == "sum":
                    final_update = update_sum
                elif args.ensemble_reduction == "snr":
                    stack = torch.stack(proposal_updates[(module, layer_idx)], dim=0)
                    mean_update = stack.mean(dim=0)
                    std_update = stack.std(dim=0, unbiased=False).clamp_min(1e-8)
                    snr = mean_update.abs() / std_update
                    gate = torch.sigmoid((snr - args.ensemble_snr_threshold) * args.ensemble_snr_temperature)
                    final_update = mean_update * gate
                    applied_extra = {
                        "snr_threshold": args.ensemble_snr_threshold,
                        "snr_temperature": args.ensemble_snr_temperature,
                        "mean_gate": float(gate.mean().item()),
                        "gate_gt_0_5_fraction": float((gate > 0.5).float().mean().item()),
                        "mean_update_fro": float(torch.linalg.vector_norm(mean_update).item()),
                    }
                    applied_extra.update(proposal_alignment_stats(proposal_updates[(module, layer_idx)]))
                elif args.ensemble_reduction == "directional":
                    final_update, applied_extra = directional_consensus_update(
                        proposal_updates[(module, layer_idx)],
                        min_agreement=args.ensemble_directional_min_agreement,
                    )
                    applied_extra.update(proposal_alignment_stats(proposal_updates[(module, layer_idx)]))
                elif args.ensemble_reduction == "anchored_directional":
                    final_update, applied_extra = anchored_directional_update(
                        anchor_updates[(module, layer_idx)],
                        proposal_updates[(module, layer_idx)],
                        residual_scale=args.ensemble_anchor_residual_scale,
                        min_agreement=args.ensemble_directional_min_agreement,
                    )
                    applied_extra.update(proposal_alignment_stats(proposal_updates[(module, layer_idx)]))
                elif args.ensemble_reduction == "subspace":
                    final_update, applied_extra = subspace_consensus_update(
                        proposal_updates[(module, layer_idx)],
                        rank=args.ensemble_subspace_rank,
                    )
                    applied_extra.update(proposal_alignment_stats(proposal_updates[(module, layer_idx)]))
                elif args.ensemble_reduction == "anchored_subspace":
                    final_update, applied_extra = anchored_subspace_update(
                        anchor_updates[(module, layer_idx)],
                        proposal_updates[(module, layer_idx)],
                        rank=args.ensemble_subspace_rank,
                        residual_scale=args.ensemble_anchor_residual_scale,
                    )
                    applied_extra.update(proposal_alignment_stats(proposal_updates[(module, layer_idx)]))
                else:
                    raise ValueError(f"Unknown ensemble reduction: {args.ensemble_reduction}")
                lexical_update = lexical_updates.get((module, layer_idx))
                if lexical_update is not None:
                    grammar_update = final_update
                    scaled_lexical_update = args.lexical_channel_scale * lexical_update
                    final_update = final_update + scaled_lexical_update
                    applied_extra.update(
                        {
                            "lexical_channel_scale": args.lexical_channel_scale,
                            "lexical_trace_rows": lexical_trace_rows.get((module, layer_idx), 0),
                            "lexical_update_fro": float(torch.linalg.vector_norm(lexical_update).item()),
                            "lexical_scaled_update_fro": float(
                                torch.linalg.vector_norm(scaled_lexical_update).item()
                            ),
                            "lexical_grammar_cosine": update_cosine(grammar_update, lexical_update),
                        }
                    )
                span_update = context_span_updates.get((module, layer_idx))
                if span_update is not None:
                    grammar_update = final_update
                    span_component = span_update
                    span_projection_stats: dict[str, float] = {}
                    if args.context_span_channel_mode == "aligned":
                        span_component, span_projection_stats = project_update_onto_reference(
                            span_update,
                            grammar_update,
                            positive_only=True,
                        )
                    scaled_span_update = args.context_span_channel_scale * span_component
                    final_update = final_update + scaled_span_update
                    applied_extra.update(
                        {
                            "context_span_channel_mode": args.context_span_channel_mode,
                            "context_span_channel_scale": args.context_span_channel_scale,
                            "context_span_trace_rows": context_span_trace_rows.get((module, layer_idx), 0),
                            "context_span_update_fro": float(torch.linalg.vector_norm(span_update).item()),
                            "context_span_component_fro": float(torch.linalg.vector_norm(span_component).item()),
                            "context_span_scaled_update_fro": float(
                                torch.linalg.vector_norm(scaled_span_update).item()
                            ),
                            "context_span_grammar_cosine": update_cosine(grammar_update, span_update),
                        }
                    )
                    applied_extra.update(
                        {f"context_span_{key}": value for key, value in span_projection_stats.items()}
                    )
                if module == "attention_o":
                    if args.activation_object_gate:
                        assert object_gate_attn is not None
                        configure_object_gate(attention_wrappers[layer_idx], object_gate_attn[layer_idx].keys, args)
                        applied_extra["object_gate_rows"] = float(object_gate_attn[layer_idx].keys.shape[0])
                    attention_wrappers[layer_idx].add_memory_(final_update, slot_id=slot_id)
                elif module.startswith("attention_"):
                    projection = module.removeprefix("attention_")
                    if args.activation_object_gate:
                        configure_object_gate(
                            attention_projection_wrappers[projection][layer_idx],
                            object_gate_projections[projection][layer_idx].keys,
                            args,
                        )
                        applied_extra["object_gate_rows"] = float(
                            object_gate_projections[projection][layer_idx].keys.shape[0]
                        )
                    attention_projection_wrappers[projection][layer_idx].add_memory_(final_update, slot_id=slot_id)
                else:
                    if args.activation_object_gate:
                        assert object_gate_mlp is not None
                        gate_stats = configure_object_gate(
                            wrappers[layer_idx],
                            object_gate_mlp[layer_idx].keys,
                            args,
                            negative_groups=(
                                object_gate_density_neg_mlp[layer_idx]
                                if object_gate_density_neg_mlp is not None
                                else None
                            ),
                            calibration_keys=(
                                guard_captures[layer_idx].keys
                                if object_gate_density_neg_mlp is not None
                                else None
                            ),
                        )
                        applied_extra["object_gate_rows"] = float(object_gate_mlp[layer_idx].keys.shape[0])
                        applied_extra.update({f"object_gate_{key}": value for key, value in gate_stats.items()})
                    wrappers[layer_idx].add_memory_(final_update, slot_id=slot_id)
                append_jsonl(
                    updates_path,
                    {
                        "ensemble_phase": "applied",
                        "lesson_idx": write_lesson_idx,
                        "module": module,
                        "layer": layer_idx,
                        "count": count,
                        "reduction": args.ensemble_reduction,
                        "update_fro": float(torch.linalg.vector_norm(final_update).item()),
                        **applied_extra,
                    },
                )

        prompt_object_router_stats: dict[str, float] = {}
        if args.object_gate_prompt_router and args.activation_object_gate and args.object_gate_mode == "density_ratio":
            prompt_object_router_stats = install_prompt_object_gate_router(
                model,
                tokenizer,
                wrappers,
                eval_questions,
                sentinels,
                args,
                device,
            )
            append_jsonl(
                updates_path,
                {
                    "ensemble_phase": "prompt_object_router",
                    **prompt_object_router_stats,
                },
            )

        edited = evaluate_mc(
            model,
            tokenizer,
            eval_questions,
            device,
            context=None,
            max_length=args.max_length,
            use_chat_template=args.chat_template,
        )
        sentinel_after = (
            evaluate_generic_mc(model, tokenizer, sentinels, device, args.max_length, args.chat_template)
            if sentinels
            else None
        )
        final_row = {"stage": "after_write", "seconds": time.time() - started}
        final_row.update(eval_metadata)
        final_row.update(prompt_object_router_stats)
        add_metrics(final_row, "baseline", baseline)
        add_metrics(final_row, "context", context)
        add_metrics(final_row, "edited", edited)
        if sentinel_before is not None and sentinel_after is not None:
            add_metrics(final_row, "sentinel_before", sentinel_before)
            add_metrics(final_row, "sentinel_after", sentinel_after)
            final_row["sentinel_accuracy_delta"] = sentinel_after["accuracy"] - sentinel_before["accuracy"]
            final_row["sentinel_margin_delta"] = sentinel_after["mean_margin"] - sentinel_before["mean_margin"]
            add_sentinel_shift_metrics(final_row, sentinel_before, sentinel_after)
        final_row["accuracy_delta"] = edited["accuracy"] - baseline["accuracy"]
        final_row["internalization_ratio"] = (
            (edited["accuracy"] - baseline["accuracy"])
            / (context["accuracy"] - baseline["accuracy"] + 1e-12)
        )
        append_jsonl(metrics_path, final_row)
        for idx, detail in enumerate(edited["details"]):
            append_jsonl(eval_details_path, {"stage": "edited", "idx": idx, **detail})
        if sentinel_after is not None:
            for idx, detail in enumerate(sentinel_after["details"]):
                append_jsonl(eval_details_path, {"stage": "sentinel_after", "idx": idx, **detail})
        print(f"Wrote mini-language metrics to {metrics_path}")
        print(f"Wrote mini-language updates to {updates_path}")
        return

    for lesson_idx, lesson_text in enumerate(lesson_texts):
        if args.write_only_final and lesson_idx != final_lesson_idx:
            continue
        lesson_started = time.time()
        if args.trace_context == "lesson":
            teacher_context = lesson_text
        elif args.trace_context == "cumulative":
            teacher_context = "\n\n".join(lesson_texts[: lesson_idx + 1])
        else:
            teacher_context = full_context
        build_trace_questions = build_balanced_questions if args.balanced_trace else build_questions
        trace_language_idx = (
            min(lesson_idx, args.freeze_language_after)
            if args.freeze_language_after is not None
            else lesson_idx
        )
        probes = build_trace_questions(
            args.trace_probes,
            args.seed + args.trace_seed_offset + lesson_idx * 10_000,
            lesson_idx,
            "trace_translation",
            language_idx=trace_language_idx,
        )
        if args.packed_use_trace:
            packed_items = packed_items_from_questions(probes)
            if args.packed_use_span_items > 0:
                packed_items.extend(
                    packed_items_from_context_spans(
                        context_span_traces(
                            teacher_context,
                            seed=args.seed + lesson_idx * 53_921 + 17,
                            max_items=args.packed_use_span_items,
                        )
                    )
                )
            full_prompts = packed_use_prompts(
                tokenizer,
                packed_items,
                teacher_context,
                args.chat_template,
                args.teacher_forcing_trace,
                args.token_teacher_forcing_trace,
                args.packed_use_mode,
            )
            key_prompts = packed_use_prompts(
                tokenizer,
                packed_items,
                None,
                args.chat_template,
                args.teacher_forcing_trace,
                args.token_teacher_forcing_trace,
                args.packed_use_mode,
            )
        else:
            full_prompts = [
                prompt
                for question in probes
                for perspective in args.trace_perspectives
                for prompt in trace_prompts_for_question(
                    tokenizer,
                    question,
                    teacher_context,
                    args.chat_template,
                    args.teacher_forcing_trace,
                    args.token_teacher_forcing_trace,
                    perspective=perspective,
                )
            ]
            key_prompts = [
                prompt
                for question in probes
                for perspective in args.trace_perspectives
                for prompt in trace_prompts_for_question(
                    tokenizer,
                    question,
                    None,
                    args.chat_template,
                    args.teacher_forcing_trace,
                    args.token_teacher_forcing_trace,
                    perspective="direct" if args.teacher_perspectives_only else perspective,
                )
            ]
        option_negative_prompts: list[str] = []
        if args.option_negative_keys:
            for question in probes:
                for option in question.options:
                    if option == question.answer:
                        continue
                    option_negative_prompts.extend(
                        trace_prompts_for_answer(
                            tokenizer,
                            question,
                            option,
                            None,
                            args.chat_template,
                            args.teacher_forcing_trace,
                            args.token_teacher_forcing_trace,
                        )
                    )
            if len(option_negative_prompts) > args.max_option_negative_prompts:
                rng = random.Random(args.seed + lesson_idx * 101)
                option_negative_prompts = rng.sample(
                    option_negative_prompts,
                    args.max_option_negative_prompts,
                )
        object_gate_mlp = None
        object_gate_attn = None
        object_gate_projections: dict[str, dict[int, object]] = {}
        if args.activation_object_gate:
            gate_probe_count = args.object_gate_probes if args.object_gate_probes > 0 else args.trace_probes
            gate_questions = (
                probes
                if gate_probe_count == args.trace_probes and args.object_gate_probes <= 0
                else build_trace_questions(
                    gate_probe_count,
                    args.seed + args.trace_seed_offset + lesson_idx * 10_000 + 31_337,
                    lesson_idx,
                    "object_gate",
                    language_idx=trace_language_idx,
                )
            )
            gate_prompts, gate_token_indices = object_gate_prompts_and_indices(
                tokenizer,
                gate_questions,
                args.chat_template,
                max_prompts=args.object_gate_max_prompts,
                seed=args.seed + args.trace_seed_offset + lesson_idx * 10_000 + 977,
            )
            if args.write_mlp:
                object_gate_mlp = capture_layer_io_at_token_indices(
                    model,
                    tokenizer,
                    gate_prompts,
                    gate_token_indices,
                    args.layers,
                    device,
                    args.max_length,
                    capture_window=args.object_gate_token_window,
                )
            if args.write_attention_o:
                object_gate_attn = capture_attention_io_at_token_indices(
                    model,
                    tokenizer,
                    gate_prompts,
                    gate_token_indices,
                    args.layers,
                    device,
                    args.max_length,
                    capture_window=args.object_gate_token_window,
                )
            object_gate_projections = {
                projection: capture_attention_projection_io_at_token_indices(
                    model,
                    tokenizer,
                    gate_prompts,
                    gate_token_indices,
                    args.layers,
                    projection,
                    device,
                    args.max_length,
                    capture_window=args.object_gate_token_window,
                )
                for projection in attention_projection_names
            }
        full_blocks = capture_block_io(
            model,
            tokenizer,
            full_prompts,
            args.layers,
            device,
            args.batch_size,
            args.max_length,
            capture_last_tokens=args.trace_last_tokens,
        )
        full_projection_all = {
            projection: capture_attention_projection_io(
                model,
                tokenizer,
                full_prompts,
                args.layers,
                projection,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
            for projection in attention_projection_names
        }
        option_negative_captures = None
        if option_negative_prompts:
            option_negative_captures = capture_layer_io(
                model,
                tokenizer,
                option_negative_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
        current_blocks_all = None
        current_mlp_all = None
        current_attn_all = None
        current_projection_all: dict[str, dict[int, object]] = {}
        if args.cache_current_captures:
            current_blocks_all = capture_block_io(
                model,
                tokenizer,
                key_prompts,
                args.layers,
                device,
                args.batch_size,
                args.max_length,
                capture_last_tokens=args.trace_last_tokens,
            )
            if args.write_mlp:
                current_mlp_all = capture_layer_io(
                    model,
                    tokenizer,
                    key_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
            if args.write_attention_o:
                current_attn_all = capture_attention_io(
                    model,
                    tokenizer,
                    key_prompts,
                    args.layers,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
            current_projection_all = {
                projection: capture_attention_projection_io(
                    model,
                    tokenizer,
                    key_prompts,
                    args.layers,
                    projection,
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
                for projection in attention_projection_names
            }
        updates: dict[int, torch.Tensor] = {}
        for layer_idx in args.layers:
            if current_blocks_all is None:
                current_blocks = capture_block_io(
                    model,
                    tokenizer,
                    key_prompts,
                    [layer_idx],
                    device,
                    args.batch_size,
                    args.max_length,
                    capture_last_tokens=args.trace_last_tokens,
                )
            else:
                current_blocks = current_blocks_all
            if args.write_mlp:
                if current_mlp_all is None:
                    current_mlp = capture_layer_io(
                        model,
                        tokenizer,
                        key_prompts,
                        [layer_idx],
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                else:
                    current_mlp = current_mlp_all
            if args.target_mode == "source":
                targets = block_source_targets(
                    full_blocks[layer_idx].inputs,
                    full_blocks[layer_idx].outputs,
                    current_blocks[layer_idx].inputs,
                    current_blocks[layer_idx].outputs,
                )
            else:
                targets = full_blocks[layer_idx].outputs.float() - current_blocks[layer_idx].outputs.float()
            if args.shuffle_targets:
                generator = torch.Generator().manual_seed(args.seed + lesson_idx * 10_000 + layer_idx)
                targets = targets[torch.randperm(targets.shape[0], generator=generator)]
            negative_keys = guard_captures[layer_idx].keys
            if option_negative_captures is not None and args.option_negative_mode == "ridge":
                negative_keys = torch.cat([negative_keys, option_negative_captures[layer_idx].keys], dim=0)
            energy_keys = current_mlp[layer_idx].keys if args.write_mlp else None
            separation_stats = (
                key_separation_stats(energy_keys, negative_keys)
                if energy_keys is not None
                else {}
            )
            layer_scale = 1.0
            if args.key_separation_filter and separation_stats:
                layer_scale = key_separation_scale(
                    separation_stats,
                    threshold=args.key_separation_threshold,
                    temperature=args.key_separation_temperature,
                    floor=args.key_separation_floor,
                )
            energy_weights, energy_stats = activation_energy_weights(
                args.activation_energy_weighting,
                full_inputs=full_blocks[layer_idx].inputs,
                full_outputs=full_blocks[layer_idx].outputs,
                current_inputs=current_blocks[layer_idx].inputs,
                current_outputs=current_blocks[layer_idx].outputs,
                targets=targets,
                keys=energy_keys,
                floor=args.activation_energy_weight_floor,
                temperature=args.activation_energy_weight_temperature,
                top_k=args.activation_energy_top_k,
            )
            if args.write_attention_o:
                if current_attn_all is None:
                    current_attn = capture_attention_io(
                        model,
                        tokenizer,
                        key_prompts,
                        [layer_idx],
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                else:
                    current_attn = current_attn_all
                if guard_attn_captures is None:
                    guard_attn = capture_attention_io(
                        model,
                        tokenizer,
                        guard,
                        [layer_idx],
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                else:
                    guard_attn = guard_attn_captures
                attn_update, attn_stats = protected_ridge_update(
                    current_attn[layer_idx].keys,
                    targets,
                    negative_keys=guard_attn[layer_idx].keys,
                    positive_weights=energy_weights,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta * layer_scale,
                    max_update_norm=args.max_update_norm,
                )
                if args.memory_gate:
                    attention_wrappers[layer_idx].set_gate_last_token_only_(args.memory_gate_final_token_only)
                    attention_wrappers[layer_idx].set_gate_keys_(
                        current_attn[layer_idx].keys,
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                        append=True,
                    )
                object_gate_rows = 0
                if args.activation_object_gate:
                    assert object_gate_attn is not None
                    configure_object_gate(attention_wrappers[layer_idx], object_gate_attn[layer_idx].keys, args)
                    object_gate_rows = int(object_gate_attn[layer_idx].keys.shape[0])
                attention_wrappers[layer_idx].add_memory_(attn_update, slot_id=slot_id)
                update_row = {
                    "lesson_idx": lesson_idx,
                    "layer": layer_idx,
                    "module": "attention_o",
                    "trace_rows": int(current_attn[layer_idx].keys.shape[0]),
                    "guard_rows": int(guard_attn[layer_idx].keys.shape[0]),
                    "object_gate_rows": object_gate_rows,
                    "option_negative_rows": 0,
                    "seconds": time.time() - lesson_started,
                }
                update_row.update(attn_stats.__dict__)
                update_row.update(energy_stats)
                update_row.update(separation_stats)
                update_row["key_separation_layer_scale"] = layer_scale
                append_jsonl(updates_path, update_row)
            for projection in attention_projection_names:
                if projection in current_projection_all:
                    current_projection = current_projection_all[projection]
                else:
                    current_projection = capture_attention_projection_io(
                        model,
                        tokenizer,
                        key_prompts,
                        [layer_idx],
                        projection,
                        device,
                        args.batch_size,
                        args.max_length,
                        capture_last_tokens=args.trace_last_tokens,
                    )
                projection_targets = (
                    full_projection_all[projection][layer_idx].outputs.float()
                    - current_projection[layer_idx].outputs.float()
                )
                if args.shuffle_targets:
                    generator = torch.Generator().manual_seed(
                        args.seed + lesson_idx * 10_000 + layer_idx * 10 + ord(projection)
                    )
                    projection_targets = projection_targets[
                        torch.randperm(projection_targets.shape[0], generator=generator)
                    ]
                projection_update, projection_stats = protected_ridge_update(
                    current_projection[layer_idx].keys,
                    projection_targets,
                    negative_keys=guard_projection_captures[projection][layer_idx].keys,
                    positive_weights=energy_weights,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta * layer_scale * args.attention_projection_eta_scale,
                    max_update_norm=(
                        args.attention_projection_max_update_norm
                        if args.attention_projection_max_update_norm is not None
                        else args.max_update_norm
                    ),
                )
                if args.memory_gate:
                    attention_projection_wrappers[projection][layer_idx].set_gate_last_token_only_(
                        args.memory_gate_final_token_only
                    )
                    attention_projection_wrappers[projection][layer_idx].set_gate_keys_(
                        current_projection[layer_idx].keys,
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                        append=True,
                    )
                object_gate_rows = 0
                if args.activation_object_gate:
                    configure_object_gate(
                        attention_projection_wrappers[projection][layer_idx],
                        object_gate_projections[projection][layer_idx].keys,
                        args,
                    )
                    object_gate_rows = int(object_gate_projections[projection][layer_idx].keys.shape[0])
                attention_projection_wrappers[projection][layer_idx].add_memory_(projection_update, slot_id=slot_id)
                update_row = {
                    "lesson_idx": lesson_idx,
                    "layer": layer_idx,
                    "module": f"attention_{projection}",
                    "trace_rows": int(current_projection[layer_idx].keys.shape[0]),
                    "guard_rows": int(guard_projection_captures[projection][layer_idx].keys.shape[0]),
                    "object_gate_rows": object_gate_rows,
                    "option_negative_rows": 0,
                    "seconds": time.time() - lesson_started,
                    "projection_target_mode": "projection_output_delta",
                    "attention_projection_eta_scale": args.attention_projection_eta_scale,
                }
                update_row.update(projection_stats.__dict__)
                update_row.update(energy_stats)
                update_row.update(separation_stats)
                update_row["key_separation_layer_scale"] = layer_scale
                append_jsonl(updates_path, update_row)
            if args.write_mlp:
                update, stats = protected_ridge_update(
                    current_mlp[layer_idx].keys,
                    targets,
                    negative_keys=negative_keys,
                    positive_weights=energy_weights,
                    ridge=args.ridge,
                    negative_weight=args.negative_weight,
                    eta=args.eta * layer_scale,
                    max_update_norm=args.max_update_norm,
                )
                option_projection_stats = {}
                if option_negative_captures is not None and args.option_negative_mode == "project":
                    update, option_projection_stats = suppress_update_on_keys(
                        update,
                        option_negative_captures[layer_idx].keys,
                        strength=args.option_negative_project_strength,
                        ridge=args.option_negative_project_ridge,
                    )
                updates[layer_idx] = update
                if args.memory_gate:
                    wrappers[layer_idx].set_gate_last_token_only_(args.memory_gate_final_token_only)
                    wrappers[layer_idx].set_gate_keys_(
                        current_mlp[layer_idx].keys,
                        threshold=args.memory_gate_threshold,
                        temperature=args.memory_gate_temperature,
                        append=True,
                    )
                object_gate_rows = 0
                object_gate_stats = {}
                if args.activation_object_gate:
                    assert object_gate_mlp is not None
                    object_gate_stats = configure_object_gate(
                        wrappers[layer_idx],
                        object_gate_mlp[layer_idx].keys,
                        args,
                        negative_groups=(
                            object_gate_density_neg_mlp[layer_idx]
                            if object_gate_density_neg_mlp is not None
                            else None
                        ),
                        calibration_keys=(
                            guard_captures[layer_idx].keys
                            if object_gate_density_neg_mlp is not None
                            else None
                        ),
                    )
                    object_gate_rows = int(object_gate_mlp[layer_idx].keys.shape[0])
                    if args.object_gate_mode == "density_ratio" and wrappers[layer_idx].object_density_gates:
                        gate_params = wrappers[layer_idx].object_density_gates[-1]
                        object_gate_stats.update(
                            density_gate_group_stats(
                                gate_params,
                                object_gate_mlp[layer_idx].keys,
                                group="train_positive",
                                floor=args.object_gate_floor,
                            )
                        )
                        object_gate_stats.update(
                            density_gate_group_stats(
                                gate_params,
                                current_mlp[layer_idx].keys,
                                group="write_keys",
                                floor=args.object_gate_floor,
                            )
                        )
                        if object_gate_density_neg_mlp is not None:
                            for group_name, group_keys in object_gate_density_neg_mlp[layer_idx].items():
                                object_gate_stats.update(
                                    density_gate_group_stats(
                                        gate_params,
                                        group_keys,
                                        group=f"neg_{group_name}",
                                        floor=args.object_gate_floor,
                                    )
                                )
                        if object_gate_diagnostic_mlp is not None:
                            for group_name, captures in object_gate_diagnostic_mlp.items():
                                object_gate_stats.update(
                                    density_gate_group_stats(
                                        gate_params,
                                        captures[layer_idx].keys,
                                        group=group_name,
                                        floor=args.object_gate_floor,
                                    )
                                )
                wrappers[layer_idx].add_memory_(update, slot_id=slot_id)
                update_row = {
                    "lesson_idx": lesson_idx,
                    "layer": layer_idx,
                    "module": "mlp_down",
                    "trace_rows": int(current_mlp[layer_idx].keys.shape[0]),
                    "guard_rows": int(guard_captures[layer_idx].keys.shape[0]),
                    "object_gate_rows": object_gate_rows,
                    "option_negative_rows": int(
                        option_negative_captures[layer_idx].keys.shape[0]
                        if option_negative_captures is not None
                        else 0
                    ),
                    "seconds": time.time() - lesson_started,
                }
                update_row.update(stats.__dict__)
                update_row.update(energy_stats)
                update_row.update(separation_stats)
                update_row["key_separation_layer_scale"] = layer_scale
                update_row.update(option_projection_stats)
                update_row.update({f"object_gate_{key}": value for key, value in object_gate_stats.items()})
                append_jsonl(updates_path, update_row)

    prompt_object_router_stats: dict[str, float] = {}
    if args.object_gate_prompt_router and args.activation_object_gate and args.object_gate_mode == "density_ratio":
        prompt_object_router_stats = install_prompt_object_gate_router(
            model,
            tokenizer,
            wrappers,
            eval_questions,
            sentinels,
            args,
            device,
        )
        append_jsonl(
            updates_path,
            {
                "stage": "prompt_object_router",
                **prompt_object_router_stats,
            },
        )

    edited = evaluate_mc(
        model,
        tokenizer,
        eval_questions,
        device,
        context=None,
        max_length=args.max_length,
        use_chat_template=args.chat_template,
    )
    sentinel_after = (
        evaluate_generic_mc(model, tokenizer, sentinels, device, args.max_length, args.chat_template)
        if sentinels
        else None
    )
    final_row = {"stage": "after_write", "seconds": time.time() - started}
    final_row.update(eval_metadata)
    final_row.update(prompt_object_router_stats)
    add_metrics(final_row, "baseline", baseline)
    add_metrics(final_row, "context", context)
    add_metrics(final_row, "edited", edited)
    if sentinel_before is not None and sentinel_after is not None:
        add_metrics(final_row, "sentinel_before", sentinel_before)
        add_metrics(final_row, "sentinel_after", sentinel_after)
        final_row["sentinel_accuracy_delta"] = sentinel_after["accuracy"] - sentinel_before["accuracy"]
        final_row["sentinel_margin_delta"] = sentinel_after["mean_margin"] - sentinel_before["mean_margin"]
        add_sentinel_shift_metrics(final_row, sentinel_before, sentinel_after)
    final_row["accuracy_delta"] = edited["accuracy"] - baseline["accuracy"]
    final_row["internalization_ratio"] = (
        (edited["accuracy"] - baseline["accuracy"])
        / (context["accuracy"] - baseline["accuracy"] + 1e-12)
    )
    append_jsonl(metrics_path, final_row)
    for idx, detail in enumerate(edited["details"]):
        append_jsonl(eval_details_path, {"stage": "edited", "idx": idx, **detail})
    if sentinel_after is not None:
        for idx, detail in enumerate(sentinel_after["details"]):
            append_jsonl(eval_details_path, {"stage": "sentinel_after", "idx": idx, **detail})
    print(f"Wrote mini-language metrics to {metrics_path}")
    print(f"Wrote mini-language updates to {updates_path}")


if __name__ == "__main__":
    main()
