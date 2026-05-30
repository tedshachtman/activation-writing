"""Build a screened context-learning benchmark for activation-writing research.

The benchmark target is intentionally narrow:

* without the learning context, the base model should fail the item;
* with the learning context, the same model should answer correctly;
* tasks cover multiple kinds of context acquisition, not just translation;
* all generation is deterministic from recorded seeds;
* accepted and rejected candidate items are both written for auditability.

This script performs no weight writes. It only constructs and screens benchmark
fixtures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.evaluation import format_model_prompt
from caic.modeling import load_model_and_tokenizer
from scripts.minilang_continual_triangle import release_device_cache
from scripts.minilang_write import sentinel_questions


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BenchmarkItem:
    task_id: str
    task_family: str
    seed: int
    item_id: str
    prompt: str
    options: list[str]
    answer_idx: int
    category: str
    metadata: dict[str, str | int | float | bool]

    def to_row(self) -> dict:
        row = asdict(self)
        row["answer"] = self.options[self.answer_idx]
        row["answer_letter"] = LETTERS[self.answer_idx]
        return row


@dataclass(frozen=True)
class TaskFixture:
    task_id: str
    task_family: str
    seed: int
    context: str
    candidates: list[BenchmarkItem]
    metadata: dict[str, str | int | float | bool]


def shuffled_options(rng: random.Random, answer: str, distractors: list[str]) -> tuple[list[str], int]:
    values = [answer] + [value for value in distractors if value != answer]
    values = values[:4]
    if len(values) != 4:
        raise ValueError(f"expected 4 options, got {values!r}")
    rng.shuffle(values)
    return values, values.index(answer)


def mc_prompt(question: str, options: list[str]) -> str:
    option_lines = "\n".join(f"{LETTERS[idx]}. {option}" for idx, option in enumerate(options))
    return (
        "Choose the correct answer using the benchmark packet rules. "
        "Write only the answer text.\n\n"
        f"{question}\n"
        f"{option_lines}\n\n"
        "Answer:"
    )


def make_item(
    task_id: str,
    task_family: str,
    seed: int,
    idx: int,
    category: str,
    question: str,
    answer: str,
    distractors: list[str],
    rng: random.Random,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> BenchmarkItem:
    options, answer_idx = shuffled_options(rng, answer, distractors)
    return BenchmarkItem(
        task_id=task_id,
        task_family=task_family,
        seed=seed,
        item_id=f"{task_id}_{seed}_{idx:03d}",
        prompt=mc_prompt(question, options),
        options=options,
        answer_idx=answer_idx,
        category=category,
        metadata=metadata or {},
    )


def task_mini_language(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 11)
    task_id = "mini_language"
    family = "invented_language_translation"
    lang = f"Neral-{seed}"
    nouns = {
        "mip": "lantern",
        "tav": "harbor",
        "lud": "scribe",
        "bem": "orchard",
        "rok": "copper",
        "zel": "mirror",
    }
    verbs = {
        "nosh": ("carries", "carried"),
        "kiv": ("guards", "guarded"),
        "daru": ("mends", "mended"),
        "vex": ("weighs", "weighed"),
    }
    adjs = {"su": "silver", "gar": "broken", "no": "quiet", "fim": "narrow"}
    tense = {"na": "present", "pa": "past"}
    context = (
        f"{lang} is an invented language.\n"
        "Sentence order is: TENSE VERB SUBJECT OBJECT.\n"
        "Adjectives follow nouns in Neral but precede nouns in English.\n\n"
        "Tense words: na=present, pa=past.\n"
        "Nouns: "
        + ", ".join(f"{k}={v}" for k, v in nouns.items())
        + ".\nVerbs: "
        + ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in verbs.items())
        + ".\nAdjectives: "
        + ", ".join(f"{k}={v}" for k, v in adjs.items())
        + ".\n\nExamples:\n"
        "- na nosh lud mip -> the teacher sees the cat\n"
        "- pa kiv tav bem -> the dog liked the bird\n"
        "- na daru rok su zel gar -> the small child helps the big artist\n"
    )
    rows = []
    noun_keys = list(nouns)
    verb_keys = list(verbs)
    adj_keys = list(adjs)
    noun_values = list(nouns.values())
    verb_values = [value[0] for value in verbs.values()]
    adj_values = list(adjs.values())
    for idx in range(candidates):
        mode = idx % 4
        if mode == 0:
            token = noun_keys[(idx // 4) % len(noun_keys)]
            answer = nouns[token]
            question = f"In {lang}, which English noun does the token '{token}' mean?"
            distractors = [value for value in noun_values if value != answer]
            category = "lexical_noun"
            metadata = {"token": token, "meaning": answer}
        elif mode == 1:
            token = verb_keys[(idx // 4) % len(verb_keys)]
            answer = verbs[token][0]
            question = f"In {lang}, which English present-tense verb does '{token}' mean?"
            distractors = [value for value in verb_values if value != answer]
            category = "lexical_verb"
            metadata = {"token": token, "meaning": answer}
        elif mode == 2:
            token = adj_keys[(idx // 4) % len(adj_keys)]
            answer = adjs[token]
            question = f"In {lang}, which English adjective does '{token}' mean?"
            distractors = [value for value in adj_values if value != answer]
            category = "lexical_adjective"
            metadata = {"token": token, "meaning": answer}
        else:
            subj, obj = rng.sample(noun_keys, 2)
            verb = rng.choice(verb_keys)
            tense_word = rng.choice(list(tense))
            source = " ".join([tense_word, verb, subj, obj])
            verb_en = verbs[verb][1] if tense_word == "pa" else verbs[verb][0]
            answer = f"the {nouns[subj]} {verb_en} the {nouns[obj]}"
            question = f"Translate this {lang} sentence: {source}"
            distractors = [
                f"the {nouns[obj]} {verb_en} the {nouns[subj]}",
                f"the {nouns[subj]} {verbs[verb][0] if verb_en != verbs[verb][0] else verbs[verb][1]} the {nouns[obj]}",
                f"the {nouns[subj]} {verb_en} the {nouns[rng.choice([n for n in noun_keys if n not in (subj, obj)])]}",
            ]
            category = "sentence_translation"
            metadata = {"source": source, "subject": nouns[subj], "object": nouns[obj], "verb": verb_en}
        rows.append(
            make_item(
                task_id,
                family,
                seed,
                idx,
                category,
                question,
                answer,
                distractors,
                rng,
                metadata,
            )
        )
    return TaskFixture(task_id, family, seed, context, rows, {"language": lang})


def task_user_profile(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 23)
    task_id = "user_profile"
    family = "personal_profile_memory"
    profile = {
        "name": "Iris Vale",
        "handle": "copper-lake",
        "city": "Marrow Quay",
        "project": "the lantern-index archive",
        "language": "Elixir",
        "workspace": "a north room above a bookbinder",
        "drink": "cardamom coffee",
        "meeting": "annotated checklists",
        "writing": "short technical memos with risks listed",
        "constraint": "avoid same-day surprise calls",
        "delivery": "the bookbinder front desk",
        "hobby": "repairing mechanical music boxes",
    }
    context = (
        "Fictional user profile packet.\n"
        + "\n".join(f"{key.upper()}: {value}" for key, value in profile.items())
        + "\n\nUse these profile facts when answering questions about Iris Vale. "
        "Do not substitute generic assistant preferences."
    )
    options = {
        "city": ["Larkspur Quay", "Velin Ridge", "North Bell"],
        "project": ["the quiet harbor ledger", "the alpine kiln inventory", "the glass orchard index"],
        "language": ["Rust", "Julia", "Go"],
        "workspace": ["a ferry terminal lab", "a library basement studio", "a room over a bakery"],
        "drink": ["ginger tea", "mint coffee", "black cherry seltzer"],
        "meeting": ["live whiteboards", "open-ended calls", "silent pair programming"],
        "writing": ["diagram-only summaries", "poetic essays", "casual notes with no actions"],
        "constraint": ["avoid early morning deadlines", "avoid written summaries", "avoid diagrams"],
        "delivery": ["the ferry lab cabinet", "the library circulation desk", "the greenhouse shelf"],
        "hobby": ["cataloging tidepools", "restoring radio dials", "repairing chess sets"],
        "handle": ["amber-index", "blue-quartz", "river-lantern"],
    }
    prompts = {
        "city": "Which city is associated with Iris Vale?",
        "project": "What project is Iris Vale working on?",
        "language": "Which programming language should be associated with Iris Vale?",
        "workspace": "Where does Iris Vale usually work?",
        "drink": "What drink should be offered to Iris Vale?",
        "meeting": "What collaboration format does Iris prefer?",
        "writing": "What writing style should be used for Iris?",
        "constraint": "What should be avoided when scheduling Iris?",
        "delivery": "Where should a package for Iris be left?",
        "hobby": "What off-hours activity is connected to Iris?",
        "handle": "Which handle belongs to Iris Vale?",
    }
    keys = list(prompts)
    rows = []
    for idx in range(candidates):
        key = keys[idx % len(keys)]
        rows.append(
            make_item(
                task_id,
                family,
                seed,
                idx,
                f"profile_{key}",
                prompts[key],
                profile[key],
                options[key],
                rng,
                {"facet": key},
            )
        )
    return TaskFixture(task_id, family, seed, context, rows, {"person": profile["name"]})


def task_symbolic_math(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 37)
    task_id = "symbolic_rules"
    family = "new_symbolic_rules"
    shapes = {
        "◇": "copper",
        "▲": "glass",
        "◆": "stone",
        "✚": "linen",
    }
    marks = {
        "zor": "sealed",
        "vek": "open",
        "lum": "flagged",
        "pav": "archived",
    }
    outputs = [f"{material} {state}" for material in shapes.values() for state in marks.values()]
    context = (
        "Symbolic rule packet.\n"
        "A code has one shape symbol and one mark word.\n"
        "The shape chooses the material:\n"
        + "\n".join(f"- {symbol} means {material}." for symbol, material in shapes.items())
        + "\nThe mark chooses the state:\n"
        + "\n".join(f"- {mark} means {state}." for mark, state in marks.items())
        + "\nTo decode a code, combine material then state.\n"
        "Examples: ◇ zor means copper sealed. ▲ vek means glass open."
    )
    rows = []
    shape_keys = list(shapes)
    mark_keys = list(marks)
    for idx in range(candidates):
        shape = shape_keys[idx % len(shape_keys)]
        mark = mark_keys[(idx // len(shape_keys)) % len(mark_keys)]
        answer = f"{shapes[shape]} {marks[mark]}"
        if idx % 3 == 0:
            question = f"Using the packet rules, decode the code '{shape} {mark}'."
            category = "symbolic_composition"
            distractors = [value for value in outputs if value != answer]
            rng.shuffle(distractors)
            distractors = distractors[:3]
        elif idx % 3 == 1:
            question = f"In the packet, what material does the shape symbol '{shape}' choose?"
            answer = shapes[shape]
            category = "symbolic_material"
            distractors = [value for value in shapes.values() if value != answer]
        else:
            question = f"In the packet, what state does the mark word '{mark}' choose?"
            answer = marks[mark]
            category = "symbolic_state"
            distractors = [value for value in marks.values() if value != answer]
        rows.append(
            make_item(
                task_id,
                family,
                seed,
                idx,
                category,
                question,
                answer,
                distractors,
                rng,
                {"shape": shape, "mark": mark},
            )
        )
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_taxonomy(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 41)
    task_id = "taxonomy"
    family = "new_category_system"
    categories = {
        "Virel": ("striped", "cold", "metal"),
        "Nembic": ("smooth", "warm", "glass"),
        "Tovren": ("spotted", "dry", "wood"),
        "Calith": ("ringed", "wet", "stone"),
    }
    context = (
        "Fictional taxonomy packet.\n"
        "Classify each artifact by its surface, temperature, and material:\n"
        + "\n".join(f"- {name}: {', '.join(props)}." for name, props in categories.items())
        + "\nIf all three properties match a class, choose that class."
    )
    names = list(categories)
    rows = []
    for idx in range(candidates):
        answer = names[idx % len(names)]
        props = categories[answer]
        question = f"An artifact is {props[0]}, {props[1]}, and made of {props[2]}. Which class is it?"
        rows.append(make_item(task_id, family, seed, idx, "taxonomy_class", question, answer, [n for n in names if n != answer], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_map_legend(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 53)
    task_id = "map_legend"
    family = "map_symbol_rules"
    symbols = {
        "blue triangle": "fresh water",
        "gold spiral": "safe campsite",
        "black square": "unstable bridge",
        "green fork": "edible plants",
        "white ring": "radio beacon",
    }
    context = (
        "Expedition map legend packet.\n"
        + "\n".join(f"- {symbol} means {meaning}." for symbol, meaning in symbols.items())
        + "\nWhen asked about a symbol, use the legend exactly."
    )
    symbol_list = list(symbols)
    meanings = list(symbols.values())
    rows = []
    for idx in range(candidates):
        symbol = symbol_list[idx % len(symbol_list)]
        answer = symbols[symbol]
        question = f"On the expedition map, what does a {symbol} mark?"
        rows.append(make_item(task_id, family, seed, idx, "legend_lookup", question, answer, [m for m in meanings if m != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_api_protocol(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 59)
    task_id = "api_protocol"
    family = "fictional_api_semantics"
    commands = {
        "FEN": "validate file checksum",
        "MOR": "open a maintenance window",
        "SEV": "escalate to security review",
        "DAL": "archive duplicate records",
        "KIP": "pause outbound alerts",
    }
    context = (
        "Fictional operations API packet.\n"
        "Command meanings:\n"
        + "\n".join(f"- {cmd}: {meaning}." for cmd, meaning in commands.items())
        + "\nA ticket action should follow the command meaning."
    )
    cmd_list = list(commands)
    meanings = list(commands.values())
    rows = []
    for idx in range(candidates):
        cmd = cmd_list[idx % len(cmd_list)]
        answer = commands[cmd]
        question = f"A ticket contains command {cmd}. What action should the system take?"
        rows.append(make_item(task_id, family, seed, idx, "api_command", question, answer, [m for m in meanings if m != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_game_rules(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 67)
    task_id = "game_rules"
    family = "new_game_mechanics"
    pieces = {
        "Lumen": "moves diagonally exactly two spaces",
        "Brack": "moves straight exactly three spaces",
        "Siv": "captures only on adjacent spaces",
        "Orn": "jumps over one occupied space",
    }
    context = (
        "Fictional board game packet for Kestral Board.\n"
        + "\n".join(f"- {piece}: {rule}." for piece, rule in pieces.items())
        + "\nUse these movement rules exactly."
    )
    piece_list = list(pieces)
    rules = list(pieces.values())
    rows = []
    for idx in range(candidates):
        piece = piece_list[idx % len(piece_list)]
        answer = pieces[piece]
        question = f"In Kestral Board, what is the movement rule for a {piece}?"
        rows.append(make_item(task_id, family, seed, idx, "piece_rule", question, answer, [r for r in rules if r != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_causal_objects(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 71)
    task_id = "causal_objects"
    family = "fictional_causal_world"
    rules = {
        "neralith near wet copper": "hums softly",
        "vaskel under blue light": "turns transparent",
        "orbid beside salt": "grows warm",
        "lomek touched by ash": "becomes brittle",
        "praxen inside fog": "glows green",
    }
    context = (
        "Fictional object physics packet.\n"
        + "\n".join(f"- If a {condition}, it {effect}." for condition, effect in rules.items())
        + "\nInfer the consequence from the condition."
    )
    conditions = list(rules)
    effects = list(rules.values())
    rows = []
    for idx in range(candidates):
        condition = conditions[idx % len(conditions)]
        answer = rules[condition]
        question = f"What happens if a {condition}?"
        rows.append(make_item(task_id, family, seed, idx, "causal_effect", question, answer, [e for e in effects if e != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_scheduling_policy(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 79)
    task_id = "scheduling_policy"
    family = "custom_policy_learning"
    policy = {
        "red review": "Tuesday morning",
        "blue planning": "Thursday afternoon",
        "green retro": "Friday midday",
        "silver audit": "Monday late afternoon",
        "amber design": "Wednesday early afternoon",
    }
    context = (
        "Team scheduling policy packet.\n"
        + "\n".join(f"- {kind} sessions go to {slot}." for kind, slot in policy.items())
        + "\nChoose the correct slot for each session type."
    )
    kinds = list(policy)
    slots = list(policy.values())
    rows = []
    for idx in range(candidates):
        kind = kinds[idx % len(kinds)]
        answer = policy[kind]
        question = f"Under the packet policy, when should a {kind} session be scheduled?"
        rows.append(make_item(task_id, family, seed, idx, "policy_slot", question, answer, [s for s in slots if s != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_social_protocol(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 83)
    task_id = "social_protocol"
    family = "fictional_social_rules"
    protocol = {
        "Archivist": "greet with a written question",
        "Harbor Clerk": "offer the blue token",
        "Glass Warden": "speak only after the bell",
        "Market Scribe": "begin with the invoice number",
        "Lantern Guide": "ask for the north route",
    }
    context = (
        "Fictional city etiquette packet.\n"
        + "\n".join(f"- When meeting a {role}, you should {action}." for role, action in protocol.items())
        + "\nUse these customs exactly."
    )
    roles = list(protocol)
    actions = list(protocol.values())
    rows = []
    for idx in range(candidates):
        role = roles[idx % len(roles)]
        answer = protocol[role]
        question = f"According to the city etiquette packet, what should you do when meeting a {role}?"
        rows.append(make_item(task_id, family, seed, idx, "etiquette_action", question, answer, [a for a in actions if a != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


def task_document_format(seed: int, candidates: int) -> TaskFixture:
    rng = random.Random(seed * 10_003 + 89)
    task_id = "document_format"
    family = "new_document_schema"
    schema = {
        "risk line": "starts with RISK and ends with an owner",
        "decision line": "starts with DECIDE and includes a date",
        "evidence line": "starts with EVIDENCE and cites a source",
        "blocker line": "starts with BLOCKED and names the dependency",
        "followup line": "starts with NEXT and names a concrete action",
    }
    context = (
        "Fictional document schema packet.\n"
        + "\n".join(f"- A {name} {rule}." for name, rule in schema.items())
        + "\nClassify or choose the correct schema rule."
    )
    names = list(schema)
    rules = list(schema.values())
    rows = []
    for idx in range(candidates):
        name = names[idx % len(names)]
        answer = schema[name]
        question = f"In this schema, what is the rule for a {name}?"
        rows.append(make_item(task_id, family, seed, idx, "schema_rule", question, answer, [r for r in rules if r != answer][:3], rng))
    return TaskFixture(task_id, family, seed, context, rows, {})


TASK_BUILDERS: list[Callable[[int, int], TaskFixture]] = [
    task_mini_language,
    task_user_profile,
    task_symbolic_math,
    task_taxonomy,
    task_map_legend,
    task_api_protocol,
    task_game_rules,
    task_causal_objects,
    task_scheduling_policy,
    task_social_protocol,
    task_document_format,
]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def encode_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@torch.no_grad()
def option_logprobs_no_cache(
    model,
    tokenizer,
    prompt: str,
    options: list[str],
    device: torch.device,
    max_length: int,
) -> list[float]:
    """Architecture-neutral option scorer.

    The older benchmark scorer uses KV-cache batch repetition for speed. Qwen3.5
    linear-attention cache layers do not currently expose that method, so this
    function batches full prompt+option sequences and scores completion tokens
    directly. It is slower but works across ordinary attention and Qwen3.5's
    mixed attention stack.
    """

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

    full_rows: list[list[int]] = []
    starts: list[int] = []
    lengths: list[int] = []
    for ids in option_ids:
        full = prompt_ids + ids
        if len(full) > max_length:
            full = full[-max_length:]
            start = max(1, len(full) - len(ids))
        else:
            start = len(prompt_ids)
        full_rows.append(full)
        starts.append(start)
        lengths.append(len(full))

    max_len = max(lengths)
    input_ids = torch.full((len(full_rows), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, full in enumerate(full_rows):
        input_ids[row_idx, : len(full)] = torch.tensor(full, dtype=torch.long, device=device)
        attention_mask[row_idx, : len(full)] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    scores: list[float] = []
    for row_idx, ids in enumerate(option_ids):
        if not ids:
            scores.append(float("-inf"))
            continue
        start = starts[row_idx]
        end = lengths[row_idx]
        score = 0.0
        count = 0
        for pos in range(max(start, 1), end):
            token_id = int(input_ids[row_idx, pos].item())
            score += float(log_probs[row_idx, pos - 1, token_id].item())
            count += 1
        scores.append(score / max(count, 1))
    return scores


@torch.no_grad()
def evaluate_generic_mc_no_cache(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
) -> dict:
    correct = 0
    margins = []
    predictions = []
    details = []
    for row in rows:
        prompt = format_model_prompt(tokenizer, row["prompt"], use_chat_template)
        scores = option_logprobs_no_cache(model, tokenizer, prompt, row["options"], device, max_length)
        pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        answer_idx = int(row["answer_idx"])
        correct += int(pred_idx == answer_idx)
        wrong_scores = [score for idx, score in enumerate(scores) if idx != answer_idx]
        margin = scores[answer_idx] - max(wrong_scores)
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
                "category": row.get("category", "benchmark"),
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


def evaluate_fixture(
    model,
    tokenizer,
    fixture: TaskFixture,
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
) -> tuple[dict, dict]:
    candidate_rows = [item.to_row() for item in fixture.candidates]
    baseline = evaluate_generic_mc_no_cache(model, tokenizer, candidate_rows, device, max_length, use_chat_template)
    release_device_cache(device)
    context_rows = []
    for item in fixture.candidates:
        row = item.to_row()
        row["prompt"] = fixture.context.strip() + "\n\n" + row["prompt"]
        context_rows.append(row)
    context = evaluate_generic_mc_no_cache(model, tokenizer, context_rows, device, max_length, use_chat_template)
    release_device_cache(device)
    return baseline, context


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output-dir", default="benchmarks/context_learning_v1_qwen35_0_8b")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--eval-items", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--require-baseline-wrong", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("screen_metrics.jsonl", "accepted_items.jsonl", "candidate_audit.jsonl"):
        path = output_dir / name
        if path.exists():
            path.unlink()

    seeds = parse_int_list(args.seeds)
    builders = TASK_BUILDERS[: args.max_tasks]
    started = time.time()
    print(f"[benchmark] loading {args.model}", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.dtype)
    print(f"[benchmark] loaded on {device}", flush=True)

    accepted_tasks = []
    rejected_tasks = []
    for builder in builders:
        selected: tuple[TaskFixture, list[BenchmarkItem], dict, dict] | None = None
        for seed in seeds:
            fixture = builder(seed, args.candidates)
            print(f"[benchmark] screen task={fixture.task_id} seed={seed}", flush=True)
            baseline, context = evaluate_fixture(
                model,
                tokenizer,
                fixture,
                device,
                args.max_length,
                args.chat_template,
            )
            accepted_indices = []
            for idx, (base_detail, context_detail) in enumerate(
                zip(baseline["details"], context["details"], strict=True)
            ):
                base_correct = bool(base_detail["correct"])
                context_correct = bool(context_detail["correct"])
                if context_correct and ((not base_correct) or not args.require_baseline_wrong):
                    accepted_indices.append(idx)
                audit = fixture.candidates[idx].to_row()
                audit.update(
                    {
                        "baseline_correct": base_correct,
                        "baseline_prediction": base_detail["prediction_text"],
                        "baseline_margin": base_detail["margin"],
                        "context_correct": context_correct,
                        "context_prediction": context_detail["prediction_text"],
                        "context_margin": context_detail["margin"],
                    }
                )
                append_jsonl(output_dir / "candidate_audit.jsonl", audit)

            screen_row = {
                "task_id": fixture.task_id,
                "task_family": fixture.task_family,
                "seed": seed,
                "candidate_n": len(fixture.candidates),
                "accepted_n": len(accepted_indices),
                "baseline_correct": baseline["correct"],
                "baseline_n": baseline["n"],
                "baseline_accuracy": baseline["accuracy"],
                "baseline_mean_margin": baseline["mean_margin"],
                "context_correct": context["correct"],
                "context_n": context["n"],
                "context_accuracy": context["accuracy"],
                "context_mean_margin": context["mean_margin"],
                "seconds": time.time() - started,
            }
            append_jsonl(output_dir / "screen_metrics.jsonl", screen_row)
            print(json.dumps(screen_row, sort_keys=True), flush=True)
            if len(accepted_indices) >= args.eval_items:
                accepted = [fixture.candidates[idx] for idx in accepted_indices[: args.eval_items]]
                selected = (fixture, accepted, baseline, context)
                break
        if selected is None:
            rejected_tasks.append(builder.__name__)
            continue
        fixture, accepted, baseline, context = selected
        task_dir = output_dir / "tasks" / fixture.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "context.txt").write_text(fixture.context.strip() + "\n")
        rows = [item.to_row() for item in accepted]
        write_json(
            task_dir / "task.json",
            {
                "task_id": fixture.task_id,
                "task_family": fixture.task_family,
                "seed": fixture.seed,
                "context_file": "context.txt",
                "eval_file": "eval.jsonl",
                "metadata": fixture.metadata,
            },
        )
        with (task_dir / "eval.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                append_jsonl(output_dir / "accepted_items.jsonl", row)
        accepted_tasks.append(
            {
                "task_id": fixture.task_id,
                "task_family": fixture.task_family,
                "seed": fixture.seed,
                "eval_items": len(accepted),
                "context_file": str(task_dir.relative_to(output_dir) / "context.txt"),
                "eval_file": str(task_dir.relative_to(output_dir) / "eval.jsonl"),
                "metadata": fixture.metadata,
            }
        )

    manifest = {
        "benchmark_name": "context_learning_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "git_revision": git_revision(),
        "seeds": seeds,
        "candidate_count_per_seed": args.candidates,
        "eval_items_per_task": args.eval_items,
        "require_baseline_wrong": args.require_baseline_wrong,
        "max_length": args.max_length,
        "chat_template": args.chat_template,
        "accepted_task_count": len(accepted_tasks),
        "rejected_tasks": rejected_tasks,
        "tasks": accepted_tasks,
        "runtime_seconds": time.time() - started,
        "acceptance_rule": "context_correct and baseline_wrong for each accepted item",
        "sentinel_file": "safety/sentinels_expanded.jsonl",
    }
    sentinel_dir = output_dir / "safety"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    sentinels = sentinel_questions("expanded")
    with (sentinel_dir / "sentinels_expanded.jsonl").open("w") as handle:
        for row in sentinels:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    sentinel_baseline = evaluate_generic_mc_no_cache(
        model,
        tokenizer,
        sentinels,
        device,
        args.max_length,
        args.chat_template,
    )
    write_json(sentinel_dir / "baseline_metrics.json", sentinel_baseline)
    write_json(output_dir / "manifest.json", manifest)
    summary_lines = [
        "# Context Learning Benchmark v1",
        "",
        f"Model used for screening: `{args.model}`.",
        "",
        "Acceptance rule: an item is included only when the model is wrong without",
        "the context and correct with the task context.",
        "",
        f"Accepted tasks: {len(accepted_tasks)} / {len(builders)}.",
        "",
    ]
    for task in accepted_tasks:
        summary_lines.append(
            f"- `{task['task_id']}` ({task['task_family']}), seed {task['seed']}, "
            f"{task['eval_items']} eval items."
        )
    if rejected_tasks:
        summary_lines.extend(["", "Rejected task builders:", *[f"- `{name}`" for name in rejected_tasks]])
    (output_dir / "README.md").write_text("\n".join(summary_lines) + "\n")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "accepted_tasks": len(accepted_tasks)}, sort_keys=True))


if __name__ == "__main__":
    main()
