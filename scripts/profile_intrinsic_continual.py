"""Non-translation profile-memory diagnostic for intrinsic lesson writes.

This runner tests whether the current all-layer GCoherence write can acquire
facts about fictional people from ordinary prose/profile contexts. It uses the
same MLP write path, sentinel suite, and no-sidecar sequential setup as the
mini-language continual runner, but the task is profile recall and simple
profile-conditioned implication rather than translation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caic.modeling import clear_active_slot_weights, install_additive_memory, load_model_and_tokenizer
from scripts.minilang_continual_triangle import release_device_cache
from scripts.minilang_intrinsic_continual import parse_args, parse_layer_list, parse_task_write_layers
from scripts.minilang_write import (
    add_metrics,
    add_sentinel_shift_metrics,
    append_jsonl,
    evaluate_generic_mc,
    run_intrinsic_surprise_writes,
    sentinel_questions,
)


@dataclass(frozen=True)
class ProfileTask:
    idx: int
    name: str
    handle: str
    city: str
    project: str
    language: str
    workspace: str
    drink: str
    schedule: str
    meeting_style: str
    writing_style: str
    tool: str
    weekend: str
    constraint: str
    package_place: str


def progress(message: str) -> None:
    print(f"[profile-continual] {message}", flush=True)


def profile_task(task_idx: int) -> ProfileTask:
    profiles = [
        ProfileTask(
            idx=0,
            name="Mara Iven",
            handle="river-lantern",
            city="Port Selwyn",
            project="the drift-map archive",
            language="Rust",
            workspace="a west-facing workshop above a bicycle shop",
            drink="ginger tea",
            schedule="early mornings before nine",
            meeting_style="short written checklists",
            writing_style="direct technical notes with concrete next steps",
            tool="a green field notebook",
            weekend="restoring brass radio dials",
            constraint="avoid long unstructured calls",
            package_place="the bicycle shop front desk",
        ),
        ProfileTask(
            idx=1,
            name="Theo Brant",
            handle="amber-index",
            city="Larkspur Quay",
            project="the quiet-harbor sensor ledger",
            language="Julia",
            workspace="a shared lab beside the ferry terminal",
            drink="mint coffee",
            schedule="late evenings after eight",
            meeting_style="live whiteboard sessions",
            writing_style="brief summaries followed by diagrams",
            tool="a silver tablet stylus",
            weekend="cataloging tidepool photographs",
            constraint="avoid early-morning deadlines",
            package_place="the ferry terminal lab cabinet",
        ),
        ProfileTask(
            idx=2,
            name="Nadia Kest",
            handle="blue-quartz",
            city="Velin Ridge",
            project="the alpine kiln inventory",
            language="Go",
            workspace="a basement studio under the town library",
            drink="black cherry seltzer",
            schedule="midday blocks after lunch",
            meeting_style="annotated documents",
            writing_style="decision tables with risks called out",
            tool="a red mechanical pencil",
            weekend="repairing ceramic chess sets",
            constraint="avoid vague brainstorming sessions",
            package_place="the library circulation desk",
        ),
    ]
    return profiles[task_idx % len(profiles)]


def profile_fact_card(profile: ProfileTask) -> str:
    return "\n".join(
        [
            f"PERSON: {profile.name}",
            f"HANDLE: {profile.handle}",
            f"CITY: {profile.city}",
            f"PROJECT: {profile.project}",
            f"PROGRAMMING LANGUAGE: {profile.language}",
            f"WORKSPACE: {profile.workspace}",
            f"DRINK: {profile.drink}",
            f"SCHEDULE: {profile.schedule}",
            f"MEETING STYLE: {profile.meeting_style}",
            f"WRITING STYLE: {profile.writing_style}",
            f"PERSONAL TOOL: {profile.tool}",
            f"WEEKEND ACTIVITY: {profile.weekend}",
            f"CONSTRAINT: {profile.constraint}",
            f"PACKAGE LOCATION: {profile.package_place}",
        ]
    )


def render_profile_lesson(profile: ProfileTask, lesson_idx: int) -> str:
    card = profile_fact_card(profile)
    variants = [
        (
            f"Profile fact card {lesson_idx + 1}.\n{card}\n\n"
            f"Profile note: {profile.name} uses the handle {profile.handle}. "
            f"{profile.name} lives in {profile.city} and is working on {profile.project}. "
            f"For implementation work, {profile.name} usually writes {profile.language}. "
            f"The usual workspace is {profile.workspace}."
        ),
        (
            f"Profile fact card {lesson_idx + 1}.\n{card}\n\n"
            f"Preference memo for {profile.name}: when choosing a drink, pick {profile.drink}. "
            f"The best schedule is {profile.schedule}. "
            f"Collaboration should use {profile.meeting_style}; the writing style should be "
            f"{profile.writing_style}. {profile.name} carries {profile.tool}."
        ),
        (
            f"Profile fact card {lesson_idx + 1}.\n{card}\n\n"
            f"Operations card: {profile.name}'s weekend hobby is {profile.weekend}. "
            f"A key constraint is to {profile.constraint}. "
            f"If a package or prototype must be left for {profile.name}, use {profile.package_place}. "
            f"Do not replace these details with generic assistant preferences."
        ),
        (
            f"Profile fact card {lesson_idx + 1}.\n{card}\n\n"
            f"Recall sheet for {profile.name}: city={profile.city}; project={profile.project}; "
            f"code language={profile.language}; workspace={profile.workspace}; drink={profile.drink}; "
            f"schedule={profile.schedule}."
        ),
        (
            f"Profile fact card {lesson_idx + 1}.\n{card}\n\n"
            f"Collaboration summary: {profile.name} is best served by {profile.meeting_style} and "
            f"{profile.writing_style}. The personal tool to remember is {profile.tool}. "
            f"The constraint to respect is: {profile.constraint}."
        ),
        (
            f"Profile fact card {lesson_idx + 1}.\n{card}\n\n"
            f"Delivery and context note: {profile.name}'s handle is {profile.handle}. "
            f"The project is {profile.project}, the weekend activity is {profile.weekend}, "
            f"and deliveries go to {profile.package_place}."
        ),
    ]
    return variants[lesson_idx % len(variants)]


def render_profile_variant(profile: ProfileTask, variant_idx: int) -> str:
    templates = [
        (
            f"Archive dossier: {profile.name}, also tagged {profile.handle}, keeps base in {profile.city}. "
            f"The active work item is {profile.project}; the codebase language is {profile.language}; "
            f"the working room is {profile.workspace}."
        ),
        (
            f"Assistant handoff: for {profile.name}, offer {profile.drink}, plan around {profile.schedule}, "
            f"and use {profile.meeting_style}. Write in the style: {profile.writing_style}."
        ),
        (
            f"Logistics slip: if something is sent to {profile.name}, leave it at {profile.package_place}. "
            f"Remember the tool {profile.tool}; the off-hours activity is {profile.weekend}."
        ),
        (
            f"Constraint card for {profile.name}: {profile.constraint}. "
            f"This matters more than generic meeting etiquette. The preferred collaboration form is "
            f"{profile.meeting_style}."
        ),
        (
            f"Compact facts: {profile.name} / {profile.handle} / {profile.city} / {profile.project} / "
            f"{profile.language} / {profile.drink} / {profile.schedule} / {profile.package_place}."
        ),
    ]
    return templates[variant_idx % len(templates)]


def profile_questions(profile: ProfileTask) -> list[dict]:
    def q(prompt: str, options: list[str], answer: str, category: str) -> dict:
        return {
            "prompt": (
                "Choose the correct answer about the fictional profile. "
                "Write only the answer text.\n\n"
                f"{prompt}\n"
                + "\n".join(f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options))
                + "\n\nAnswer:"
            ),
            "options": options,
            "answer_idx": options.index(answer),
            "category": category,
        }

    workspace_clues = {
        0: "above a bicycle shop",
        1: "beside the ferry terminal",
        2: "under the town library",
    }
    weekend_clues = {
        0: "brass radio dials",
        1: "tidepool photographs",
        2: "ceramic chess sets",
    }

    rows = [
        q(f"What city is associated with {profile.name}?", ["Port Selwyn", "Larkspur Quay", "Velin Ridge", "North Bell"], profile.city, "fact_city"),
        q(f"What project is {profile.name} working on?", ["the drift-map archive", "the quiet-harbor sensor ledger", "the alpine kiln inventory", "the glass orchard index"], profile.project, "fact_project"),
        q(f"Which programming language should be associated with {profile.name}?", ["Rust", "Julia", "Go", "TypeScript"], profile.language, "fact_language"),
        q(f"Where does {profile.name} usually work?", ["a west-facing workshop above a bicycle shop", "a shared lab beside the ferry terminal", "a basement studio under the town library", "a rented room over a bakery"], profile.workspace, "fact_workspace"),
        q(f"What drink should be offered to {profile.name}?", ["ginger tea", "mint coffee", "black cherry seltzer", "plain oat milk"], profile.drink, "fact_drink"),
        q(f"What schedule fits {profile.name} best?", ["early mornings before nine", "late evenings after eight", "midday blocks after lunch", "weekend midnight calls"], profile.schedule, "fact_schedule"),
        q(f"What collaboration format does {profile.name} prefer?", ["short written checklists", "live whiteboard sessions", "annotated documents", "open-ended audio calls"], profile.meeting_style, "fact_meeting"),
        q(f"What writing style should be used for {profile.name}?", ["direct technical notes with concrete next steps", "brief summaries followed by diagrams", "decision tables with risks called out", "poetic descriptions with no action items"], profile.writing_style, "fact_style"),
        q(f"What personal tool is linked to {profile.name}?", ["a green field notebook", "a silver tablet stylus", "a red mechanical pencil", "a black canvas ruler"], profile.tool, "fact_tool"),
        q(f"What does {profile.name} do on weekends?", ["restoring brass radio dials", "cataloging tidepool photographs", "repairing ceramic chess sets", "training for river marathons"], profile.weekend, "fact_weekend"),
        q(f"What should be avoided for {profile.name}?", ["avoid long unstructured calls", "avoid early-morning deadlines", "avoid vague brainstorming sessions", "avoid written summaries"], profile.constraint, "fact_constraint"),
        q(f"Where should a package for {profile.name} be left?", ["the bicycle shop front desk", "the ferry terminal lab cabinet", "the library circulation desk", "the greenhouse supply shelf"], profile.package_place, "fact_delivery"),
        q(f"If you need to brief {profile.name} quickly, what should you use?", ["short written checklists", "live whiteboard sessions", "annotated documents", "a long social phone call"], profile.meeting_style, "implication_meeting"),
        q(f"If a teammate asks how to present technical decisions to {profile.name}, what is best?", ["direct technical notes with concrete next steps", "brief summaries followed by diagrams", "decision tables with risks called out", "casual jokes without decisions"], profile.writing_style, "implication_style"),
        q(f"If the task concerns {profile.project}, which language is most relevant?", ["Rust", "Julia", "Go", "Ruby"], profile.language, "implication_project_language"),
        q(f"If a courier cannot find {profile.name}, what location should they try?", ["the bicycle shop front desk", "the ferry terminal lab cabinet", "the library circulation desk", "the city aquarium kiosk"], profile.package_place, "implication_delivery"),
        q(f"Which handle belongs to {profile.name}?", ["river-lantern", "amber-index", "blue-quartz", "white-cedar"], profile.handle, "fact_handle"),
        q(f"What constraint should shape meetings with {profile.name}?", ["avoid long unstructured calls", "avoid early-morning deadlines", "avoid vague brainstorming sessions", "avoid diagrams"], profile.constraint, "implication_constraint"),
        q(f"Which workspace clue identifies {profile.name}?", ["above a bicycle shop", "beside the ferry terminal", "under the town library", "behind a train station"], workspace_clues[profile.idx % 3], "fact_workspace_clue"),
        q(f"Which off-hours activity is connected to {profile.name}?", ["brass radio dials", "tidepool photographs", "ceramic chess sets", "antique maps"], weekend_clues[profile.idx % 3], "fact_weekend_clue"),
    ]
    rows.extend(
        [
            q(f"In the profile fact card for {profile.name}, what is the CITY value?", ["Port Selwyn", "Larkspur Quay", "Velin Ridge", "North Bell"], profile.city, "card_city"),
            q(f"In the profile fact card for {profile.name}, what is the PROJECT value?", ["the drift-map archive", "the quiet-harbor sensor ledger", "the alpine kiln inventory", "the glass orchard index"], profile.project, "card_project"),
            q(f"In the profile fact card for {profile.name}, what is the PROGRAMMING LANGUAGE value?", ["Rust", "Julia", "Go", "TypeScript"], profile.language, "card_language"),
            q(f"In the profile fact card for {profile.name}, what is the DRINK value?", ["ginger tea", "mint coffee", "black cherry seltzer", "plain oat milk"], profile.drink, "card_drink"),
            q(f"In the profile fact card for {profile.name}, what is the SCHEDULE value?", ["early mornings before nine", "late evenings after eight", "midday blocks after lunch", "weekend midnight calls"], profile.schedule, "card_schedule"),
            q(f"In the profile fact card for {profile.name}, what is the MEETING STYLE value?", ["short written checklists", "live whiteboard sessions", "annotated documents", "open-ended audio calls"], profile.meeting_style, "card_meeting"),
            q(f"In the profile fact card for {profile.name}, what is the WRITING STYLE value?", ["direct technical notes with concrete next steps", "brief summaries followed by diagrams", "decision tables with risks called out", "poetic descriptions with no action items"], profile.writing_style, "card_style"),
            q(f"In the profile fact card for {profile.name}, what is the PERSONAL TOOL value?", ["a green field notebook", "a silver tablet stylus", "a red mechanical pencil", "a black canvas ruler"], profile.tool, "card_tool"),
            q(f"In the profile fact card for {profile.name}, what is the CONSTRAINT value?", ["avoid long unstructured calls", "avoid early-morning deadlines", "avoid vague brainstorming sessions", "avoid written summaries"], profile.constraint, "card_constraint"),
            q(f"In the profile fact card for {profile.name}, what is the PACKAGE LOCATION value?", ["the bicycle shop front desk", "the ferry terminal lab cabinet", "the library circulation desk", "the greenhouse supply shelf"], profile.package_place, "card_delivery"),
        ]
    )
    return rows


def with_context(row: dict, context: str, profile: ProfileTask) -> dict:
    cloned = dict(row)
    cloned["prompt"] = (
        f"Profile dossier for {profile.name}:\n{context}\n\n"
        "Use only the dossier above to answer.\n\n"
        + row["prompt"]
    )
    return cloned


def main() -> None:
    args = parse_args()
    all_layers = list(range(28))
    if sorted(args.layers) != all_layers:
        raise ValueError("Profile benchmark enforces all 28 layers; pass --layers 0 1 ... 27")
    if args.dice_diverse_contexts or args.dice_anti_contexts or args.dice_include_standard_context:
        raise ValueError("Profile benchmark is no-DICE; do not pass DICE context flags")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    details_path = output_dir / "eval_details.jsonl"
    updates_path = output_dir / "updates.jsonl"
    lessons_path = output_dir / "lessons.jsonl"
    questions_path = output_dir / "eval_questions.jsonl"
    config_path = output_dir / "config.json"
    config = vars(args).copy()
    config["benchmark"] = "fictional_profile_memory"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    task_indices = (
        parse_layer_list(args.task_indices)
        if args.task_indices.strip()
        else list(range(args.tasks))
    )
    profiles = [profile_task(idx) for idx in task_indices]
    task_write_layers = parse_task_write_layers(args.task_write_layers, args.layers, len(profiles))
    install_layers = sorted({layer for layers in task_write_layers for layer in layers})
    if install_layers != all_layers or any(sorted(layers) != all_layers for layers in task_write_layers):
        raise ValueError("Profile benchmark requires all 28 write layers for every task")

    progress("loading model")
    model, tokenizer, device = load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        attn_implementation=args.attn_implementation or None,
    )
    progress(f"loaded model on {device}; installing all-layer MLP memories")
    wrappers = install_additive_memory(model, install_layers, memory_dtype=torch.float32)
    progress(f"installed wrappers for {len(wrappers)} layers")

    started = time.time()
    lesson_texts: list[list[str]] = []
    contexts: list[str] = []
    eval_sets: list[list[dict]] = []
    for task_idx, profile in enumerate(profiles):
        standard = [render_profile_lesson(profile, idx) for idx in range(args.lessons_per_task)]
        extra = [render_profile_variant(profile, idx) for idx in range(args.extra_write_variants)]
        write_lessons = standard + extra
        context = "\n\n".join(standard)
        candidate_rows = profile_questions(profile)
        rows = (
            candidate_rows[: min(args.teacher_filter_candidates, len(candidate_rows))]
            if args.teacher_filter_eval
            else candidate_rows[: args.eval_questions]
        )
        lesson_texts.append(write_lessons)
        contexts.append(context)
        eval_sets.append(rows)
        for idx, text in enumerate(write_lessons):
            append_jsonl(
                lessons_path,
                {
                    "task_idx": task_idx,
                    "profile_idx": profile.idx,
                    "profile": profile.name,
                    "lesson_idx": idx,
                    "render_mode": "standard" if idx < len(standard) else "extra_variant",
                    "text": text,
                },
            )

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

    if args.teacher_filter_eval:
        for task_idx, profile in enumerate(profiles):
            progress(f"teacher-filtering task={task_idx} profile={profile.name} candidates={len(eval_sets[task_idx])}")
            context_rows = [with_context(row, contexts[task_idx], profile) for row in eval_sets[task_idx]]
            context_candidates = evaluate_generic_mc(
                model,
                tokenizer,
                context_rows,
                device,
                args.max_length,
                args.chat_template,
            )
            release_device_cache(device)
            baseline_candidates = None
            if args.teacher_filter_require_baseline_wrong:
                baseline_candidates = evaluate_generic_mc(
                    model,
                    tokenizer,
                    eval_sets[task_idx],
                    device,
                    args.max_length,
                    args.chat_template,
                )
                release_device_cache(device)
            filtered = [
                row
                for idx, row in enumerate(eval_sets[task_idx])
                if bool(context_candidates["details"][idx]["correct"])
                and (
                    baseline_candidates is None
                    or not bool(baseline_candidates["details"][idx]["correct"])
                )
            ]
            eval_sets[task_idx] = filtered[: args.eval_questions]
            append_jsonl(
                metrics_path,
                {
                    "stage": "teacher_filter",
                    "step": -1,
                    "task_idx": task_idx,
                    "profile_idx": profile.idx,
                    "profile": profile.name,
                    "teacher_filter_candidates": len(context_candidates["details"]),
                    "teacher_filter_correct": len(filtered),
                    "teacher_filter_selected": len(eval_sets[task_idx]),
                    "teacher_filter_require_baseline_wrong": bool(args.teacher_filter_require_baseline_wrong),
                    "seconds": time.time() - started,
                },
            )

    for task_idx, (profile, rows) in enumerate(zip(profiles, eval_sets, strict=True)):
        for idx, row in enumerate(rows):
            append_jsonl(
                questions_path,
                {
                    "task_idx": task_idx,
                    "profile_idx": profile.idx,
                    "profile": profile.name,
                    "idx": idx,
                    "prompt": row["prompt"],
                    "options": row["options"],
                    "answer_idx": row["answer_idx"],
                    "category": row["category"],
                },
            )

    baselines: list[dict] = []
    contexts_metrics: list[dict] = []
    for task_idx, profile in enumerate(profiles):
        progress(f"scoring before-write task={task_idx} profile={profile.name}")
        baseline = evaluate_generic_mc(
            model,
            tokenizer,
            eval_sets[task_idx],
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)
        context_rows = [with_context(row, contexts[task_idx], profile) for row in eval_sets[task_idx]]
        context = evaluate_generic_mc(
            model,
            tokenizer,
            context_rows,
            device,
            args.max_length,
            args.chat_template,
        )
        release_device_cache(device)
        baselines.append(baseline)
        contexts_metrics.append(context)
        row = {
            "stage": "before_write",
            "step": -1,
            "task_idx": task_idx,
            "profile_idx": profile.idx,
            "profile": profile.name,
            "seconds": time.time() - started,
        }
        add_metrics(row, "baseline", baseline)
        add_metrics(row, "context", context)
        append_jsonl(metrics_path, row)
        for stage, metrics in (("baseline", baseline), ("context", context)):
            for idx, detail in enumerate(metrics["details"]):
                append_jsonl(
                    details_path,
                    {"stage": stage, "step": -1, "task_idx": task_idx, "idx": idx, **detail},
                )

    if args.screen_before_write_only:
        append_jsonl(metrics_path, {"stage": "screen_complete", "step": -1, "seconds": time.time() - started})
        progress(f"screen complete; wrote metrics to {metrics_path}")
        return

    acquisition_accuracy: list[float | None] = [None for _ in profiles]
    acquisition_margin: list[float | None] = [None for _ in profiles]
    try:
        for step, profile in enumerate(profiles):
            step_layers = task_write_layers[step]
            progress(f"writing task={step} profile={profile.name} from profile notes layers={step_layers}")
            step_started = time.time()
            original_layers = args.layers
            args.layers = step_layers
            try:
                run_intrinsic_surprise_writes(
                    model,
                    tokenizer,
                    wrappers,
                    lesson_texts[step],
                    args,
                    device,
                    updates_path,
                    slot_id=None,
                    dice_anti_lesson_texts=None,
                )
            finally:
                args.layers = original_layers
            release_device_cache(device)
            append_jsonl(
                metrics_path,
                {
                    "stage": "write_complete",
                    "step": step,
                    "task_idx": step,
                    "profile_idx": profile.idx,
                    "profile": profile.name,
                    "write_layers": step_layers,
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

            for eval_task_idx in range(step + 1):
                eval_profile = profiles[eval_task_idx]
                progress(f"after task={step}, evaluating profile task={eval_task_idx}")
                edited = evaluate_generic_mc(
                    model,
                    tokenizer,
                    eval_sets[eval_task_idx],
                    device,
                    args.max_length,
                    args.chat_template,
                )
                release_device_cache(device)
                if eval_task_idx == step:
                    acquisition_accuracy[eval_task_idx] = edited["accuracy"]
                    acquisition_margin[eval_task_idx] = edited["mean_margin"]
                row = {
                    "stage": "after_step",
                    "step": step,
                    "task_idx": eval_task_idx,
                    "profile_idx": eval_profile.idx,
                    "profile": eval_profile.name,
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
                    append_jsonl(
                        details_path,
                        {"stage": "edited", "step": step, "task_idx": eval_task_idx, "idx": idx, **detail},
                    )

            if args.early_stop_c2w_over >= 0 and sentinel_c2w > args.early_stop_c2w_over:
                progress(f"early stopping after task={step}: sentinel_c2w={sentinel_c2w}>{args.early_stop_c2w_over}")
                break
    finally:
        clear_active_slot_weights(model)

    progress(f"wrote profile continual metrics to {metrics_path}")
    progress(f"wrote profile continual updates to {updates_path}")


if __name__ == "__main__":
    main()
