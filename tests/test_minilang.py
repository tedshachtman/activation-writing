from argparse import Namespace

import torch
from torch import nn

from scripts.minilang_write import (
    build_eval_questions,
    build_exhaustive_modified_questions,
    build_unique_random_questions,
    cap_row_norms,
    center_targets,
    intrinsic_input_penalty_keys,
    intrinsic_span_readout_selection,
    lesson_example_keys,
    merge_negative_keys,
    object_gate_prompts_for_questions,
    question_key,
    render_lesson,
)


def base_args(**overrides):
    values = dict(
        balanced_trace=False,
        ensemble_corpora=1,
        ensemble_seed_stride=100_000,
        ensemble_shared_probes=False,
        eval_max_attempts=10_000,
        eval_mode="random",
        eval_questions=16,
        eval_questions_jsonl="",
        exclude_eval_lesson_overlaps=False,
        exclude_eval_trace_overlaps=False,
        freeze_language_after=None,
        lesson_examples=8,
        lessons=4,
        seed=1,
        trace_probes=4,
    )
    values.update(overrides)
    return Namespace(**values)


def test_unique_random_eval_has_no_duplicate_source_answers():
    questions = build_unique_random_questions(
        12,
        seed=91_001,
        lesson_idx=3,
        category="heldout_translation",
    )
    assert len(questions) == 12
    assert len({question_key(question) for question in questions}) == 12


def test_exhaustive_modified_eval_covers_final_four_lesson_grid():
    questions = build_exhaustive_modified_questions(
        seed=91_001,
        lesson_idx=3,
        category="heldout_translation_exhaustive",
    )
    assert len(questions) == 36
    assert len({question_key(question) for question in questions}) == 36


def test_strict_eval_filters_lesson_and_trace_overlaps():
    args = base_args(
        eval_mode="exhaustive_modified",
        exclude_eval_lesson_overlaps=True,
        exclude_eval_trace_overlaps=True,
    )
    lesson_texts = [
        render_lesson(idx, args.lesson_examples, args.seed)
        for idx in range(args.lessons)
    ]
    questions, metadata = build_eval_questions(args, lesson_texts)
    lesson_keys = lesson_example_keys(lesson_texts)
    assert metadata["eval_original_count"] == 36
    assert metadata["eval_duplicate_removed"] == 0
    assert metadata["eval_final_count"] == len(questions)
    assert metadata["eval_lesson_overlap_count"] > 0
    assert all(question_key(question) not in lesson_keys for question in questions)


def test_object_gate_prompts_end_on_source_sentence():
    class DummyTokenizer:
        pass

    questions = build_unique_random_questions(
        2,
        seed=91_001,
        lesson_idx=3,
        category="heldout_translation",
    )
    prompts = object_gate_prompts_for_questions(DummyTokenizer(), questions, use_chat_template=False)
    assert len(prompts) == 2
    assert all("English:" not in prompt for prompt in prompts)
    assert all(prompt.rstrip().endswith(question.sentence) for prompt, question in zip(prompts, questions, strict=True))


def test_intrinsic_span_readout_selection_uses_lesson_span_positions():
    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) % 128 for char in text]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            return {
                "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
                "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
            }

    model = nn.Module()
    model.lm_head = nn.Linear(3, 128, bias=False)
    prompt = "Nouns: dax=cat.\n"
    input_ids = torch.tensor(CharTokenizer().encode(prompt), dtype=torch.long)
    keys = torch.arange(len(prompt) * 5, dtype=torch.float32).reshape(len(prompt), 5)

    selection = intrinsic_span_readout_selection(
        model,
        CharTokenizer(),
        prompt,
        prompt,
        input_ids,
        keys,
        seed=0,
        max_items=0,
        target_scale=0.25,
    )

    assert selection is not None
    assert selection.keys.shape == (1, 5)
    assert selection.targets.shape == (1, 3)
    assert selection.token_indices.tolist() == [prompt.index("=")]
    assert torch.allclose(selection.keys[0], keys[prompt.index("=")])


def test_cap_row_norms_only_clips_large_rows():
    rows = torch.tensor([[3.0, 4.0], [30.0, 40.0]])
    capped = cap_row_norms(rows, 10.0)

    assert torch.allclose(capped[0], rows[0])
    assert torch.allclose(torch.linalg.vector_norm(capped[1]), torch.tensor(10.0))


def test_center_targets_uses_positive_weights():
    targets = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    centered, mean = center_targets(targets, torch.tensor([3.0, 1.0]))

    assert torch.allclose(mean, torch.tensor([2.5, 0.0]))
    assert torch.allclose(centered.mean(dim=0), torch.tensor([2.5, 0.0]))
    assert torch.allclose((centered * torch.tensor([[3.0], [1.0]])).sum(dim=0), torch.zeros(2))


def test_intrinsic_input_penalty_svd_mode_returns_dense_key_basis():
    selection_keys = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    down = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    keys = intrinsic_input_penalty_keys(
        selection_keys,
        down,
        output_basis=None,
        feature_count=2,
        mode="svd",
    )

    assert keys is not None
    assert keys.shape == (2, 3)
    assert torch.allclose(torch.linalg.vector_norm(keys, dim=1), torch.full((2,), 2.0))


def test_intrinsic_input_penalty_hybrid_mode_combines_svd_and_onehot_rows():
    selection_keys = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    down = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 1.0]])

    keys = intrinsic_input_penalty_keys(
        selection_keys,
        down,
        output_basis=None,
        feature_count=3,
        mode="hybrid",
    )

    assert keys is not None
    assert keys.shape == (3, 3)
    assert torch.count_nonzero(keys[-1]).item() == 1


def test_merge_negative_keys_caps_and_scales_extra_rows():
    primary = torch.tensor([[1.0, 0.0]])
    extra = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])

    merged = merge_negative_keys(primary, extra, max_extra_rows=2, extra_scale=0.5)

    assert merged is not None
    assert merged.shape == (3, 2)
    assert torch.allclose(merged[0], primary[0])
    assert torch.allclose(merged[1:], torch.tensor([[0.0, 0.5], [0.0, 1.5]]))
