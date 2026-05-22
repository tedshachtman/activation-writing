import torch
from torch import nn

from caic.experiment import (
    answer_direction_targets_like,
    fit_linear_probe_direction,
    project_rows_away_from_direction,
)
from caic.memory import PlasticityState, RLSConfig, select_d_optimal
from caic.modeling import AdditiveMemoryLinear, clear_active_slot_weights, set_active_slot_weights_for_prompts
from caic.synthetic import QuestionRecord


def test_rls_update_moves_memory_toward_targets():
    config = RLSConfig(ridge=1e-1, eta=1.0, max_update_norm=None)
    state = PlasticityState(in_features=3, config=config)
    memory = torch.zeros(2, 3)
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    before = torch.linalg.vector_norm(targets - memory @ keys)
    delta, next_p = state.propose(memory, keys, targets)
    after = torch.linalg.vector_norm(targets - (memory + delta) @ keys)
    assert after < before
    assert next_p.shape == state.p.shape


def test_select_d_optimal_returns_unique_indices():
    keys = torch.randn(10, 4)
    selected = select_d_optimal(keys, k=5, ridge=1.0)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert all(0 <= idx < 10 for idx in selected)


def test_activation_slot_router_overrides_lexical_slot_weights():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            base = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                base.weight.zero_()
            self.memory = AdditiveMemoryLinear(base)
            first = self.memory.add_slot_(["not-in-prompts"])
            second = self.memory.add_slot_(["also-not-in-prompts"])
            with torch.no_grad():
                self.memory.slot_memories[first].copy_(torch.tensor([[1.0, 0.0]]))
                self.memory.slot_memories[second].copy_(torch.tensor([[0.0, 1.0]]))

        def forward(self, x):
            return self.memory(x)

    model = TinyModel()
    model._caic_activation_slot_router = lambda prompts: torch.tensor(  # noqa: SLF001
        [[1.0, 0.0], [0.0, 1.0]][: len(prompts)]
    )
    set_active_slot_weights_for_prompts(model, ["first prompt", "second prompt"])
    x = torch.tensor([[[2.0, 10.0]], [[2.0, 10.0]]])
    out = model(x)
    assert torch.allclose(out[:, 0, 0], torch.tensor([2.0, 10.0]))
    clear_active_slot_weights(model)
    assert model.memory._active_slot_weights is None  # noqa: SLF001


def test_activation_object_router_overrides_internal_object_gate():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            base = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                base.weight.zero_()
            self.memory = AdditiveMemoryLinear(base)
            self.memory.copy_memory_(torch.tensor([[1.0, 0.0]]))
            self.memory.set_object_gate_keys_(torch.tensor([[0.0, 1.0]]), threshold=0.5, temperature=50.0)

        def forward(self, x):
            return self.memory(x)

    model = TinyModel()
    x = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    assert torch.all(model(x)[:, 0, 0] < 1e-3)

    model._caic_object_gate_router = lambda prompts: torch.tensor([1.0, 0.25][: len(prompts)])  # noqa: SLF001
    set_active_slot_weights_for_prompts(model, ["object prompt", "weak object prompt"])
    out = model(x)
    assert torch.allclose(out[:, 0, 0], torch.tensor([1.0, 0.25]))
    clear_active_slot_weights(model)
    assert model.memory._active_object_gate is None  # noqa: SLF001


def test_additive_memory_gate_can_be_limited_to_final_token():
    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    memory = AdditiveMemoryLinear(base)
    memory.copy_memory_(torch.tensor([[1.0, 0.0]]))
    memory.set_gate_keys_(torch.tensor([[1.0, 0.0]]), threshold=0.0, temperature=100.0)
    memory.set_gate_last_token_only_(True)

    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    out = memory(x)
    assert torch.allclose(out[0, :, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)


def test_additive_memory_object_gate_uses_sequence_context():
    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    memory = AdditiveMemoryLinear(base)
    memory.copy_memory_(torch.tensor([[1.0, 0.0]]))
    memory.set_gate_keys_(torch.tensor([[1.0, 0.0]]), threshold=0.5, temperature=50.0)
    memory.set_gate_last_token_only_(True)
    memory.set_object_gate_keys_(torch.tensor([[0.0, 1.0]]), threshold=0.5, temperature=50.0)

    with_object = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    without_object = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])

    assert memory(with_object)[0, -1, 0] > 0.99
    assert memory(without_object)[0, -1, 0] < 1e-3


def test_additive_memory_object_gate_floor_keeps_damped_write_active():
    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    memory = AdditiveMemoryLinear(base)
    memory.copy_memory_(torch.tensor([[1.0, 0.0]]))
    memory.set_gate_keys_(torch.tensor([[1.0, 0.0]]), threshold=0.5, temperature=50.0)
    memory.set_gate_last_token_only_(True)
    memory.set_object_gate_keys_(
        torch.tensor([[0.0, 1.0]]),
        threshold=0.5,
        temperature=50.0,
        floor=0.25,
    )

    without_object = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])

    assert torch.allclose(memory(without_object)[0, -1, 0], torch.tensor(0.25), atol=1e-3)


def test_project_rows_away_from_direction_removes_component():
    rows = torch.tensor([[2.0, 3.0], [4.0, -1.0]])
    direction = torch.tensor([1.0, 0.0])
    projected = project_rows_away_from_direction(rows, direction)
    assert torch.allclose(projected[:, 0], torch.zeros(2))
    assert torch.allclose(projected[:, 1], rows[:, 1])


def test_answer_direction_targets_follow_question_labels():
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == " Yes":
                return [0]
            if text == " No":
                return [1]
            raise AssertionError(text)

    model = nn.Module()
    model.lm_head = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    questions = [
        QuestionRecord("positive?", True, [], "test"),
        QuestionRecord("negative?", False, [], "test"),
    ]
    targets = answer_direction_targets_like(
        {0: torch.zeros(4, 3)},
        questions,
        capture_last_tokens=2,
        model=model,
        tokenizer=DummyTokenizer(),
        scale=2.0,
    )
    assert torch.allclose(targets[0][0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(targets[0][1], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(targets[0][2], torch.tensor([-2.0, 0.0, 0.0]))
    assert torch.allclose(targets[0][3], torch.tensor([-2.0, 0.0, 0.0]))


def test_fit_linear_probe_direction_separates_simple_labels():
    features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    labels = torch.tensor([1.0, 1.0, -1.0, -1.0])
    direction, bias = fit_linear_probe_direction(features, labels, ridge=1e-3)
    scores = features @ direction + bias
    assert torch.all(scores[:2] > 0)
    assert torch.all(scores[2:] < 0)
