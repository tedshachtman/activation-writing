import torch
from torch import nn

from caic.contrastive_gate import fit_contrastive_density_gate, score_tokens, sequence_gate
from caic.modeling import AdditiveMemoryLinear


def test_contrastive_density_gate_separates_protected_axis():
    pos = torch.tensor(
        [
            [2.0, 0.0, 0.1],
            [2.2, 0.1, 0.0],
            [1.8, -0.1, 0.1],
            [2.1, 0.0, -0.1],
        ]
    )
    neg = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.1, 2.2, 0.1],
            [-0.1, 1.8, 0.0],
            [0.0, 2.1, -0.1],
        ]
    )
    params, stats = fit_contrastive_density_gate(
        pos,
        {"protected": neg},
        torch.cat([pos, neg], dim=0),
        rank_q=3,
        rank_k=2,
    )

    assert stats["density_pos_score_mean"] > stats["density_neg_score_mean"]
    assert score_tokens(pos, params).mean() > score_tokens(neg, params).mean()
    assert sequence_gate(pos.unsqueeze(0), params).item() > sequence_gate(neg.unsqueeze(0), params).item()


def test_additive_memory_density_object_gate_uses_sequence_context():
    pos = torch.tensor(
        [
            [0.0, 2.0],
            [0.1, 2.1],
            [-0.1, 1.9],
            [0.0, 2.2],
        ]
    )
    neg = torch.tensor(
        [
            [0.0, -2.0],
            [0.1, -2.1],
            [-0.1, -1.9],
            [0.0, -2.2],
        ]
    )
    params, _stats = fit_contrastive_density_gate(
        pos,
        {"protected": neg},
        torch.cat([pos, neg], dim=0),
        rank_q=2,
        rank_k=1,
    )

    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    memory = AdditiveMemoryLinear(base)
    memory.copy_memory_(torch.tensor([[1.0, 0.0]]))
    memory.set_gate_keys_(torch.tensor([[1.0, 0.0]]), threshold=0.5, temperature=50.0)
    memory.set_gate_last_token_only_(True)
    memory.add_density_object_gate_(params)

    with_object = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])
    without_object = torch.tensor([[[0.0, -2.0], [1.0, 0.0]]])

    assert memory(with_object)[0, -1, 0] > memory(without_object)[0, -1, 0]
