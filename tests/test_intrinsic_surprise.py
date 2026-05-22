import torch
from torch import nn
from torch.nn import functional as F

from caic.intrinsic_surprise import (
    attention_flow_values,
    apply_mlp_gauge_seal_,
    down_output_basis_specificity,
    down_value_specificity,
    gauge_canonical_key_scale,
    karp_purify_update,
    lesson_persistence_weights,
    mlp_gauge_salience,
    mlp_activation_normals,
    mlp_weight_prior_scale,
    orca_karp_purify_update,
    ocep_project_update,
    ocep_purify_update,
    prism_q_purify_update,
    qrico_purify_update,
    tdmi_q_transport_scores,
    trace_q_purify_update,
    select_intrinsic_associative_binding_write,
    select_intrinsic_compatibility_residual_write,
    select_intrinsic_conditional_relation_innovation_write,
    select_intrinsic_conjunctive_feature_birth_update,
    select_intrinsic_feature_birth_update,
    select_intrinsic_predictive_residual_write,
    select_intrinsic_relational_aggregate_write,
    select_intrinsic_relational_residual_write,
    select_intrinsic_schur_transport_actuator_write,
    select_intrinsic_surprise_write,
    seal_qrico_purify_update,
    sharp_karp_purify_update,
    shape_surprise_weights,
    signed_anti_erase_update,
    spectra_purify_update,
)
from caic.intrinsic_surprise import _joint_basis_projection, _rank_one_metric_project, _solve_sylvester_psd
from caic.modeling import AdditiveMemoryLinear


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(3, 4, bias=False)
        self.mlp.gate_proj = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.mlp.up_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 4.0],
                        [1.0, 1.0, 0.0],
                    ]
                )
            )
            self.mlp.gate_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 4.0],
                        [1.0, -1.0, 0.0],
                    ]
                )
            )
        self.self_attn = nn.Module()
        self.self_attn.num_heads = 1
        self.self_attn.num_key_value_heads = 1
        self.self_attn.head_dim = 3
        self.self_attn.v_proj = nn.Linear(3, 3, bias=False)
        self.self_attn.o_proj = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.self_attn.v_proj.weight.copy_(torch.eye(3))
            self.self_attn.o_proj.weight.copy_(torch.eye(3))


class GaugeLayer(nn.Module):
    def __init__(self, d_model: int = 5, d_ff: int = 7, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d_model, d_ff, bias=True)
        self.mlp.gate_proj = nn.Linear(d_model, d_ff, bias=True)
        down = nn.Linear(d_ff, d_model, bias=False)
        self.mlp.down_proj = AdditiveMemoryLinear(down)


def _gauge_layer_output(layer: GaugeLayer, x: torch.Tensor) -> torch.Tensor:
    hidden = layer.mlp.up_proj(x) * F.silu(layer.mlp.gate_proj(x))
    return layer.mlp.down_proj(hidden)


def test_mlp_gauge_seal_preserves_swiglu_output_with_memory():
    layer = GaugeLayer()
    wrapper = layer.mlp.down_proj
    with torch.no_grad():
        wrapper.memory.copy_(0.01 * torch.randn_like(wrapper.memory))
    x = torch.randn(3, 5)
    before = _gauge_layer_output(layer, x)
    scales = torch.linspace(1.0, 1.1, steps=wrapper.in_features)

    stats = apply_mlp_gauge_seal_(layer, wrapper, scales)
    after = _gauge_layer_output(layer, x)

    assert torch.allclose(after, before, atol=2e-6, rtol=2e-6)
    assert stats["seal_scaled_channels"] == 6.0


def test_mlp_gauge_salience_detects_sealed_channels():
    torch.manual_seed(1)
    up = torch.randn(6, 4)
    down = torch.randn(4, 6)
    sealed = torch.tensor([1, 4])
    scales = torch.ones(6)
    scales[sealed] = torch.tensor([2.5, 3.0])
    up_sealed = up * scales.unsqueeze(1)
    down_sealed = down / scales.unsqueeze(0)

    salience = mlp_gauge_salience(up_sealed, down_sealed, tau=0.0)
    other = torch.tensor([idx for idx in range(6) if idx not in set(sealed.tolist())])

    assert salience[sealed].mean() > salience[other].mean()
    assert set(torch.topk(salience, k=2).indices.tolist()) == set(sealed.tolist())


def test_signed_anti_erase_shrinks_only_negative_parallel_component():
    current_down = torch.eye(3)
    update = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )
    salience = torch.tensor([4.0, 4.0, 4.0])

    purified, diag = signed_anti_erase_update(
        update,
        current_down,
        salience,
        eta_erase=3.0,
        return_diagnostics=True,
    )

    assert abs(float(purified[0, 0].item())) < abs(float(update[0, 0].item()))
    assert torch.allclose(purified[1], update[1], atol=1e-6)
    assert torch.allclose(purified[2], update[2], atol=1e-6)
    assert diag["seal_destructive_parallel_energy_after"] < diag["seal_destructive_parallel_energy_before"]


def test_gauge_canonical_activation_is_invariant_to_up_down_scaling():
    activations = torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    down = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.5],
        ]
    )
    scales = torch.tensor([1.2, 2.0, 0.75])

    before = activations * gauge_canonical_key_scale(down).unsqueeze(0)
    after = (activations * scales.unsqueeze(0)) * gauge_canonical_key_scale(
        down / scales.unsqueeze(0)
    ).unsqueeze(0)

    assert torch.allclose(after, before, atol=1e-6)


def test_mlp_weight_prior_scale_uses_up_and_gate_norms():
    scale = mlp_weight_prior_scale(DummyLayer(), 4)
    assert scale.shape == (4,)
    assert scale[2] > scale[1] > scale[0]


def test_select_intrinsic_surprise_write_uses_last_token_and_top_features():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
        ]
    )
    down = torch.tensor([[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, -1.0, 0.0]])
    selection = select_intrinsic_surprise_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=1,
        target_feature_top_k=1,
    )
    assert selection.token_indices.tolist() == [1]
    assert selection.keys.shape == (1, 4)
    assert torch.count_nonzero(selection.keys).item() == 1
    assert selection.keys[0, 2] == 10.0
    assert torch.allclose(selection.targets, torch.tensor([[30.0, -10.0]]))


def test_final_aligned_token_mode_excludes_final_token_but_uses_its_feature_space():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 12.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
        ]
    )
    down = torch.eye(2, 4)
    selection = select_intrinsic_surprise_write(
        keys,
        layer,
        down,
        token_mode="final_aligned",
        top_tokens=1,
        feature_top_k=1,
        target_feature_top_k=1,
    )
    assert selection.token_indices.tolist() == [0]


def test_associative_binding_writes_other_surprising_feature_value_to_sparse_key():
    layer = DummyLayer()
    keys = torch.tensor([[2.0, 5.0, 0.0, 0.0]])
    down = torch.tensor(
        [
            [10.0, 0.0, 0.5, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    )
    selection = select_intrinsic_associative_binding_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=2,
        key_feature_top_k=1,
        value_feature_top_k=2,
    )
    assert selection.keys.shape == (1, 4)
    assert torch.count_nonzero(selection.keys).item() == 1
    key_feature = int(selection.feature_indices[0].item())
    assert key_feature in {0, 1}
    if key_feature == 0:
        assert torch.allclose(selection.targets, torch.tensor([[0.0, 15.0]]))
    else:
        assert torch.allclose(selection.targets, torch.tensor([[20.0, 0.0]]))


def test_predictive_residual_write_targets_unpredicted_feature_value():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 4.0, 0.0],
            [3.0, 5.0, 6.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
        ]
    )
    selection = select_intrinsic_predictive_residual_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=3,
        key_feature_top_k=1,
        value_feature_top_k=1,
        prediction_ridge=0.01,
    )
    assert selection.keys.shape == (1, 4)
    assert selection.feature_indices is not None
    assert int(selection.feature_indices[0].item()) == 0
    assert selection.targets[0, 1].abs() > selection.targets[0, 2].abs()
    assert selection.targets[0, 1].abs() > 1.0


def test_relational_residual_write_targets_unexpected_feature_binding():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 4.0, 0.0],
            [3.0, 5.0, 6.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
        ]
    )
    selection = select_intrinsic_relational_residual_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=3,
        key_feature_top_k=1,
        value_feature_top_k=2,
        pair_top_k=1,
        prediction_ridge=0.01,
    )
    assert selection.keys.shape == (1, 4)
    assert selection.feature_indices is not None
    assert int(selection.feature_indices[0].item()) == 0
    assert selection.targets[0, 1].abs() > selection.targets[0, 2].abs()
    assert selection.targets[0, 1].abs() > 1.0


def test_relational_residual_bidirectional_pairs_add_reverse_row():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 4.0, 0.0],
            [3.0, 5.0, 6.0, 0.0],
        ]
    )
    down = torch.eye(3, 4)
    selection = select_intrinsic_relational_residual_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=3,
        key_feature_top_k=1,
        value_feature_top_k=2,
        pair_top_k=1,
        prediction_ridge=0.01,
        bidirectional_pairs=True,
    )
    assert selection.keys.shape == (2, 4)
    assert torch.count_nonzero(selection.keys, dim=1).tolist() == [1, 1]
    assert set(selection.feature_indices.tolist()) == {0, 1}


def test_relational_aggregate_write_combines_pair_values_by_trigger():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [1.0, 0.0, 2.0, 2.0],
            [2.0, 0.0, 4.0, 4.0],
            [3.0, 5.0, 6.0, 7.0],
        ]
    )
    down = torch.eye(4)
    selection = select_intrinsic_relational_aggregate_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=4,
        key_feature_top_k=1,
        value_feature_top_k=3,
        pair_top_k=2,
        prediction_ridge=0.01,
    )
    assert selection.keys.shape == (1, 4)
    assert selection.feature_indices is not None
    assert torch.count_nonzero(selection.target_keys[0]).item() >= 2
    key_idx = int(selection.feature_indices[0].item())
    assert torch.count_nonzero(selection.keys[0]).item() == 1
    assert selection.keys[0, key_idx].abs() > 0
    assert torch.linalg.vector_norm(selection.targets[0]).item() > 1.0


def test_relational_aggregate_context_mode_writes_full_value_context():
    layer = DummyLayer()
    keys = torch.tensor(
        [
            [1.0, 0.0, 2.0, 2.0],
            [2.0, 0.0, 4.0, 4.0],
            [3.0, 5.0, 6.0, 7.0],
        ]
    )
    down = torch.eye(4)
    selection = select_intrinsic_relational_aggregate_write(
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=4,
        key_feature_top_k=1,
        value_feature_top_k=3,
        pair_top_k=2,
        prediction_ridge=0.01,
        relation_value_mode="context",
    )
    key_idx = int(selection.feature_indices[0].item())
    assert selection.target_keys[0, key_idx] == 0
    assert torch.count_nonzero(selection.target_keys[0]).item() >= 2
    assert torch.allclose(selection.targets, selection.target_keys)


def test_lesson_persistence_downweights_one_off_spike():
    scores = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
            [0.0, 4.0],
        ]
    )
    weights = lesson_persistence_weights(scores, threshold_fraction=0.25, min_tokens=2)
    assert weights[1] > weights[0]


def test_down_value_specificity_marks_top_pc_column_as_generic():
    down = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    specificity, basis = down_value_specificity(down, rank=1)
    assert basis.shape == (1, 2)
    assert specificity.shape == (3,)
    assert specificity[2] > specificity[0]


def test_down_output_basis_specificity_downweights_readout_aligned_columns():
    down = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    specificity = down_output_basis_specificity(down, torch.tensor([[1.0, 0.0]]))
    assert specificity.shape == (3,)
    assert specificity[0] < specificity[2]


def test_exponential_surprise_weights_concentrate_on_high_tail():
    raw = torch.tensor([1.0, 1.2, 10.0])
    linear = shape_surprise_weights(raw, mode="linear")
    exponential = shape_surprise_weights(raw, mode="exponential", temperature=1.0, max_weight=100.0)
    assert torch.isclose(exponential.mean(), torch.tensor(1.0))
    assert exponential[-1] / exponential[0] > linear[-1] / linear[0]


def test_mlp_activation_normals_match_swiglu_gradient():
    layer = DummyLayer()
    x = torch.tensor([[1.0, 2.0, 0.5]])
    feature_indices = torch.tensor([[0, 1]])

    normals = mlp_activation_normals(x, layer, feature_indices)

    x_var = x.clone().requires_grad_(True)
    up = layer.mlp.up_proj(x_var)
    gate = layer.mlp.gate_proj(x_var)
    acts = torch.nn.functional.silu(gate) * up
    acts[0, 0].backward(retain_graph=True)
    grad0 = x_var.grad.detach().clone()
    x_var.grad.zero_()
    acts[0, 1].backward()
    grad1 = x_var.grad.detach().clone()

    assert normals.shape == (1, 2, 3)
    assert torch.allclose(normals[0, 0], grad0[0], atol=1e-6)
    assert torch.allclose(normals[0, 1], grad1[0], atol=1e-6)


def test_compatibility_residual_write_selects_unsupported_binding():
    layer = DummyLayer()
    mlp_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 5.0, 0.0, 0.0],
        ]
    )
    # Feature 0 value is orthogonal to the feature-1 normal, so their strong
    # same-token co-instantiation should be selected as an unsupported binding.
    down = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    selection = select_intrinsic_compatibility_residual_write(
        mlp_inputs,
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=2,
        key_feature_top_k=1,
        value_feature_top_k=2,
        pair_top_k=2,
        compatibility_threshold=0.2,
        compatibility_temperature=0.1,
    )

    assert selection.keys.shape[0] == 1
    assert selection.targets.shape == (1, 3)
    assert selection.feature_indices is not None
    assert torch.count_nonzero(selection.keys[0]).item() >= 1
    assert torch.linalg.vector_norm(selection.targets[0]).item() > 0


def test_compatibility_residual_value_mode_uses_down_value_direction():
    layer = DummyLayer()
    mlp_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 5.0, 0.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    selection = select_intrinsic_compatibility_residual_write(
        mlp_inputs,
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=2,
        key_feature_top_k=1,
        value_feature_top_k=2,
        pair_top_k=1,
        compatibility_threshold=0.2,
        compatibility_temperature=0.1,
        target_vector_mode="value",
    )

    assert selection.targets[0, 1].abs() > selection.targets[0, 0].abs()


def test_attention_flow_values_applies_value_output_path():
    layer = DummyLayer()
    source_values = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])

    flowed = attention_flow_values(layer, source_values, torch.tensor([0.25]), mode="vo")

    assert torch.allclose(flowed, 0.25 * source_values)


def test_compatibility_residual_attention_edges_key_source_tokens():
    layer = DummyLayer()
    mlp_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    attn = torch.zeros(1, 3, 3)
    attn[0, 2, 0] = 1.0

    selection = select_intrinsic_compatibility_residual_write(
        mlp_inputs,
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=2,
        key_feature_top_k=1,
        value_feature_top_k=2,
        pair_top_k=1,
        compatibility_threshold=0.2,
        compatibility_temperature=0.1,
        target_vector_mode="value",
        attention_probs=attn,
        attention_edge_top_k=1,
        include_same_token_edges=False,
    )

    assert selection.token_indices.tolist() == [0]
    assert selection.keys[0, 0] == keys[0, 0]
    assert torch.linalg.vector_norm(selection.targets[0]).item() > 0


def test_conditional_relation_innovation_returns_dense_relation_rows():
    layer = DummyLayer()
    mlp_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.1],
        ]
    )
    keys = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [6.0, 6.0, 0.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )

    selection = select_intrinsic_conditional_relation_innovation_write(
        mlp_inputs,
        keys,
        layer,
        down,
        feature_top_k=2,
        relation_rank=2,
        beta=1.0,
        target_mode="svd_value",
        target_scale=0.5,
    )

    assert 1 <= selection.keys.shape[0] <= 2
    assert selection.keys.shape[1] == 4
    assert selection.targets.shape == (selection.keys.shape[0], 3)
    assert selection.weights.shape == (selection.keys.shape[0],)
    assert torch.linalg.vector_norm(selection.keys).item() > 0
    assert torch.linalg.vector_norm(selection.targets).item() > 0


def test_schur_transport_actuator_uses_future_transport_target():
    layer = DummyLayer()
    mlp_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.2],
        ]
    )
    keys = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [6.0, 6.0, 0.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ]
    )
    future = {
        1: mlp_inputs
        + torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, 0.5, 1.0],
            ]
        )
    }

    selection = select_intrinsic_schur_transport_actuator_write(
        mlp_inputs,
        keys,
        layer,
        down,
        layer_idx=0,
        future_mlp_inputs_by_layer=future,
        feature_top_k=2,
        relation_rank=2,
        beta=1.0,
        target_scale=1.0,
        future_layer_horizon=2,
        future_token_top_k=2,
        ordinary_key_rank=1,
        value_projection_features=2,
        schur_ridge=1e-3,
        map_ridge=1e-3,
    )

    assert 1 <= selection.keys.shape[0] <= 2
    assert selection.targets.shape == (selection.keys.shape[0], 3)
    assert selection.negative_keys is not None
    assert selection.negative_keys.shape == selection.keys.shape
    assert selection.diagnostics is not None
    assert selection.diagnostics["star_explained_ratio"] >= 0.0
    assert torch.linalg.vector_norm(selection.targets).item() > 0


def test_karp_shrinks_generic_key_generic_output_atom_more_than_specific_readout():
    # update = generic-key x generic-output + specific-key x same generic-output.
    # KARP should keep the readout-sensitive value direction when it is paired
    # with a specific key, while shrinking the same value paired with a generic
    # key direction.
    update = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 1.0, 4.0, 0.0],
            [0.0, -1.0, 4.0, 0.0],
        ]
    )
    targets = keys @ update.T
    weights = torch.ones(keys.shape[0])
    all_keys = torch.cat(
        [
            keys,
            torch.tensor(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                ]
            ),
        ],
        dim=0,
    )
    all_outputs = torch.cat(
        [
            targets,
            torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ),
        ],
        dim=0,
    )

    result = karp_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=weights,
        all_keys=all_keys,
        all_outputs=all_outputs,
        negative_keys=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        output_basis=torch.tensor([[1.0, 0.0, 0.0]]),
        key_rank=2,
        value_rank=2,
        eta_cross=2.0,
        eta_key=0.0,
        eta_value=0.0,
        low_surprise_quantile=0.2,
    )

    purified = result.update
    assert abs(float(purified[0, 0].item())) < abs(float(update[0, 0].item()))
    assert abs(float(purified[0, 2].item())) > 0.5 * abs(float(update[0, 2].item()))
    assert result.diagnostics["karp_removed_update_ratio"] > 0.1


def test_sharp_karp_shrinks_candidate_shadow_margin_drop_and_keeps_signal():
    # Candidate update has one useful atom (feature 2 -> value 0) and one
    # harmful atom (feature 0 -> value 0) that lowers same-pass shadow margins.
    update = torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    shadow_keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, shadow_keys], dim=0)
    all_outputs = torch.zeros(all_keys.shape[0], 3)
    logit_top_indices = torch.tensor([[0, 1]] * all_keys.shape[0])
    logit_top_values = torch.tensor([[5.0, 0.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1])
    lm_head_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = sharp_karp_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        key_rank=2,
        value_rank=1,
        low_surprise_quantile=0.75,
        confidence_quantile=0.0,
        max_anchors=3,
        signal_top_k=2,
        eta_sharp=0.5,
        shadow_weight=10.0,
        ridge=1e-4,
    )

    purified = result.update
    before_shadow = float((shadow_keys[0] @ update.T)[0].item())
    after_shadow = float((shadow_keys[0] @ purified.T)[0].item())
    after_signal = float((keys[0] @ purified.T)[0].item())
    assert before_shadow < 0.0
    assert after_shadow > before_shadow
    assert after_signal > 4.0
    assert result.diagnostics["sharp_anchor_rows"] > 0
    assert result.diagnostics["sharp_shadow_drop_after"] < result.diagnostics["sharp_shadow_drop_before"]


def test_orca_karp_keeps_target_parallel_option_atom_more_than_orthogonal_atom():
    # The candidate has two atoms under the same specific key: one moves local
    # option contrasts parallel to the relational target, and one creates
    # target-orthogonal option churn. ORCA should shrink the orthogonal value
    # atom more aggressively without needing sentinel examples.
    update = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    low_keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, low_keys], dim=0)
    all_outputs = torch.zeros(all_keys.shape[0], 3)
    logit_top_indices = torch.tensor([[0, 1, 2]] * all_keys.shape[0])
    logit_top_values = torch.tensor([[3.0, 2.0, 1.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1, 2])
    lm_head_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = orca_karp_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        key_rank=2,
        value_rank=2,
        option_top_k=3,
        object_rank=2,
        off_object_rank=2,
        eta_orth=2.0,
        eta_posture=0.0,
        eta_off_object=0.0,
        eta_karp=0.0,
        risk_ratio_cap=100.0,
    )

    purified = result.update
    useful_ratio = abs(float(purified[0, 2].item())) / abs(float(update[0, 2].item()))
    orth_ratio = abs(float(purified[1, 2].item())) / abs(float(update[1, 2].item()))
    assert useful_ratio > orth_ratio
    assert useful_ratio > 0.25
    assert result.diagnostics["orca_signal_retention"] > 0.0
    assert result.diagnostics["orca_atoms_shrunk_gt50"] > 0

    removed = orca_karp_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        key_rank=2,
        value_rank=2,
        option_top_k=3,
        object_rank=2,
        off_object_rank=2,
        eta_orth=2.0,
        eta_posture=0.0,
        eta_off_object=0.0,
        eta_karp=0.0,
        ablation_mode="removed_only",
        risk_ratio_cap=100.0,
    )
    removed_update = removed.update
    assert abs(float(removed_update[1, 2].item())) > abs(float(removed_update[0, 2].item()))
    assert removed.diagnostics["orca_ablation_mode_code"] == 2.0


def test_qrico_joint_deflation_preserves_basis_residual():
    update = torch.zeros(4, 5)
    update[0, 0] = 3.0
    update[3, 4] = 2.0
    key_basis_rows = torch.eye(5)[:1]
    value_basis_rows = torch.eye(4)[:1]

    residual, projected, coeff = _joint_basis_projection(
        update,
        key_basis_rows,
        value_basis_rows,
        mode="joint",
    )

    assert torch.allclose(projected[0, 0], torch.tensor(3.0), atol=1e-5)
    assert torch.allclose(residual[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(residual[3, 4], torch.tensor(2.0), atol=1e-5)
    assert coeff.shape == (1, 1)


def test_qrico_sylvester_matches_vectorized_solve():
    left = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    right = torch.tensor([[1.5, -0.1], [-0.1, 2.5]])
    rhs = torch.tensor([[1.0, 0.5], [-0.25, 0.75]])
    solved, _denom = _solve_sylvester_psd(left, right, rhs)

    assert torch.allclose(left @ solved + solved @ right, rhs, atol=1e-4)


def test_qrico_penalizes_target_orthogonal_option_scramble():
    update = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    low_keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, low_keys], dim=0)
    all_outputs = torch.zeros(all_keys.shape[0], 3)
    logit_top_indices = torch.tensor([[0, 1, 2]] * all_keys.shape[0])
    logit_top_values = torch.tensor([[3.0, 2.0, 1.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1, 2])
    lm_head_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = qrico_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        deflate_key_rank=0,
        deflate_value_rank=0,
        rank=2,
        option_sketch_rank=3,
        target_parallel_rank=1,
        scramble_weight=20.0,
        residual_row_weight_power=0.0,
        layer_evidence_min=0.0,
        layer_evidence_target=0.01,
        output_basis=torch.eye(3)[:2],
        output_weight=0.0,
    )

    purified = result.update
    useful_ratio = abs(float(purified[0, 2].item())) / abs(float(update[0, 2].item()))
    orth_ratio = abs(float(purified[1, 2].item())) / abs(float(update[1, 2].item()))
    assert useful_ratio > orth_ratio
    assert result.diagnostics["qrico_scramble_metric_trace"] > 0.0
    assert result.diagnostics["qrico_capture_ratio"] > 0.0


def test_seal_qrico_returns_signed_anti_erase_update_and_scales():
    update = torch.zeros(3, 4)
    update[0, 2] = 2.0
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, torch.eye(4)[:3]], dim=0)
    all_outputs = torch.zeros(all_keys.shape[0], 3)
    logit_top_indices = torch.tensor([[0, 1, 2]] * all_keys.shape[0])
    logit_top_values = torch.tensor([[3.0, 2.0, 1.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1, 2])
    lm_head_rows = torch.eye(3)
    up_weight = torch.randn(4, 3)
    current_down = torch.eye(3, 4)

    result = seal_qrico_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        up_weight=up_weight,
        current_down_weight=current_down,
        deflate_key_rank=0,
        deflate_value_rank=0,
        rank=2,
        option_sketch_rank=3,
        target_parallel_rank=1,
        scramble_weight=0.0,
        residual_row_weight_power=0.0,
        layer_evidence_min=0.0,
        layer_evidence_target=0.01,
        apply_layer_trust=False,
    )

    assert result.update.shape == update.shape
    assert result.seal_scales.shape == (4,)
    assert result.diagnostics["seal_qrico_enabled"] == 1.0


def test_rank_one_metric_project_clips_hazard_and_preserves_tail():
    m0 = torch.zeros(4, 3)
    m0[0, 0] = 2.0
    m0[1, 1] = 3.0
    g = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    c = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = torch.tensor([2.0, 0.5])
    betas = torch.tensor([100.0, 100.0])

    projected = _rank_one_metric_project(
        m0,
        left_metric_rows=torch.empty(0, 4),
        constraint_keys=g,
        constraint_values=c,
        targets=targets,
        betas=betas,
        ridge=1.0,
    )

    assert abs(float(projected[0, 0].item()) - 2.0) < 1e-4
    assert abs(float(projected[1, 1].item()) - 0.5) < 0.05


def test_spectra_preserves_tail_more_than_hazard():
    update = torch.zeros(3, 4)
    update[0, 2] = 2.0
    update[1, 0] = 3.0
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    low_keys = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, low_keys], dim=0)
    all_outputs = torch.zeros(all_keys.shape[0], 3)
    logit_top_indices = torch.tensor([[0, 1, 2]] * all_keys.shape[0])
    logit_top_values = torch.tensor([[3.0, 2.0, 1.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1, 2])
    lm_head_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = spectra_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        negative_keys=low_keys,
        output_basis=torch.eye(3)[:2],
        quotient_rank=0,
        contrast_rank=2,
        tail_anchors=2,
        hazard_rank=1,
        hazard_budget=0.05,
        beta_tail=100.0,
        beta_hazard=100.0,
        generic_key_rank=1,
        option_top_k=3,
        input_metric_weight=1.0,
        use_orca_quotient=False,
    )

    purified = result.update
    useful_ratio = abs(float(purified[0, 2].item())) / abs(float(update[0, 2].item()))
    hazard_ratio = abs(float(purified[1, 0].item())) / abs(float(update[1, 0].item()))
    assert useful_ratio > hazard_ratio
    assert result.diagnostics["spectra_tail_mass_retention"] > 0.5
    assert result.diagnostics["spectra_hazard_constraints"] >= 1


def test_prism_q_clips_generic_hazard_preserving_signal():
    update = torch.zeros(3, 4)
    update[0, 2] = 2.0
    update[1, 0] = 3.0
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    low_keys = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, low_keys], dim=0)
    all_outputs = torch.zeros(all_keys.shape[0], 3)
    logit_top_indices = torch.tensor([[0, 1, 2]] * all_keys.shape[0])
    logit_top_values = torch.tensor([[3.0, 2.0, 1.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1, 2])
    lm_head_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = prism_q_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        layer_idx=0,
        future_outputs_by_layer={},
        negative_keys=low_keys,
        output_basis=torch.eye(3)[:2],
        horizon=0,
        signal_rank=1,
        hazard_rank=1,
        option_top_k=3,
        generic_key_rank=1,
        low_surprise_rows=3,
        budget=0.05,
        correction_cap=1.0,
        signal_retention_min=0.95,
        use_future_outputs=False,
    )

    purified = result.update
    useful_ratio = abs(float(purified[0, 2].item())) / abs(float(update[0, 2].item()))
    hazard_ratio = abs(float(purified[1, 0].item())) / abs(float(update[1, 0].item()))
    assert useful_ratio > 0.95
    assert useful_ratio > hazard_ratio
    assert result.diagnostics["prism_hazard_spectral_after"] < result.diagnostics["prism_hazard_spectral_before"]
    assert result.diagnostics["prism_signal_retention"] >= 0.95


def test_trace_q_keeps_object_contrast_and_shrinks_ambient_collateral():
    update = torch.zeros(3, 4)
    update[0, 2] = 2.0  # object key -> object contrast
    update[1, 0] = 3.0  # generic key -> ambient contrast
    keys = torch.tensor(
        [
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    low_keys = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    all_keys = torch.cat([keys, low_keys], dim=0)
    logit_top_indices = torch.tensor(
        [
            [0, 2, 3],
            [0, 2, 3],
            [0, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
    logit_top_values = torch.tensor([[3.0, 2.0, 1.0]] * all_keys.shape[0])
    lm_head_indices = torch.tensor([0, 1, 2, 3])
    lm_head_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = trace_q_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=torch.ones(keys.shape[0]),
        all_keys=all_keys,
        token_indices=torch.tensor([0, 1, 2]),
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        negative_keys=low_keys,
        object_endpoints=3,
        ambient_endpoints=3,
        option_top_k=3,
        option_contrasts=1,
        object_rank=1,
        ambient_rank=1,
        generic_key_rank=1,
        target_floor=0.05,
        collateral_weight=4.0,
        layer_trust_threshold=0.01,
    )

    purified = result.update
    useful_ratio = abs(float(purified[0, 2].item())) / abs(float(update[0, 2].item()))
    hazard_ratio = abs(float(purified[1, 0].item())) / abs(float(update[1, 0].item()))
    assert useful_ratio > hazard_ratio
    assert result.diagnostics["trace_q_collateral_after"] < result.diagnostics["trace_q_collateral_before"]
    assert result.diagnostics["trace_q_object_rank"] >= 1


def test_tdmi_q_trusts_object_transport_over_default_manifold():
    update = torch.zeros(3, 4)
    update[0, 2] = 2.0  # object row effect
    update[1, 0] = 2.0  # default row effect
    keys = torch.tensor(
        [
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    all_keys = keys.clone()
    all_outputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.2, 0.0],
        ]
    )
    future = {
        1: torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.2, 0.0],
            ]
        )
    }

    result = tdmi_q_transport_scores(
        update,
        keys=keys,
        targets=targets,
        weights=torch.tensor([10.0, 9.0, 1.0, 1.0]),
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=torch.tensor([0, 1, 2, 3]),
        future_outputs_by_layer=future,
        layer_idx=0,
        object_endpoints=2,
        ambient_endpoints=2,
        object_rank=1,
        ambient_rank=1,
        horizon=1,
        trust_temperature=0.25,
        trust_floor=0.05,
    )

    assert result.row_trust[:2].mean() > result.row_trust[2:].mean()
    assert result.row_signal[:2].mean() > result.row_ambient[:2].mean()
    assert result.row_ambient[2:].mean() > result.row_signal[2:].mean()
    assert result.diagnostics["tdmi_q_object_rank"] >= 1


def test_feature_birth_creates_closed_form_trigger_on_low_impact_neuron():
    layer = DummyLayer()
    mlp_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [10.0, 0.0, 0.5, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    )
    update = select_intrinsic_feature_birth_update(
        mlp_inputs,
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=1,
        value_feature_top_k=1,
        trigger_scale=2.0,
        trigger_ridge=0.0,
    )
    assert update.neuron_indices.tolist() == [3]
    assert update.token_indices.tolist() == [1]
    new_up_row = layer.mlp.up_proj.weight.detach()[3] + update.up_row_delta[0]
    assert torch.allclose(new_up_row @ mlp_inputs[1], torch.tensor(2.0))
    assert update.down_col_delta.shape == (2, 1)
    assert torch.linalg.vector_norm(update.targets).item() > 0


def test_conjunctive_feature_birth_copies_surprising_feature_rows():
    layer = DummyLayer()
    mlp_inputs = torch.tensor([[2.0, 3.0, 0.0]])
    keys = torch.tensor([[4.0, 5.0, 0.0, 0.0]])
    down = torch.tensor(
        [
            [10.0, 0.0, 0.5, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    )
    update = select_intrinsic_conjunctive_feature_birth_update(
        mlp_inputs,
        keys,
        layer,
        down,
        token_mode="last",
        feature_top_k=2,
        key_feature_top_k=2,
        value_feature_top_k=2,
        pair_count=1,
    )
    assert update.neuron_indices.tolist() == [3]
    new_up_row = layer.mlp.up_proj.weight.detach()[3] + update.up_row_delta[0]
    new_gate_row = layer.mlp.gate_proj.weight.detach()[3] + update.gate_row_delta[0]
    copied_up = [
        torch.allclose(new_up_row, layer.mlp.up_proj.weight.detach()[0]),
        torch.allclose(new_up_row, layer.mlp.up_proj.weight.detach()[1]),
    ]
    copied_gate = [
        torch.allclose(new_gate_row, layer.mlp.gate_proj.weight.detach()[0]),
        torch.allclose(new_gate_row, layer.mlp.gate_proj.weight.detach()[1]),
    ]
    assert any(copied_up)
    assert any(copied_gate)
    assert update.trigger_response.abs().item() > 0


def test_ocep_projection_reduces_generic_leakage_preserving_object_effect():
    # update is [d, m]; internally OCEP uses M = update.T [m, d].
    update = torch.tensor(
        [
            [2.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    keys = torch.tensor([[1.0, 0.0, 0.0]])
    weights = torch.tensor([1.0])
    object_basis = torch.tensor([[1.0, 0.0, 0.0]])
    generic_basis = torch.tensor([[0.0, 1.0, 0.0]])
    option_basis = torch.tensor([[1.0, 0.0]])

    purified, diagnostics = ocep_project_update(
        update,
        keys=keys,
        weights=weights,
        object_basis=object_basis,
        generic_basis=generic_basis,
        option_basis=option_basis,
        ridge=1e-6,
        correction_cap=10.0,
    )

    before_object = keys @ update.T
    after_object = keys @ purified.T
    assert torch.allclose(after_object, before_object, atol=1e-4)
    assert abs(float(purified[0, 1].item())) < abs(float(update[0, 1].item())) * 0.01
    assert diagnostics["ocep_leakage_reduction"] > 0.95
    assert diagnostics["ocep_object_delta_ratio"] < 1e-4


def test_ocep_purifier_smoke_builds_current_weight_bases():
    update = torch.tensor(
        [
            [1.5, 2.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ]
    )
    keys = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.8, 0.1, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    weights = torch.ones(2)
    all_keys = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
        ]
    )
    down = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
        ]
    )
    result = ocep_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=weights,
        all_keys=all_keys,
        down_weight=down,
        output_basis=torch.tensor([[1.0, 0.0]]),
        object_rank=2,
        generic_rank=3,
        option_rank=2,
        correction_cap=10.0,
    )
    assert result.update.shape == update.shape
    assert result.object_basis.shape[1] == update.shape[1]
    assert result.generic_basis.shape[1] == update.shape[1]
    assert result.option_basis.shape[1] == update.shape[0]
    assert result.diagnostics["ocep_enabled"] == 1.0
