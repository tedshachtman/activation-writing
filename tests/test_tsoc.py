import torch

from caic.tsoc import (
    block_source_targets,
    mean_row_cosine,
    mean_row_l2,
    principal_components,
    projection_energy_fraction,
    project_rows_away_from_basis,
    protected_metric_update,
    protected_ridge_update,
)


def test_block_source_targets_remove_incoming_delta():
    full_in = torch.tensor([[2.0, 0.0], [3.0, 1.0]])
    null_in = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    local_source = torch.tensor([[0.5, -0.5], [1.0, 2.0]])
    full_out = full_in + torch.tensor([[4.0, 4.0], [4.0, 4.0]]) + local_source
    null_out = null_in + torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    assert torch.allclose(block_source_targets(full_in, full_out, null_in, null_out), local_source)


def test_project_rows_away_from_basis_removes_basis_component():
    rows = torch.tensor([[3.0, 2.0], [-1.0, 4.0]])
    projected = project_rows_away_from_basis(rows, torch.tensor([1.0, 0.0]))
    assert torch.allclose(projected[:, 0], torch.zeros(2), atol=1e-5)
    assert torch.allclose(projected[:, 1], rows[:, 1], atol=1e-5)


def test_principal_components_identify_dominant_direction():
    rows = torch.tensor([[3.0, 0.1], [2.0, -0.1], [-3.0, 0.0], [-2.0, 0.1]])
    pcs = principal_components(rows, 1)
    assert pcs.shape == (1, 2)
    assert abs(float(torch.dot(pcs[0], torch.tensor([1.0, 0.0])))) > 0.99
    assert projection_energy_fraction(rows, pcs) > 0.98


def test_protected_ridge_update_fits_positive_keys_and_suppresses_negatives():
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    negative_keys = torch.tensor([[1.0, 1.0]])
    update_loose, loose_stats = protected_ridge_update(
        keys,
        targets,
        negative_keys=None,
        ridge=1e-3,
        eta=1.0,
        max_update_norm=None,
    )
    update_protected, protected_stats = protected_ridge_update(
        keys,
        targets,
        negative_keys=negative_keys,
        ridge=1e-3,
        negative_weight=100.0,
        eta=1.0,
        max_update_norm=None,
    )
    loose_negative_norm = torch.linalg.vector_norm(negative_keys @ update_loose.T)
    protected_negative_norm = torch.linalg.vector_norm(negative_keys @ update_protected.T)
    assert loose_stats.fit_rmse < 0.01
    assert protected_negative_norm < loose_negative_norm
    assert protected_stats.negative_rows == 1


def test_protected_metric_update_penalizes_output_basis_predictions():
    keys = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[10.0, 1.0]])
    update_loose, _loose_stats = protected_metric_update(
        keys,
        targets,
        output_penalty_basis=torch.tensor([[1.0, 0.0]]),
        output_penalty_weight=0.0,
        ridge=1e-3,
        eta=1.0,
        max_update_norm=None,
    )
    update_protected, protected_stats = protected_metric_update(
        keys,
        targets,
        output_penalty_basis=torch.tensor([[1.0, 0.0]]),
        output_penalty_weight=100.0,
        ridge=1e-3,
        eta=1.0,
        max_update_norm=None,
    )
    loose_prediction = keys @ update_loose.T
    protected_prediction = keys @ update_protected.T
    assert protected_prediction[0, 0].abs() < loose_prediction[0, 0].abs()
    assert protected_prediction[0, 1] > 0.9
    assert protected_stats.positive_rows == 1


def test_row_metrics_are_directional_and_distance_sensitive():
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    right = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert mean_row_cosine(left, left) > mean_row_cosine(left, right)
    assert mean_row_l2(left, left) < mean_row_l2(left, right)
