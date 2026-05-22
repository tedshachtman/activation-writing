"""Trajectory-Source Operator Consolidation helpers.

TSOC treats the context effect as a local source term in residual dynamics.
These helpers are intentionally small and tensor-only so scripts can compare
source-term writes against older raw activation-delta writes without dragging
the full experiment runner into tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .memory import clip_frobenius


@dataclass
class TSOCUpdateStats:
    positive_rows: int
    negative_rows: int
    ridge: float
    negative_weight: float
    eta: float
    update_fro: float
    target_fro: float
    fit_rmse: float
    negative_rmse: float


def block_source_targets(
    full_inputs: torch.Tensor,
    full_outputs: torch.Tensor,
    null_inputs: torch.Tensor,
    null_outputs: torch.Tensor,
) -> torch.Tensor:
    """Approximate the context source term contributed inside a block.

    For residual blocks, a cheap source proxy is:

        (full_out - null_out) - (full_in - null_in)

    This is not a full Jacobian replay, but it removes the incoming cumulative
    residual difference and leaves the block-local contribution delta.
    """

    for name, tensor in {
        "full_inputs": full_inputs,
        "full_outputs": full_outputs,
        "null_inputs": null_inputs,
        "null_outputs": null_outputs,
    }.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be [rows, dim], got {tuple(tensor.shape)}")
    if full_inputs.shape != full_outputs.shape:
        raise ValueError("full_inputs and full_outputs must have matching shapes.")
    if null_inputs.shape != null_outputs.shape:
        raise ValueError("null_inputs and null_outputs must have matching shapes.")
    if full_inputs.shape != null_inputs.shape:
        raise ValueError("full and null captures must have matching shapes.")
    incoming = full_inputs.float() - null_inputs.float()
    outgoing = full_outputs.float() - null_outputs.float()
    return outgoing - incoming


def project_rows_away_from_basis(rows: torch.Tensor, basis: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Remove the row-space component spanned by `basis`.

    `rows` is `[n, d]`; `basis` is `[k, d]` or `[d]`. The projection is ordinary
    Euclidean by default. A small ridge keeps near-collinear nuisance bases
    numerically stable.
    """

    if rows.numel() == 0 or basis.numel() == 0:
        return rows
    rows_f = rows.float()
    basis_f = basis.float()
    if basis_f.ndim == 1:
        basis_f = basis_f.unsqueeze(0)
    if basis_f.ndim != 2 or basis_f.shape[1] != rows_f.shape[1]:
        raise ValueError(f"Expected basis [k, {rows_f.shape[1]}], got {tuple(basis_f.shape)}")
    gram = basis_f @ basis_f.T + ridge * torch.eye(basis_f.shape[0], dtype=basis_f.dtype)
    coeff = rows_f @ basis_f.T @ torch.linalg.pinv(gram)
    return rows_f - coeff @ basis_f


def principal_components(
    rows: torch.Tensor,
    k: int,
    center: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return top right-singular vectors of a row matrix as `[k, dim]`."""

    if k <= 0 or rows.numel() == 0:
        dim = rows.shape[-1] if rows.ndim >= 1 else 0
        return torch.empty(0, dim, dtype=torch.float32)
    if rows.ndim != 2:
        raise ValueError(f"Expected rows [n, dim], got {tuple(rows.shape)}")
    x = rows.detach().float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    if float(torch.linalg.vector_norm(x).item()) <= eps:
        return torch.empty(0, x.shape[1], dtype=torch.float32)
    _u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[: min(k, vh.shape[0])].contiguous()


def projection_energy_fraction(rows: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of row energy lying in the span of `basis`."""

    if rows.numel() == 0 or basis.numel() == 0:
        return 0.0
    rows_f = rows.float()
    projected = rows_f - project_rows_away_from_basis(rows_f, basis)
    num = torch.linalg.vector_norm(projected).square()
    den = torch.linalg.vector_norm(rows_f).square().clamp_min(1e-12)
    return float((num / den).item())


def solve_ridge_system(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a symmetric ridge system with a Hermitian pseudoinverse fallback."""

    try:
        return torch.linalg.pinv(system, hermitian=True) @ rhs
    except RuntimeError:
        eye = torch.eye(system.shape[0], dtype=system.dtype, device=system.device)
        jitter = system.diag().abs().mean().clamp_min(1.0) * 1e-6
        return torch.linalg.pinv(system + jitter * eye, hermitian=True) @ rhs


def protected_ridge_update(
    keys: torch.Tensor,
    targets: torch.Tensor,
    negative_keys: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
    ridge: float = 1e-2,
    negative_weight: float = 1.0,
    eta: float = 1.0,
    max_update_norm: float | None = None,
) -> tuple[torch.Tensor, TSOCUpdateStats]:
    """Solve a protected closed-form down-projection update.

    Shapes are row-major:

    - `keys`: `[n, in_features]`
    - `targets`: `[n, out_features]`
    - `negative_keys`: `[m, in_features]`

    The solve is equivalent to:

        min_W ||keys W^T - targets||^2
              + negative_weight ||negative_keys W^T||^2
              + ridge ||W||^2

    and returns `W` as `[out_features, in_features]`.
    """

    if keys.ndim != 2:
        raise ValueError(f"keys must be [n, in_features], got {tuple(keys.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must be [n, out_features], got {tuple(targets.shape)}")
    if keys.shape[0] != targets.shape[0]:
        raise ValueError(f"key rows {keys.shape[0]} != target rows {targets.shape[0]}")
    if keys.shape[0] == 0:
        raise ValueError("Cannot solve a TSOC update with zero positive keys.")

    k_pos = keys.detach().float()
    y_pos = targets.detach().float()
    if positive_weights is not None:
        if positive_weights.ndim != 1 or positive_weights.shape[0] != k_pos.shape[0]:
            raise ValueError(
                f"positive_weights must be [{k_pos.shape[0]}], got {tuple(positive_weights.shape)}"
            )
        row_scale = positive_weights.detach().float().clamp_min(0).sqrt().unsqueeze(1)
        k_solve = k_pos * row_scale
        y_solve = y_pos * row_scale
    else:
        k_solve = k_pos
        y_solve = y_pos
    matrices = [k_solve]
    values = [y_solve]
    neg_rows = 0
    if negative_keys is not None and negative_keys.numel() > 0 and negative_weight > 0:
        if negative_keys.ndim != 2 or negative_keys.shape[1] != k_pos.shape[1]:
            raise ValueError(
                f"negative_keys must be [m, {k_pos.shape[1]}], got {tuple(negative_keys.shape)}"
            )
        scale = negative_weight**0.5
        neg = negative_keys.detach().float() * scale
        matrices.append(neg)
        values.append(torch.zeros(neg.shape[0], y_pos.shape[1], dtype=y_pos.dtype))
        neg_rows = neg.shape[0]

    k_all = torch.cat(matrices, dim=0)
    y_all = torch.cat(values, dim=0)
    system = k_all @ k_all.T + ridge * torch.eye(k_all.shape[0], dtype=k_all.dtype)
    update = y_all.T @ solve_ridge_system(system, k_all)
    update = eta * update
    if max_update_norm is not None:
        update = clip_frobenius(update, max_update_norm)

    fit = k_pos @ update.T
    fit_rmse = torch.sqrt(torch.mean((fit - y_pos).square())).item()
    if neg_rows:
        neg_fit = negative_keys.detach().float() @ update.T
        negative_rmse = torch.sqrt(torch.mean(neg_fit.square())).item()
    else:
        negative_rmse = 0.0
    stats = TSOCUpdateStats(
        positive_rows=k_pos.shape[0],
        negative_rows=neg_rows,
        ridge=ridge,
        negative_weight=negative_weight,
        eta=eta,
        update_fro=float(torch.linalg.vector_norm(update).item()),
        target_fro=float(torch.linalg.vector_norm(y_pos).item()),
        fit_rmse=float(fit_rmse),
        negative_rmse=float(negative_rmse),
    )
    return update.contiguous(), stats


def protected_metric_update(
    keys: torch.Tensor,
    targets: torch.Tensor,
    negative_keys: torch.Tensor | None = None,
    output_penalty_basis: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
    ridge: float = 1e-2,
    negative_weight: float = 1.0,
    output_penalty_weight: float = 0.0,
    eta: float = 1.0,
    max_update_norm: float | None = None,
) -> tuple[torch.Tensor, TSOCUpdateStats]:
    """Closed-form ridge update with a protected output metric.

    This solves the same row-major problem as :func:`protected_ridge_update`,
    but adds a low-rank output-side penalty:

        min_W ||K W^T - Y||^2
              + gamma ||K W^T P||^2
              + negative_weight ||B W^T||^2
              + ridge ||W||^2

    where ``P`` is the projection onto ``output_penalty_basis``. Because the
    penalty basis is orthonormalized internally, the solve decomposes into the
    protected output subspace and its complement. No iterative optimizer or
    gradient step is used.
    """

    if output_penalty_basis is None or output_penalty_basis.numel() == 0 or output_penalty_weight <= 0:
        return protected_ridge_update(
            keys,
            targets,
            negative_keys=negative_keys,
            positive_weights=positive_weights,
            ridge=ridge,
            negative_weight=negative_weight,
            eta=eta,
            max_update_norm=max_update_norm,
        )
    if keys.ndim != 2:
        raise ValueError(f"keys must be [n, in_features], got {tuple(keys.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must be [n, out_features], got {tuple(targets.shape)}")
    if keys.shape[0] != targets.shape[0]:
        raise ValueError(f"key rows {keys.shape[0]} != target rows {targets.shape[0]}")
    if keys.shape[0] == 0:
        raise ValueError("Cannot solve a TSOC update with zero positive keys.")

    k_pos = keys.detach().float()
    y_pos = targets.detach().float()
    if positive_weights is not None:
        if positive_weights.ndim != 1 or positive_weights.shape[0] != k_pos.shape[0]:
            raise ValueError(
                f"positive_weights must be [{k_pos.shape[0]}], got {tuple(positive_weights.shape)}"
            )
        row_scale = positive_weights.detach().float().clamp_min(0).sqrt().unsqueeze(1)
        k_solve = k_pos * row_scale
        y_solve = y_pos * row_scale
    else:
        k_solve = k_pos
        y_solve = y_pos

    matrices = [k_solve]
    values = [y_solve]
    neg_rows = 0
    if negative_keys is not None and negative_keys.numel() > 0 and negative_weight > 0:
        if negative_keys.ndim != 2 or negative_keys.shape[1] != k_pos.shape[1]:
            raise ValueError(
                f"negative_keys must be [m, {k_pos.shape[1]}], got {tuple(negative_keys.shape)}"
            )
        scale = negative_weight**0.5
        neg = negative_keys.detach().float() * scale
        matrices.append(neg)
        values.append(torch.zeros(neg.shape[0], y_pos.shape[1], dtype=y_pos.dtype))
        neg_rows = neg.shape[0]

    basis = output_penalty_basis.detach().float()
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    if basis.ndim != 2 or basis.shape[1] != y_pos.shape[1]:
        raise ValueError(f"Expected output_penalty_basis [r, {y_pos.shape[1]}], got {tuple(basis.shape)}")
    q, _r = torch.linalg.qr(basis.T, mode="reduced")
    basis = q.T.contiguous()

    k_all = torch.cat(matrices, dim=0)
    y_all = torch.cat(values, dim=0)
    protected_targets = (y_all @ basis.T) @ basis
    free_targets = y_all - protected_targets

    kk = k_all @ k_all.T
    eye = torch.eye(kk.shape[0], dtype=kk.dtype)
    free_system = kk + ridge * eye
    protected_system = (1.0 + float(output_penalty_weight)) * kk + ridge * eye
    update_free = free_targets.T @ solve_ridge_system(free_system, k_all)
    update_protected = protected_targets.T @ solve_ridge_system(protected_system, k_all)
    update = eta * (update_free + update_protected)
    if max_update_norm is not None:
        update = clip_frobenius(update, max_update_norm)

    fit = k_pos @ update.T
    fit_rmse = torch.sqrt(torch.mean((fit - y_pos).square())).item()
    if neg_rows:
        neg_fit = negative_keys.detach().float() @ update.T
        negative_rmse = torch.sqrt(torch.mean(neg_fit.square())).item()
    else:
        negative_rmse = 0.0
    stats = TSOCUpdateStats(
        positive_rows=k_pos.shape[0],
        negative_rows=neg_rows,
        ridge=ridge,
        negative_weight=negative_weight,
        eta=eta,
        update_fro=float(torch.linalg.vector_norm(update).item()),
        target_fro=float(torch.linalg.vector_norm(y_pos).item()),
        fit_rmse=float(fit_rmse),
        negative_rmse=float(negative_rmse),
    )
    return update.contiguous(), stats


def mean_row_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.numel() == 0:
        return 0.0
    left_n = torch.nn.functional.normalize(left.float(), dim=1)
    right_n = torch.nn.functional.normalize(right.float(), dim=1)
    return float((left_n * right_n).sum(dim=1).mean().item())


def mean_row_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(left.float() - right.float(), dim=1).mean().item())
