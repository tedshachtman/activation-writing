"""Contrastive density-ratio gates for activation-local writes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class DensityRatioGateParams:
    """Runtime parameters for a whitened low-rank density-ratio gate.

    Tensors use row-major activation convention: activations are shaped
    ``[..., d]`` and the low-rank coordinate is ``[..., k]``.
    """

    mu: torch.Tensor
    inv_std: torch.Tensor
    projection: torch.Tensor
    pos_mean: torch.Tensor
    pos_cov_inv: torch.Tensor
    pos_logdet: torch.Tensor
    pos_logp_floor: torch.Tensor
    neg_means: dict[str, torch.Tensor]
    neg_cov_invs: dict[str, torch.Tensor]
    neg_logdets: dict[str, torch.Tensor]
    tau: float
    temperature: float
    kappa: float
    pool_top_k: int
    eigvals: torch.Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "DensityRatioGateParams":
        return DensityRatioGateParams(
            mu=self.mu.to(device=device, dtype=dtype),
            inv_std=self.inv_std.to(device=device, dtype=dtype),
            projection=self.projection.to(device=device, dtype=dtype),
            pos_mean=self.pos_mean.to(device=device, dtype=dtype),
            pos_cov_inv=self.pos_cov_inv.to(device=device, dtype=dtype),
            pos_logdet=self.pos_logdet.to(device=device, dtype=dtype),
            pos_logp_floor=self.pos_logp_floor.to(device=device, dtype=dtype),
            neg_means={key: value.to(device=device, dtype=dtype) for key, value in self.neg_means.items()},
            neg_cov_invs={key: value.to(device=device, dtype=dtype) for key, value in self.neg_cov_invs.items()},
            neg_logdets={key: value.to(device=device, dtype=dtype) for key, value in self.neg_logdets.items()},
            tau=float(self.tau),
            temperature=float(self.temperature),
            kappa=float(self.kappa),
            pool_top_k=int(self.pool_top_k),
            eigvals=self.eigvals.to(device=device, dtype=dtype),
        )


def _as_rows(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [n, d], got {tuple(value.shape)}")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return value


def _covariance(values: torch.Tensor, *, shrink: float, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError(f"Expected [n, k] values, got {tuple(values.shape)}")
    k = values.shape[1]
    mean = values.mean(dim=0)
    centered = values - mean
    denom = max(values.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    diag = torch.diag(torch.diagonal(cov).clamp_min(0.0))
    cov = (1.0 - shrink) * cov + shrink * diag
    scale = torch.diagonal(cov).mean().clamp_min(1.0)
    cov = cov + (ridge * scale) * torch.eye(k, dtype=cov.dtype, device=cov.device)
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def _gaussian_params(
    values: torch.Tensor,
    *,
    shrink: float,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, cov = _covariance(values, shrink=shrink, ridge=ridge)
    chol = torch.linalg.cholesky(cov)
    cov_inv = torch.cholesky_inverse(chol)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return mean, cov_inv, logdet


def _log_gaussian(values: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    diff = values - mean
    mahal = torch.sum((diff @ cov_inv) * diff, dim=-1)
    return -0.5 * (mahal + logdet)


def fit_contrastive_density_gate(
    pos: torch.Tensor,
    neg_groups: dict[str, torch.Tensor],
    cal: torch.Tensor | None = None,
    *,
    rank_q: int = 64,
    rank_k: int = 8,
    beta: float = 1e-3,
    shrink: float = 0.2,
    gaussian_ridge: float = 1e-3,
    target_neg_fpr: float = 1e-3,
    kappa: float = 1.0,
    pool_top_k: int = 8,
) -> tuple[DensityRatioGateParams, dict[str, float]]:
    """Fit a protected low-rank density-ratio gate.

    ``pos`` and all negative groups are row-major activation matrices shaped
    ``[n, d]``. The learned subspace is obtained by calibration whitening,
    compact SVD, then generalized eigenvalue separation of positive scatter
    against protected scatter.
    """

    pos = _as_rows("pos", pos)
    neg_groups = {name: _as_rows(f"neg_groups[{name!r}]", value) for name, value in neg_groups.items() if value.numel()}
    if not neg_groups:
        raise ValueError("fit_contrastive_density_gate requires at least one negative group")
    if cal is None or cal.numel() == 0:
        cal = torch.cat([pos, *neg_groups.values()], dim=0)
    cal = _as_rows("cal", cal)
    if any(value.shape[1] != pos.shape[1] for value in [cal, *neg_groups.values()]):
        raise ValueError("All gate activation matrices must have the same feature dimension")

    mu = cal.mean(dim=0)
    var = cal.var(dim=0, unbiased=False)
    median_var = torch.median(var).clamp_min(1e-6)
    inv_std = torch.rsqrt(var + 1e-3 * median_var)

    def whiten(values: torch.Tensor) -> torch.Tensor:
        return (values - mu) * inv_std

    y_pos = whiten(pos)
    y_neg_groups = {name: whiten(value) for name, value in neg_groups.items()}
    y_all = torch.cat([y_pos, *y_neg_groups.values()], dim=0)

    q = min(rank_q, y_all.shape[0], y_all.shape[1])
    if q <= 0:
        raise ValueError("No usable rank for density gate")
    _u, _s, vh = torch.linalg.svd(y_all, full_matrices=False)
    basis = vh[:q].T.contiguous()

    p = y_pos @ basis
    n_projected = {name: value @ basis for name, value in y_neg_groups.items()}
    s_pos = p.T @ p / max(p.shape[0], 1)
    s_neg = torch.zeros_like(s_pos)
    for value in n_projected.values():
        s_neg = s_neg + value.T @ value / max(value.shape[0], 1)
    s_neg = s_neg / max(len(n_projected), 1)
    scale = torch.diagonal(s_neg).mean().clamp_min(1.0)
    s_neg = 0.5 * (s_neg + s_neg.T) + (beta * scale) * torch.eye(q, dtype=s_neg.dtype, device=s_neg.device)
    s_pos = 0.5 * (s_pos + s_pos.T)

    chol = torch.linalg.cholesky(s_neg)
    chol_inv = torch.linalg.inv(chol)
    sym = chol_inv @ s_pos @ chol_inv.T
    sym = 0.5 * (sym + sym.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    order = torch.argsort(eigvals, descending=True)
    k = min(rank_k, q)
    eigvals = eigvals[order[:k]].clamp_min(0.0)
    eigvecs = eigvecs[:, order[:k]]
    gen_vecs = torch.linalg.solve(chol.T, eigvecs)
    gen_vecs = F.normalize(gen_vecs, dim=0)
    projection = basis @ gen_vecs
    projection = F.normalize(projection, dim=0)

    q_pos = y_pos @ projection
    q_neg_groups = {name: value @ projection for name, value in y_neg_groups.items()}
    pos_mean, pos_cov_inv, pos_logdet = _gaussian_params(q_pos, shrink=shrink, ridge=gaussian_ridge)
    neg_means: dict[str, torch.Tensor] = {}
    neg_cov_invs: dict[str, torch.Tensor] = {}
    neg_logdets: dict[str, torch.Tensor] = {}
    for name, value in q_neg_groups.items():
        mean, cov_inv, logdet = _gaussian_params(value, shrink=shrink, ridge=gaussian_ridge)
        neg_means[name] = mean
        neg_cov_invs[name] = cov_inv
        neg_logdets[name] = logdet

    params = DensityRatioGateParams(
        mu=mu,
        inv_std=inv_std,
        projection=projection,
        pos_mean=pos_mean,
        pos_cov_inv=pos_cov_inv,
        pos_logdet=pos_logdet,
        pos_logp_floor=torch.tensor(0.0),
        neg_means=neg_means,
        neg_cov_invs=neg_cov_invs,
        neg_logdets=neg_logdets,
        tau=0.0,
        temperature=1.0,
        kappa=float(kappa),
        pool_top_k=int(pool_top_k),
        eigvals=eigvals,
    )

    q_pos_for_floor = low_rank_coordinates(pos, params)
    pos_logp_for_floor = _log_gaussian(q_pos_for_floor, params.pos_mean, params.pos_cov_inv, params.pos_logdet)
    params.pos_logp_floor = torch.quantile(pos_logp_for_floor, 0.05)

    pos_scores = score_tokens(pos, params)
    neg_scores_by_group = {name: score_tokens(value, params) for name, value in neg_groups.items()}
    neg_scores = torch.cat(list(neg_scores_by_group.values()), dim=0)
    neg_quantile = torch.quantile(neg_scores, min(max(1.0 - target_neg_fpr, 0.0), 1.0))
    pos_low = torch.quantile(pos_scores, 0.20)
    tau = 0.5 * (neg_quantile + pos_low) if pos_low > neg_quantile else neg_quantile
    separation = (pos_low - neg_quantile).clamp_min(0.0)
    if separation > 0:
        spread = (separation / 8.0).clamp_min(1.0)
    else:
        spread = torch.cat([pos_scores, neg_scores], dim=0).std(unbiased=False).clamp_min(1.0)
    params.tau = float(tau.item())
    params.temperature = float(spread.item())

    pos_active = torch.sigmoid((pos_scores - params.tau) / params.temperature)
    neg_active = torch.sigmoid((neg_scores - params.tau) / params.temperature)
    stats: dict[str, float] = {
        "density_rank_q": float(q),
        "density_rank_k": float(k),
        "density_pool_top_k": float(params.pool_top_k),
        "density_pos_rows": float(pos.shape[0]),
        "density_neg_rows": float(neg_scores.shape[0]),
        "density_tau": params.tau,
        "density_temperature": params.temperature,
        "density_threshold_separation": float(separation.item()),
        "density_pos_score_mean": float(pos_scores.mean().item()),
        "density_pos_score_q20": float(pos_low.item()),
        "density_pos_logp_floor": float(params.pos_logp_floor.item()),
        "density_neg_score_mean": float(neg_scores.mean().item()),
        "density_neg_score_q999": float(neg_quantile.item()),
        "density_pos_gate_mean": float(pos_active.mean().item()),
        "density_neg_gate_mean": float(neg_active.mean().item()),
        "density_top_eig": float(eigvals[0].item()) if eigvals.numel() else 0.0,
    }
    for name, values in neg_scores_by_group.items():
        stats[f"density_{name}_score_mean"] = float(values.mean().item())
        stats[f"density_{name}_gate_mean"] = float(torch.sigmoid((values - params.tau) / params.temperature).mean().item())
        stats[f"density_{name}_rows"] = float(values.shape[0])
    return params, stats


def low_rank_coordinates(values: torch.Tensor, params: DensityRatioGateParams) -> torch.Tensor:
    values = values.float()
    y = (values - params.mu.to(device=values.device, dtype=values.dtype)) * params.inv_std.to(
        device=values.device,
        dtype=values.dtype,
    )
    return y @ params.projection.to(device=values.device, dtype=values.dtype)


def score_tokens(values: torch.Tensor, params: DensityRatioGateParams) -> torch.Tensor:
    """Return token-level log p(pos) - max_g log p(neg_g)."""

    original_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1]).float()
    q = low_rank_coordinates(flat, params)
    pos = _log_gaussian(
        q,
        params.pos_mean.to(device=q.device, dtype=q.dtype),
        params.pos_cov_inv.to(device=q.device, dtype=q.dtype),
        params.pos_logdet.to(device=q.device, dtype=q.dtype),
    )
    neg_values = []
    for name, mean in params.neg_means.items():
        neg_values.append(
            _log_gaussian(
                q,
                mean.to(device=q.device, dtype=q.dtype),
                params.neg_cov_invs[name].to(device=q.device, dtype=q.dtype),
                params.neg_logdets[name].to(device=q.device, dtype=q.dtype),
            )
        )
    neg = torch.stack(neg_values, dim=0).amax(dim=0)
    support_penalty = (pos - params.pos_logp_floor.to(device=q.device, dtype=q.dtype)).clamp_max(0.0)
    return (pos - neg + support_penalty).reshape(original_shape)


def score_sequence(values: torch.Tensor, params: DensityRatioGateParams) -> torch.Tensor:
    """Pool token-level scores into one sequence score.

    Accepts ``[B, T, d]`` or ``[T, d]`` and returns ``[B]`` or a scalar.
    The default top-k pooling is intentional: the object gate is a
    sequence-level "is the learned object present anywhere?" decision, so a few
    content/source tokens should not be averaged away by answer-format tokens.
    """

    scores = score_tokens(values, params)
    kappa = max(float(params.kappa), 1e-6)
    if scores.ndim == 1:
        pool = scores
        if params.pool_top_k > 0 and scores.shape[0] > params.pool_top_k:
            pool = torch.topk(scores, k=params.pool_top_k, dim=0).values
        return kappa * torch.logsumexp(pool / kappa, dim=0) - kappa * torch.log(
            torch.tensor(pool.shape[0], device=scores.device, dtype=scores.dtype)
        )
    pool = scores
    if params.pool_top_k > 0 and scores.shape[1] > params.pool_top_k:
        pool = torch.topk(scores, k=params.pool_top_k, dim=1).values
    return kappa * torch.logsumexp(pool / kappa, dim=1) - kappa * torch.log(
        torch.tensor(pool.shape[1], device=scores.device, dtype=scores.dtype)
    )


def sequence_gate(values: torch.Tensor, params: DensityRatioGateParams) -> torch.Tensor:
    seq = score_sequence(values, params)
    return torch.sigmoid((seq - float(params.tau)) / max(float(params.temperature), 1e-6))
