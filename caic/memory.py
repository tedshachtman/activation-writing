"""Closed-form memory updates for CAIC."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RLSConfig:
    ridge: float = 1e-2
    eta: float = 0.35
    max_update_norm: float | None = 5.0
    min_weight: float = 0.05
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


class PlasticityState:
    """Per-layer inverse covariance used by recursive least squares."""

    def __init__(self, in_features: int, config: RLSConfig):
        self.config = config
        device = torch.device(config.device)
        self.p = torch.eye(in_features, device=device, dtype=config.dtype) / config.ridge

    @property
    def device(self) -> torch.device:
        return self.p.device

    def init_from_keys_diagonal(self, keys: torch.Tensor, mu: float = 1.0) -> None:
        """Initialize P from diagonal background covariance.

        Full covariance inversion is expensive for wide MLPs; diagonal covariance
        keeps the useful "common directions are less plastic" behavior while
        staying cheap enough for the first prototype.
        """

        if keys.numel() == 0:
            return
        keys = _as_column_matrix(keys, self.device, self.config.dtype)
        var = keys.square().mean(dim=1)
        diag = 1.0 / (mu * var + self.config.ridge)
        self.p = torch.diag(diag).to(device=self.device, dtype=self.config.dtype)

    def propose(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(delta_memory, next_p)` without mutating state.

        Shapes use the linear-memory convention:

        - `memory`: `[out_features, in_features]`
        - `keys`: `[in_features, n]`
        - `targets`: `[out_features, n]`
        """

        keys = _as_column_matrix(keys, self.device, self.config.dtype)
        targets = _as_column_matrix(targets, self.device, self.config.dtype)
        memory = memory.to(device=self.device, dtype=self.config.dtype)
        if keys.shape[1] == 0:
            return torch.zeros_like(memory), self.p.clone()
        if targets.shape[1] != keys.shape[1]:
            raise ValueError(f"targets columns {targets.shape[1]} != key columns {keys.shape[1]}")

        if weights is not None:
            weights = weights.to(device=self.device, dtype=self.config.dtype).flatten()
            weights = torch.clamp(weights, min=self.config.min_weight)
            if weights.numel() != keys.shape[1]:
                raise ValueError(f"weights length {weights.numel()} != key columns {keys.shape[1]}")
            scale = torch.sqrt(weights).unsqueeze(0)
            keys = keys * scale
            targets = targets * scale

        p = self.p
        pa = p @ keys
        system = torch.eye(keys.shape[1], device=self.device, dtype=self.config.dtype) + keys.T @ pa
        gain = pa @ torch.linalg.pinv(system)
        error = targets - memory @ keys
        delta = self.config.eta * (error @ gain.T)
        if self.config.max_update_norm is not None:
            delta = clip_frobenius(delta, self.config.max_update_norm)
        next_p = p - gain @ keys.T @ p
        next_p = 0.5 * (next_p + next_p.T)
        return delta, next_p

    def commit(self, next_p: torch.Tensor) -> None:
        self.p = next_p.detach().to(device=self.device, dtype=self.config.dtype)


def _as_column_matrix(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = tensor.detach().to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        return tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def clip_frobenius(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(tensor)
    if norm <= max_norm:
        return tensor
    return tensor * (max_norm / (norm + 1e-12))


def stack_columns(vectors: list[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        raise ValueError("Cannot stack an empty vector list.")
    cols = [vec.detach().float().flatten() for vec in vectors]
    return torch.stack(cols, dim=1)


def select_d_optimal(
    keys_by_row: torch.Tensor,
    k: int,
    weights: torch.Tensor | None = None,
    ridge: float = 1.0,
) -> list[int]:
    """Greedy D-optimal-ish probe selection over row-major keys.

    `keys_by_row` has shape `[n, d]`. The update uses Sherman-Morrison on a
    regularized inverse covariance and scores each candidate by log determinant
    gain times causal weight.
    """

    keys = keys_by_row.detach().float().cpu()
    if keys.ndim != 2:
        raise ValueError(f"Expected [n, d] keys, got {tuple(keys.shape)}")
    n, d = keys.shape
    if n == 0:
        return []
    k = min(k, n)
    keys = torch.nn.functional.normalize(keys, dim=1)
    if weights is None:
        weights = torch.ones(n)
    else:
        weights = torch.clamp(weights.detach().float().cpu().flatten(), min=1e-6)

    inv_cov = torch.eye(d) / ridge
    remaining = set(range(n))
    selected: list[int] = []
    for _ in range(k):
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            a = keys[idx].unsqueeze(1)
            gain = (a.T @ inv_cov @ a).item()
            score = torch.log1p(torch.tensor(gain)).item() + torch.log(weights[idx]).item()
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
        a = keys[best_idx].unsqueeze(1)
        inv_a = inv_cov @ a
        inv_cov = inv_cov - (inv_a @ inv_a.T) / (1.0 + (a.T @ inv_a).item())
    return selected
