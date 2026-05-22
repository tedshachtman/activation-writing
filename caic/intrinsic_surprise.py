"""One-pass, weight-intrinsic surprise utilities.

These helpers deliberately avoid null prompts, generated probes, task labels,
or empirical calibration sets. The only prior is induced by the current model
weights, and the only activations come from the single lesson forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from caic.modeling import AdditiveMemoryLinear


@dataclass
class IntrinsicSurpriseSelection:
    keys: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    token_indices: torch.Tensor
    row_scores: torch.Tensor
    feature_scores: torch.Tensor
    target_keys: torch.Tensor
    feature_indices: torch.Tensor | None = None
    negative_keys: torch.Tensor | None = None
    diagnostics: dict[str, float] | None = None


@dataclass
class IntrinsicFeatureBirthUpdate:
    """Sparse paired MLP update for creating lesson-conditioned features."""

    neuron_indices: torch.Tensor
    token_indices: torch.Tensor
    up_row_delta: torch.Tensor
    gate_row_delta: torch.Tensor | None
    down_col_delta: torch.Tensor
    row_scores: torch.Tensor
    feature_scores: torch.Tensor
    target_keys: torch.Tensor
    targets: torch.Tensor
    trigger_rows: torch.Tensor
    trigger_response: torch.Tensor
    feature_indices: torch.Tensor


@dataclass
class KarpPurificationResult:
    """KARP-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    """

    update: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class SharpKarpPurificationResult:
    """SHARP-KARP-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    """

    update: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class OrcaKarpPurificationResult:
    """ORCA-KARP-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    """

    update: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class QricoPurificationResult:
    """Q-RICO-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    """

    update: torch.Tensor
    residual_update: torch.Tensor
    projected_update: torch.Tensor
    key_basis: torch.Tensor
    value_basis: torch.Tensor
    coeff: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class SpectraPurificationResult:
    """SPECTRA-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    """

    update: torch.Tensor
    residual_update: torch.Tensor
    projected_update: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class SealQricoResult:
    """SEAL-Q-RICO-purified update plus gauge-seal diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    ``seal_scales`` is one scale per MLP hidden channel. Applying the scales is
    function-preserving only when the matching up-row/down-column gauge transform
    is applied to the live model weights.
    """

    update: torch.Tensor
    seal_scales: torch.Tensor
    salience: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class PrismPurificationResult:
    """PRISM-Q-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    PRISM-Q keeps the high-rank relational/context-value candidate and clips
    generic-key -> propagated option-hazard functionals outside the same-pass
    innovation cone.
    """

    update: torch.Tensor
    signal_basis: torch.Tensor
    hazard_basis: torch.Tensor
    generic_basis: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class TraceQResult:
    """TRACE-Q-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    This first implementation uses same-pass local option contrasts as a cheap
    endpoint tangent proxy, then separates object-supported readout movement
    from ambient/generic collateral movement.
    """

    update: torch.Tensor
    object_basis: torch.Tensor
    ambient_basis: torch.Tensor
    generic_basis: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class TdmiQResult:
    """TDMI-Q transported/default-manifold row scores.

    ``row_trust`` is one multiplicative weight per selected relational row.
    The first implementation uses same-pass hidden-state transport proxies
    rather than exact downstream VJPs: rows are trusted when their proposed
    residual effect lies more in transported object/default-separated hidden
    manifolds than in low-surprise default manifolds.
    """

    row_trust: torch.Tensor
    row_signal: torch.Tensor
    row_ambient: torch.Tensor
    object_basis: torch.Tensor
    ambient_basis: torch.Tensor
    diagnostics: dict[str, float]


@dataclass
class OcepPurificationResult:
    """OCEP-purified update plus diagnostics.

    ``update`` is row-major MLP down delta, shaped ``[d_model, d_ff]``.
    """

    update: torch.Tensor
    object_basis: torch.Tensor
    generic_basis: torch.Tensor
    option_basis: torch.Tensor
    diagnostics: dict[str, float]


def shape_surprise_weights(
    raw_weights: torch.Tensor,
    *,
    mode: str = "linear",
    temperature: float = 1.0,
    max_weight: float = 100.0,
) -> torch.Tensor:
    """Normalize selected surprise weights for a closed-form write.

    ``linear`` keeps the previous behavior: row weights are proportional to the
    selected surprise score. ``exponential`` robust-standardizes the selected
    scores and exponentiates them, so moderately surprising rows contribute very
    little while the high-surprise tail dominates the fit.
    """

    weights = raw_weights.detach().float().clamp_min(1e-12)
    if weights.numel() == 0:
        return weights
    if mode == "linear":
        shaped = weights
    elif mode == "exponential":
        centered = weights - weights.median()
        scale = centered.abs().median()
        if float(scale.item()) <= 1e-12:
            scale = weights.std(unbiased=False).clamp_min(1e-12)
        z = centered / scale.clamp_min(1e-12)
        shaped = torch.exp((z / max(float(temperature), 1e-6)).clamp(max=20.0))
        if max_weight > 0:
            shaped = shaped.clamp(max=float(max_weight))
    else:
        raise ValueError(f"Unknown surprise weight mode {mode!r}")
    return (shaped / shaped.mean().clamp_min(1e-12)).contiguous()


def _linear_weight(module: nn.Module | None) -> torch.Tensor | None:
    if module is None:
        return None
    if isinstance(module, AdditiveMemoryLinear):
        return module.base.weight.detach().float().cpu()
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight.detach().float().cpu()
    return None


def _mlp_linear(module: nn.Module, names: tuple[str, ...]) -> nn.Module | None:
    for name in names:
        child = getattr(module, name, None)
        if child is not None:
            return child
    return None


def _mlp_up_module(layer: nn.Module) -> nn.Module | None:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    return _mlp_linear(mlp, ("up_proj", "fc1", "c_fc", "dense_h_to_4h"))


def _mlp_gate_module(layer: nn.Module) -> nn.Module | None:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    return _mlp_linear(mlp, ("gate_proj",))


def _attention_module(layer: nn.Module) -> nn.Module | None:
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)


def _attention_v_module(layer: nn.Module) -> nn.Module | None:
    attn = _attention_module(layer)
    if attn is None:
        return None
    return _mlp_linear(attn, ("v_proj", "value", "c_attn_v"))


def _attention_o_module(layer: nn.Module) -> nn.Module | None:
    attn = _attention_module(layer)
    if attn is None:
        return None
    return _mlp_linear(attn, ("o_proj", "out_proj", "c_proj", "dense"))


def attention_flow_values(
    layer: nn.Module,
    source_values: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    mode: str = "vo",
) -> torch.Tensor:
    """Flow residual value columns through one frozen attention edge.

    ``source_values`` is ``[d_model, n]``. ``head_weights`` is one attention
    weight per query head. ``vo`` applies the layer's value and output
    projections; ``identity`` uses mean attention mass times the source value.
    """

    values = source_values.detach().float().cpu()
    heads = head_weights.detach().float().cpu()
    if values.ndim != 2:
        raise ValueError(f"source_values must be [d_model, n], got {tuple(values.shape)}")
    if mode == "identity":
        return heads.mean().clamp_min(0.0) * values
    if mode != "vo":
        raise ValueError(f"Unknown attention flow mode {mode!r}")
    v_weight = _linear_weight(_attention_v_module(layer))
    o_weight = _linear_weight(_attention_o_module(layer))
    attn = _attention_module(layer)
    if v_weight is None or o_weight is None or attn is None:
        return heads.mean().clamp_min(0.0) * values
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    if head_dim <= 0:
        num_heads_attr = int(getattr(attn, "num_heads", 0) or getattr(attn, "num_attention_heads", 0) or 0)
        if num_heads_attr > 0 and o_weight.shape[1] % num_heads_attr == 0:
            head_dim = o_weight.shape[1] // num_heads_attr
    if head_dim <= 0:
        return heads.mean().clamp_min(0.0) * values
    q_heads = min(int(heads.numel()), o_weight.shape[1] // head_dim)
    kv_heads = v_weight.shape[0] // head_dim
    if q_heads <= 0 or kv_heads <= 0:
        return heads.mean().clamp_min(0.0) * values
    if q_heads % kv_heads != 0:
        repeat = max(1, q_heads // kv_heads)
    else:
        repeat = q_heads // kv_heads
    projected = v_weight @ values
    projected = projected[: kv_heads * head_dim].reshape(kv_heads, head_dim, values.shape[1])
    repeated = projected.repeat_interleave(repeat, dim=0)[:q_heads]
    weighted = repeated * heads[:q_heads].reshape(q_heads, 1, 1)
    flat = weighted.reshape(q_heads * head_dim, values.shape[1])
    o = o_weight[:, : q_heads * head_dim]
    return (o @ flat).contiguous()


def _select_tokens_from_scores(
    row_scores: torch.Tensor,
    *,
    token_mode: str,
    top_tokens: int,
) -> torch.Tensor:
    if token_mode == "last":
        return torch.tensor([row_scores.shape[0] - 1], dtype=torch.long)
    if token_mode == "top":
        row_k = max(1, min(int(top_tokens), row_scores.shape[0]))
        return torch.topk(row_scores, k=row_k, dim=0).indices.sort().values
    if token_mode == "all":
        return torch.arange(row_scores.shape[0], dtype=torch.long)
    raise ValueError(f"Unknown token_mode {token_mode!r}")


def _select_tokens_from_feature_scores(
    feature_scores: torch.Tensor,
    row_scores: torch.Tensor,
    *,
    token_mode: str,
    top_tokens: int,
    feature_top_k: int,
) -> torch.Tensor:
    if token_mode != "final_aligned":
        return _select_tokens_from_scores(row_scores, token_mode=token_mode, top_tokens=top_tokens)
    if feature_scores.shape[0] == 1:
        return torch.tensor([0], dtype=torch.long)
    final_scores = feature_scores[-1].clamp_min(0.0)
    k = max(1, min(int(feature_top_k), final_scores.shape[0]))
    final_idx = torch.topk(final_scores, k=k, dim=0).indices
    final_vec = torch.zeros_like(final_scores)
    final_vec[final_idx] = final_scores[final_idx]
    overlap = (feature_scores.clamp_min(0.0) * final_vec.unsqueeze(0)).sum(dim=1)
    denom = (
        torch.linalg.vector_norm(feature_scores.clamp_min(0.0), dim=1)
        * torch.linalg.vector_norm(final_vec).clamp_min(1e-12)
    )
    aligned = overlap / denom.clamp_min(1e-12)
    combined = row_scores.clamp_min(0.0).sqrt() * aligned.clamp_min(0.0)
    combined[-1] = -torch.inf
    row_k = max(1, min(int(top_tokens), combined.shape[0] - 1))
    return torch.topk(combined, k=row_k, dim=0).indices.sort().values


def _closed_form_trigger_rows(
    inputs: torch.Tensor,
    *,
    trigger_scale: float,
    ridge: float,
) -> torch.Tensor:
    """Minimum-norm rows R with R @ inputs.T approximately trigger_scale * I."""

    x = inputs.detach().float().cpu()
    gram = x @ x.T
    system = gram + float(ridge) * torch.eye(gram.shape[0], dtype=gram.dtype)
    return float(trigger_scale) * torch.linalg.solve(system, x)


def choose_low_impact_neurons(
    layer: nn.Module,
    down_weight: torch.Tensor,
    keys: torch.Tensor,
    count: int,
    avoid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick MLP neurons whose current value path appears cheapest to repurpose.

    This uses only weights plus the lesson forward. A neuron with a small down
    column and low lesson activation is less likely to carry broadly useful
    behavior than a large, active value column.
    """

    down = down_weight.detach().float().cpu()
    if down.ndim != 2:
        raise ValueError(f"down_weight must be [out, features], got {tuple(down.shape)}")
    count = max(1, min(int(count), down.shape[1]))
    down_norm = torch.linalg.vector_norm(down, dim=0)
    keys_f = keys.detach().float().cpu()
    lesson_rms = torch.sqrt(keys_f.square().mean(dim=0).clamp_min(0.0)) if keys_f.numel() else torch.zeros_like(down_norm)
    up_weight = _linear_weight(_mlp_up_module(layer))
    gate_weight = _linear_weight(_mlp_gate_module(layer))
    if up_weight is not None and up_weight.shape[0] == down.shape[1]:
        upstream_norm = torch.linalg.vector_norm(up_weight, dim=1)
        if gate_weight is not None and gate_weight.shape[0] == down.shape[1]:
            upstream_norm = torch.sqrt(upstream_norm * torch.linalg.vector_norm(gate_weight, dim=1).clamp_min(1e-12))
        upstream_norm = upstream_norm / torch.median(upstream_norm[upstream_norm > 0]).clamp_min(1e-12)
    else:
        upstream_norm = torch.ones_like(down_norm)
    down_scaled = down_norm / torch.median(down_norm[down_norm > 0]).clamp_min(1e-12)
    lesson_scaled = lesson_rms / torch.median(lesson_rms[lesson_rms > 0]).clamp_min(1e-12) if torch.any(lesson_rms > 0) else lesson_rms
    impact = down_scaled * upstream_norm + 0.05 * lesson_scaled
    if avoid is not None and avoid.numel() > 0:
        avoid_idx = avoid.detach().long().cpu()
        avoid_idx = avoid_idx[(avoid_idx >= 0) & (avoid_idx < impact.shape[0])]
        impact[avoid_idx] = torch.inf
    return torch.topk(impact, k=count, largest=False).indices.sort().values


def mlp_weight_prior_scale(layer: nn.Module, in_features: int, eps: float = 1e-3) -> torch.Tensor:
    """Return a per-MLP-feature scale derived only from current weights.

    For gated MLPs, the down-projection input is roughly
    ``act(gate_proj(x)) * up_proj(x)``. Under an isotropic residual prior, a
    feature's natural scale is therefore proportional to the product of the
    corresponding gate/up row norms. For non-gated MLPs we fall back to the up
    row norm, then to a unit scale.
    """

    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return torch.ones(in_features)
    up_weight = _linear_weight(getattr(mlp, "up_proj", None) or getattr(mlp, "fc1", None) or getattr(mlp, "c_fc", None))
    gate_weight = _linear_weight(getattr(mlp, "gate_proj", None))
    if up_weight is not None and up_weight.shape[0] == in_features:
        scale = torch.linalg.vector_norm(up_weight, dim=1)
        if gate_weight is not None and gate_weight.shape[0] == in_features:
            gate_scale = torch.linalg.vector_norm(gate_weight, dim=1)
            scale = scale * gate_scale
    else:
        scale = torch.ones(in_features)
    median = torch.median(scale[scale > 0]) if torch.any(scale > 0) else torch.tensor(1.0)
    median = median.clamp_min(1e-12)
    return scale.clamp_min(float(eps) * median) / median


def effective_down_weight(wrapper: AdditiveMemoryLinear) -> torch.Tensor:
    """Current down-projection weight, including additive memory."""

    return (wrapper.base.weight.detach().float().cpu() + wrapper.memory.detach().float().cpu()).contiguous()


def base_down_weight(wrapper: AdditiveMemoryLinear) -> torch.Tensor:
    """Frozen/base down-projection weight, excluding additive memory."""

    return wrapper.base.weight.detach().float().cpu().contiguous()


def metric_column_norm_sq(
    values: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Column-wise squared norm under ``I + B^T B``.

    ``values`` is ``[d, m]``. ``output_metric_basis`` is a row basis
    ``[r, d]``. The identity term keeps the metric well-conditioned and avoids
    treating non-readout directions as zero-norm.
    """

    vals = torch.nan_to_num(values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if vals.ndim != 2:
        raise ValueError(f"values must be [d,m], got {tuple(vals.shape)}")
    norm_sq = vals.square().sum(dim=0)
    if output_metric_basis is not None and output_metric_basis.numel() > 0:
        basis = torch.nan_to_num(
            output_metric_basis.detach().float().cpu(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)
        if basis.ndim == 2 and basis.shape[1] == vals.shape[0]:
            proj = basis @ vals
            norm_sq = norm_sq + proj.square().sum(dim=0)
    return norm_sq.clamp_min(float(eps))


def metric_row_norm_sq(
    rows: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Row-wise squared norm under ``I + B^T B`` for ``rows`` shaped ``[m,d]``."""

    row_f = torch.nan_to_num(rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if row_f.ndim != 2:
        raise ValueError(f"rows must be [m,d], got {tuple(row_f.shape)}")
    norm_sq = row_f.square().sum(dim=1)
    if output_metric_basis is not None and output_metric_basis.numel() > 0:
        basis = torch.nan_to_num(
            output_metric_basis.detach().float().cpu(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)
        if basis.ndim == 2 and basis.shape[1] == row_f.shape[1]:
            proj = row_f @ basis.T
            norm_sq = norm_sq + proj.square().sum(dim=1)
    return norm_sq.clamp_min(float(eps))


def gauge_canonical_key_scale(
    down_weight: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the gauge-invariant activation multiplier ``||down[:,j]||_G``."""

    return torch.sqrt(metric_column_norm_sq(down_weight, output_metric_basis, eps=eps)).contiguous()


def mlp_gauge_salience(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
    *,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Detect function-preserving up/down gauge imbalance from weights only."""

    up = torch.nan_to_num(up_weight.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    down = torch.nan_to_num(down_weight.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if up.ndim != 2 or down.ndim != 2 or up.shape[0] != down.shape[1]:
        raise ValueError(f"up/down shapes must be [m,d] and [d,m], got {tuple(up.shape)} and {tuple(down.shape)}")
    up_norm = torch.linalg.vector_norm(up, dim=1).clamp_min(float(eps))
    down_norm = torch.sqrt(metric_column_norm_sq(down, output_metric_basis, eps=eps))
    z = torch.log(up_norm) - torch.log(down_norm.clamp_min(float(eps)))
    center = torch.median(z)
    mad = torch.median((z - center).abs()).clamp_min(float(eps))
    return F.softplus((z - center) / mad - float(tau)).contiguous()


def _metric_row_dot(
    rows_a: torch.Tensor,
    rows_b: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
) -> torch.Tensor:
    dot = (rows_a * rows_b).sum(dim=1)
    if output_metric_basis is not None and output_metric_basis.numel() > 0:
        basis = torch.nan_to_num(
            output_metric_basis.detach().float().cpu(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)
        if basis.ndim == 2 and basis.shape[1] == rows_a.shape[1]:
            dot = dot + ((rows_a @ basis.T) * (rows_b @ basis.T)).sum(dim=1)
    return dot


def signed_anti_erase_update(
    update_m_by_d: torch.Tensor,
    current_down_d_by_m: torch.Tensor,
    salience_m: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
    *,
    eta_erase: float = 2.0,
    eps: float = 1e-6,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Shrink only anti-parallel edits to salient/current down-value columns."""

    update = torch.nan_to_num(update_m_by_d.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    down = torch.nan_to_num(current_down_d_by_m.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    sal = torch.nan_to_num(salience_m.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if update.ndim != 2 or down.ndim != 2 or update.shape[0] != down.shape[1] or update.shape[1] != down.shape[0]:
        raise ValueError(
            "update/current_down shapes must be [m,d] and [d,m], "
            f"got {tuple(update.shape)} and {tuple(down.shape)}"
        )
    if sal.ndim != 1 or sal.shape[0] != update.shape[0]:
        raise ValueError(f"salience must be [{update.shape[0]}], got {tuple(sal.shape)}")

    down_rows = down.T.contiguous()
    denom = metric_row_norm_sq(down_rows, output_metric_basis, eps=eps)
    alpha = _metric_row_dot(update, down_rows, output_metric_basis) / denom.clamp_min(float(eps))
    median_denom = torch.median(denom).clamp_min(float(eps))
    risk = (denom / median_denom).clamp_min(0.0)
    destructive = torch.relu(-alpha)
    shrink = 1.0 + float(eta_erase) * sal * risk
    alpha_star = torch.where(alpha < 0.0, alpha / shrink.clamp_min(1.0), alpha)
    purified = update + (alpha_star - alpha).unsqueeze(1) * down_rows

    if not return_diagnostics:
        return purified.contiguous()
    after_destructive = torch.relu(-alpha_star)
    before_energy = (sal * risk * destructive.square()).sum()
    after_energy = (sal * risk * after_destructive.square()).sum()
    parallel_energy = (risk * alpha.square()).sum().clamp_min(float(eps))
    diagnostics = {
        "seal_anti_erase_ratio": float((after_energy / before_energy.clamp_min(float(eps))).item())
        if float(before_energy.item()) > 0.0
        else 0.0,
        "seal_destructive_parallel_energy_before": float(before_energy.item()),
        "seal_destructive_parallel_energy_after": float(after_energy.item()),
        "seal_parallel_energy": float(parallel_energy.item()),
        "seal_collision_fraction": float((before_energy / parallel_energy).item()),
        "seal_destructive_channels": float((alpha < 0.0).sum().item()),
        "seal_salience_mean": float(sal.mean().item()) if sal.numel() else 0.0,
        "seal_salience_max": float(sal.max().item()) if sal.numel() else 0.0,
    }
    return purified.contiguous(), diagnostics


def compute_gauge_seal_scales(
    keys: torch.Tensor,
    update_m_by_d: torch.Tensor,
    row_weights: torch.Tensor,
    output_metric_basis: torch.Tensor | None = None,
    *,
    eta_seal: float = 0.05,
    max_scale: float = 1.10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-channel function-preserving gauge-seal scales."""

    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    update = torch.nan_to_num(update_m_by_d.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(row_weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if k.ndim != 2 or update.ndim != 2 or k.shape[1] != update.shape[0]:
        raise ValueError(f"keys/update shapes must be [n,m] and [m,d], got {tuple(k.shape)} and {tuple(update.shape)}")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"row_weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if float(eta_seal) <= 0.0 or float(max_scale) <= 1.0 or k.shape[0] == 0:
        return torch.ones(update.shape[0], dtype=torch.float32)

    w_norm = w / w.mean().clamp_min(float(eps))
    key_energy = (k.square() * w_norm.unsqueeze(1)).sum(dim=0)
    update_energy = metric_row_norm_sq(update, output_metric_basis, eps=eps)
    importance = key_energy * update_energy
    positive = importance[importance > float(eps)]
    if positive.numel() == 0:
        return torch.ones(update.shape[0], dtype=torch.float32)
    q95 = torch.quantile(positive, 0.95).clamp_min(float(eps))
    raw = torch.exp(float(eta_seal) * (importance / q95).clamp(0.0, 1.0))
    return raw.clamp(min=1.0, max=float(max_scale)).contiguous()


def build_ocep_object_basis(
    keys: torch.Tensor,
    weights: torch.Tensor,
    *,
    rank: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted row basis for object keys whose effects should be preserved."""

    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if k.ndim != 2:
        raise ValueError(f"keys must be [n,m], got {tuple(k.shape)}")
    if k.shape[0] == 0 or int(rank) <= 0:
        return torch.empty(0, k.shape[1], dtype=torch.float32)
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        w = torch.ones(k.shape[0], dtype=torch.float32)
    w = w / w.mean().clamp_min(float(eps))
    rows = k * w.sqrt().unsqueeze(1)
    basis = _orthonormal_basis([rows], k.shape[1])
    if basis.shape[0] > int(rank):
        basis = basis[: int(rank)]
    return basis.contiguous()


def build_ocep_option_basis(
    targets: torch.Tensor,
    *,
    output_basis: torch.Tensor | None = None,
    token_indices: torch.Tensor | None = None,
    logit_top_values: torch.Tensor | None = None,
    logit_top_indices: torch.Tensor | None = None,
    lm_head_indices: torch.Tensor | None = None,
    lm_head_rows: torch.Tensor | None = None,
    rank: int = 64,
    output_rank: int = 32,
    local_rank: int = 32,
) -> torch.Tensor:
    """Build option/readout contrast rows for collateral-evidence detection."""

    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if y.ndim != 2:
        raise ValueError(f"targets must be [n,d], got {tuple(y.shape)}")
    d_model = y.shape[1]
    if int(rank) <= 0:
        return torch.empty(0, d_model, dtype=torch.float32)
    parts: list[torch.Tensor | None] = []
    if output_basis is not None and output_basis.numel() > 0 and int(output_rank) > 0:
        out = torch.nan_to_num(output_basis.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            parts.append(out[: min(int(output_rank), out.shape[0])])
    target_rank = max(1, min(max(1, int(rank) // 4), y.shape[0]))
    parts.append(_fast_basis_with_rows(y, target_rank, d_model))
    if (
        token_indices is not None
        and logit_top_values is not None
        and logit_top_indices is not None
        and lm_head_indices is not None
        and lm_head_rows is not None
        and int(local_rank) > 0
        and logit_top_values.numel() > 0
        and logit_top_indices.numel() > 0
    ):
        tok = token_indices.detach().cpu().long().clamp(min=0, max=logit_top_indices.shape[0] - 1)
        top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        top_idx = logit_top_indices.detach().cpu().long()
        usable_k = max(2, min(top_idx.shape[1], top_vals.shape[1], int(local_rank) * 2))
        selected_top = top_idx[tok, :usable_k]
        selected_vals = top_vals[tok, :usable_k]
        selected_rows = _lookup_weight_rows(
            selected_top,
            stored_indices=lm_head_indices.detach().cpu().long(),
            stored_rows=torch.nan_to_num(lm_head_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0),
        )
        probs = torch.softmax(selected_vals, dim=1)
        expected = (probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
        contrast_rows = (selected_rows - expected).reshape(-1, d_model)
        parts.append(_fast_basis_with_rows(contrast_rows, int(local_rank), d_model))
    basis = _orthonormal_basis(parts, d_model)
    if basis.shape[0] > int(rank):
        basis = basis[: int(rank)]
    return basis.contiguous()


def build_ocep_generic_key_basis(
    all_keys: torch.Tensor,
    down_weight: torch.Tensor,
    option_basis: torch.Tensor,
    *,
    up_weight: torch.Tensor | None = None,
    negative_keys: torch.Tensor | None = None,
    rank: int = 128,
    low_surprise_rank: int = 32,
    weight_anchor_rank: int = 96,
    protected_rank: int = 32,
    low_surprise_quantile: float = 0.35,
) -> torch.Tensor:
    """Build generic key anchors from current weights and same-pass ambient rows."""

    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    down = torch.nan_to_num(down_weight.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    c = torch.nan_to_num(option_basis.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if all_k.ndim != 2 or down.ndim != 2 or all_k.shape[1] != down.shape[1]:
        raise ValueError("OCEP generic key inputs must align as all_keys [T,m], down [d,m]")
    d_ff = all_k.shape[1]
    if int(rank) <= 0:
        return torch.empty(0, d_ff, dtype=torch.float32)

    parts: list[torch.Tensor | None] = []
    if c.numel() > 0 and c.ndim == 2 and c.shape[1] == down.shape[0] and int(weight_anchor_rank) > 0:
        option_key_rows = c @ down
        parts.append(_fast_basis_with_rows(option_key_rows, int(weight_anchor_rank), d_ff))
    if up_weight is not None and up_weight.numel() > 0 and int(weight_anchor_rank) > 0:
        up = torch.nan_to_num(up_weight.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if up.ndim == 2 and up.shape[0] == d_ff:
            rows_rank = max(1, min(int(weight_anchor_rank) // 4, d_ff))
            # A full up-projection SVD is too expensive for all-layer runs.
            # High-upstream-norm one-hot anchors give a cheap current-weight
            # salience sketch in the same key coordinates.
            top_idx = torch.topk(torch.linalg.vector_norm(up, dim=1), k=rows_rank).indices
            onehot = torch.zeros(rows_rank, d_ff, dtype=torch.float32)
            onehot[torch.arange(rows_rank), top_idx] = 1.0
            parts.append(onehot)
    if all_k.shape[0] > 1 and int(low_surprise_rank) > 0:
        row_scores = torch.linalg.vector_norm(all_k, dim=1)
        threshold = torch.quantile(row_scores, min(max(float(low_surprise_quantile), 0.0), 1.0))
        low = all_k[row_scores <= threshold]
        if low.shape[0] < 2:
            low = all_k[row_scores <= row_scores.median()]
        parts.append(_fast_basis_with_rows(low, int(low_surprise_rank), d_ff))
    if negative_keys is not None and negative_keys.numel() > 0 and int(protected_rank) > 0:
        neg = torch.nan_to_num(negative_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            parts.append(_fast_basis_with_rows(neg, int(protected_rank), d_ff))

    basis = _orthonormal_basis(parts, d_ff)
    if basis.shape[0] > int(rank):
        basis = basis[: int(rank)]
    return basis.contiguous()


def ocep_project_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    weights: torch.Tensor,
    object_basis: torch.Tensor,
    generic_basis: torch.Tensor,
    option_basis: torch.Tensor,
    ridge: float = 1e-3,
    correction_cap: float = 0.35,
    conflict_skip: float = 1.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Obliquely remove generic option leakage while preserving object keys."""

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    qk = torch.nan_to_num(object_basis.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    g = torch.nan_to_num(generic_basis.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    c = torch.nan_to_num(option_basis.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff:
        raise ValueError("OCEP keys do not match update")
    if qk.ndim != 2 or qk.shape[1] != d_ff or g.ndim != 2 or g.shape[1] != d_ff or c.ndim != 2 or c.shape[1] != d_model:
        diagnostics = {
            "ocep_enabled": 1.0,
            "ocep_fallback": 1.0,
            "ocep_reason_code": 1.0,
        }
        return update_f.contiguous(), diagnostics
    if qk.numel() == 0 or g.numel() == 0 or c.numel() == 0:
        diagnostics = {
            "ocep_enabled": 1.0,
            "ocep_fallback": 1.0,
            "ocep_reason_code": 2.0,
            "ocep_object_rank": float(qk.shape[0]),
            "ocep_generic_rank": float(g.shape[0]),
            "ocep_option_rank": float(c.shape[0]),
        }
        return update_f.contiguous(), diagnostics

    # Re-orthonormalize defensively because callers may pass truncated bases.
    qk = _orthonormal_basis([qk], d_ff)
    g = _orthonormal_basis([g], d_ff)
    c = _orthonormal_basis([c], d_model)
    m0 = update_f.T.contiguous()  # [m,d]
    g_null = g - (g @ qk.T) @ qk
    leakage_before_matrix = g @ m0 @ c.T
    leakage_before = torch.linalg.vector_norm(leakage_before_matrix)
    g_norm = torch.linalg.vector_norm(g).clamp_min(float(eps))
    conflict = torch.linalg.vector_norm(g @ qk.T).square() / g_norm.square().clamp_min(float(eps))
    skipped = bool(float(conflict.item()) >= float(conflict_skip) if conflict_skip < 1.0 else False)
    correction = torch.zeros_like(m0)
    correction_scale = torch.tensor(0.0, dtype=torch.float32)
    if not skipped and float(leakage_before.item()) > float(eps):
        system = g_null @ g_null.T + float(ridge) * torch.eye(g_null.shape[0], dtype=torch.float32)
        solved = _solve_symmetric_psd(system, leakage_before_matrix)
        y = -g_null.T @ solved
        correction = y @ c
        corr_norm = torch.linalg.vector_norm(correction)
        cap = float(correction_cap) * torch.linalg.vector_norm(m0).clamp_min(float(eps))
        if float(corr_norm.item()) > float(cap.item()) and correction_cap >= 0:
            correction_scale = cap / corr_norm.clamp_min(float(eps))
            correction = correction * correction_scale
        else:
            correction_scale = torch.tensor(1.0, dtype=torch.float32)
    m_star = m0 + correction
    leakage_after = torch.linalg.vector_norm(g @ m_star @ c.T)
    object_delta = torch.linalg.vector_norm(k @ correction)
    object_effect = torch.linalg.vector_norm(k @ m0).clamp_min(float(eps))
    correction_ratio = torch.linalg.vector_norm(correction) / torch.linalg.vector_norm(m0).clamp_min(float(eps))
    diagnostics = {
        "ocep_enabled": 1.0,
        "ocep_fallback": 0.0,
        "ocep_object_rank": float(qk.shape[0]),
        "ocep_generic_rank": float(g.shape[0]),
        "ocep_option_rank": float(c.shape[0]),
        "ocep_leakage_before": float(leakage_before.item()),
        "ocep_leakage_after": float(leakage_after.item()),
        "ocep_leakage_reduction": float((1.0 - leakage_after / leakage_before.clamp_min(float(eps))).item())
        if float(leakage_before.item()) > 0.0
        else 0.0,
        "ocep_object_delta_ratio": float((object_delta / object_effect).item()),
        "ocep_correction_ratio": float(correction_ratio.item()),
        "ocep_correction_scale": float(correction_scale.item()),
        "ocep_conflict": float(conflict.item()),
        "ocep_skipped_conflict": 1.0 if skipped else 0.0,
        "ocep_ridge": float(ridge),
        "ocep_correction_cap": float(correction_cap),
    }
    return m_star.T.contiguous(), diagnostics


def ocep_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    token_indices: torch.Tensor | None = None,
    logit_top_values: torch.Tensor | None = None,
    logit_top_indices: torch.Tensor | None = None,
    lm_head_indices: torch.Tensor | None = None,
    lm_head_rows: torch.Tensor | None = None,
    object_rank: int = 64,
    generic_rank: int = 128,
    option_rank: int = 64,
    option_output_rank: int = 32,
    option_local_rank: int = 32,
    generic_low_surprise_rank: int = 32,
    generic_weight_anchor_rank: int = 96,
    generic_protected_rank: int = 32,
    low_surprise_quantile: float = 0.35,
    ridge: float = 1e-3,
    correction_cap: float = 0.35,
    conflict_skip: float = 1.1,
) -> OcepPurificationResult:
    """Purify a candidate update by removing generic option evidence.

    OCEP is a map-level post-solve projection. It preserves the candidate's
    effects on the selected object rows while using the current weights and
    same-pass ambient rows to identify generic key directions whose option
    evidence should be suppressed.
    """

    object_basis = build_ocep_object_basis(keys, weights, rank=object_rank)
    option_basis = build_ocep_option_basis(
        targets,
        output_basis=output_basis,
        token_indices=token_indices,
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        rank=option_rank,
        output_rank=option_output_rank,
        local_rank=option_local_rank,
    )
    generic_basis = build_ocep_generic_key_basis(
        all_keys,
        down_weight,
        option_basis,
        up_weight=up_weight,
        negative_keys=negative_keys,
        rank=generic_rank,
        low_surprise_rank=generic_low_surprise_rank,
        weight_anchor_rank=generic_weight_anchor_rank,
        protected_rank=generic_protected_rank,
        low_surprise_quantile=low_surprise_quantile,
    )
    purified, diagnostics = ocep_project_update(
        update,
        keys=keys,
        weights=weights,
        object_basis=object_basis,
        generic_basis=generic_basis,
        option_basis=option_basis,
        ridge=ridge,
        correction_cap=correction_cap,
        conflict_skip=conflict_skip,
    )
    diagnostics.update(
        {
            "ocep_mode_code": 0.0,
            "ocep_update_fro_before": float(torch.linalg.vector_norm(update.detach().float().cpu()).item()),
            "ocep_update_fro_after": float(torch.linalg.vector_norm(purified).item()),
        }
    )
    return OcepPurificationResult(
        update=purified,
        object_basis=object_basis,
        generic_basis=generic_basis,
        option_basis=option_basis,
        diagnostics=diagnostics,
    )


def ocep_qrico_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor | None = None,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    deflate_key_rank: int = 16,
    deflate_value_rank: int = 16,
    rank: int = 64,
    option_sketch_rank: int = 256,
    target_parallel_rank: int = 4,
    scramble_weight: float = 0.35,
    residual_row_weight_power: float = 0.5,
    quotient_mode: str = "joint",
    solve_mode: str = "sylvester",
    low_surprise_quantile: float = 0.35,
    negative_weight: float = 40.0,
    output_weight: float = 10.0,
    cca_ridge: float = 1e-3,
    layer_evidence_min: float = 0.03,
    layer_evidence_target: float = 0.20,
    apply_layer_trust: bool = True,
    risk_ratio_cap: float = 100.0,
    object_rank: int = 64,
    generic_rank: int = 128,
    ocep_option_rank: int = 64,
    option_output_rank: int = 32,
    option_local_rank: int = 32,
    generic_low_surprise_rank: int = 32,
    generic_weight_anchor_rank: int = 96,
    generic_protected_rank: int = 32,
    ocep_ridge: float = 1e-3,
    correction_cap: float = 0.35,
    conflict_skip: float = 1.1,
) -> OcepPurificationResult:
    """Run Q-RICO first, then OCEP on the resulting map."""

    qrico = qrico_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=weights,
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=token_indices,
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        layer=layer,
        negative_keys=negative_keys,
        output_basis=output_basis,
        deflate_key_rank=deflate_key_rank,
        deflate_value_rank=deflate_value_rank,
        rank=rank,
        option_sketch_rank=option_sketch_rank,
        target_parallel_rank=target_parallel_rank,
        scramble_weight=scramble_weight,
        residual_row_weight_power=residual_row_weight_power,
        quotient_mode=quotient_mode,
        solve_mode=solve_mode,
        low_surprise_quantile=low_surprise_quantile,
        negative_weight=negative_weight,
        output_weight=output_weight,
        cca_ridge=cca_ridge,
        layer_evidence_min=layer_evidence_min,
        layer_evidence_target=layer_evidence_target,
        apply_layer_trust=apply_layer_trust,
        risk_ratio_cap=risk_ratio_cap,
    )
    ocep = ocep_purify_update(
        qrico.update,
        keys=keys,
        targets=targets,
        weights=weights,
        all_keys=all_keys,
        down_weight=down_weight,
        up_weight=up_weight,
        negative_keys=negative_keys,
        output_basis=output_basis,
        token_indices=token_indices,
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        object_rank=object_rank,
        generic_rank=generic_rank,
        option_rank=ocep_option_rank,
        option_output_rank=option_output_rank,
        option_local_rank=option_local_rank,
        generic_low_surprise_rank=generic_low_surprise_rank,
        generic_weight_anchor_rank=generic_weight_anchor_rank,
        generic_protected_rank=generic_protected_rank,
        low_surprise_quantile=low_surprise_quantile,
        ridge=ocep_ridge,
        correction_cap=correction_cap,
        conflict_skip=conflict_skip,
    )
    diagnostics = dict(qrico.diagnostics)
    diagnostics.update(ocep.diagnostics)
    diagnostics["ocep_mode_code"] = 1.0
    diagnostics["ocep_qrico_update_fro_before"] = float(torch.linalg.vector_norm(qrico.update).item())
    diagnostics["ocep_qrico_base_fallback"] = float(qrico.diagnostics.get("qrico_fallback", 0.0))
    return OcepPurificationResult(
        update=ocep.update,
        object_basis=ocep.object_basis,
        generic_basis=ocep.generic_basis,
        option_basis=ocep.option_basis,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def apply_mlp_gauge_seal_(
    layer: nn.Module,
    wrapper: AdditiveMemoryLinear,
    scales: torch.Tensor,
) -> dict[str, float]:
    """Apply the SwiGLU up/down channel gauge transform in-place."""

    up_module = _mlp_up_module(layer)
    if up_module is None or not hasattr(up_module, "weight"):
        raise AttributeError("Cannot apply MLP gauge seal without an up projection weight")
    up_weight = getattr(up_module, "weight")
    if not isinstance(up_weight, torch.Tensor):
        raise AttributeError("MLP up projection weight is not a tensor")
    scale_cpu = torch.nan_to_num(scales.detach().float().cpu(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-12)
    if scale_cpu.ndim != 1 or scale_cpu.shape[0] != wrapper.in_features or up_weight.shape[0] != wrapper.in_features:
        raise ValueError(
            f"seal scales must be [{wrapper.in_features}], got {tuple(scale_cpu.shape)} "
            f"for up weight {tuple(up_weight.shape)}"
        )
    scale_up = scale_cpu.to(device=up_weight.device, dtype=up_weight.dtype)
    up_weight.mul_(scale_up.unsqueeze(1))
    up_bias = getattr(up_module, "bias", None)
    if isinstance(up_bias, torch.Tensor):
        up_bias.mul_(scale_up)

    inv_base = (1.0 / scale_cpu).to(device=wrapper.base.weight.device, dtype=wrapper.base.weight.dtype)
    wrapper.base.weight.mul_(inv_base.unsqueeze(0))
    inv_memory = (1.0 / scale_cpu).to(device=wrapper.memory.device, dtype=wrapper.memory.dtype)
    wrapper.memory.mul_(inv_memory.unsqueeze(0))
    for slot_memory in wrapper.slot_memories:
        slot_memory.mul_(inv_memory.unsqueeze(0))
    return {
        "seal_applied": 1.0,
        "seal_scale_mean": float(scale_cpu.mean().item()),
        "seal_scale_max": float(scale_cpu.max().item()),
        "seal_scaled_channels": float((scale_cpu > 1.000001).sum().item()),
    }


def intrinsic_feature_scores(keys: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """Weight-relative feature energy for MLP down-input activations."""

    keys_f = keys.detach().float().cpu()
    scale = mlp_weight_prior_scale(layer, keys_f.shape[1]).to(keys_f.device)
    return (keys_f / scale.unsqueeze(0)).square()


def lesson_persistence_weights(
    feature_scores: torch.Tensor,
    *,
    threshold_fraction: float = 0.25,
    min_tokens: int = 2,
) -> torch.Tensor:
    """Feature weights from one lesson's recurrence structure.

    One-off spikes are likely punctuation/readout/posture. Features that are
    repeatedly surprising, but not uniformly on for every token, are better
    candidates for context-instantiated latent objects.
    """

    scores = feature_scores.detach().float()
    if scores.ndim != 2:
        raise ValueError(f"feature_scores must be [tokens, features], got {tuple(scores.shape)}")
    max_score = scores.max(dim=0).values.clamp_min(1e-12)
    mean_score = scores.mean(dim=0).clamp_min(1e-12)
    threshold = float(threshold_fraction) * max_score
    support = (scores >= threshold.unsqueeze(0)).float().sum(dim=0)
    support_weight = (support / max(float(min_tokens), 1.0)).clamp(max=1.0)
    contrast = torch.log1p(max_score / mean_score)
    return support_weight * contrast


def down_value_specificity(
    down_weight: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight-only feature specificity from the down-projection value geometry.

    Top singular directions of the down columns are treated as generic value
    directions. A feature is specific when its value column has substantial
    residual outside that generic subspace.
    """

    down = down_weight.detach().float().cpu()
    if rank <= 0:
        return torch.ones(down.shape[1]), torch.empty(0, down.shape[0])
    rank = min(int(rank), down.shape[0], down.shape[1])
    columns = down.T.contiguous()
    _u, _s, vh = torch.linalg.svd(columns, full_matrices=False)
    basis = vh[:rank].contiguous()
    projected = columns @ basis.T @ basis
    residual = columns - projected
    specificity = torch.linalg.vector_norm(residual, dim=1) / torch.linalg.vector_norm(columns, dim=1).clamp_min(1e-12)
    return specificity.clamp_min(1e-6).contiguous(), basis


def down_output_basis_specificity(
    down_weight: torch.Tensor,
    output_basis: torch.Tensor | None,
) -> torch.Tensor:
    """Feature specificity against a protected output/readout basis.

    Returns the fraction of each down-projection value column that remains
    outside ``output_basis``. Features whose value columns point mostly into
    generic readout directions get low weight; features with value directions
    outside that basis get high weight.
    """

    down = down_weight.detach().float().cpu()
    if output_basis is None or output_basis.numel() == 0:
        return torch.ones(down.shape[1])
    basis = output_basis.detach().float().cpu()
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    if basis.ndim != 2 or basis.shape[1] != down.shape[0]:
        raise ValueError(f"Expected output_basis [rank, {down.shape[0]}], got {tuple(basis.shape)}")
    q, _r = torch.linalg.qr(basis.T, mode="reduced")
    basis = q.T.contiguous()
    columns = down.T.contiguous()
    projected = columns @ basis.T @ basis
    residual = columns - projected
    specificity = torch.linalg.vector_norm(residual, dim=1) / torch.linalg.vector_norm(columns, dim=1).clamp_min(1e-12)
    return specificity.clamp_min(1e-6).contiguous()


def project_rows_away_from_basis(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return rows
    rows_f = rows.float()
    basis_f = basis.to(device=rows_f.device, dtype=rows_f.dtype)
    return rows_f - rows_f @ basis_f.T @ basis_f


def _row_pcs(rows: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0 or rows.numel() == 0:
        return torch.empty(0, rows.shape[-1] if rows.ndim else 0, dtype=torch.float32)
    rows_f = rows.detach().float().cpu()
    if rows_f.ndim != 2 or rows_f.shape[0] < 2:
        return torch.empty(0, rows_f.shape[-1] if rows_f.ndim else 0, dtype=torch.float32)
    rows_f = torch.nan_to_num(rows_f, nan=0.0, posinf=0.0, neginf=0.0)
    rows_f = rows_f - rows_f.mean(dim=0, keepdim=True)
    if float(torch.linalg.vector_norm(rows_f).item()) <= 1e-12:
        return torch.empty(0, rows_f.shape[1], dtype=torch.float32)
    try:
        _u, _s, vh = torch.linalg.svd(rows_f, full_matrices=False)
        return vh[: min(int(rank), vh.shape[0])].contiguous()
    except RuntimeError:
        gram = rows_f @ rows_f.T
        gram = torch.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        keep = min(int(rank), eigvecs.shape[1])
        order = torch.argsort(eigvals, descending=True)[:keep]
        vals = eigvals[order].clamp_min(1e-12).sqrt().unsqueeze(1)
        pcs = eigvecs[:, order].T @ rows_f / vals
        return F.normalize(pcs, dim=1).contiguous()


def _orthonormal_basis(rows: list[torch.Tensor | None], dim: int) -> torch.Tensor:
    valid = [row.detach().float().cpu() for row in rows if row is not None and row.numel() > 0]
    if not valid:
        return torch.empty(0, dim, dtype=torch.float32)
    basis = torch.cat(valid, dim=0)
    if basis.ndim != 2 or basis.shape[1] != dim:
        raise ValueError(f"Expected basis rows [rank, {dim}], got {tuple(basis.shape)}")
    q, _r = torch.linalg.qr(basis.T, mode="reduced")
    return q.T.contiguous()


def _normalize_mass(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values_f = values.detach().float().cpu().clamp_min(0.0)
    total = values_f.sum()
    if float(total.item()) <= eps:
        return torch.full_like(values_f, 1.0 / max(values_f.numel(), 1))
    return values_f / total.clamp_min(eps)


def _integration_relevance(
    residuals: torch.Tensor,
    row_scores: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """One-pass estimate of which lesson tokens feed the integrated state."""

    x = residuals.detach().float().cpu()
    scores = row_scores.detach().float().cpu().clamp_min(0.0)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"residuals must be non-empty [tokens, dim], got {tuple(x.shape)}")
    if scores.shape[0] != x.shape[0]:
        raise ValueError(f"row_scores must have {x.shape[0]} rows, got {tuple(scores.shape)}")
    if x.shape[0] == 1:
        return torch.ones(1, dtype=torch.float32)
    final = x[-1]
    cos = (x @ final) / (
        torch.linalg.vector_norm(x, dim=1).clamp_min(1e-12)
        * torch.linalg.vector_norm(final).clamp_min(1e-12)
    )
    cos = cos.clamp(min=-1.0, max=1.0)
    score_scale = scores / scores.mean().clamp_min(1e-12)
    logits = float(temperature) * cos + torch.log1p(score_scale)
    logits = logits - logits.max()
    relevance = torch.softmax(logits, dim=0)
    return relevance.contiguous()


def relation_edge_matrix(
    residuals: torch.Tensor,
    row_scores: torch.Tensor,
    attn_probs: torch.Tensor | None = None,
    *,
    edge_top_k: int = 0,
    attention_scale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse source-token to target-token relation matrix.

    The returned ``E`` is ``[T, T]`` with ``E[source, target]``. The diagonal
    carries the token integration relevance; optional attention entries add
    same-forward relation mass from source tokens into target tokens.
    """

    rho = _integration_relevance(residuals, row_scores)
    tokens = rho.shape[0]
    edge = torch.diag(rho)
    attn = attn_probs.detach().float().cpu() if attn_probs is not None else None
    if attn is not None and attn.ndim == 4:
        attn = attn[0]
    if attn is not None:
        if attn.ndim != 3 or attn.shape[1] != tokens or attn.shape[2] != tokens:
            raise ValueError(f"attention_probs must be [heads, tokens, tokens], got {tuple(attn.shape)}")
        if edge_top_k > 0 and attention_scale > 0:
            mean_attn = attn.mean(dim=0)
            for target_idx in range(tokens):
                scores = mean_attn[target_idx].clone()
                scores[target_idx] = -torch.inf
                finite = torch.isfinite(scores)
                if not torch.any(finite):
                    continue
                keep = min(int(edge_top_k), int(torch.count_nonzero(finite).item()))
                source_indices = torch.topk(scores, k=keep, dim=0).indices
                masses = scores[source_indices].clamp_min(0.0)
                total = masses.sum()
                if float(total.item()) <= 1e-12:
                    continue
                edge[source_indices, target_idx] += (
                    float(attention_scale) * rho[target_idx] * masses / total.clamp_min(1e-12)
                )
    return edge.contiguous(), rho.contiguous()


def default_relation_prior(
    row_marginal: torch.Tensor,
    col_marginal: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    beta: float = 3.0,
    sinkhorn_steps: int = 0,
) -> torch.Tensor:
    """Maximum-entropy relation prior with empirical marginals and weight bias."""

    r = _normalize_mass(row_marginal)
    c = _normalize_mass(col_marginal)
    comp = compatibility.detach().float().cpu()
    if comp.shape != (r.shape[0], c.shape[0]):
        raise ValueError(f"compatibility must be [{r.shape[0]}, {c.shape[0]}], got {tuple(comp.shape)}")
    logits = (float(beta) * comp).clamp(min=-30.0, max=30.0)
    kernel = torch.exp(logits).clamp_min(1e-30)
    if sinkhorn_steps > 0:
        prior = kernel
        for _ in range(int(sinkhorn_steps)):
            prior = prior * (r / prior.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
            prior = prior * (c / prior.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)
        return (prior / prior.sum().clamp_min(1e-12)).contiguous()
    prior = r.unsqueeze(1) * c.unsqueeze(0) * kernel
    return (prior / prior.sum().clamp_min(1e-12)).contiguous()


def mlp_activation_normals(
    mlp_inputs: torch.Tensor,
    layer: nn.Module,
    feature_indices: torch.Tensor,
) -> torch.Tensor:
    """Activation normals for selected SwiGLU MLP features.

    ``feature_indices`` is ``[tokens, selected_features]``. The returned tensor
    is ``[tokens, selected_features, d_model]`` and gives the local residual
    direction that would increase each selected down-input feature.
    """

    x = mlp_inputs.detach().float().cpu()
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"mlp_inputs must be non-empty [tokens, dim], got {tuple(x.shape)}")
    if feature_indices.ndim != 2 or feature_indices.shape[0] != x.shape[0]:
        raise ValueError(
            f"feature_indices must be [tokens, p] with token count {x.shape[0]}, "
            f"got {tuple(feature_indices.shape)}"
        )
    up_weight = _linear_weight(_mlp_up_module(layer))
    gate_weight = _linear_weight(_mlp_gate_module(layer))
    if up_weight is None or up_weight.ndim != 2:
        raise AttributeError(f"Layer {layer.__class__.__name__} has no supported MLP up projection.")
    if gate_weight is None or gate_weight.shape != up_weight.shape:
        # Non-gated fallback: derivative of up_j x is simply up_j.
        idx = feature_indices.detach().long().cpu().clamp(0, up_weight.shape[0] - 1)
        return up_weight[idx].contiguous()
    if up_weight.shape[1] != x.shape[1]:
        raise ValueError(f"MLP up width {up_weight.shape[1]} != input dim {x.shape[1]}")

    idx = feature_indices.detach().long().cpu().clamp(0, up_weight.shape[0] - 1)
    up_pre = x @ up_weight.T
    gate_pre = x @ gate_weight.T
    selected_up_pre = up_pre.gather(1, idx)
    selected_gate_pre = gate_pre.gather(1, idx)
    selected_up_rows = up_weight[idx]
    selected_gate_rows = gate_weight[idx]
    sigmoid_gate = torch.sigmoid(selected_gate_pre)
    silu_gate = selected_gate_pre * sigmoid_gate
    silu_prime = sigmoid_gate * (1.0 + selected_gate_pre * (1.0 - sigmoid_gate))
    normals = (
        silu_gate.unsqueeze(-1) * selected_up_rows
        + selected_up_pre.unsqueeze(-1) * silu_prime.unsqueeze(-1) * selected_gate_rows
    )
    return normals.contiguous()


def select_intrinsic_compatibility_residual_write(
    mlp_inputs: torch.Tensor,
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    key_feature_top_k: int = 8,
    value_feature_top_k: int = 32,
    pair_top_k: int = 64,
    compatibility_threshold: float = 0.15,
    compatibility_temperature: float = 0.15,
    posture_pcs: int = 0,
    target_vector_mode: str = "normal",
    attention_probs: torch.Tensor | None = None,
    attention_edge_top_k: int = 0,
    attention_flow_mode: str = "vo",
    include_same_token_edges: bool = True,
    target_scale: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    surprise_weight_mode: str = "linear",
    surprise_weight_temperature: float = 1.0,
    surprise_weight_cap: float = 100.0,
) -> IntrinsicSurpriseSelection:
    """Write residual repairs for unsupported same-token feature bindings.

    This is the first WICR implementation. It replaces raw feature-energy
    surprise with a weight-induced compatibility residual: a same-token feature
    pair is surprising only when the source feature's down-value is poorly
    aligned with the activation normal of the target feature, yet the lesson
    strongly co-instantiates both features.
    """

    inputs_f = mlp_inputs.detach().float().cpu()
    keys_f = keys.detach().float().cpu()
    if inputs_f.ndim != 2 or inputs_f.shape[0] == 0:
        raise ValueError(f"mlp_inputs must be non-empty [tokens, dim], got {tuple(inputs_f.shape)}")
    if keys_f.ndim != 2 or keys_f.shape[0] != inputs_f.shape[0]:
        raise ValueError(
            f"keys must be [tokens, features] with same token count as mlp_inputs, "
            f"got inputs={tuple(inputs_f.shape)} keys={tuple(keys_f.shape)}"
        )
    down = down_weight.detach().float().cpu()
    if down.ndim != 2 or down.shape[1] != keys_f.shape[1] or down.shape[0] != inputs_f.shape[1]:
        raise ValueError(
            f"down_weight must be [input_dim, features] = [{inputs_f.shape[1]}, {keys_f.shape[1]}], "
            f"got {tuple(down.shape)}"
        )
    if target_vector_mode not in {"normal", "value"}:
        raise ValueError(f"Unknown target_vector_mode {target_vector_mode!r}")

    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )

    candidate_k = max(2, min(int(feature_top_k), keys_f.shape[1]))
    all_top_features = torch.topk(feature_scores, k=candidate_k, dim=1).indices
    normals = mlp_activation_normals(inputs_f, layer, all_top_features)
    scale = mlp_weight_prior_scale(layer, keys_f.shape[1]).to(keys_f.device).clamp_min(1e-12)
    z_keys = keys_f / scale.unsqueeze(0)
    down_norms = torch.linalg.vector_norm(down, dim=0).clamp_min(1e-12)

    dim = down.shape[0]
    target_basis = target_projection_basis.detach().float().cpu() if target_projection_basis is not None else None
    busy_low_novelty: list[torch.Tensor] = []
    grouped: dict[int, dict[str, object]] = {}
    pair_scores_for_weights: dict[int, list[torch.Tensor]] = {}

    cause_k = max(1, min(int(key_feature_top_k), candidate_k - 1))
    value_k = max(1, min(int(value_feature_top_k), candidate_k - 1))
    max_pairs = max(1, int(pair_top_k))
    threshold = float(compatibility_threshold)
    temperature = max(float(compatibility_temperature), 1e-6)

    attn = attention_probs.detach().float().cpu() if attention_probs is not None else None
    if attn is not None and attn.ndim == 4:
        attn = attn[0]
    if attn is not None and (attn.ndim != 3 or attn.shape[1] != keys_f.shape[0] or attn.shape[2] != keys_f.shape[0]):
        raise ValueError(f"attention_probs must be [heads, tokens, tokens], got {tuple(attn.shape)}")

    for target_token_idx in token_indices.tolist():
        target_key = keys_f[target_token_idx]
        target_z = z_keys[target_token_idx]
        target_scores = feature_scores[target_token_idx]
        target_candidates = all_top_features[target_token_idx]
        value_features = target_candidates[: max(cause_k + 1, min(candidate_k, cause_k + value_k))]
        feature_to_local = {int(feature.item()): pos for pos, feature in enumerate(target_candidates)}
        edges: list[tuple[int, torch.Tensor, float]] = []
        if include_same_token_edges:
            edges.append((target_token_idx, torch.ones(1, dtype=torch.float32), 1.0))
        if attn is not None and attention_edge_top_k > 0:
            edge_scores = attn[:, target_token_idx, : target_token_idx + 1].mean(dim=0)
            if include_same_token_edges and target_token_idx < edge_scores.shape[0]:
                edge_scores = edge_scores.clone()
                edge_scores[target_token_idx] = -torch.inf
            valid = torch.isfinite(edge_scores)
            if torch.any(valid):
                keep = min(int(attention_edge_top_k), int(torch.count_nonzero(valid).item()))
                top_sources = torch.topk(edge_scores, k=keep, dim=0).indices
                for source_idx_tensor in top_sources:
                    source_idx = int(source_idx_tensor.item())
                    mass = float(edge_scores[source_idx].clamp_min(0.0).item())
                    if mass <= 0:
                        continue
                    edges.append((source_idx, attn[:, target_token_idx, source_idx], mass))

        for source_token_idx, head_weights, edge_mass in edges:
            source_key = keys_f[source_token_idx]
            source_z = z_keys[source_token_idx]
            source_scores = feature_scores[source_token_idx]
            source_candidates = all_top_features[source_token_idx]
            cause_features = source_candidates[:cause_k]
            pair_rows: list[tuple[float, int, int, float, torch.Tensor]] = []
            flowed_cache: dict[tuple[int, ...], torch.Tensor] = {}
            for cause_idx in cause_features.tolist():
                source_value = down[:, cause_idx]
                if source_token_idx == target_token_idx:
                    flowed_value = source_value
                else:
                    cache_key = tuple(cause_features.tolist())
                    if cache_key not in flowed_cache:
                        source_values = down[:, cause_features]
                        flowed_cache[cache_key] = attention_flow_values(
                            layer,
                            source_values,
                            head_weights,
                            mode=attention_flow_mode,
                        )
                    cause_pos = cause_features.tolist().index(cause_idx)
                    flowed_value = flowed_cache[cache_key][:, cause_pos]
                source_norm = torch.linalg.vector_norm(flowed_value).clamp_min(1e-12)
                for value_idx in value_features.tolist():
                    if source_token_idx == target_token_idx and value_idx == cause_idx:
                        continue
                    local_pos = feature_to_local.get(value_idx)
                    if local_pos is None:
                        continue
                    target_normal = normals[target_token_idx, local_pos]
                    normal_norm = torch.linalg.vector_norm(target_normal).clamp_min(1e-12)
                    compatibility = torch.dot(flowed_value, target_normal) / (source_norm * normal_norm)
                    coinstantiation = source_z[cause_idx] * target_z[value_idx] * max(edge_mass, 1e-6)
                    if float(coinstantiation.abs().item()) <= 1e-12:
                        continue
                    signed_compat = torch.sign(coinstantiation) * compatibility
                    mismatch = torch.nn.functional.softplus((threshold - signed_compat) / temperature)
                    feature_pair_score = torch.sqrt(
                        source_scores[cause_idx].clamp_min(1e-12)
                        * target_scores[value_idx].clamp_min(1e-12)
                    )
                    pair_score = coinstantiation.abs() * mismatch * feature_pair_score.sqrt()
                    if not torch.isfinite(pair_score):
                        continue
                    if target_vector_mode == "value":
                        target_vector = target_key[value_idx] * down[:, value_idx]
                        target_unit = target_vector
                    else:
                        target_unit = target_normal / normal_norm
                    if float(pair_score.item()) > 0:
                        pair_rows.append(
                            (
                                float(pair_score.item()),
                                cause_idx,
                                value_idx,
                                float(torch.sign(coinstantiation).item()),
                                target_unit,
                            )
                        )
                    if signed_compat >= threshold and target_scores[value_idx] > 0:
                        busy_low_novelty.append(target_unit)
            if not pair_rows:
                continue
            pair_rows.sort(key=lambda item: item[0], reverse=True)
            selected_pairs = pair_rows[:max_pairs]
            score_tensor = torch.tensor([row[0] for row in selected_pairs], dtype=torch.float32).clamp_min(1e-12)
            pair_gains = torch.sqrt(score_tensor / score_tensor.mean().clamp_min(1e-12)).clamp(max=4.0)
            for pair_gain, (score_value, cause_idx, value_idx, sign_value, target_unit) in zip(
                pair_gains,
                selected_pairs,
                strict=False,
            ):
                row = grouped.setdefault(
                    source_token_idx,
                    {
                        "key": torch.zeros_like(source_key),
                        "target": torch.zeros(dim, dtype=torch.float32),
                        "features": [],
                    },
                )
                row["key"][cause_idx] = source_key[cause_idx]
                row["target"].add_(sign_value * float(pair_gain.item()) * target_unit)
                row["features"].append(cause_idx)
                pair_scores_for_weights.setdefault(source_token_idx, []).append(
                    torch.tensor(score_value, dtype=torch.float32)
                )

    if not grouped:
        raise ValueError("No nonzero compatibility residual examples were selected")

    posture_basis = (
        _row_pcs(torch.stack(busy_low_novelty, dim=0), posture_pcs)
        if busy_low_novelty and posture_pcs > 0
        else torch.empty(0, dim, dtype=torch.float32)
    )
    safe_basis = _orthonormal_basis([target_basis, posture_basis], dim)

    examples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    example_tokens: list[torch.Tensor] = []
    example_features: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_target_keys: list[torch.Tensor] = []
    for token_idx in sorted(grouped):
        row = grouped[token_idx]
        key = row["key"]
        target = row["target"]
        if float(torch.linalg.vector_norm(key).item()) <= 1e-12:
            continue
        if safe_basis.numel() > 0:
            target = project_rows_away_from_basis(target.unsqueeze(0), safe_basis).squeeze(0)
        if float(torch.linalg.vector_norm(target).item()) <= 1e-12:
            continue
        score_values = torch.stack(pair_scores_for_weights[token_idx], dim=0)
        raw_weight = score_values.mean().clamp_min(1e-12)
        targets.append(float(target_scale) * target)
        examples.append(key)
        weights.append(raw_weight)
        features = row["features"]
        feature_idx = max(set(features), key=features.count) if features else int(torch.argmax(key.abs()).item())
        example_features.append(torch.tensor(feature_idx, dtype=torch.long))
        example_tokens.append(torch.tensor(token_idx, dtype=torch.long))
        selected_scores.append(raw_weight)
        selected_feature_scores.append(feature_scores[token_idx])
        selected_target_keys.append(key)

    if not examples:
        raise ValueError("All compatibility residual targets vanished after projection")
    raw_weights = torch.stack(weights, dim=0)
    normed_weights = shape_surprise_weights(
        raw_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=torch.stack(examples, dim=0).contiguous(),
        targets=torch.stack(targets, dim=0).contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=torch.stack(example_tokens, dim=0).contiguous(),
        row_scores=torch.stack(selected_scores, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(selected_target_keys, dim=0).contiguous(),
        feature_indices=torch.stack(example_features, dim=0).contiguous(),
    )


def select_intrinsic_conditional_relation_innovation_write(
    mlp_inputs: torch.Tensor,
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    feature_top_k: int = 128,
    relation_rank: int = 16,
    beta: float = 3.0,
    edge_top_k: int = 0,
    edge_attention_scale: float = 0.5,
    sinkhorn_steps: int = 0,
    target_mode: str = "svd_value",
    target_scale: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    attention_probs: torch.Tensor | None = None,
    surprise_weight_mode: str = "exponential",
    surprise_weight_temperature: float = 2.0,
    surprise_weight_cap: float = 20.0,
) -> IntrinsicSurpriseSelection:
    """Select CORI dense relation-state rows from a single lesson forward.

    CORI scores relation innovation rather than raw feature energy. It builds
    an empirical feature-relation field, conditions out the empirical feature
    marginals plus a weight-induced compatibility prior, then writes dense
    relation-state keys for the top innovation modes.
    """

    inputs_f = mlp_inputs.detach().float().cpu()
    keys_f = keys.detach().float().cpu()
    if inputs_f.ndim != 2 or inputs_f.shape[0] == 0:
        raise ValueError(f"mlp_inputs must be non-empty [tokens, dim], got {tuple(inputs_f.shape)}")
    if keys_f.ndim != 2 or keys_f.shape[0] != inputs_f.shape[0]:
        raise ValueError(
            f"keys must be [tokens, features] with same token count as mlp_inputs, "
            f"got inputs={tuple(inputs_f.shape)} keys={tuple(keys_f.shape)}"
        )
    down = down_weight.detach().float().cpu()
    if down.ndim != 2 or down.shape[1] != keys_f.shape[1] or down.shape[0] != inputs_f.shape[1]:
        raise ValueError(
            f"down_weight must be [input_dim, features] = [{inputs_f.shape[1]}, {keys_f.shape[1]}], "
            f"got {tuple(down.shape)}"
        )
    if target_mode not in {"svd_value", "innovation_value"}:
        raise ValueError(f"Unknown CORI target_mode {target_mode!r}")

    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    edge, rho = relation_edge_matrix(
        inputs_f,
        row_scores,
        attention_probs,
        edge_top_k=edge_top_k,
        attention_scale=edge_attention_scale,
    )

    scale = mlp_weight_prior_scale(layer, keys_f.shape[1]).to(keys_f.device).clamp_min(1e-12)
    z_keys = keys_f / scale.unsqueeze(0)
    abs_z = z_keys.abs()
    inside = (rho.unsqueeze(1) * z_keys.square()).sum(dim=0)
    outside = ((1.0 - rho).clamp_min(0.0).unsqueeze(1) * z_keys.square()).sum(dim=0)
    participation = (rho.unsqueeze(1) * abs_z).sum(dim=0)
    feature_mass_score = participation * torch.log1p(inside / outside.clamp_min(1e-8))
    if feature_weights is not None:
        feature_mass_score = feature_mass_score * feature_weights.detach().float().cpu().clamp_min(0.0)
    selected_features = torch.topk(feature_mass_score, k=score_k, dim=0).indices.sort().values

    b = z_keys[:, selected_features]
    abs_b = b.abs()
    raw_q = abs_b.T @ edge @ abs_b
    q_total = raw_q.sum().clamp_min(1e-12)
    empirical = raw_q / q_total
    signed_den = raw_q.clamp_min(1e-12)
    signed = (b.T @ edge @ b) / signed_den
    row_marginal = empirical.sum(dim=1)
    col_marginal = empirical.sum(dim=0)

    repeated_features = selected_features.unsqueeze(0).repeat(inputs_f.shape[0], 1)
    normals = mlp_activation_normals(inputs_f, layer, repeated_features)
    normal_weights = rho.unsqueeze(1) * abs_b
    normal_denom = normal_weights.sum(dim=0).clamp_min(1e-12)
    avg_normals = (normal_weights.unsqueeze(2) * normals).sum(dim=0) / normal_denom.unsqueeze(1)

    target_basis = target_projection_basis.detach().float().cpu() if target_projection_basis is not None else None
    safe_basis = _orthonormal_basis([target_basis], down.shape[0])
    selected_values = down[:, selected_features].T.contiguous()
    sem_values = project_rows_away_from_basis(selected_values, safe_basis) if safe_basis.numel() else selected_values
    sem_normals = project_rows_away_from_basis(avg_normals, safe_basis) if safe_basis.numel() else avg_normals
    compatibility = sem_values @ sem_normals.T
    compatibility = compatibility / (
        torch.linalg.vector_norm(sem_values, dim=1).clamp_min(1e-12).unsqueeze(1)
        * torch.linalg.vector_norm(sem_normals, dim=1).clamp_min(1e-12).unsqueeze(0)
    )
    compatibility = compatibility.nan_to_num(0.0).clamp(min=-1.0, max=1.0)

    prior = default_relation_prior(
        row_marginal,
        col_marginal,
        compatibility,
        beta=beta,
        sinkhorn_steps=sinkhorn_steps,
    )
    innovation = (empirical - prior) / torch.sqrt(prior.clamp_min(1e-12))
    if float(torch.linalg.vector_norm(innovation).item()) <= 1e-12:
        raise ValueError("CORI relation innovation vanished")

    u, singular, vh = torch.linalg.svd(innovation, full_matrices=False)
    keep = min(max(1, int(relation_rank)), singular.shape[0])
    u = u[:, :keep].contiguous()
    singular = singular[:keep].contiguous()
    vh = vh[:keep].contiguous()

    examples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    example_tokens: list[torch.Tensor] = []
    example_features: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_target_keys: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_values_t = down[:, selected_features]
    signed_innovation = innovation * signed
    for component_idx in range(keep):
        left = u[:, component_idx]
        right = vh[component_idx]
        source_signal = b @ left
        target_signal = b @ right
        token_weights = target_signal * (edge.T @ source_signal)
        denom = token_weights.abs().sum().clamp_min(1e-12)
        if float(denom.item()) <= 1e-12:
            continue
        dense_key = (token_weights.unsqueeze(1) * keys_f).sum(dim=0) / denom
        if float(torch.linalg.vector_norm(dense_key).item()) <= 1e-12:
            continue
        if target_mode == "innovation_value":
            coeff = signed_innovation.T @ left
        else:
            coeff = singular[component_idx] * right
        target = selected_values_t @ coeff
        packet_value = dense_key @ down.T
        if safe_basis.numel() > 0:
            target = project_rows_away_from_basis(target.unsqueeze(0), safe_basis).squeeze(0)
            packet_value = project_rows_away_from_basis(packet_value.unsqueeze(0), safe_basis).squeeze(0)
        if float(torch.dot(target, packet_value).item()) < 0.0:
            target = -target
        target = float(target_scale) * target
        if float(torch.linalg.vector_norm(target).item()) <= 1e-12:
            continue
        token_idx = int(torch.argmax(token_weights.abs()).item())
        examples.append(dense_key)
        targets.append(target)
        weights.append(singular[component_idx].clamp_min(1e-12))
        example_tokens.append(torch.tensor(token_idx, dtype=torch.long))
        selected_scores.append(singular[component_idx].clamp_min(1e-12))
        selected_feature_scores.append(feature_scores[token_idx])
        selected_target_keys.append(dense_key)
        feature_pos = int(torch.argmax(right.abs()).item())
        example_features.append(selected_features[feature_pos].detach().long())

    if not examples:
        raise ValueError("No nonzero CORI relation innovation rows were selected")
    raw_weights = torch.stack(weights, dim=0)
    normed_weights = shape_surprise_weights(
        raw_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=torch.stack(examples, dim=0).contiguous(),
        targets=torch.stack(targets, dim=0).contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=torch.stack(example_tokens, dim=0).contiguous(),
        row_scores=torch.stack(selected_scores, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(selected_target_keys, dim=0).contiguous(),
        feature_indices=torch.stack(example_features, dim=0).contiguous(),
    )


def _weighted_residualize_rows(
    rows: torch.Tensor,
    nuisance: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Remove row-wise variation explained by nuisance covariates.

    ``rows`` is ``[n, d]`` and ``nuisance`` is ``[n, z]``. This computes the
    weighted Schur residual ``rows - Z (Z' W Z + ridge I)^-1 Z' W rows``.
    """

    if rows.numel() == 0 or nuisance.numel() == 0:
        return rows
    rows_f = rows.detach().float().cpu()
    z = nuisance.detach().float().cpu()
    if rows_f.ndim != 2 or z.ndim != 2 or z.shape[0] != rows_f.shape[0]:
        raise ValueError(f"Expected rows [n,d] and nuisance [n,z], got {tuple(rows_f.shape)} {tuple(z.shape)}")
    w = weights.detach().float().cpu().clamp_min(0.0)
    if w.ndim != 1 or w.shape[0] != rows_f.shape[0]:
        raise ValueError(f"weights must be [{rows_f.shape[0]}], got {tuple(w.shape)}")
    keep = torch.linalg.vector_norm(z, dim=0) > 1e-8
    if not torch.any(keep):
        return rows_f
    z = z[:, keep]
    wz = z * w.unsqueeze(1)
    system = z.T @ wz + float(ridge) * torch.eye(z.shape[1], dtype=z.dtype)
    rhs = z.T @ (rows_f * w.unsqueeze(1))
    coeff = torch.linalg.pinv(system, hermitian=True) @ rhs
    return (rows_f - z @ coeff).contiguous()


def _safe_row_pcs_from_scores(rows: torch.Tensor, scores: torch.Tensor, rank: int, *, low: bool) -> torch.Tensor:
    if rank <= 0 or rows.numel() == 0 or rows.shape[0] < 2:
        return torch.empty(0, rows.shape[-1], dtype=torch.float32)
    scores_f = scores.detach().float().cpu()
    if scores_f.numel() != rows.shape[0]:
        return torch.empty(0, rows.shape[-1], dtype=torch.float32)
    median = scores_f.median()
    mask = scores_f <= median if low else scores_f >= median
    subset = rows.detach().float().cpu()[mask]
    if subset.shape[0] < 2:
        subset = rows.detach().float().cpu()
    return _row_pcs(subset, rank)


def _basis_with_mean(rows: torch.Tensor | None, rank: int, dim: int) -> torch.Tensor:
    if rows is None or rows.numel() == 0 or rank <= 0:
        return torch.empty(0, dim, dtype=torch.float32)
    rows_f = rows.detach().float().cpu()
    if rows_f.ndim != 2 or rows_f.shape[1] != dim:
        return torch.empty(0, dim, dtype=torch.float32)
    parts: list[torch.Tensor] = []
    mean = rows_f.mean(dim=0, keepdim=True)
    if float(torch.linalg.vector_norm(mean).item()) > 1e-12:
        parts.append(mean)
    pc_rank = max(0, int(rank) - len(parts))
    pcs = _row_pcs(rows_f, pc_rank)
    if pcs.numel() > 0:
        parts.append(pcs)
    return _orthonormal_basis(parts, dim)[:rank]


def _reduced_generalized_risk_basis(
    signal_rows: torch.Tensor,
    risk_rows: torch.Tensor,
    *,
    dim: int,
    rank: int,
    eps: float = 1e-4,
    ratio_cap: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top directions where risk covariance exceeds signal covariance.

    Returns Euclidean-orthonormal basis rows ``[r, dim]`` and nonnegative risk
    ratios ``[r]``.  The generalized problem is solved in a compact span built
    from signal/risk row PCs, avoiding dense ``dim x dim`` eigensolves.
    """

    if rank <= 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    signal = signal_rows.detach().float().cpu()
    risk = risk_rows.detach().float().cpu()
    if signal.ndim != 2 or signal.shape[1] != dim or risk.ndim != 2 or risk.shape[1] != dim:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if signal.shape[0] == 0 or risk.shape[0] == 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)

    basis_rank = max(1, int(rank))
    signal_basis = _basis_with_mean(signal, basis_rank, dim)
    risk_basis = _basis_with_mean(risk, basis_rank, dim)
    span = _orthonormal_basis([signal_basis, risk_basis], dim)
    if span.numel() == 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)

    signal_proj = signal @ span.T
    risk_proj = risk @ span.T
    p = span.shape[0]
    eye = torch.eye(p, dtype=torch.float32)
    signal_cov = (signal_proj.T @ signal_proj) / max(float(signal_proj.shape[0]), 1.0)
    risk_cov = (risk_proj.T @ risk_proj) / max(float(risk_proj.shape[0]), 1.0)
    signal_cov = signal_cov + float(eps) * eye
    risk_cov = risk_cov + float(eps) * eye

    eig_s, vec_s = torch.linalg.eigh(signal_cov)
    keep = eig_s > float(eps) * 0.1
    if not torch.any(keep):
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    invsqrt = vec_s[:, keep] @ torch.diag(eig_s[keep].clamp_min(float(eps)).rsqrt()) @ vec_s[:, keep].T
    whitened = invsqrt.T @ risk_cov @ invsqrt
    whitened = 0.5 * (whitened + whitened.T)
    eig, vec = torch.linalg.eigh(whitened)
    coords_all = invsqrt @ vec
    raw_all = coords_all.T @ span
    raw_norm = torch.linalg.vector_norm(raw_all, dim=1).clamp_min(1e-12)
    raw_all = raw_all / raw_norm.unsqueeze(1)
    risk_energy = (risk @ raw_all.T).square().mean(dim=0)
    score = eig.clamp_min(0.0) * risk_energy.clamp_min(0.0)
    keep_rank = min(int(rank), eig.numel())
    valid = risk_energy > float(eps)
    if not torch.any(valid):
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    order = torch.argsort(score.masked_fill(~valid, -1.0), descending=True)[:keep_rank]
    order = order[score[order] >= 0.0]
    if order.numel() == 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    ratios = eig[order].clamp_min(0.0).clamp_max(float(ratio_cap)).contiguous()
    raw_basis = raw_all[order]
    basis = _orthonormal_basis([raw_basis], dim)
    if basis.shape[0] > ratios.shape[0]:
        basis = basis[: ratios.shape[0]]
    if ratios.shape[0] > basis.shape[0]:
        ratios = ratios[: basis.shape[0]]
    return basis.contiguous(), ratios.contiguous()


def _mixed_signal_risk_basis(
    signal_rows: torch.Tensor,
    risk_rows: torch.Tensor,
    *,
    dim: int,
    rank: int,
    eps: float = 1e-4,
    ratio_cap: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact basis spanning both candidate signal and generic-risk rows.

    Unlike the generalized high-risk basis, this intentionally keeps
    signal-dominant directions so KARP can measure and preserve useful
    key-attributable readout atoms instead of missing the candidate update.
    """

    if rank <= 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    signal = signal_rows.detach().float().cpu()
    risk = risk_rows.detach().float().cpu()
    if signal.ndim != 2 or signal.shape[1] != dim or risk.ndim != 2 or risk.shape[1] != dim:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    signal_rank = max(1, int(rank) // 2)
    risk_rank = max(1, int(rank) - signal_rank)
    signal_basis = _basis_with_mean(signal, signal_rank, dim)
    risk_basis = _basis_with_mean(risk, risk_rank, dim)
    basis = _orthonormal_basis([signal_basis, risk_basis], dim)
    if basis.numel() == 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if basis.shape[0] > int(rank):
        basis = basis[: int(rank)]
    signal_energy = (signal @ basis.T).square().mean(dim=0)
    risk_energy = (risk @ basis.T).square().mean(dim=0)
    ratios = (risk_energy / signal_energy.clamp_min(float(eps))).clamp_min(0.0).clamp_max(float(ratio_cap))
    return basis.contiguous(), ratios.contiguous()


def _mixed_signal_risk_candidate_basis(
    signal_rows: torch.Tensor,
    risk_rows: torch.Tensor,
    candidate_rows: torch.Tensor | None,
    *,
    dim: int,
    rank: int,
    eps: float = 1e-4,
    ratio_cap: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact basis spanning signal, risk, and candidate-update directions."""

    if rank <= 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    signal = signal_rows.detach().float().cpu()
    risk = risk_rows.detach().float().cpu()
    candidate = (
        candidate_rows.detach().float().cpu()
        if candidate_rows is not None and candidate_rows.numel() > 0
        else torch.empty(0, dim, dtype=torch.float32)
    )
    if signal.ndim != 2 or signal.shape[1] != dim or risk.ndim != 2 or risk.shape[1] != dim:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if candidate.numel() > 0 and (candidate.ndim != 2 or candidate.shape[1] != dim):
        candidate = torch.empty(0, dim, dtype=torch.float32)

    cand_rank = max(1, int(rank) // 3) if candidate.numel() > 0 else 0
    remaining = max(1, int(rank) - cand_rank)
    signal_rank = max(1, remaining // 2)
    risk_rank = max(1, remaining - signal_rank)
    parts = [
        _basis_with_mean(signal, signal_rank, dim),
        _basis_with_mean(risk, risk_rank, dim),
    ]
    if cand_rank > 0:
        parts.append(_basis_with_mean(candidate, cand_rank, dim))
    basis = _orthonormal_basis(parts, dim)
    if basis.numel() == 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if basis.shape[0] > int(rank):
        basis = basis[: int(rank)]
    signal_energy = (signal @ basis.T).square().mean(dim=0)
    risk_energy = (risk @ basis.T).square().mean(dim=0)
    ratios = (risk_energy / signal_energy.clamp_min(float(eps))).clamp_min(0.0).clamp_max(float(ratio_cap))
    return basis.contiguous(), ratios.contiguous()


def karp_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    key_rank: int = 64,
    value_rank: int = 64,
    eta_cross: float = 10.0,
    eta_key: float = 0.15,
    eta_value: float = 0.05,
    low_surprise_quantile: float = 0.35,
    eps: float = 1e-4,
    risk_ratio_cap: float = 100.0,
) -> KarpPurificationResult:
    """Key-Attributable Readout Purification for a candidate down update.

    KARP is deliberately a local purifier around an already useful candidate
    map. It shrinks only atoms that are simultaneously generic-key active and
    generic-output observable, preserving readout-sensitive atoms when their key
    direction is specific to the lesson's relational surprise rows.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    all_y = torch.nan_to_num(all_outputs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("KARP keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"KARP weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("KARP all_keys/all_outputs do not match update shape")

    row_scale = w.sqrt().unsqueeze(1)
    signal_key_rows = k * row_scale
    signal_value_rows = y * row_scale

    if layer is not None:
        feature_scores = intrinsic_feature_scores(all_k, layer)
        score_k = max(1, min(32, feature_scores.shape[1]))
        row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    else:
        row_scores = torch.linalg.vector_norm(all_k, dim=1)
    threshold = torch.quantile(row_scores, float(low_surprise_quantile))
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    generic_key_parts = [all_k[low_mask]]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_key_parts.append(neg)
    generic_key_rows = torch.cat([part for part in generic_key_parts if part.numel() > 0], dim=0)

    generic_value_parts = [all_y[low_mask]]
    if output_basis is not None and output_basis.numel() > 0:
        basis = output_basis.detach().float().cpu()
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)
        if basis.ndim == 2 and basis.shape[1] == d_model:
            generic_value_parts.append(basis)
    generic_value_rows = torch.cat([part for part in generic_value_parts if part.numel() > 0], dim=0)

    key_basis, key_ratios = _mixed_signal_risk_basis(
        signal_key_rows,
        generic_key_rows,
        dim=d_ff,
        rank=key_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    value_basis, value_ratios = _mixed_signal_risk_basis(
        signal_value_rows,
        generic_value_rows,
        dim=d_model,
        rank=value_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    diagnostics: dict[str, float] = {
        "karp_enabled": 1.0,
        "karp_key_rank": float(key_basis.shape[0]),
        "karp_value_rank": float(value_basis.shape[0]),
        "karp_low_rows": float(low_mask.sum().item()),
        "karp_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if key_basis.numel() == 0 or value_basis.numel() == 0:
        diagnostics.update(
            {
                "karp_update_fro_after": float(torch.linalg.vector_norm(update_f).item()),
                "karp_removed_fro": 0.0,
                "karp_cross_risk_before": 0.0,
                "karp_cross_risk_after": 0.0,
                "karp_kept_coeff_energy_ratio": 1.0,
            }
        )
        return KarpPurificationResult(update=update_f.contiguous(), diagnostics=diagnostics)

    # Coefficients of U=update.T in risky key x risky value coordinates.  The
    # computation is arranged as update-side products to avoid materializing
    # unnecessary dense intermediate matrices.
    coeff = (update_f @ key_basis.T).T @ value_basis.T
    key_ratio = key_ratios.clamp_min(0.0)
    value_ratio = value_ratios.clamp_min(0.0)
    shrink = 1.0 / (
        1.0
        + float(eta_cross) * key_ratio.unsqueeze(1) * value_ratio.unsqueeze(0)
        + float(eta_key) * key_ratio.unsqueeze(1)
        + float(eta_value) * value_ratio.unsqueeze(0)
    )
    removed_coeff = (1.0 - shrink) * coeff
    removed_update = value_basis.T @ removed_coeff.T @ key_basis
    purified = update_f - removed_update

    coeff_energy = coeff.square().sum().clamp_min(1e-12)
    before_risk = (
        coeff.square() * key_ratio.unsqueeze(1) * value_ratio.unsqueeze(0)
    ).sum() / coeff_energy
    after_coeff = shrink * coeff
    after_energy = after_coeff.square().sum().clamp_min(1e-12)
    after_risk = (
        after_coeff.square() * key_ratio.unsqueeze(1) * value_ratio.unsqueeze(0)
    ).sum() / after_energy
    diagnostics.update(
        {
            "karp_update_fro_after": float(torch.linalg.vector_norm(purified).item()),
            "karp_removed_fro": float(torch.linalg.vector_norm(removed_update).item()),
            "karp_removed_update_ratio": float(
                torch.linalg.vector_norm(removed_update).item()
                / max(torch.linalg.vector_norm(update_f).item(), 1e-12)
            ),
            "karp_coeff_energy": float(coeff_energy.item()),
            "karp_kept_coeff_energy_ratio": float((after_energy / coeff_energy).item()),
            "karp_cross_risk_before": float(before_risk.item()),
            "karp_cross_risk_after": float(after_risk.item()),
            "karp_key_ratio_mean": float(key_ratio.mean().item()),
            "karp_key_ratio_max": float(key_ratio.max().item()),
            "karp_value_ratio_mean": float(value_ratio.mean().item()),
            "karp_value_ratio_max": float(value_ratio.max().item()),
            "karp_atoms_shrunk_gt50": float((shrink < 0.5).sum().item()),
            "karp_atoms_shrunk_gt90": float((shrink < 0.1).sum().item()),
            "karp_eta_cross": float(eta_cross),
            "karp_eta_key": float(eta_key),
            "karp_eta_value": float(eta_value),
        }
    )
    return KarpPurificationResult(update=purified.contiguous(), diagnostics=diagnostics)


def _lookup_weight_rows(
    indices: torch.Tensor,
    *,
    stored_indices: torch.Tensor,
    stored_rows: torch.Tensor,
) -> torch.Tensor:
    """Return LM-head rows for ``indices`` from a compact sorted row table."""

    flat = indices.detach().cpu().long().reshape(-1)
    if stored_indices.ndim != 1 or stored_rows.ndim != 2:
        raise ValueError("stored_indices/stored_rows must be [u] and [u,d]")
    if stored_indices.numel() != stored_rows.shape[0]:
        raise ValueError("stored_indices and stored_rows length mismatch")
    positions = torch.searchsorted(stored_indices.detach().cpu().long(), flat)
    safe_positions = positions.clamp(max=max(stored_indices.numel() - 1, 0))
    missing_mask = (positions >= stored_indices.numel()) | (stored_indices[safe_positions] != flat)
    if torch.any(missing_mask):
        missing = flat[missing_mask]
        raise ValueError(f"Missing {int(missing.numel())} LM-head rows for SHARP-KARP")
    rows = stored_rows.detach().float().cpu()[positions]
    return rows.reshape(*indices.shape, stored_rows.shape[1]).contiguous()


def _solve_symmetric_psd(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve a small symmetric PSD system with jitter and pseudoinverse fallback."""

    eye = torch.eye(system.shape[0], dtype=system.dtype, device=system.device)
    jitter = system.diag().abs().mean().clamp_min(1.0) * 1e-6
    try:
        return torch.linalg.solve(system + jitter * eye, rhs)
    except RuntimeError:
        return torch.linalg.pinv(system + jitter * eye, hermitian=True) @ rhs


def _token_row_surprise(all_keys: torch.Tensor, layer: nn.Module | None) -> torch.Tensor:
    if layer is not None:
        feature_scores = intrinsic_feature_scores(all_keys, layer)
        score_k = max(1, min(32, feature_scores.shape[1]))
        return torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    return torch.linalg.vector_norm(all_keys, dim=1)


def orca_karp_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    layer: nn.Module | None = None,
    down_weight: torch.Tensor | None = None,
    feature_indices: torch.Tensor | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    key_rank: int = 48,
    value_rank: int = 48,
    option_top_k: int = 16,
    object_rank: int = 128,
    off_object_rank: int = 512,
    low_surprise_quantile: float = 0.35,
    eta_orth: float = 0.5,
    eta_posture: float = 0.25,
    eta_off_object: float = 0.5,
    eta_karp: float = 0.25,
    eta_key: float = 0.0,
    eta_value: float = 0.0,
    signal_floor_quantile: float = 0.0,
    ablation_mode: str = "purified",
    ablation_fraction: float = 0.25,
    nuisance_ridge: float = 1e-3,
    eps: float = 1e-6,
    risk_ratio_cap: float = 100.0,
) -> OrcaKarpPurificationResult:
    """Object-relative contrastive actuator purification for relational writes.

    ORCA-KARP is a local atom shrinker around the current relational candidate
    update. It keeps readout-sensitive atoms when their same-pass option-space
    movement is parallel to the relational target, and shrinks atoms whose
    movement is target-orthogonal, common-posture explained, or off the
    lesson's object-supported readout subspace.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    all_y = torch.nan_to_num(all_outputs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    tok = token_indices.detach().cpu().long()
    top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = logit_top_indices.detach().cpu().long()
    lm_idx = lm_head_indices.detach().cpu().long()
    lm_rows = torch.nan_to_num(lm_head_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("ORCA-KARP keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"ORCA-KARP weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("ORCA-KARP all_keys/all_outputs do not match update shape")
    if tok.numel() != k.shape[0]:
        raise ValueError("token_indices must align with selected rows")
    if top_vals.ndim != 2 or top_idx.ndim != 2 or top_vals.shape != top_idx.shape:
        raise ValueError("logit_top_values/logit_top_indices must be matching [T,k]")
    if top_vals.shape[0] < all_k.shape[0]:
        raise ValueError("logit top-k rows must cover all lesson keys")

    def cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        if rows.shape[0] <= max_rows or max_rows <= 0:
            return rows
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long().unique()
        return rows[idx]

    row_scale = w.sqrt().unsqueeze(1)
    signal_key_rows = k * row_scale
    signal_value_rows = y * row_scale
    row_scores = _token_row_surprise(all_k, layer)
    threshold = torch.quantile(row_scores, float(low_surprise_quantile))
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    generic_key_parts = [all_k[low_mask]]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_key_parts.append(cap_rows(neg, 128))
    generic_key_rows = torch.cat([part for part in generic_key_parts if part.numel() > 0], dim=0)
    generic_value_parts = [all_y[low_mask]]
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            generic_value_parts.append(cap_rows(out, 256))
    generic_value_rows = torch.cat([part for part in generic_value_parts if part.numel() > 0], dim=0)

    try:
        cand_u, _cand_s, cand_vh = torch.linalg.svd(update_f, full_matrices=False)
        candidate_key_rows = cand_vh[: max(1, min(int(key_rank), cand_vh.shape[0]))]
        candidate_value_rows = cand_u[:, : max(1, min(int(value_rank), cand_u.shape[1]))].T
    except RuntimeError:
        candidate_key_rows = torch.empty(0, d_ff)
        candidate_value_rows = torch.empty(0, d_model)

    key_basis_rows, key_ratios = _mixed_signal_risk_candidate_basis(
        signal_key_rows,
        generic_key_rows,
        candidate_key_rows,
        dim=d_ff,
        rank=key_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    value_basis_rows, value_ratios = _mixed_signal_risk_candidate_basis(
        signal_value_rows,
        generic_value_rows,
        candidate_value_rows,
        dim=d_model,
        rank=value_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    diagnostics: dict[str, float] = {
        "orca_enabled": 1.0,
        "orca_key_rank": float(key_basis_rows.shape[0]),
        "orca_value_rank": float(value_basis_rows.shape[0]),
        "orca_low_rows": float(low_mask.sum().item()),
        "orca_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if key_basis_rows.numel() == 0 or value_basis_rows.numel() == 0:
        diagnostics["orca_fallback"] = 1.0
        return OrcaKarpPurificationResult(update=update_f.contiguous(), diagnostics=diagnostics)

    p_basis = key_basis_rows.T.contiguous()  # [m,a]
    q_basis = value_basis_rows.T.contiguous()  # [d,b]
    a_rank = p_basis.shape[1]
    b_rank = q_basis.shape[1]
    x = k @ p_basis  # [n,a]
    m0 = torch.nan_to_num(p_basis.T @ update_f.T @ q_basis, nan=0.0, posinf=0.0, neginf=0.0)  # [a,b]
    projected_update = q_basis @ m0.T @ p_basis.T
    update_residual = update_f - projected_update

    usable_top_k = max(2, min(int(option_top_k), top_idx.shape[1]))
    selected_tokens = tok.clamp(min=0, max=top_idx.shape[0] - 1)
    selected_top = top_idx[selected_tokens, :usable_top_k]
    selected_vals = top_vals[selected_tokens, :usable_top_k]
    selected_rows = _lookup_weight_rows(selected_top, stored_indices=lm_idx, stored_rows=lm_rows)
    probs = torch.softmax(selected_vals, dim=1)
    expected_rows = (probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
    contrast_rows = selected_rows - expected_rows  # [n,c,d]
    target_option = torch.einsum("ncd,nd->nc", contrast_rows, y)
    value_option = torch.einsum("ncd,db->ncb", contrast_rows, q_basis)

    def standard_col(col: torch.Tensor) -> torch.Tensor:
        col_f = torch.nan_to_num(col.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        return (col_f - col_f.mean()) / col_f.std(unbiased=False).clamp_min(1e-6)

    n_rows = k.shape[0]
    nuisance_cols = [torch.ones(n_rows, dtype=torch.float32)]
    if n_rows >= 6:
        pos = selected_tokens.float() / max(float(all_k.shape[0] - 1), 1.0)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        nuisance_cols.extend(
            [
                standard_col(pos),
                standard_col(ent),
                standard_col(torch.linalg.vector_norm(k, dim=1)),
                standard_col(torch.linalg.vector_norm(y, dim=1)),
                standard_col(w),
                standard_col(row_scores[selected_tokens]),
            ]
        )
    nuisance = torch.stack(nuisance_cols, dim=1)
    w_norm = w / w.mean().clamp_min(1e-12)

    def residualize_rows(rows: torch.Tensor) -> torch.Tensor:
        original_shape = rows.shape
        flat = rows.reshape(n_rows, -1)
        system = nuisance.T @ (nuisance * w_norm.unsqueeze(1))
        system = system + float(nuisance_ridge) * torch.eye(system.shape[0], dtype=torch.float32)
        rhs = nuisance.T @ (flat * w_norm.unsqueeze(1))
        coef = _solve_symmetric_psd(system, rhs)
        return (flat - nuisance @ coef).reshape(original_shape)

    target_res = residualize_rows(target_option)
    z_all = torch.einsum("na,ncb->nabc", x, value_option)
    z_res = residualize_rows(z_all)
    z_post = z_all - z_res
    w4 = w_norm.view(-1, 1, 1, 1)
    target_energy = (w_norm.unsqueeze(1) * target_res.square()).sum().clamp_min(float(eps))
    atom_energy = (w4 * z_res.square()).sum(dim=(0, 3)).clamp_min(float(eps))
    atom_total_energy = (w4 * z_all.square()).sum(dim=(0, 3)).clamp_min(float(eps))
    inner = (w4 * z_res * target_res.view(n_rows, 1, 1, usable_top_k)).sum(dim=(0, 3))
    positive_inner = inner.clamp_min(0.0)
    signal = torch.nan_to_num(
        positive_inner.square() / (target_energy * atom_energy).clamp_min(float(eps)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    projected_energy = (positive_inner.square() / target_energy).clamp(max=atom_energy)
    orthogonal = torch.nan_to_num(
        (atom_energy - projected_energy).clamp_min(0.0) / atom_energy,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    posture = torch.nan_to_num(
        (w4 * z_post.square()).sum(dim=(0, 3)) / atom_total_energy,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    object_parts: list[torch.Tensor | None] = [
        _basis_with_mean(signal_value_rows, max(1, min(int(object_rank) // 3, signal_value_rows.shape[0])), d_model),
        _basis_with_mean(selected_rows.reshape(-1, d_model), max(1, min(int(object_rank) // 3, selected_rows.numel() // max(d_model, 1))), d_model),
    ]
    if down_weight is not None and feature_indices is not None and feature_indices.numel() > 0:
        down = down_weight.detach().float().cpu()
        feat = feature_indices.detach().cpu().long().reshape(-1).unique()
        feat = feat[(feat >= 0) & (feat < down.shape[1])]
        if down.ndim == 2 and down.shape == (d_model, d_ff) and feat.numel() > 0:
            object_parts.append(_basis_with_mean(down[:, feat].T, max(1, min(int(object_rank) // 3, feat.numel())), d_model))
    object_basis = _orthonormal_basis(object_parts, d_model)
    if object_basis.shape[0] > int(object_rank) > 0:
        object_basis = object_basis[: int(object_rank)]
    # Prefer the already-computed global output basis when it is available. A
    # fresh SVD over a 1024 x d LM cloud for every layer makes this purifier too
    # slow to iterate on, and the output basis is already an orthonormal
    # cold-readout cloud.
    lm_basis_parts: list[torch.Tensor | None] = []
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            lm_basis_parts.append(out[: max(1, min(int(off_object_rank), out.shape[0]))])
    lm_row_rank = max(1, min(32, int(off_object_rank), lm_rows.shape[0]))
    lm_basis_parts.append(_basis_with_mean(lm_rows, lm_row_rank, d_model))
    lm_basis = _orthonormal_basis(lm_basis_parts, d_model)
    if lm_basis.shape[0] > int(off_object_rank) > 0:
        lm_basis = lm_basis[: int(off_object_rank)]
    if lm_basis.numel() > 0 and object_basis.numel() > 0:
        lm_basis = lm_basis - (lm_basis @ object_basis.T) @ object_basis
    off_object_basis = _orthonormal_basis([lm_basis], d_model)
    if off_object_basis.shape[0] > int(off_object_rank) > 0:
        off_object_basis = off_object_basis[: int(off_object_rank)]
    if off_object_basis.numel() > 0:
        value_off = (q_basis.T @ off_object_basis.T).square().sum(dim=1).clamp_min(0.0)
    else:
        value_off = torch.zeros(b_rank, dtype=torch.float32)
    key_support = (w_norm.unsqueeze(1) * x.square()).sum(dim=0) / w_norm.sum().clamp_min(1e-12)
    key_support = key_support / key_support.mean().clamp_min(1e-12)
    off_object = key_support.unsqueeze(1) * value_off.unsqueeze(0)

    key_ratio = key_ratios[:a_rank].clamp_min(0.0)
    value_ratio = value_ratios[:b_rank].clamp_min(0.0)
    karp_atom = key_ratio.unsqueeze(1) * value_ratio.unsqueeze(0)
    positive_signal = signal[signal > float(eps)]
    signal_floor = torch.tensor(0.0, dtype=torch.float32)
    if positive_signal.numel() > 0 and signal_floor_quantile > 0:
        q = min(max(float(signal_floor_quantile), 0.0), 1.0)
        signal_floor = torch.quantile(positive_signal, q).clamp_min(float(eps))
    denom = (signal + signal_floor).clamp_min(float(eps))
    atom_diag = (
        float(eta_orth) * orthogonal / denom
        + float(eta_posture) * posture / denom
        + float(eta_off_object) * off_object / denom
        + float(eta_karp) * karp_atom
        + float(eta_key) * key_ratio.unsqueeze(1)
        + float(eta_value) * value_ratio.unsqueeze(0)
    )
    atom_diag = torch.nan_to_num(
        atom_diag.clamp_min(0.0).clamp_max(float(risk_ratio_cap)),
        nan=float(risk_ratio_cap),
        posinf=float(risk_ratio_cap),
        neginf=0.0,
    )
    shrink = 1.0 / (1.0 + atom_diag)
    m_star = torch.nan_to_num(m0 * shrink, nan=0.0, posinf=0.0, neginf=0.0)
    purified_projected = q_basis @ m_star.T @ p_basis.T
    removed_coeff = torch.nan_to_num(m0 - m_star, nan=0.0, posinf=0.0, neginf=0.0)
    removed_projected = q_basis @ removed_coeff.T @ p_basis.T
    mode_codes = {
        "purified": 0.0,
        "kept_only": 1.0,
        "removed_only": 2.0,
        "residual_only": 3.0,
        "top_signal_kept": 4.0,
        "top_risk_removed": 5.0,
    }
    if ablation_mode == "purified":
        purified = update_residual + purified_projected
    elif ablation_mode == "kept_only":
        purified = purified_projected
    elif ablation_mode == "removed_only":
        purified = removed_projected
    elif ablation_mode == "residual_only":
        purified = update_residual
    elif ablation_mode in {"top_signal_kept", "top_risk_removed"}:
        fraction = min(max(float(ablation_fraction), 0.0), 1.0)
        keep_count = max(1, int(round(fraction * m0.numel())))
        if ablation_mode == "top_signal_kept":
            scores = (signal * m_star.square()).reshape(-1)
            source = m_star.reshape(-1)
        else:
            scores = (atom_diag * removed_coeff.square()).reshape(-1)
            source = removed_coeff.reshape(-1)
        if keep_count < scores.numel():
            threshold_score = torch.topk(scores, k=keep_count, largest=True).values[-1]
            mask = scores >= threshold_score
        else:
            mask = torch.ones_like(scores, dtype=torch.bool)
        coeff = torch.zeros_like(source)
        coeff[mask] = source[mask]
        coeff = coeff.reshape_as(m0)
        purified = q_basis @ coeff.T @ p_basis.T
    else:
        raise ValueError(f"Unknown ORCA ablation_mode {ablation_mode!r}")

    before_norm = torch.linalg.vector_norm(update_f).clamp_min(1e-12)
    after_norm = torch.linalg.vector_norm(purified).clamp_min(1e-12)
    if float(after_norm.item()) > float(before_norm.item()):
        purified = purified * (before_norm / after_norm)
        after_norm = before_norm
    coeff_energy = m0.square().sum().clamp_min(float(eps))
    after_coeff_energy = m_star.square().sum().clamp_min(float(eps))
    signal_before = (m0.square() * signal).sum().clamp_min(float(eps))
    signal_after = (m_star.square() * signal).sum().clamp_min(float(eps))
    diagnostics.update(
        {
            "orca_option_top_k": float(usable_top_k),
            "orca_object_basis_rank": float(object_basis.shape[0]),
            "orca_off_object_basis_rank": float(off_object_basis.shape[0]),
            "orca_candidate_capture_ratio": float(
                torch.linalg.vector_norm(projected_update).item() / max(float(before_norm.item()), 1e-12)
            ),
            "orca_update_fro_after": float(after_norm.item()),
            "orca_removed_update_ratio": float(
                torch.linalg.vector_norm(update_f - purified).item() / max(float(before_norm.item()), 1e-12)
            ),
            "orca_coeff_energy": float(coeff_energy.item()),
            "orca_kept_coeff_energy_ratio": float((after_coeff_energy / coeff_energy).item()),
            "orca_signal_retention": float((signal_after / signal_before).item()),
            "orca_signal_mean": float(signal.mean().item()),
            "orca_signal_max": float(signal.max().item()),
            "orca_signal_floor": float(signal_floor.item()),
            "orca_orthogonal_mean": float(orthogonal.mean().item()),
            "orca_posture_mean": float(posture.mean().item()),
            "orca_off_object_mean": float(off_object.mean().item()),
            "orca_atom_diag_mean": float(atom_diag.mean().item()),
            "orca_atom_diag_max": float(atom_diag.max().item()),
            "orca_atoms_shrunk_gt50": float((shrink < 0.5).sum().item()),
            "orca_atoms_shrunk_gt90": float((shrink < 0.1).sum().item()),
            "orca_eta_orth": float(eta_orth),
            "orca_eta_posture": float(eta_posture),
            "orca_eta_off_object": float(eta_off_object),
            "orca_eta_karp": float(eta_karp),
            "orca_ablation_mode_code": mode_codes.get(ablation_mode, -1.0),
            "orca_ablation_fraction": float(ablation_fraction),
            "orca_projected_fro": float(torch.linalg.vector_norm(projected_update).item()),
            "orca_basis_residual_fro": float(torch.linalg.vector_norm(update_residual).item()),
            "orca_kept_projected_fro": float(torch.linalg.vector_norm(purified_projected).item()),
            "orca_removed_projected_fro": float(torch.linalg.vector_norm(removed_projected).item()),
        }
    )
    return OrcaKarpPurificationResult(update=purified.contiguous(), diagnostics=diagnostics)


def _solve_sylvester_psd(
    left: torch.Tensor,
    right: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Solve ``left @ X + X @ right = rhs`` for small symmetric PSD matrices."""

    left_f = torch.nan_to_num(left.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    right_f = torch.nan_to_num(right.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    rhs_f = torch.nan_to_num(rhs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    left_f = 0.5 * (left_f + left_f.T)
    right_f = 0.5 * (right_f + right_f.T)
    eye_l = torch.eye(left_f.shape[0], dtype=torch.float32)
    eye_r = torch.eye(right_f.shape[0], dtype=torch.float32)
    jitter_l = left_f.diag().abs().mean().clamp_min(1.0) * float(eps)
    jitter_r = right_f.diag().abs().mean().clamp_min(1.0) * float(eps)
    eval_l, basis_l = torch.linalg.eigh(left_f + jitter_l * eye_l)
    eval_r, basis_r = torch.linalg.eigh(right_f + jitter_r * eye_r)
    eval_l = eval_l.clamp_min(float(eps))
    eval_r = eval_r.clamp_min(float(eps))
    rotated = basis_l.T @ rhs_f @ basis_r
    denom = eval_l.unsqueeze(1) + eval_r.unsqueeze(0)
    solved = rotated / denom.clamp_min(float(eps))
    return (basis_l @ solved @ basis_r.T).contiguous(), float(denom.min().item())


def _fast_basis_with_rows(rows: torch.Tensor | None, rank: int, dim: int) -> torch.Tensor:
    """QR-only basis from a mean row plus high-norm rows.

    This is deliberately cheaper than PCA/SVD and is used in Q-RICO's inner
    loop where the basis is built for every layer/lesson.
    """

    if rows is None or rows.numel() == 0 or rank <= 0:
        return torch.empty(0, dim, dtype=torch.float32)
    rows_f = torch.nan_to_num(rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if rows_f.ndim != 2 or rows_f.shape[1] != dim:
        return torch.empty(0, dim, dtype=torch.float32)
    parts: list[torch.Tensor] = []
    mean = rows_f.mean(dim=0, keepdim=True)
    if float(torch.linalg.vector_norm(mean).item()) > 1e-12:
        parts.append(mean)
    remaining = max(0, int(rank) - len(parts))
    if remaining > 0:
        norms = torch.linalg.vector_norm(rows_f, dim=1)
        keep = min(remaining, rows_f.shape[0])
        if keep < rows_f.shape[0]:
            idx = torch.topk(norms, k=keep, largest=True).indices
            parts.append(rows_f[idx])
        else:
            parts.append(rows_f)
    return _orthonormal_basis(parts, dim)[: int(rank)]


def _fast_mixed_signal_risk_candidate_basis(
    signal_rows: torch.Tensor,
    risk_rows: torch.Tensor,
    candidate_rows: torch.Tensor | None,
    *,
    dim: int,
    rank: int,
    eps: float = 1e-4,
    ratio_cap: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast signal/risk/candidate basis with risk ratios.

    This avoids the SVD-based PCA path in ``_mixed_signal_risk_candidate_basis``.
    """

    if rank <= 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    signal = torch.nan_to_num(signal_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    risk = torch.nan_to_num(risk_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    candidate = (
        torch.nan_to_num(candidate_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if candidate_rows is not None and candidate_rows.numel() > 0
        else torch.empty(0, dim, dtype=torch.float32)
    )
    if signal.ndim != 2 or signal.shape[1] != dim or risk.ndim != 2 or risk.shape[1] != dim:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if candidate.numel() > 0 and (candidate.ndim != 2 or candidate.shape[1] != dim):
        candidate = torch.empty(0, dim, dtype=torch.float32)
    cand_rank = max(1, int(rank) // 3) if candidate.numel() > 0 else 0
    remaining = max(1, int(rank) - cand_rank)
    signal_rank = max(1, remaining // 2)
    risk_rank = max(1, remaining - signal_rank)
    parts = [
        _fast_basis_with_rows(signal, signal_rank, dim),
        _fast_basis_with_rows(risk, risk_rank, dim),
    ]
    if cand_rank > 0:
        parts.append(_fast_basis_with_rows(candidate, cand_rank, dim))
    basis = _orthonormal_basis(parts, dim)
    if basis.numel() == 0:
        return torch.empty(0, dim, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    if basis.shape[0] > int(rank):
        basis = basis[: int(rank)]
    signal_energy = (signal @ basis.T).square().mean(dim=0)
    risk_energy = (risk @ basis.T).square().mean(dim=0)
    ratios = (risk_energy / signal_energy.clamp_min(float(eps))).clamp_min(0.0).clamp_max(float(ratio_cap))
    return basis.contiguous(), ratios.contiguous()


def _joint_basis_projection(
    update: torch.Tensor,
    key_basis_rows: torch.Tensor,
    value_basis_rows: torch.Tensor,
    *,
    mode: str = "joint",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a row-major update onto a key/value basis block.

    The returned tensors are row-major ``[d_model, d_ff]``:
    ``residual_update``, ``projected_update``, and coefficient map ``[a,b]``.
    ``joint`` removes only the two-sided low-rank block ``UAV^T``. ``two_sided``
    removes every component with either key- or value-basis support.
    """

    update_f = update.detach().float().cpu()
    d_model, d_ff = update_f.shape
    if key_basis_rows.numel() == 0 or value_basis_rows.numel() == 0:
        coeff = torch.empty(0, 0, dtype=torch.float32)
        return update_f.contiguous(), torch.zeros_like(update_f), coeff
    p_basis = key_basis_rows.T.contiguous()  # [m,a]
    q_basis = value_basis_rows.T.contiguous()  # [d,b]
    coeff = torch.nan_to_num(p_basis.T @ update_f.T @ q_basis, nan=0.0, posinf=0.0, neginf=0.0)
    projected = q_basis @ coeff.T @ p_basis.T
    if mode == "joint":
        residual = update_f - projected
    elif mode == "two_sided":
        m0 = update_f.T
        key_proj = p_basis @ (p_basis.T @ m0)
        value_proj = (m0 @ q_basis) @ q_basis.T
        joint = p_basis @ coeff @ q_basis.T
        residual_m = m0 - key_proj - value_proj + joint
        residual = residual_m.T
        projected = update_f - residual
    else:
        raise ValueError(f"Unknown Q-RICO quotient mode {mode!r}")
    return residual.contiguous(), projected.contiguous(), coeff.contiguous()


def _apply_lowrank_left_metric_inverse(
    rows: torch.Tensor,
    rhs: torch.Tensor,
    *,
    ridge: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply ``(ridge I + rows.T @ rows)^-1`` without forming the big matrix."""

    rhs_f = torch.nan_to_num(rhs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if rhs_f.ndim != 2:
        raise ValueError(f"rhs must be [dim, cols], got {tuple(rhs_f.shape)}")
    lam = max(float(ridge), float(eps))
    if rows.numel() == 0:
        return rhs_f / lam
    r = torch.nan_to_num(rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if r.ndim != 2 or r.shape[1] != rhs_f.shape[0]:
        raise ValueError(f"rows must be [n,{rhs_f.shape[0]}], got {tuple(r.shape)}")
    if r.shape[0] == 0:
        return rhs_f / lam
    inv_lam_rhs = rhs_f / lam
    gram = torch.eye(r.shape[0], dtype=torch.float32) + (r @ r.T) / lam
    middle = _solve_symmetric_psd(0.5 * (gram + gram.T), r @ inv_lam_rhs)
    return inv_lam_rhs - (r.T @ middle) / lam


def _rank_one_metric_project(
    m0: torch.Tensor,
    *,
    left_metric_rows: torch.Tensor,
    constraint_keys: torch.Tensor,
    constraint_values: torch.Tensor,
    targets: torch.Tensor,
    betas: torch.Tensor,
    ridge: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Closed-form rank-one functional projection around a full map.

    Solves the finite-rank KKT system for:

    ``min_M 0.5 ||M-M0||_A^2 + 0.5 sum beta_i (g_i.T M c_i - y_i)^2``

    where ``A = ridge I + left_metric_rows.T @ left_metric_rows``.
    """

    m = torch.nan_to_num(m0.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    g = torch.nan_to_num(constraint_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    c = torch.nan_to_num(constraint_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    beta = torch.nan_to_num(betas.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(float(eps))
    if g.numel() == 0 or c.numel() == 0:
        return m.contiguous()
    if g.ndim != 2 or c.ndim != 2 or g.shape[0] != c.shape[0] or y.shape[0] != g.shape[0]:
        raise ValueError("rank-one constraints must align")
    if g.shape[1] != m.shape[0] or c.shape[1] != m.shape[1]:
        raise ValueError("rank-one constraints do not match map shape")
    if beta.shape[0] != g.shape[0]:
        raise ValueError("betas must align with constraints")
    raw = torch.einsum("qm,md,qd->q", g, m, c)
    residual = raw - y
    if float(torch.linalg.vector_norm(residual).item()) <= float(eps):
        return m.contiguous()
    x = _apply_lowrank_left_metric_inverse(left_metric_rows, g.T, ridge=ridge, eps=eps)
    gamma = (g @ x) * (c @ c.T)
    system = gamma + torch.diag(1.0 / beta)
    z = _solve_symmetric_psd(0.5 * (system + system.T), residual)
    correction = x @ (z.unsqueeze(1) * c)
    return torch.nan_to_num(m - correction, nan=0.0, posinf=0.0, neginf=0.0).contiguous()


def spectra_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    quotient_rank: int = 16,
    contrast_rank: int = 128,
    tail_anchors: int = 32,
    tail_quantile: float = 0.80,
    hazard_rank: int = 4,
    hazard_budget: float = 0.25,
    beta_tail: float = 100.0,
    beta_hazard: float = 10.0,
    generic_key_rank: int = 256,
    option_top_k: int = 128,
    low_surprise_quantile: float = 0.35,
    input_metric_weight: float = 20.0,
    quotient_mode: str = "joint",
    use_orca_quotient: bool = True,
    ablation_mode: str = "none",
    ridge: float = 1e-3,
    eps: float = 1e-6,
    risk_ratio_cap: float = 100.0,
) -> SpectraPurificationResult:
    """High-rank residual-map purifier with low-rank hazard correction.

    SPECTRA starts from the direct relational candidate, optionally removes the
    ORCA mixed key/value block, and then preserves high-tail target-parallel
    functionals while clipping top generic option-contrast hazards. Unlike
    Q-RICO it does not reconstruct the map in a low-rank basis.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    all_y = torch.nan_to_num(all_outputs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    tok = token_indices.detach().cpu().long()
    top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = logit_top_indices.detach().cpu().long()
    lm_idx = lm_head_indices.detach().cpu().long()
    lm_rows = torch.nan_to_num(lm_head_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("SPECTRA keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"SPECTRA weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("SPECTRA all_keys/all_outputs do not match update shape")
    if tok.numel() != k.shape[0]:
        raise ValueError("token_indices must align with selected rows")
    if top_vals.ndim != 2 or top_idx.ndim != 2 or top_vals.shape != top_idx.shape:
        raise ValueError("logit_top_values/logit_top_indices must be matching [T,k]")
    if top_vals.shape[0] < all_k.shape[0]:
        raise ValueError("logit top-k rows must cover all lesson keys")

    def cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        if rows.shape[0] <= max_rows or max_rows <= 0:
            return rows
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long().unique()
        return rows[idx]

    diagnostics: dict[str, float] = {
        "spectra_enabled": 1.0,
        "spectra_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if k.shape[0] == 0 or float(torch.linalg.vector_norm(update_f).item()) <= 1e-12:
        diagnostics["spectra_fallback"] = 1.0
        return SpectraPurificationResult(
            update=update_f.contiguous(),
            residual_update=update_f.contiguous(),
            projected_update=torch.zeros_like(update_f),
            diagnostics=diagnostics,
        )

    w_norm = w / w.mean().clamp_min(1e-12)
    row_scale = w_norm.sqrt().unsqueeze(1)
    row_scores = _token_row_surprise(all_k, layer)
    threshold = torch.quantile(row_scores, float(low_surprise_quantile))
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    generic_key_parts = [all_k[low_mask]]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_key_parts.append(cap_rows(neg, 512))
    generic_key_rows = torch.cat([part for part in generic_key_parts if part.numel() > 0], dim=0)
    generic_value_parts = [all_y[low_mask]]
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            generic_value_parts.append(cap_rows(out, 256))
    generic_value_rows = torch.cat([part for part in generic_value_parts if part.numel() > 0], dim=0)

    if use_orca_quotient and int(quotient_rank) > 0:
        signal_key_rows = k * row_scale
        signal_value_rows = y * row_scale
        try:
            cand_u, _cand_s, cand_vh = torch.linalg.svd(update_f, full_matrices=False)
            candidate_key_rows = cand_vh[: max(1, min(int(quotient_rank), cand_vh.shape[0]))]
            candidate_value_rows = cand_u[:, : max(1, min(int(quotient_rank), cand_u.shape[1]))].T
        except RuntimeError:
            candidate_key_rows = torch.empty(0, d_ff)
            candidate_value_rows = torch.empty(0, d_model)
        # Match the ORCA residual-only coordinate exactly. Q-RICO's faster
        # deflation basis was safe but lost the threshold-crossing component.
        deflate_key_rows, _ = _mixed_signal_risk_candidate_basis(
            signal_key_rows,
            generic_key_rows,
            candidate_key_rows,
            dim=d_ff,
            rank=quotient_rank,
            eps=eps,
            ratio_cap=risk_ratio_cap,
        )
        deflate_value_rows, _ = _mixed_signal_risk_candidate_basis(
            signal_value_rows,
            generic_value_rows,
            candidate_value_rows,
            dim=d_model,
            rank=quotient_rank,
            eps=eps,
            ratio_cap=risk_ratio_cap,
        )
        residual_update, projected_update, _coeff = _joint_basis_projection(
            update_f,
            deflate_key_rows,
            deflate_value_rows,
            mode=quotient_mode,
        )
    else:
        deflate_key_rows = torch.empty(0, d_ff)
        deflate_value_rows = torch.empty(0, d_model)
        residual_update = update_f.contiguous()
        projected_update = torch.zeros_like(update_f)

    mr = residual_update.T.contiguous()  # [m,d]
    usable_top_k = max(2, min(int(option_top_k), top_idx.shape[1]))
    selected_tokens = tok.clamp(min=0, max=top_idx.shape[0] - 1)
    selected_top = top_idx[selected_tokens, :usable_top_k]
    selected_vals = top_vals[selected_tokens, :usable_top_k]
    selected_rows = _lookup_weight_rows(selected_top, stored_indices=lm_idx, stored_rows=lm_rows)
    probs = torch.softmax(selected_vals, dim=1)
    expected_rows = (probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
    contrast_parts: list[torch.Tensor | None] = [
        (selected_rows - expected_rows).reshape(-1, d_model),
        y,
    ]
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            contrast_parts.append(cap_rows(out, max(1, int(contrast_rank) // 2)))
    contrast_rows = _fast_basis_with_rows(
        torch.cat([part for part in contrast_parts if part is not None and part.numel() > 0], dim=0),
        max(1, int(contrast_rank)),
        d_model,
    )
    if contrast_rows.numel() == 0:
        diagnostics["spectra_fallback"] = 1.0
        return SpectraPurificationResult(
            update=residual_update.contiguous(),
            residual_update=residual_update.contiguous(),
            projected_update=projected_update.contiguous(),
            diagnostics=diagnostics,
        )
    contrast = contrast_rows.T.contiguous()  # [d,a]
    effect = k @ mr
    ze = effect @ contrast
    zr = y @ contrast
    target_dot = (ze * zr).sum(dim=1).clamp_min(0.0)
    tail_score = w_norm * target_dot / torch.linalg.vector_norm(zr, dim=1).clamp_min(float(eps))
    tail_score = tail_score * torch.log1p(torch.linalg.vector_norm(ze, dim=1))
    tail_score = torch.nan_to_num(tail_score, nan=0.0, posinf=0.0, neginf=0.0)
    max_tail = max(1, min(int(tail_anchors), k.shape[0]))
    threshold_score = torch.quantile(tail_score, min(max(float(tail_quantile), 0.0), 1.0))
    tail_idx = torch.nonzero(tail_score >= threshold_score, as_tuple=False).flatten()
    if tail_idx.numel() > max_tail:
        tail_idx = torch.topk(tail_score, k=max_tail, largest=True).indices
    if tail_idx.numel() == 0:
        tail_idx = torch.topk(tail_score, k=max_tail, largest=True).indices
    k_tail = k[tail_idx]
    c_tail_raw = (y[tail_idx] @ contrast) @ contrast.T
    c_tail_norm = torch.linalg.vector_norm(c_tail_raw, dim=1, keepdim=True)
    c_tail = torch.where(c_tail_norm > float(eps), c_tail_raw / c_tail_norm.clamp_min(float(eps)), torch.zeros_like(c_tail_raw))
    y_tail = torch.einsum("tm,md,td->t", k_tail, mr, c_tail)

    tail_key_basis = _orthonormal_basis([k_tail], d_ff)
    generic_basis = _fast_basis_with_rows(generic_key_rows, max(1, int(generic_key_rank)), d_ff)
    if tail_key_basis.numel() > 0 and generic_basis.numel() > 0:
        generic_basis = generic_basis - (generic_basis @ tail_key_basis.T) @ tail_key_basis
    generic_basis = _orthonormal_basis([generic_basis], d_ff)

    tail_value_basis = _orthonormal_basis([c_tail], d_model)
    contrast_perp_rows = contrast_rows
    if tail_value_basis.numel() > 0:
        contrast_perp_rows = contrast_perp_rows - (contrast_perp_rows @ tail_value_basis.T) @ tail_value_basis
    contrast_perp_rows = _orthonormal_basis([contrast_perp_rows], d_model)
    if generic_basis.numel() == 0 or contrast_perp_rows.numel() == 0 or int(hazard_rank) <= 0:
        ghaz = torch.empty(0, d_ff)
        chaz = torch.empty(0, d_model)
        raw_haz = torch.empty(0)
        y_haz = torch.empty(0)
        hazard_before = torch.tensor(0.0)
        hazard_after_base = torch.tensor(0.0)
    else:
        c_perp = contrast_perp_rows.T.contiguous()
        hazard_matrix = torch.nan_to_num(generic_basis @ mr @ c_perp, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            u_h, s_h, vh_h = torch.linalg.svd(hazard_matrix, full_matrices=False)
        except RuntimeError:
            u_h = torch.empty(hazard_matrix.shape[0], 0)
            s_h = torch.empty(0)
            vh_h = torch.empty(0, hazard_matrix.shape[1])
        h_keep = min(max(0, int(hazard_rank)), s_h.numel())
        if h_keep > 0:
            ghaz = u_h[:, :h_keep].T @ generic_basis
            chaz = (c_perp @ vh_h[:h_keep].T).T
            chaz = F.normalize(chaz, dim=1)
            raw_haz = torch.einsum("hm,md,hd->h", ghaz, mr, chaz)
            base_budget = torch.median(y_tail.abs()).clamp_min(float(eps)) if y_tail.numel() > 0 else raw_haz.abs().median().clamp_min(float(eps))
            budget = float(hazard_budget) * base_budget
            y_haz = raw_haz.clamp(min=-float(budget.item()), max=float(budget.item()))
            hazard_before = s_h[0].abs() if s_h.numel() > 0 else torch.tensor(0.0)
            hazard_after_base = torch.linalg.vector_norm(y_haz, ord=float("inf")) if y_haz.numel() > 0 else torch.tensor(0.0)
        else:
            ghaz = torch.empty(0, d_ff)
            chaz = torch.empty(0, d_model)
            raw_haz = torch.empty(0)
            y_haz = torch.empty(0)
            hazard_before = torch.tensor(0.0)
            hazard_after_base = torch.tensor(0.0)

    if ablation_mode == "no_tail":
        g_constraints = ghaz
        c_constraints = chaz
        targets_all = y_haz
        betas = torch.full((ghaz.shape[0],), float(beta_hazard), dtype=torch.float32)
    elif ablation_mode == "no_hazard":
        g_constraints = k_tail
        c_constraints = c_tail
        targets_all = y_tail
        betas = torch.full((k_tail.shape[0],), float(beta_tail), dtype=torch.float32)
    elif ablation_mode == "hazard_only":
        g_constraints = ghaz
        c_constraints = chaz
        targets_all = y_haz
        betas = torch.full((ghaz.shape[0],), float(beta_hazard), dtype=torch.float32)
    else:
        tail_targets = y_tail
        tail_values = c_tail
        if ablation_mode == "shuffled_tail" and tail_targets.numel() > 1:
            perm = torch.roll(torch.arange(tail_targets.numel()), shifts=1)
            tail_targets = tail_targets[perm]
            tail_values = tail_values[perm]
        g_constraints = torch.cat([k_tail, ghaz], dim=0) if ghaz.numel() > 0 else k_tail
        c_constraints = torch.cat([tail_values, chaz], dim=0) if chaz.numel() > 0 else tail_values
        targets_all = torch.cat([tail_targets, y_haz], dim=0) if y_haz.numel() > 0 else tail_targets
        betas = torch.cat(
            [
                torch.full((tail_targets.shape[0],), float(beta_tail), dtype=torch.float32),
                torch.full((y_haz.shape[0],), float(beta_hazard), dtype=torch.float32),
            ],
            dim=0,
        )

    metric_rows_parts = [k * row_scale]
    if negative_keys is not None and negative_keys.numel() > 0 and input_metric_weight > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            metric_rows_parts.append(cap_rows(neg, 1024) * (float(input_metric_weight) ** 0.5))
    if generic_basis.numel() > 0:
        metric_rows_parts.append(generic_basis)
    metric_rows = torch.cat([part for part in metric_rows_parts if part.numel() > 0], dim=0)

    m_star = _rank_one_metric_project(
        mr,
        left_metric_rows=metric_rows,
        constraint_keys=g_constraints,
        constraint_values=c_constraints,
        targets=targets_all,
        betas=betas,
        ridge=ridge,
        eps=eps,
    )
    purified = m_star.T.contiguous()
    before_norm = torch.linalg.vector_norm(residual_update).clamp_min(1e-12)
    after_norm = torch.linalg.vector_norm(purified).clamp_min(1e-12)
    if float(after_norm.item()) > float(before_norm.item()):
        purified = purified * (before_norm / after_norm)
        m_star = purified.T.contiguous()
        after_norm = before_norm

    tail_after = torch.einsum("tm,md,td->t", k_tail, m_star, c_tail) if k_tail.numel() > 0 else torch.empty(0)
    if generic_basis.numel() > 0 and contrast_perp_rows.numel() > 0:
        hazard_after_matrix = generic_basis @ m_star @ contrast_perp_rows.T
        try:
            hazard_after = torch.linalg.svdvals(hazard_after_matrix)[0].abs()
        except RuntimeError:
            hazard_after = torch.linalg.vector_norm(hazard_after_matrix)
    else:
        hazard_after = torch.tensor(0.0)
    tail_mass_before = y_tail.clamp_min(0.0).sum().clamp_min(float(eps))
    tail_mass_after = tail_after.clamp_min(0.0).sum() if tail_after.numel() > 0 else torch.tensor(0.0)
    correction_fro = torch.linalg.vector_norm(m_star - mr)
    diagnostics.update(
        {
            "spectra_deflate_key_rank": float(deflate_key_rows.shape[0]),
            "spectra_deflate_value_rank": float(deflate_value_rows.shape[0]),
            "spectra_contrast_rank": float(contrast_rows.shape[0]),
            "spectra_tail_constraints": float(k_tail.shape[0]),
            "spectra_hazard_constraints": float(ghaz.shape[0]),
            "spectra_generic_key_rank": float(generic_basis.shape[0]),
            "spectra_residual_map_fro": float(torch.linalg.vector_norm(mr).item()),
            "spectra_projected_update_fro": float(torch.linalg.vector_norm(projected_update).item()),
            "spectra_purified_map_fro": float(after_norm.item()),
            "spectra_tail_mass_before": float(tail_mass_before.item()),
            "spectra_tail_mass_after": float(tail_mass_after.item()),
            "spectra_tail_mass_retention": float((tail_mass_after / tail_mass_before).item()),
            "spectra_hazard_spectral_before": float(hazard_before.item()),
            "spectra_hazard_spectral_after": float(hazard_after.item()),
            "spectra_hazard_spectral_ratio": float((hazard_after / hazard_before.clamp_min(float(eps))).item()),
            "spectra_hazard_budget_raw": float(hazard_after_base.item()),
            "spectra_correction_fro": float(correction_fro.item()),
            "spectra_beta_tail": float(beta_tail),
            "spectra_beta_hazard": float(beta_hazard),
            "spectra_hazard_budget": float(hazard_budget),
            "spectra_ablation_mode_code": {
                "none": 0.0,
                "no_tail": 1.0,
                "no_hazard": 2.0,
                "hazard_only": 3.0,
                "shuffled_tail": 4.0,
            }.get(ablation_mode, -1.0),
        }
    )
    return SpectraPurificationResult(
        update=purified.contiguous(),
        residual_update=residual_update.contiguous(),
        projected_update=projected_update.contiguous(),
        diagnostics=diagnostics,
    )


def prism_q_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    layer_idx: int,
    layer: nn.Module | None = None,
    future_outputs_by_layer: dict[int, torch.Tensor] | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    horizon: int = 4,
    signal_rank: int = 16,
    hazard_rank: int = 16,
    option_top_k: int = 8,
    generic_key_rank: int = 128,
    low_surprise_rows: int = 64,
    budget: float = 0.25,
    correction_cap: float = 0.35,
    signal_retention_min: float = 0.90,
    low_surprise_quantile: float = 0.35,
    residualize_hazard: bool = True,
    use_future_outputs: bool = True,
    ablation_mode: str = "none",
    ridge: float = 1e-3,
    eps: float = 1e-6,
    risk_ratio_cap: float = 100.0,
) -> PrismPurificationResult:
    """PRISM-Q propagated residual innovation-safety purifier.

    This first implementation uses the available single-pass layer captures as
    a cheap frozen-tangent proxy: downstream same-token/high-surprise MLP output
    rows augment the innovation cone, while local logit contrast rows and
    output-protection rows form the hazard cone. The hazard cone is quotiented
    by the innovation cone before clipping generic-key -> hazard singular modes.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    all_y = torch.nan_to_num(all_outputs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    tok = token_indices.detach().cpu().long()
    top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = logit_top_indices.detach().cpu().long()
    lm_idx = lm_head_indices.detach().cpu().long()
    lm_rows = torch.nan_to_num(lm_head_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("PRISM-Q keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"PRISM-Q weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("PRISM-Q all_keys/all_outputs do not match update shape")
    if tok.numel() != k.shape[0]:
        raise ValueError("token_indices must align with selected rows")
    if top_vals.ndim != 2 or top_idx.ndim != 2 or top_vals.shape != top_idx.shape:
        raise ValueError("logit_top_values/logit_top_indices must be matching [T,k]")

    def cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        if rows.shape[0] <= max_rows or max_rows <= 0:
            return rows
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long().unique()
        return rows[idx]

    diagnostics: dict[str, float] = {
        "prism_enabled": 1.0,
        "prism_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if k.shape[0] == 0 or float(torch.linalg.vector_norm(update_f).item()) <= 1e-12:
        diagnostics["prism_fallback"] = 1.0
        return PrismPurificationResult(
            update=update_f.contiguous(),
            signal_basis=torch.empty(0, d_model),
            hazard_basis=torch.empty(0, d_model),
            generic_basis=torch.empty(0, d_ff),
            diagnostics=diagnostics,
        )

    update_m = update_f.T.contiguous()  # [m,d]
    w_norm = w / w.mean().clamp_min(1e-12)
    effect = torch.nan_to_num(k @ update_m, nan=0.0, posinf=0.0, neginf=0.0)

    row_scores = _token_row_surprise(all_k, layer)
    q = min(max(float(low_surprise_quantile), 0.0), 1.0)
    threshold = torch.quantile(row_scores, q)
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    low_keys = all_k[low_mask]
    if int(low_surprise_rows) > 0:
        low_keys = cap_rows(low_keys, int(low_surprise_rows))

    generic_parts: list[torch.Tensor | None] = [low_keys]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_parts.append(cap_rows(neg, max(int(low_surprise_rows), int(generic_key_rank))))
    generic_basis = _fast_basis_with_rows(
        torch.cat([part for part in generic_parts if part is not None and part.numel() > 0], dim=0),
        max(1, int(generic_key_rank)),
        d_ff,
    )

    signal_parts: list[torch.Tensor | None] = [y, effect]
    if use_future_outputs and ablation_mode != "local_only" and future_outputs_by_layer:
        selected_tokens = tok.clamp(min=0, max=max(all_y.shape[0] - 1, 0))
        for future_idx in sorted(future_outputs_by_layer):
            if future_idx < layer_idx or future_idx > layer_idx + max(0, int(horizon)):
                continue
            future = torch.nan_to_num(
                future_outputs_by_layer[future_idx].detach().float().cpu(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if future.ndim != 2 or future.shape[1] != d_model or future.shape[0] == 0:
                continue
            clipped = selected_tokens.clamp(max=future.shape[0] - 1)
            signal_parts.append(future[clipped])
            keep = min(max(1, int(signal_rank)), future.shape[0])
            norms = torch.linalg.vector_norm(future, dim=1)
            signal_parts.append(future[torch.topk(norms, k=keep, largest=True).indices])
    signal_rows = torch.cat([part for part in signal_parts if part is not None and part.numel() > 0], dim=0)
    if ablation_mode == "shuffled_signal" and signal_rows.shape[0] > 1:
        signal_rows = torch.roll(signal_rows, shifts=1, dims=0)
    signal_basis = _fast_basis_with_rows(signal_rows, max(1, int(signal_rank)), d_model)

    usable_top_k = max(2, min(int(option_top_k), top_idx.shape[1]))
    hazard_parts: list[torch.Tensor | None] = []
    if top_idx.shape[0] > 0 and usable_top_k > 0:
        low_token_idx = torch.nonzero(low_mask, as_tuple=False).flatten()
        if low_token_idx.numel() == 0:
            low_token_idx = torch.arange(min(all_k.shape[0], top_idx.shape[0]))
        low_token_idx = cap_rows(low_token_idx.unsqueeze(1).float(), max(int(low_surprise_rows), 1)).flatten().long()
        low_token_idx = low_token_idx.clamp(min=0, max=top_idx.shape[0] - 1)
        low_top = top_idx[low_token_idx, :usable_top_k]
        low_vals = top_vals[low_token_idx, :usable_top_k]
        low_rows = _lookup_weight_rows(low_top, stored_indices=lm_idx, stored_rows=lm_rows)
        low_probs = torch.softmax(low_vals, dim=1)
        low_expected = (low_probs.unsqueeze(-1) * low_rows).sum(dim=1, keepdim=True)
        hazard_parts.append((low_rows - low_expected).reshape(-1, d_model))
        selected_tokens = tok.clamp(min=0, max=top_idx.shape[0] - 1)
        selected_top = top_idx[selected_tokens, :usable_top_k]
        selected_vals = top_vals[selected_tokens, :usable_top_k]
        selected_rows = _lookup_weight_rows(selected_top, stored_indices=lm_idx, stored_rows=lm_rows)
        selected_probs = torch.softmax(selected_vals, dim=1)
        selected_expected = (selected_probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
        hazard_parts.append((selected_rows - selected_expected).reshape(-1, d_model))
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            hazard_parts.append(cap_rows(out, max(int(hazard_rank) * 4, int(hazard_rank))))
    if use_future_outputs and ablation_mode != "local_only" and future_outputs_by_layer:
        for future_idx in sorted(future_outputs_by_layer):
            if future_idx < layer_idx or future_idx > layer_idx + max(0, int(horizon)):
                continue
            future = torch.nan_to_num(
                future_outputs_by_layer[future_idx].detach().float().cpu(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if future.ndim == 2 and future.shape[1] == d_model and future.shape[0] > 1:
                hazard_parts.append(cap_rows(future, max(1, int(hazard_rank))))
    raw_hazard = _fast_basis_with_rows(
        torch.cat([part for part in hazard_parts if part is not None and part.numel() > 0], dim=0),
        max(1, int(hazard_rank) * 2),
        d_model,
    )
    if residualize_hazard and ablation_mode != "no_residualize" and signal_basis.numel() > 0 and raw_hazard.numel() > 0:
        hazard_rows = raw_hazard - (raw_hazard @ signal_basis.T) @ signal_basis
    else:
        hazard_rows = raw_hazard
    hazard_basis = _fast_basis_with_rows(hazard_rows, max(1, int(hazard_rank)), d_model)

    if generic_basis.numel() == 0 or hazard_basis.numel() == 0 or signal_basis.numel() == 0:
        diagnostics["prism_fallback"] = 1.0
        return PrismPurificationResult(
            update=update_f.contiguous(),
            signal_basis=signal_basis.contiguous(),
            hazard_basis=hazard_basis.contiguous(),
            generic_basis=generic_basis.contiguous(),
            diagnostics=diagnostics,
        )

    hazard_before_matrix = torch.nan_to_num(generic_basis @ update_m @ hazard_basis.T, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        u_h, s_h, vh_h = torch.linalg.svd(hazard_before_matrix, full_matrices=False)
    except RuntimeError:
        u_h = torch.empty(hazard_before_matrix.shape[0], 0)
        s_h = torch.empty(0)
        vh_h = torch.empty(0, hazard_before_matrix.shape[1])
    if s_h.numel() == 0:
        diagnostics["prism_fallback"] = 1.0
        return PrismPurificationResult(
            update=update_f.contiguous(),
            signal_basis=signal_basis.contiguous(),
            hazard_basis=hazard_basis.contiguous(),
            generic_basis=generic_basis.contiguous(),
            diagnostics=diagnostics,
        )
    signal_before_matrix = torch.nan_to_num(k @ update_m @ signal_basis.T, nan=0.0, posinf=0.0, neginf=0.0)
    signal_scale = torch.linalg.vector_norm(signal_before_matrix, dim=1).median().clamp_min(float(eps))
    spectral_budget = float(budget) * signal_scale
    clipped = s_h.clamp(max=float(spectral_budget.item()))
    hazard_delta = u_h @ torch.diag(s_h - clipped) @ vh_h
    if ablation_mode == "no_hazard":
        hazard_delta.zero_()
    elif ablation_mode == "correction_only":
        pass
    elif ablation_mode == "removed_hazard_only":
        pass

    left_system = generic_basis @ generic_basis.T + max(float(ridge), float(eps)) * torch.eye(
        generic_basis.shape[0], dtype=torch.float32
    )
    left = _solve_symmetric_psd(0.5 * (left_system + left_system.T), hazard_delta)
    correction_m = generic_basis.T @ left @ hazard_basis
    correction_norm = torch.linalg.vector_norm(correction_m)
    base_norm = torch.linalg.vector_norm(update_m).clamp_min(float(eps))
    cap = max(float(correction_cap), 0.0) * base_norm
    if float(correction_norm.item()) > float(cap.item()) and float(cap.item()) > 0.0:
        correction_m = correction_m * (cap / correction_norm.clamp_min(float(eps)))
        correction_norm = torch.linalg.vector_norm(correction_m)

    if ablation_mode == "correction_only":
        candidate_m = correction_m
    elif ablation_mode == "removed_hazard_only":
        candidate_m = update_m - correction_m
        candidate_m = update_m - candidate_m
    else:
        candidate_m = update_m - correction_m

    def signal_retention(candidate: torch.Tensor) -> torch.Tensor:
        before = torch.linalg.vector_norm(signal_before_matrix).clamp_min(float(eps))
        after = torch.linalg.vector_norm(k @ candidate @ signal_basis.T)
        return after / before

    retention = signal_retention(candidate_m)
    scale = torch.tensor(1.0)
    if ablation_mode not in {"correction_only", "removed_hazard_only"}:
        for _ in range(8):
            if float(retention.item()) >= float(signal_retention_min):
                break
            scale = scale * 0.5
            candidate_m = update_m - scale * correction_m
            retention = signal_retention(candidate_m)

    hazard_after_matrix = torch.nan_to_num(generic_basis @ candidate_m @ hazard_basis.T, nan=0.0, posinf=0.0, neginf=0.0)
    hazard_before = torch.linalg.matrix_norm(hazard_before_matrix, ord=2)
    hazard_after = torch.linalg.matrix_norm(hazard_after_matrix, ord=2)
    after_norm = torch.linalg.vector_norm(candidate_m)
    if ablation_mode != "correction_only" and float(after_norm.item()) > float(base_norm.item()):
        candidate_m = candidate_m * (base_norm / after_norm.clamp_min(float(eps)))
        after_norm = torch.linalg.vector_norm(candidate_m)

    diagnostics.update(
        {
            "prism_fallback": 0.0,
            "prism_signal_rank": float(signal_basis.shape[0]),
            "prism_hazard_rank": float(hazard_basis.shape[0]),
            "prism_generic_key_rank": float(generic_basis.shape[0]),
            "prism_horizon": float(horizon),
            "prism_option_top_k": float(usable_top_k),
            "prism_budget": float(budget),
            "prism_correction_cap": float(correction_cap),
            "prism_spectral_budget": float(spectral_budget.item()),
            "prism_hazard_spectral_before": float(hazard_before.item()),
            "prism_hazard_spectral_after": float(hazard_after.item()),
            "prism_hazard_spectral_ratio": float((hazard_after / hazard_before.clamp_min(float(eps))).item()),
            "prism_correction_fro": float(correction_norm.item()),
            "prism_correction_scale": float(scale.item()),
            "prism_signal_retention": float(retention.item()),
            "prism_update_fro_after": float(after_norm.item()),
            "prism_residualize_hazard": 1.0 if residualize_hazard and ablation_mode != "no_residualize" else 0.0,
            "prism_use_future_outputs": 1.0 if use_future_outputs and ablation_mode != "local_only" else 0.0,
            "prism_ablation_mode_code": {
                "none": 0.0,
                "no_residualize": 1.0,
                "local_only": 2.0,
                "shuffled_signal": 3.0,
                "correction_only": 4.0,
                "removed_hazard_only": 5.0,
                "no_hazard": 6.0,
            }.get(ablation_mode, -1.0),
        }
    )
    return PrismPurificationResult(
        update=candidate_m.T.contiguous(),
        signal_basis=signal_basis.contiguous(),
        hazard_basis=hazard_basis.contiguous(),
        generic_basis=generic_basis.contiguous(),
        diagnostics=diagnostics,
    )


def trace_q_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    object_endpoints: int = 16,
    ambient_endpoints: int = 32,
    option_top_k: int = 8,
    option_contrasts: int = 4,
    object_rank: int = 16,
    ambient_rank: int = 16,
    generic_key_rank: int = 128,
    low_surprise_quantile: float = 0.35,
    target_tau: float = 1.0,
    target_floor: float = 0.10,
    collateral_weight: float = 0.25,
    layer_trust_threshold: float = 2.0,
    eps: float = 1e-6,
) -> TraceQResult:
    """TRACE-Q local endpoint-tangent purifier.

    This is the first cheap implementation of the TRACE-Q idea. It does not
    backpropagate exact downstream VJPs. Instead, it builds object and ambient
    local option-contrast bases from the same pass, residualizes ambient
    contrasts against object contrasts, keeps update components that live in an
    object-predominant readout subspace, and shrinks generic-key to ambient
    residual movement.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    tok = token_indices.detach().cpu().long()
    top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = logit_top_indices.detach().cpu().long()
    lm_idx = lm_head_indices.detach().cpu().long()
    lm_rows = torch.nan_to_num(lm_head_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("TRACE-Q keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"TRACE-Q weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff:
        raise ValueError("TRACE-Q all_keys do not match update shape")
    if tok.numel() != k.shape[0]:
        raise ValueError("TRACE-Q token_indices must align with selected rows")
    if top_vals.ndim != 2 or top_idx.ndim != 2 or top_vals.shape != top_idx.shape:
        raise ValueError("TRACE-Q logit_top_values/logit_top_indices must be matching [T,k]")

    diagnostics: dict[str, float] = {
        "trace_q_enabled": 1.0,
        "trace_q_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if k.shape[0] == 0 or float(torch.linalg.vector_norm(update_f).item()) <= 1e-12:
        diagnostics["trace_q_fallback"] = 1.0
        return TraceQResult(
            update=update_f.contiguous(),
            object_basis=torch.empty(0, d_model),
            ambient_basis=torch.empty(0, d_model),
            generic_basis=torch.empty(0, d_ff),
            diagnostics=diagnostics,
        )

    def cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        if rows.shape[0] <= max_rows or max_rows <= 0:
            return rows
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long().unique()
        return rows[idx]

    def contrast_rows_for_tokens(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if tokens.numel() == 0 or top_idx.numel() == 0:
            return torch.empty(0, d_model, dtype=torch.float32)
        tokens = tokens.flatten().long().unique()
        if tokens.numel() > max_tokens > 0:
            tokens = tokens[:max_tokens]
        tokens = tokens.clamp(min=0, max=top_idx.shape[0] - 1)
        usable_top_k = max(2, min(int(option_top_k), top_idx.shape[1]))
        keep_contrasts = max(1, min(int(option_contrasts), usable_top_k))
        selected_top = top_idx[tokens, :usable_top_k]
        selected_vals = top_vals[tokens, :usable_top_k]
        selected_rows = _lookup_weight_rows(selected_top, stored_indices=lm_idx, stored_rows=lm_rows)
        probs = torch.softmax(selected_vals, dim=1)
        expected = (probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
        contrasts = selected_rows - expected
        if keep_contrasts < usable_top_k:
            contrast_norms = torch.linalg.vector_norm(contrasts, dim=2)
            local_idx = torch.topk(contrast_norms, k=keep_contrasts, dim=1, largest=True).indices
            contrasts = torch.gather(
                contrasts,
                1,
                local_idx.unsqueeze(-1).expand(-1, -1, d_model),
            )
        return contrasts.reshape(-1, d_model)

    w_rank = torch.argsort(w, descending=True)
    object_tokens = tok[w_rank[: max(1, min(int(object_endpoints), tok.numel()))]]
    object_rows = contrast_rows_for_tokens(object_tokens, int(object_endpoints))

    row_scores = _token_row_surprise(all_k, layer)
    q = min(max(float(low_surprise_quantile), 0.0), 1.0)
    threshold = torch.quantile(row_scores, q)
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    low_idx = torch.nonzero(low_mask, as_tuple=False).flatten()
    if top_vals.shape[0] > 0 and top_vals.shape[1] >= 2 and low_idx.numel() > 0:
        capped_low = low_idx.clamp(max=top_vals.shape[0] - 1)
        margins = top_vals[capped_low, 0] - top_vals[capped_low, 1]
        order = torch.argsort(margins, descending=True)
        ambient_tokens = capped_low[order[: max(1, min(int(ambient_endpoints), capped_low.numel()))]]
    else:
        ambient_tokens = low_idx[: max(1, min(int(ambient_endpoints), low_idx.numel()))]
    if ambient_tokens.numel() > 0 and object_tokens.numel() > 0:
        object_set = set(int(v) for v in object_tokens.flatten().tolist())
        ambient_tokens = torch.tensor(
            [int(v) for v in ambient_tokens.flatten().tolist() if int(v) not in object_set],
            dtype=torch.long,
        )
    ambient_rows = contrast_rows_for_tokens(ambient_tokens, int(ambient_endpoints))
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            ambient_rows = torch.cat([ambient_rows, cap_rows(out, max(int(ambient_rank) * 2, int(ambient_rank)))], dim=0)

    object_basis = _fast_basis_with_rows(object_rows, max(1, int(object_rank)), d_model)
    ambient_raw = _fast_basis_with_rows(ambient_rows, max(1, int(ambient_rank) * 2), d_model)
    if object_basis.numel() > 0 and ambient_raw.numel() > 0:
        ambient_resid = ambient_raw - (ambient_raw @ object_basis.T) @ object_basis
    else:
        ambient_resid = ambient_raw
    ambient_basis = _fast_basis_with_rows(ambient_resid, max(1, int(ambient_rank)), d_model)

    signal_key_basis = _fast_basis_with_rows(k * (w / w.mean().clamp_min(1e-12)).sqrt().unsqueeze(1), max(1, min(k.shape[0], 64)), d_ff)
    generic_parts = [all_k[low_mask]]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = torch.nan_to_num(negative_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_parts.append(cap_rows(neg, max(int(generic_key_rank), 1)))
    generic_raw = _fast_basis_with_rows(
        torch.cat([part for part in generic_parts if part.numel() > 0], dim=0),
        max(1, int(generic_key_rank) * 2),
        d_ff,
    )
    if signal_key_basis.numel() > 0 and generic_raw.numel() > 0:
        generic_resid = generic_raw - (generic_raw @ signal_key_basis.T) @ signal_key_basis
    else:
        generic_resid = generic_raw
    generic_basis = _fast_basis_with_rows(generic_resid, max(1, int(generic_key_rank)), d_ff)

    if object_basis.numel() == 0 or ambient_basis.numel() == 0 or generic_basis.numel() == 0:
        diagnostics["trace_q_fallback"] = 1.0
        return TraceQResult(
            update=update_f.contiguous(),
            object_basis=object_basis.contiguous(),
            ambient_basis=ambient_basis.contiguous(),
            generic_basis=generic_basis.contiguous(),
            diagnostics=diagnostics,
        )

    # Object-predominant target/update projector.  Work in the low-rank union
    # of object and ambient endpoint contrasts to avoid a dense d x d eigensolve.
    union_basis = _fast_basis_with_rows(
        torch.cat([object_basis, ambient_basis], dim=0),
        max(1, min(int(object_rank) + int(ambient_rank), d_model)),
        d_model,
    )
    obj_u = object_basis @ union_basis.T
    amb_u = ambient_basis @ union_basis.T
    mo = obj_u.T @ obj_u
    mc = amb_u.T @ amb_u
    eye = torch.eye(union_basis.shape[0], dtype=torch.float32)
    try:
        evals_b, evecs_b = torch.linalg.eigh(0.5 * (mc + mc.T) + float(eps) * eye)
        inv_sqrt = evecs_b @ torch.diag(evals_b.clamp_min(float(eps)).rsqrt()) @ evecs_b.T
        sym = inv_sqrt.T @ (0.5 * (mo + mo.T)) @ inv_sqrt
        evals, evecs = torch.linalg.eigh(0.5 * (sym + sym.T))
        modes = inv_sqrt @ evecs
        tau = max(float(target_tau), float(eps))
        gains = (evals.clamp_min(0.0) / (evals.clamp_min(0.0) + tau)).clamp(0.0, 1.0)
        coord_projector = modes @ torch.diag(gains) @ modes.T
        coord_projector = 0.5 * (coord_projector + coord_projector.T)
    except RuntimeError:
        coord_projector = torch.eye(union_basis.shape[0], dtype=torch.float32)
        evals = torch.ones(union_basis.shape[0], dtype=torch.float32)
        gains = torch.ones_like(evals)
    update_m = update_f.T.contiguous()
    projected_part = (update_m @ union_basis.T) @ coord_projector @ union_basis
    full_union_part = (update_m @ union_basis.T) @ union_basis
    outside_part = update_m - full_union_part
    floor = min(max(float(target_floor), 0.0), 1.0)
    candidate_m = outside_part * floor + projected_part + floor * (full_union_part - projected_part)

    # Two-sided collateral shrink in the residualized generic-key x ambient
    # contrast block.  Bases are orthonormal rows, so the proximal shrink has a
    # simple closed form in that block.
    collateral_before_matrix = generic_basis @ candidate_m @ ambient_basis.T
    shrink = float(collateral_weight) / (1.0 + max(float(collateral_weight), 0.0))
    if shrink > 0.0:
        correction = shrink * (generic_basis.T @ collateral_before_matrix @ ambient_basis)
        candidate_m = candidate_m - correction
    else:
        correction = torch.zeros_like(candidate_m)

    object_after = torch.linalg.vector_norm(k @ candidate_m @ object_basis.T).square()
    collateral_after = torch.linalg.vector_norm(generic_basis @ candidate_m @ ambient_basis.T).square()
    quotient = object_after / collateral_after.clamp_min(float(eps))
    trust_threshold = max(float(layer_trust_threshold), float(eps))
    trust = torch.sqrt((quotient / trust_threshold).clamp(min=0.0, max=1.0))
    candidate_m = candidate_m * trust

    before_norm = torch.linalg.vector_norm(update_f).clamp_min(float(eps))
    after_norm = torch.linalg.vector_norm(candidate_m).clamp_min(float(eps))
    if float(after_norm.item()) > float(before_norm.item()):
        candidate_m = candidate_m * (before_norm / after_norm)
        after_norm = torch.linalg.vector_norm(candidate_m).clamp_min(float(eps))

    diagnostics.update(
        {
            "trace_q_fallback": 0.0,
            "trace_q_object_rank": float(object_basis.shape[0]),
            "trace_q_ambient_rank": float(ambient_basis.shape[0]),
            "trace_q_generic_key_rank": float(generic_basis.shape[0]),
            "trace_q_union_rank": float(union_basis.shape[0]),
            "trace_q_object_endpoints": float(object_tokens.numel()),
            "trace_q_ambient_endpoints": float(ambient_tokens.numel()),
            "trace_q_target_floor": float(floor),
            "trace_q_target_tau": float(target_tau),
            "trace_q_collateral_weight": float(collateral_weight),
            "trace_q_collateral_before": float(torch.linalg.vector_norm(collateral_before_matrix).item()),
            "trace_q_collateral_after": float(torch.linalg.vector_norm(generic_basis @ candidate_m @ ambient_basis.T).item()),
            "trace_q_correction_fro": float(torch.linalg.vector_norm(correction).item()),
            "trace_q_object_gain_after": float(object_after.item()),
            "trace_q_collateral_gain_after": float(collateral_after.item()),
            "trace_q_quotient": float(quotient.item()),
            "trace_q_layer_trust": float(trust.item()),
            "trace_q_update_fro_after": float(after_norm.item()),
            "trace_q_generalized_eval_max": float(evals.max().item()) if evals.numel() else 0.0,
            "trace_q_generalized_gain_mean": float(gains.mean().item()) if gains.numel() else 0.0,
        }
    )
    return TraceQResult(
        update=candidate_m.T.contiguous(),
        object_basis=object_basis.contiguous(),
        ambient_basis=ambient_basis.contiguous(),
        generic_basis=generic_basis.contiguous(),
        diagnostics=diagnostics,
    )


def tdmi_q_transport_scores(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    layer: nn.Module | None = None,
    future_outputs_by_layer: dict[int, torch.Tensor] | None = None,
    layer_idx: int = 0,
    object_endpoints: int = 8,
    ambient_endpoints: int = 16,
    object_rank: int = 8,
    ambient_rank: int = 16,
    horizon: int = 4,
    low_surprise_quantile: float = 0.35,
    trust_temperature: float = 0.5,
    trust_threshold: float = 0.0,
    trust_floor: float = 0.15,
    use_future_outputs: bool = True,
    eps: float = 1e-6,
) -> TdmiQResult:
    """Score selected rows by transported object/default hidden manifolds.

    This is the fast TDMI-Q row-scoring primitive.  It keeps Q-RICO's proposed
    high-rank update intact, computes each selected row's proposed residual
    effect ``u_i = k_i update.T``, and assigns trust based on whether ``u_i``
    lies in object/innovation hidden-state manifolds rather than low-surprise
    default hidden manifolds.  ``future_outputs_by_layer`` provides a cheap
    same-pass transport proxy by pooling hidden states at the same token
    indices in downstream layers.  Exact VJP transport is intentionally left as
    a stricter future mode.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    all_y = torch.nan_to_num(all_outputs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    tok = token_indices.detach().cpu().long()

    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("TDMI-Q keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"TDMI-Q weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("TDMI-Q all_keys/all_outputs do not match update shape")
    if tok.numel() != k.shape[0]:
        raise ValueError("TDMI-Q token_indices must align with selected rows")

    diagnostics: dict[str, float] = {
        "tdmi_q_enabled": 1.0,
        "tdmi_q_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if k.shape[0] == 0 or float(torch.linalg.vector_norm(update_f).item()) <= 1e-12:
        trust = torch.ones(k.shape[0], dtype=torch.float32)
        diagnostics["tdmi_q_fallback"] = 1.0
        return TdmiQResult(
            row_trust=trust,
            row_signal=torch.zeros_like(trust),
            row_ambient=torch.zeros_like(trust),
            object_basis=torch.empty(0, d_model),
            ambient_basis=torch.empty(0, d_model),
            diagnostics=diagnostics,
        )

    def cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        if rows.shape[0] <= max_rows or max_rows <= 0:
            return rows
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long().unique()
        return rows[idx]

    row_scores = _token_row_surprise(all_k, layer)
    low_q = min(max(float(low_surprise_quantile), 0.0), 1.0)
    threshold = torch.quantile(row_scores, low_q)
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    low_idx = torch.nonzero(low_mask, as_tuple=False).flatten()

    w_rank = torch.argsort(w, descending=True)
    object_row_idx = w_rank[: max(1, min(int(object_endpoints), w_rank.numel()))]
    object_tokens = tok[object_row_idx].clamp(min=0, max=max(all_y.shape[0] - 1, 0))
    ambient_tokens = low_idx[: max(1, min(int(ambient_endpoints), low_idx.numel()))].clamp(
        min=0,
        max=max(all_y.shape[0] - 1, 0),
    )
    if ambient_tokens.numel() > 0 and object_tokens.numel() > 0:
        object_set = set(int(v) for v in object_tokens.flatten().tolist())
        ambient_tokens = torch.tensor(
            [int(v) for v in ambient_tokens.flatten().tolist() if int(v) not in object_set],
            dtype=torch.long,
        )

    object_parts: list[torch.Tensor] = [
        y[object_row_idx],
        (k @ update_f.T)[object_row_idx],
        all_y[object_tokens] if object_tokens.numel() > 0 else torch.empty(0, d_model),
    ]
    ambient_parts: list[torch.Tensor] = [
        all_y[ambient_tokens] if ambient_tokens.numel() > 0 else torch.empty(0, d_model)
    ]
    if use_future_outputs and future_outputs_by_layer:
        max_future = int(layer_idx) + max(0, int(horizon))
        for future_idx in sorted(future_outputs_by_layer):
            if future_idx < int(layer_idx) or future_idx > max_future:
                continue
            future = torch.nan_to_num(
                future_outputs_by_layer[future_idx].detach().float().cpu(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if future.ndim != 2 or future.shape[1] != d_model or future.shape[0] == 0:
                continue
            if object_tokens.numel() > 0:
                object_parts.append(future[object_tokens.clamp(max=future.shape[0] - 1)])
            if ambient_tokens.numel() > 0:
                ambient_parts.append(future[ambient_tokens.clamp(max=future.shape[0] - 1)])
            keep = min(max(1, int(ambient_endpoints)), future.shape[0])
            future_norms = torch.linalg.vector_norm(future, dim=1)
            ambient_parts.append(future[torch.topk(future_norms, k=keep, largest=False).indices])

    object_rows = torch.cat([part for part in object_parts if part.numel() > 0], dim=0)
    ambient_rows = torch.cat([part for part in ambient_parts if part.numel() > 0], dim=0)
    object_basis = _fast_basis_with_rows(object_rows, max(1, int(object_rank)), d_model)
    ambient_raw = _fast_basis_with_rows(ambient_rows, max(1, int(ambient_rank) * 2), d_model)
    if object_basis.numel() > 0 and ambient_raw.numel() > 0:
        ambient_rows_resid = ambient_raw - (ambient_raw @ object_basis.T) @ object_basis
    else:
        ambient_rows_resid = ambient_raw
    ambient_basis = _fast_basis_with_rows(ambient_rows_resid, max(1, int(ambient_rank)), d_model)

    effects = torch.nan_to_num(k @ update_f.T, nan=0.0, posinf=0.0, neginf=0.0)
    if object_basis.numel() == 0 or ambient_basis.numel() == 0:
        trust = torch.ones(k.shape[0], dtype=torch.float32)
        diagnostics["tdmi_q_fallback"] = 1.0
        return TdmiQResult(
            row_trust=trust,
            row_signal=torch.zeros_like(trust),
            row_ambient=torch.zeros_like(trust),
            object_basis=object_basis.contiguous(),
            ambient_basis=ambient_basis.contiguous(),
            diagnostics=diagnostics,
        )

    row_signal = torch.linalg.vector_norm(effects @ object_basis.T, dim=1).square()
    row_ambient = torch.linalg.vector_norm(effects @ ambient_basis.T, dim=1).square()
    temp = max(float(trust_temperature), float(eps))
    logits = (torch.log(row_signal + float(eps)) - torch.log(row_ambient + float(eps)) - float(trust_threshold)) / temp
    floor = min(max(float(trust_floor), 0.0), 1.0)
    row_trust = floor + (1.0 - floor) * torch.sigmoid(logits)

    def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() < 2 or b.numel() != a.numel():
            return 0.0
        aa = a.float() - a.float().mean()
        bb = b.float() - b.float().mean()
        denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
        if float(denom.item()) <= float(eps):
            return 0.0
        return float(((aa @ bb) / denom).item())

    diagnostics.update(
        {
            "tdmi_q_fallback": 0.0,
            "tdmi_q_object_rank": float(object_basis.shape[0]),
            "tdmi_q_ambient_rank": float(ambient_basis.shape[0]),
            "tdmi_q_object_endpoints": float(object_tokens.numel()),
            "tdmi_q_ambient_endpoints": float(ambient_tokens.numel()),
            "tdmi_q_horizon": float(horizon),
            "tdmi_q_use_future_outputs": 1.0 if use_future_outputs else 0.0,
            "tdmi_q_trust_mean": float(row_trust.mean().item()),
            "tdmi_q_trust_min": float(row_trust.min().item()),
            "tdmi_q_trust_max": float(row_trust.max().item()),
            "tdmi_q_signal_mean": float(row_signal.mean().item()),
            "tdmi_q_ambient_mean": float(row_ambient.mean().item()),
            "tdmi_q_signal_kept_fraction": float(((row_trust * row_signal).sum() / row_signal.sum().clamp_min(float(eps))).item()),
            "tdmi_q_ambient_kept_fraction": float(((row_trust * row_ambient).sum() / row_ambient.sum().clamp_min(float(eps))).item()),
            "tdmi_q_trust_weight_corr": safe_corr(row_trust, w),
            "tdmi_q_trust_effect_norm_corr": safe_corr(row_trust, torch.linalg.vector_norm(effects, dim=1)),
        }
    )
    return TdmiQResult(
        row_trust=row_trust.contiguous(),
        row_signal=row_signal.contiguous(),
        row_ambient=row_ambient.contiguous(),
        object_basis=object_basis.contiguous(),
        ambient_basis=ambient_basis.contiguous(),
        diagnostics=diagnostics,
    )


def qrico_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    deflate_key_rank: int = 16,
    deflate_value_rank: int = 16,
    rank: int = 64,
    option_sketch_rank: int = 256,
    target_parallel_rank: int = 4,
    scramble_weight: float = 0.35,
    residual_row_weight_power: float = 0.5,
    quotient_mode: str = "joint",
    solve_mode: str = "sylvester",
    low_surprise_quantile: float = 0.35,
    negative_weight: float = 20.0,
    output_weight: float = 10.0,
    cca_ridge: float = 1e-3,
    layer_evidence_min: float = 0.03,
    layer_evidence_target: float = 0.20,
    apply_layer_trust: bool = True,
    eps: float = 1e-6,
    risk_ratio_cap: float = 100.0,
) -> QricoPurificationResult:
    """Q-RICO map-level purifier for relational context-value writes.

    The purifier treats the current protected relational write as a proposal,
    removes the mixed ORCA low-rank atom block, then solves a small Sylvester
    problem in the residual map's key/value geometry. The value penalty is a
    local option-scramble metric: logit contrast movement orthogonal to the
    residual target's own local option effect is costly, while target-parallel
    readout leverage is allowed.
    """

    update_f = torch.nan_to_num(update.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    k = torch.nan_to_num(keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(targets.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.nan_to_num(weights.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    all_k = torch.nan_to_num(all_keys.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    all_y = torch.nan_to_num(all_outputs.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    tok = token_indices.detach().cpu().long()
    top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = logit_top_indices.detach().cpu().long()
    lm_idx = lm_head_indices.detach().cpu().long()
    lm_rows = torch.nan_to_num(lm_head_rows.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("Q-RICO keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"Q-RICO weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("Q-RICO all_keys/all_outputs do not match update shape")
    if tok.numel() != k.shape[0]:
        raise ValueError("token_indices must align with selected rows")
    if top_vals.ndim != 2 or top_idx.ndim != 2 or top_vals.shape != top_idx.shape:
        raise ValueError("logit_top_values/logit_top_indices must be matching [T,k]")
    if top_vals.shape[0] < all_k.shape[0]:
        raise ValueError("logit top-k rows must cover all lesson keys")

    def cap_rows(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        if rows.shape[0] <= max_rows or max_rows <= 0:
            return rows
        idx = torch.linspace(0, rows.shape[0] - 1, steps=max_rows).round().long().unique()
        return rows[idx]

    diagnostics: dict[str, float] = {
        "qrico_enabled": 1.0,
        "qrico_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if k.shape[0] == 0 or float(torch.linalg.vector_norm(update_f).item()) <= 1e-12:
        diagnostics["qrico_fallback"] = 1.0
        empty = torch.empty(0, 0, dtype=torch.float32)
        return QricoPurificationResult(
            update=update_f.contiguous(),
            residual_update=update_f.contiguous(),
            projected_update=torch.zeros_like(update_f),
            key_basis=empty,
            value_basis=empty,
            coeff=empty,
            diagnostics=diagnostics,
        )

    w_norm = w / w.mean().clamp_min(1e-12)
    row_scale = w_norm.sqrt().unsqueeze(1)
    signal_key_rows = k * row_scale
    signal_value_rows = y * row_scale
    row_scores = _token_row_surprise(all_k, layer)
    threshold = torch.quantile(row_scores, float(low_surprise_quantile))
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    generic_key_parts = [all_k[low_mask]]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_key_parts.append(cap_rows(neg, 256))
    generic_key_rows = torch.cat([part for part in generic_key_parts if part.numel() > 0], dim=0)
    generic_value_parts = [all_y[low_mask]]
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            generic_value_parts.append(cap_rows(out, 256))
    generic_value_rows = torch.cat([part for part in generic_value_parts if part.numel() > 0], dim=0)

    base_deflate_key_rows, _base_deflate_key_ratios = _fast_mixed_signal_risk_candidate_basis(
        signal_key_rows,
        generic_key_rows,
        None,
        dim=d_ff,
        rank=deflate_key_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    base_deflate_value_rows, _base_deflate_value_ratios = _fast_mixed_signal_risk_candidate_basis(
        signal_value_rows,
        generic_value_rows,
        None,
        dim=d_model,
        rank=deflate_value_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    deflate_candidate_key_rows = (
        base_deflate_value_rows @ update_f if base_deflate_value_rows.numel() > 0 else torch.empty(0, d_ff)
    )
    deflate_candidate_value_rows = (
        base_deflate_key_rows @ update_f.T if base_deflate_key_rows.numel() > 0 else torch.empty(0, d_model)
    )
    deflate_key_rows, _deflate_key_ratios = _fast_mixed_signal_risk_candidate_basis(
        signal_key_rows,
        generic_key_rows,
        deflate_candidate_key_rows,
        dim=d_ff,
        rank=deflate_key_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    deflate_value_rows, _deflate_value_ratios = _fast_mixed_signal_risk_candidate_basis(
        signal_value_rows,
        generic_value_rows,
        deflate_candidate_value_rows,
        dim=d_model,
        rank=deflate_value_rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    residual_update, projected_update, _deflate_coeff = _joint_basis_projection(
        update_f,
        deflate_key_rows,
        deflate_value_rows,
        mode=quotient_mode,
    )
    m_perp = residual_update.T.contiguous()  # [m,d]
    r_perp = torch.nan_to_num(k @ m_perp, nan=0.0, posinf=0.0, neginf=0.0)
    base_fit = torch.nan_to_num(k @ update_f.T, nan=0.0, posinf=0.0, neginf=0.0)
    residual_ratio = (
        torch.linalg.vector_norm(r_perp, dim=1).square()
        / torch.linalg.vector_norm(base_fit, dim=1).square().clamp_min(float(eps))
    ).clamp_min(0.0)
    residual_power = min(max(float(residual_row_weight_power), 0.0), 2.0)
    w_tilde = w_norm * residual_ratio.clamp_min(float(eps)).pow(residual_power)
    if float(w_tilde.mean().item()) <= 1e-12:
        w_tilde = w_norm
    else:
        w_tilde = w_tilde / w_tilde.mean().clamp_min(1e-12)

    positive_key_rows = k * w_tilde.sqrt().unsqueeze(1)
    positive_value_rows = r_perp * w_tilde.sqrt().unsqueeze(1)
    base_key_rows, _base_key_ratios = _fast_mixed_signal_risk_candidate_basis(
        positive_key_rows,
        generic_key_rows,
        None,
        dim=d_ff,
        rank=rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    base_value_rows, _base_value_ratios = _fast_mixed_signal_risk_candidate_basis(
        positive_value_rows,
        generic_value_rows,
        None,
        dim=d_model,
        rank=rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    residual_candidate_key_rows = (
        base_value_rows @ residual_update if base_value_rows.numel() > 0 else torch.empty(0, d_ff)
    )
    residual_candidate_value_rows = (
        base_key_rows @ residual_update.T if base_key_rows.numel() > 0 else torch.empty(0, d_model)
    )
    key_basis_rows, key_ratios = _fast_mixed_signal_risk_candidate_basis(
        positive_key_rows,
        generic_key_rows,
        residual_candidate_key_rows,
        dim=d_ff,
        rank=rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    value_basis_rows, value_ratios = _fast_mixed_signal_risk_candidate_basis(
        positive_value_rows,
        generic_value_rows,
        residual_candidate_value_rows,
        dim=d_model,
        rank=rank,
        eps=eps,
        ratio_cap=risk_ratio_cap,
    )
    if key_basis_rows.numel() == 0 or value_basis_rows.numel() == 0:
        diagnostics["qrico_fallback"] = 1.0
        return QricoPurificationResult(
            update=residual_update.contiguous(),
            residual_update=residual_update.contiguous(),
            projected_update=projected_update.contiguous(),
            key_basis=key_basis_rows.contiguous(),
            value_basis=value_basis_rows.contiguous(),
            coeff=torch.empty(0, 0, dtype=torch.float32),
            diagnostics=diagnostics,
        )

    p_basis = key_basis_rows.T.contiguous()  # [m,rk]
    q_basis = value_basis_rows.T.contiguous()  # [d,rv]
    kp = k @ p_basis
    rq = r_perp @ q_basis

    usable_top_k = max(2, min(int(option_sketch_rank), top_idx.shape[1]))
    selected_tokens = tok.clamp(min=0, max=top_idx.shape[0] - 1)
    selected_top = top_idx[selected_tokens, :usable_top_k]
    selected_vals = top_vals[selected_tokens, :usable_top_k]
    selected_rows = _lookup_weight_rows(selected_top, stored_indices=lm_idx, stored_rows=lm_rows)
    probs = torch.softmax(selected_vals, dim=1)
    expected_rows = (probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
    contrast_rows = selected_rows - expected_rows  # [n,c,d]
    # Use the original relational context-value target as the local option
    # reference. ``r_perp`` is the candidate residual map's own output; using it
    # as the reference would let collateral output movement define itself as
    # target-parallel. The solve still fits the quotient residual map, but the
    # value-side risk asks whether that residual moves options in the direction
    # the relational target wanted locally.
    target_option = torch.einsum("ncd,nd->nc", contrast_rows, y)
    value_option = torch.einsum("ncd,dr->ncr", contrast_rows, q_basis)
    target_rank = max(1, min(int(target_parallel_rank), k.shape[0], usable_top_k))
    target_norms = torch.linalg.vector_norm(target_option, dim=1)
    if target_rank == 1:
        target_unit = target_option / target_norms.clamp_min(float(eps)).unsqueeze(1)
        target_unit = torch.where(target_norms.unsqueeze(1) > float(eps), target_unit, torch.zeros_like(target_unit))
        parallel = torch.einsum("ncr,nc->nr", value_option, target_unit)
        total = torch.einsum("ncr,ncs,n->rs", value_option, value_option, w_tilde)
        f_targ = torch.einsum("nr,ns,n->rs", parallel, parallel, w_tilde)
        norm = w_tilde.sum().clamp_min(1e-12)
        f_targ = f_targ / norm
        f_scr = total / norm - f_targ
        f_scr = 0.5 * (f_scr + f_scr.T)
    else:
        f_scr = torch.zeros(q_basis.shape[1], q_basis.shape[1], dtype=torch.float32)
        f_targ = torch.zeros_like(f_scr)
        eye_opt = torch.eye(usable_top_k, dtype=torch.float32)
        if k.shape[0] > 1:
            target_unit = F.normalize(target_option, dim=1)
            target_sim = torch.nan_to_num(target_unit @ target_unit.T, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            target_sim = torch.eye(k.shape[0], dtype=torch.float32)
        for row_idx in range(k.shape[0]):
            v_i = value_option[row_idx]
            nn_count = min(target_rank, k.shape[0])
            near = torch.topk(target_sim[row_idx], k=nn_count, largest=True).indices
            z_rows = target_option[near]
            z_basis = _orthonormal_basis([z_rows], usable_top_k).T
            if z_basis.numel() > 0:
                p_parallel = z_basis @ z_basis.T
            else:
                p_parallel = torch.zeros(usable_top_k, usable_top_k, dtype=torch.float32)
            p_off = eye_opt - p_parallel
            row_weight = w_tilde[row_idx].clamp_min(0.0)
            f_targ = f_targ + row_weight * (v_i.T @ p_parallel @ v_i)
            f_scr = f_scr + row_weight * (v_i.T @ p_off @ v_i)
        norm = w_tilde.sum().clamp_min(1e-12)
        f_targ = f_targ / norm
        f_scr = f_scr / norm
    if output_basis is not None and output_basis.numel() > 0:
        out = output_basis.detach().float().cpu()
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 2 and out.shape[1] == d_model:
            oq = cap_rows(out, 512) @ q_basis
            f_scr = f_scr + (float(output_weight) / max(float(output_weight), 1.0)) * (oq.T @ oq) / max(oq.shape[0], 1)

    if solve_mode == "sylvester":
        left = kp.T @ (kp * w_tilde.unsqueeze(1))
        if negative_keys is not None and negative_keys.numel() > 0:
            neg = negative_keys.detach().float().cpu()
            if neg.ndim == 2 and neg.shape[1] == d_ff:
                negp = cap_rows(neg, 1024) @ p_basis
                left = left + float(negative_weight) * (negp.T @ negp) / max(negp.shape[0], 1)
        left = left + float(cca_ridge) * torch.eye(left.shape[0], dtype=torch.float32)
        right = float(scramble_weight) * f_scr + float(cca_ridge) * torch.eye(f_scr.shape[0], dtype=torch.float32)
        rhs = kp.T @ (rq * w_tilde.unsqueeze(1))
        coeff, denom_min = _solve_sylvester_psd(left, right, rhs, eps=eps)
        purified = q_basis @ coeff.T @ p_basis.T
        key_sig = kp.T @ (kp * w_tilde.unsqueeze(1))
    elif solve_mode == "residual_filter":
        right = torch.eye(f_scr.shape[0], dtype=torch.float32) + float(scramble_weight) * f_scr
        right = right + float(cca_ridge) * torch.eye(f_scr.shape[0], dtype=torch.float32)
        direct_coeff = m_perp @ q_basis
        shrunk_coeff = direct_coeff @ _solve_symmetric_psd(right, torch.eye(right.shape[0], dtype=torch.float32))
        filtered_m = m_perp - direct_coeff @ q_basis.T + shrunk_coeff @ q_basis.T
        purified = filtered_m.T.contiguous()
        coeff = p_basis.T @ purified.T @ q_basis
        denom_min = float(torch.linalg.eigvalsh(0.5 * (right + right.T)).min().item())
        key_sig = kp.T @ (kp * w_tilde.unsqueeze(1))
    else:
        raise ValueError(f"Unknown Q-RICO solve_mode {solve_mode!r}")

    fit = k @ purified.T
    residual_energy = (w_tilde.unsqueeze(1) * r_perp.square()).sum().clamp_min(float(eps))
    fit_energy = (w_tilde.unsqueeze(1) * fit.square()).sum().clamp_min(float(eps))
    capture = (fit_energy / residual_energy).clamp_min(0.0)
    key_sig = kp.T @ (kp * w_tilde.unsqueeze(1))
    scramble_after = torch.trace(coeff.T @ key_sig @ coeff @ f_scr).clamp_min(0.0)
    target_after = torch.trace(coeff.T @ key_sig @ coeff @ f_targ).clamp_min(float(eps))
    scramble_quotient = scramble_after / target_after.clamp_min(float(eps))
    if layer_evidence_target > layer_evidence_min:
        evidence = ((capture - float(layer_evidence_min)) / (float(layer_evidence_target) - float(layer_evidence_min))).clamp(0.0, 1.0)
    else:
        evidence = torch.ones((), dtype=torch.float32)
    scramble_scale = torch.sqrt(torch.tensor(1.0, dtype=torch.float32) / (scramble_quotient + float(eps))).clamp(max=1.0)
    trust = torch.sqrt(evidence) * scramble_scale
    if not apply_layer_trust:
        trust = torch.ones_like(trust)
    purified = purified * trust

    before_norm = torch.linalg.vector_norm(update_f).clamp_min(1e-12)
    after_norm = torch.linalg.vector_norm(purified).clamp_min(1e-12)
    if float(after_norm.item()) > float(before_norm.item()):
        purified = purified * (before_norm / after_norm)
        after_norm = before_norm

    diagnostics.update(
        {
            "qrico_deflate_key_rank": float(deflate_key_rows.shape[0]),
            "qrico_deflate_value_rank": float(deflate_value_rows.shape[0]),
            "qrico_key_rank": float(key_basis_rows.shape[0]),
            "qrico_value_rank": float(value_basis_rows.shape[0]),
            "qrico_option_top_k": float(usable_top_k),
            "qrico_target_parallel_rank": float(target_rank),
            "qrico_projected_update_fro": float(torch.linalg.vector_norm(projected_update).item()),
            "qrico_residual_update_fro": float(torch.linalg.vector_norm(residual_update).item()),
            "qrico_residual_fit_fro": float(torch.linalg.vector_norm(r_perp).item()),
            "qrico_update_fro_after": float(after_norm.item()),
            "qrico_residual_row_weight_mean": float(w_tilde.mean().item()),
            "qrico_residual_row_weight_max": float(w_tilde.max().item()),
            "qrico_residual_ratio_mean": float(residual_ratio.mean().item()),
            "qrico_residual_ratio_max": float(residual_ratio.max().item()),
            "qrico_key_ratio_mean": float(key_ratios.mean().item()),
            "qrico_key_ratio_max": float(key_ratios.max().item()),
            "qrico_value_ratio_mean": float(value_ratios.mean().item()),
            "qrico_value_ratio_max": float(value_ratios.max().item()),
            "qrico_scramble_metric_trace": float(torch.trace(f_scr).item()),
            "qrico_target_parallel_metric_trace": float(torch.trace(f_targ).item()),
            "qrico_capture_ratio": float(capture.item()),
            "qrico_scramble_after": float(scramble_after.item()),
            "qrico_target_parallel_after": float(target_after.item()),
            "qrico_scramble_quotient": float(scramble_quotient.item()),
            "qrico_layer_trust": float(trust.item()),
            "qrico_apply_layer_trust": 1.0 if apply_layer_trust else 0.0,
            "qrico_sylvester_den_min": float(denom_min),
            "qrico_scramble_weight": float(scramble_weight),
            "qrico_residual_row_weight_power": float(residual_row_weight_power),
            "qrico_quotient_mode_code": 0.0 if quotient_mode == "joint" else 1.0,
            "qrico_solve_mode_code": 0.0 if solve_mode == "sylvester" else 1.0,
        }
    )
    return QricoPurificationResult(
        update=purified.contiguous(),
        residual_update=residual_update.contiguous(),
        projected_update=projected_update.contiguous(),
        key_basis=key_basis_rows.contiguous(),
        value_basis=value_basis_rows.contiguous(),
        coeff=coeff.contiguous(),
        diagnostics=diagnostics,
    )


def seal_qrico_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    up_weight: torch.Tensor,
    current_down_weight: torch.Tensor,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    deflate_key_rank: int = 16,
    deflate_value_rank: int = 16,
    rank: int = 64,
    option_sketch_rank: int = 256,
    target_parallel_rank: int = 4,
    scramble_weight: float = 0.35,
    residual_row_weight_power: float = 0.5,
    quotient_mode: str = "joint",
    solve_mode: str = "sylvester",
    low_surprise_quantile: float = 0.35,
    negative_weight: float = 20.0,
    output_weight: float = 10.0,
    cca_ridge: float = 1e-3,
    layer_evidence_min: float = 0.03,
    layer_evidence_target: float = 0.20,
    apply_layer_trust: bool = True,
    salience_tau: float = 1.0,
    eta_erase: float = 2.0,
    eta_seal: float = 0.05,
    max_seal_scale: float = 1.10,
    eps: float = 1e-6,
    risk_ratio_cap: float = 100.0,
) -> SealQricoResult:
    """Q-RICO plus signed anti-erasure and gauge-seal scale proposal."""

    qrico = qrico_purify_update(
        update,
        keys=keys,
        targets=targets,
        weights=weights,
        all_keys=all_keys,
        all_outputs=all_outputs,
        token_indices=token_indices,
        logit_top_values=logit_top_values,
        logit_top_indices=logit_top_indices,
        lm_head_indices=lm_head_indices,
        lm_head_rows=lm_head_rows,
        layer=layer,
        negative_keys=negative_keys,
        output_basis=output_basis,
        deflate_key_rank=deflate_key_rank,
        deflate_value_rank=deflate_value_rank,
        rank=rank,
        option_sketch_rank=option_sketch_rank,
        target_parallel_rank=target_parallel_rank,
        scramble_weight=scramble_weight,
        residual_row_weight_power=residual_row_weight_power,
        quotient_mode=quotient_mode,
        solve_mode=solve_mode,
        low_surprise_quantile=low_surprise_quantile,
        negative_weight=negative_weight,
        output_weight=output_weight,
        cca_ridge=cca_ridge,
        layer_evidence_min=layer_evidence_min,
        layer_evidence_target=layer_evidence_target,
        apply_layer_trust=apply_layer_trust,
        eps=eps,
        risk_ratio_cap=risk_ratio_cap,
    )
    salience = mlp_gauge_salience(
        up_weight,
        current_down_weight,
        output_basis,
        tau=salience_tau,
        eps=eps,
    )
    update_m = qrico.update.detach().float().cpu().T.contiguous()
    purified_m, anti_diag = signed_anti_erase_update(
        update_m,
        current_down_weight,
        salience,
        output_basis,
        eta_erase=eta_erase,
        eps=eps,
        return_diagnostics=True,
    )
    seal_scales = compute_gauge_seal_scales(
        keys,
        purified_m,
        weights,
        output_basis,
        eta_seal=eta_seal,
        max_scale=max_seal_scale,
        eps=eps,
    )
    before_norm = torch.linalg.vector_norm(qrico.update.detach().float().cpu()).clamp_min(1e-12)
    after_norm = torch.linalg.vector_norm(purified_m).clamp_min(1e-12)
    diagnostics = dict(qrico.diagnostics)
    diagnostics.update(anti_diag)
    diagnostics.update(
        {
            "seal_qrico_enabled": 1.0,
            "seal_update_fro_before": float(before_norm.item()),
            "seal_update_fro_after": float(after_norm.item()),
            "seal_update_retention": float((after_norm / before_norm).item()),
            "seal_scale_mean": float(seal_scales.mean().item()) if seal_scales.numel() else 1.0,
            "seal_scale_max": float(seal_scales.max().item()) if seal_scales.numel() else 1.0,
            "seal_scaled_channels": float((seal_scales > 1.000001).sum().item()) if seal_scales.numel() else 0.0,
            "seal_eta_erase": float(eta_erase),
            "seal_eta_seal": float(eta_seal),
            "seal_salience_tau": float(salience_tau),
        }
    )
    return SealQricoResult(
        update=purified_m.T.contiguous(),
        seal_scales=seal_scales.contiguous(),
        salience=salience.contiguous(),
        diagnostics=diagnostics,
    )


def sharp_karp_purify_update(
    update: torch.Tensor,
    *,
    keys: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    all_keys: torch.Tensor,
    all_outputs: torch.Tensor,
    token_indices: torch.Tensor,
    logit_top_values: torch.Tensor,
    logit_top_indices: torch.Tensor,
    lm_head_indices: torch.Tensor,
    lm_head_rows: torch.Tensor,
    layer: nn.Module | None = None,
    negative_keys: torch.Tensor | None = None,
    output_basis: torch.Tensor | None = None,
    key_rank: int = 48,
    value_rank: int = 48,
    low_surprise_quantile: float = 0.25,
    confidence_quantile: float = 0.60,
    max_anchors: int = 128,
    signal_top_k: int = 8,
    eta_sharp: float = 0.5,
    shadow_weight: float = 2.0,
    karp_eta_cross: float = 0.0,
    karp_eta_key: float = 0.0,
    karp_eta_value: float = 0.0,
    karp_kappa: float = 0.1,
    ridge: float = 1e-4,
    negative_weight: float = 0.0,
    output_weight: float = 0.0,
    shadow_temperature: float = 0.05,
    solve_mode: str = "ridge",
    signal_eps: float = 1e-6,
    risk_ratio_cap: float = 100.0,
) -> SharpKarpPurificationResult:
    """Signed Shadow-Anchor Readout Purification around a candidate update.

    SHARP-KARP fits the relational target in a compact key x value coefficient
    space, while penalizing atoms that the candidate itself predicts will lower
    high-confidence, low-surprise same-pass margins. Unlike a generic Fisher or
    LM-head penalty, the shadow penalty is signed and key-conditioned.
    """

    update_f = update.detach().float().cpu()
    k = keys.detach().float().cpu()
    y = targets.detach().float().cpu()
    w = weights.detach().float().cpu().clamp_min(0.0)
    all_k = all_keys.detach().float().cpu()
    all_y = all_outputs.detach().float().cpu()
    tok = token_indices.detach().cpu().long()
    top_vals = torch.nan_to_num(logit_top_values.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = logit_top_indices.detach().cpu().long()
    lm_idx = lm_head_indices.detach().cpu().long()
    lm_rows = lm_head_rows.detach().float().cpu()
    if update_f.ndim != 2:
        raise ValueError(f"update must be [d,m], got {tuple(update_f.shape)}")
    d_model, d_ff = update_f.shape
    if k.ndim != 2 or k.shape[1] != d_ff or y.ndim != 2 or y.shape[1] != d_model or k.shape[0] != y.shape[0]:
        raise ValueError("SHARP-KARP keys/targets do not match update shape")
    if w.ndim != 1 or w.shape[0] != k.shape[0]:
        raise ValueError(f"SHARP-KARP weights must be [{k.shape[0]}], got {tuple(w.shape)}")
    if all_k.ndim != 2 or all_k.shape[1] != d_ff or all_y.ndim != 2 or all_y.shape[1] != d_model:
        raise ValueError("SHARP-KARP all_keys/all_outputs do not match update shape")
    if top_vals.ndim != 2 or top_idx.ndim != 2 or top_vals.shape != top_idx.shape:
        raise ValueError("logit_top_values/logit_top_indices must be matching [T,k]")
    if top_vals.shape[0] < all_k.shape[0]:
        raise ValueError("logit top-k rows must cover all lesson keys")
    if tok.numel() != k.shape[0]:
        raise ValueError("token_indices must align with selected rows")

    row_scale = w.sqrt().unsqueeze(1)
    signal_key_rows = k * row_scale
    signal_value_rows = y * row_scale
    row_scores = _token_row_surprise(all_k, layer)
    threshold = torch.quantile(row_scores, float(low_surprise_quantile))
    low_mask = row_scores <= threshold
    if int(low_mask.sum().item()) < 2:
        low_mask = row_scores <= row_scores.median()
    generic_key_parts = [all_k[low_mask]]
    if negative_keys is not None and negative_keys.numel() > 0:
        neg = negative_keys.detach().float().cpu()
        if neg.ndim == 2 and neg.shape[1] == d_ff:
            generic_key_parts.append(neg)
    generic_key_rows = torch.cat([part for part in generic_key_parts if part.numel() > 0], dim=0)
    generic_value_parts = [all_y[low_mask]]
    if output_basis is not None and output_basis.numel() > 0:
        basis = output_basis.detach().float().cpu()
        if basis.ndim == 1:
            basis = basis.unsqueeze(0)
        if basis.ndim == 2 and basis.shape[1] == d_model:
            generic_value_parts.append(basis)
    generic_value_rows = torch.cat([part for part in generic_value_parts if part.numel() > 0], dim=0)

    key_basis_rows, key_ratios = _mixed_signal_risk_basis(
        signal_key_rows,
        generic_key_rows,
        dim=d_ff,
        rank=key_rank,
        eps=signal_eps,
        ratio_cap=risk_ratio_cap,
    )
    value_basis_rows, value_ratios = _mixed_signal_risk_basis(
        signal_value_rows,
        generic_value_rows,
        dim=d_model,
        rank=value_rank,
        eps=signal_eps,
        ratio_cap=risk_ratio_cap,
    )
    diagnostics: dict[str, float] = {
        "sharp_enabled": 1.0,
        "sharp_key_rank": float(key_basis_rows.shape[0]),
        "sharp_value_rank": float(value_basis_rows.shape[0]),
        "sharp_low_rows": float(low_mask.sum().item()),
        "sharp_update_fro_before": float(torch.linalg.vector_norm(update_f).item()),
    }
    if key_basis_rows.numel() == 0 or value_basis_rows.numel() == 0:
        diagnostics["sharp_fallback"] = 1.0
        return SharpKarpPurificationResult(update=update_f.contiguous(), diagnostics=diagnostics)

    # P and Q are column bases. Coefficient M maps key-basis coordinates to
    # value-basis coordinates through delta_W = Q M^T P^T.
    p_basis = key_basis_rows.T.contiguous()  # [m,a]
    q_basis = value_basis_rows.T.contiguous()  # [d,b]
    a_rank = p_basis.shape[1]
    b_rank = q_basis.shape[1]
    x = k @ p_basis  # [n,a]
    yq = y @ q_basis  # [n,b]
    m0 = torch.nan_to_num(p_basis.T @ update_f.T @ q_basis, nan=0.0, posinf=0.0, neginf=0.0)  # [a,b]

    # Signal denominator: target-aligned local logit movement at selected rows.
    usable_top_k = max(2, min(int(signal_top_k), top_idx.shape[1]))
    selected_top = top_idx[tok.clamp(max=top_idx.shape[0] - 1), :usable_top_k]
    selected_vals = top_vals[tok.clamp(max=top_vals.shape[0] - 1), :usable_top_k]
    selected_rows = _lookup_weight_rows(selected_top, stored_indices=lm_idx, stored_rows=lm_rows)
    probs = torch.softmax(selected_vals, dim=1)
    expected_rows = (probs.unsqueeze(-1) * selected_rows).sum(dim=1, keepdim=True)
    contrast_rows = selected_rows - expected_rows
    target_effect = torch.einsum("nkd,nd->nk", contrast_rows, y)
    value_effect = torch.einsum("nkd,db->nkb", contrast_rows, q_basis)
    denom = target_effect.square().sum(dim=1, keepdim=True).clamp_min(float(signal_eps))
    h = torch.einsum("nkb,nk->nb", value_effect, target_effect) / denom
    signal_terms = torch.relu(m0.unsqueeze(0) * x.unsqueeze(2) * h.unsqueeze(1))
    signal = torch.nan_to_num((w.view(-1, 1, 1) * signal_terms.square()).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)

    # Shadow anchors: boring, confident same-pass states outside the selected
    # relational key span. Their own top-vs-runner margin should not drop.
    margins = (top_vals[:, 0] - top_vals[:, 1]).contiguous()
    margin_threshold = torch.quantile(margins, float(confidence_quantile))
    selected_mask = torch.zeros(all_k.shape[0], dtype=torch.bool)
    selected_mask[tok.clamp(min=0, max=all_k.shape[0] - 1)] = True
    key_basis_for_overlap = _basis_with_mean(k, min(16, max(1, k.shape[0])), d_ff)
    if key_basis_for_overlap.numel() > 0:
        key_norm = torch.linalg.vector_norm(all_k, dim=1).square().clamp_min(1e-12)
        overlap = (all_k @ key_basis_for_overlap.T).square().sum(dim=1) / key_norm
        overlap_mask = overlap <= torch.quantile(overlap, 0.50)
    else:
        overlap = torch.zeros(all_k.shape[0])
        overlap_mask = torch.ones(all_k.shape[0], dtype=torch.bool)
    min_anchor_rows = max(1, min(int(max_anchors), max(8, int(max_anchors) // 4))) if max_anchors > 0 else 1
    anchor_fallback_level = 0
    anchor_mask = low_mask & (margins >= margin_threshold) & overlap_mask & (~selected_mask)
    anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).flatten()
    if anchor_idx.numel() < min_anchor_rows:
        # Do not let the overlap guard starve SHARP. A thin anchor set was too
        # weak to model collateral margin erosion, so fall back to all
        # low-surprise high-confidence rows before relaxing confidence.
        anchor_fallback_level = 1
        anchor_mask = low_mask & (margins >= margin_threshold) & (~selected_mask)
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).flatten()
    if anchor_idx.numel() < min_anchor_rows:
        anchor_fallback_level = 2
        anchor_mask = low_mask & (~selected_mask)
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).flatten()
    if anchor_idx.numel() < min_anchor_rows:
        anchor_fallback_level = 3
        anchor_mask = ~selected_mask
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).flatten()
    if anchor_idx.numel() > int(max_anchors) > 0:
        low_score = (threshold - row_scores[anchor_idx]).clamp_min(0.0)
        confidence_score = (margins[anchor_idx] - margin_threshold).clamp_min(0.0)
        overlap_penalty = overlap[anchor_idx] if overlap.numel() == all_k.shape[0] else 0.0
        anchor_priority = low_score + confidence_score - 0.1 * overlap_penalty
        order = torch.argsort(anchor_priority, descending=True)[: int(max_anchors)]
        anchor_idx = anchor_idx[order]

    if anchor_idx.numel() > 0:
        anchor_pair_idx = top_idx[anchor_idx, :2]
        anchor_rows = _lookup_weight_rows(anchor_pair_idx, stored_indices=lm_idx, stored_rows=lm_rows)
        margin_grad = anchor_rows[:, 0, :] - anchor_rows[:, 1, :]  # [g,d]
        anchor_x = all_k[anchor_idx] @ p_basis  # [g,a]
        anchor_v = margin_grad @ q_basis  # [g,b]
        delta0 = torch.einsum("ga,ab,gb->g", anchor_x, m0, anchor_v)
        temp = max(float(shadow_temperature), 1e-6)
        confidence = (margins[anchor_idx] - margin_threshold).clamp_min(0.0)
        confidence = confidence / confidence.mean().clamp_min(1e-6)
        low_surprise = (threshold - row_scores[anchor_idx]).clamp_min(0.0)
        low_surprise = low_surprise / low_surprise.mean().clamp_min(1e-6)
        alpha = confidence * low_surprise * F.softplus(-delta0 / temp)
        shadow_terms = torch.relu(-(m0.unsqueeze(0) * anchor_x.unsqueeze(2) * anchor_v.unsqueeze(1)))
        shadow_atom = torch.nan_to_num(
            (alpha.view(-1, 1, 1) * shadow_terms.square()).sum(dim=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        anchor_x = torch.empty(0, a_rank)
        anchor_v = torch.empty(0, b_rank)
        alpha = torch.empty(0)
        delta0 = torch.empty(0)
        shadow_atom = torch.zeros(a_rank, b_rank)

    key_ratio = key_ratios[:a_rank].clamp_min(0.0)
    value_ratio = value_ratios[:b_rank].clamp_min(0.0)
    karp_atom = (
        float(karp_eta_cross) * key_ratio.unsqueeze(1) * value_ratio.unsqueeze(0)
        + float(karp_eta_key) * key_ratio.unsqueeze(1)
        + float(karp_eta_value) * value_ratio.unsqueeze(0)
    )
    quotient = (shadow_atom + float(karp_kappa) * karp_atom) / signal.clamp_min(float(signal_eps))
    atom_diag = torch.nan_to_num(
        float(eta_sharp) * quotient.clamp_min(0.0).clamp_max(float(risk_ratio_cap)),
        nan=float(risk_ratio_cap),
        posinf=float(risk_ratio_cap),
        neginf=0.0,
    )

    eye_b = torch.eye(b_rank, dtype=torch.float32)
    eye_a = torch.eye(a_rank, dtype=torch.float32)
    if solve_mode == "shrink":
        theta = torch.nan_to_num(m0 / (1.0 + atom_diag), nan=0.0, posinf=0.0, neginf=0.0)
        if anchor_x.numel() > 0 and shadow_weight > 0:
            # One closed-form diagonalized shadow step. This keeps SHARP as a
            # strict purifier of the existing candidate instead of a refit that
            # can invent new high-signal/high-risk atoms.
            anchor_energy = (
                alpha.view(-1, 1, 1)
                * anchor_x.square().unsqueeze(2)
                * anchor_v.square().unsqueeze(1)
            ).sum(dim=0)
            theta = torch.nan_to_num(theta / (1.0 + float(shadow_weight) * anchor_energy), nan=0.0, posinf=0.0, neginf=0.0)
    elif solve_mode == "ridge":
        # Coefficient-space ridge solve.
        xtwx = x.T @ (x * w.unsqueeze(1))
        rhs = x.T @ (yq * w.unsqueeze(1))
        hessian = torch.kron(xtwx, eye_b)
        if negative_keys is not None and negative_keys.numel() > 0 and negative_weight > 0:
            neg = negative_keys.detach().float().cpu()
            if neg.ndim == 2 and neg.shape[1] == d_ff:
                neg_p = neg @ p_basis
                hessian = hessian + float(negative_weight) * torch.kron(neg_p.T @ neg_p, eye_b)
        if output_basis is not None and output_basis.numel() > 0 and output_weight > 0:
            out = output_basis.detach().float().cpu()
            if out.ndim == 1:
                out = out.unsqueeze(0)
            if out.ndim == 2 and out.shape[1] == d_model:
                out_q = out @ q_basis
                hessian = hessian + float(output_weight) * torch.kron(eye_a, out_q.T @ out_q)
        if anchor_x.numel() > 0 and shadow_weight > 0:
            rows = (anchor_x.unsqueeze(2) * anchor_v.unsqueeze(1)).reshape(anchor_x.shape[0], -1)
            row_scale = (float(shadow_weight) * alpha.clamp_min(0.0)).sqrt().unsqueeze(1)
            rows = rows * row_scale
            hessian = hessian + rows.T @ rows
        hessian = hessian + torch.diag(atom_diag.reshape(-1))
        hessian = hessian + float(ridge) * torch.eye(a_rank * b_rank, dtype=torch.float32)
        theta = torch.nan_to_num(
            _solve_symmetric_psd(hessian, rhs.reshape(-1, 1)).reshape(a_rank, b_rank),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        raise ValueError(f"Unknown SHARP solve_mode {solve_mode!r}")
    purified = torch.nan_to_num(q_basis @ theta.T @ p_basis.T, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep the SHARP solve within the original candidate's Frobenius budget.
    before_norm = torch.linalg.vector_norm(update_f).clamp_min(1e-12)
    after_norm = torch.linalg.vector_norm(purified).clamp_min(1e-12)
    if float(after_norm.item()) > float(before_norm.item()):
        purified = purified * (before_norm / after_norm)
        after_norm = before_norm

    pred_before = x @ m0
    pred_after = x @ theta
    fit_rmse_before = torch.sqrt(torch.mean((pred_before - yq).square()))
    fit_rmse_after = torch.sqrt(torch.mean((pred_after - yq).square()))
    signal_before = torch.relu((m0.unsqueeze(0) * x.unsqueeze(2) * h.unsqueeze(1))).square()
    signal_after = torch.relu((theta.unsqueeze(0) * x.unsqueeze(2) * h.unsqueeze(1))).square()
    signal_before_sum = (w.view(-1, 1, 1) * signal_before).sum().clamp_min(1e-12)
    signal_after_sum = (w.view(-1, 1, 1) * signal_after).sum().clamp_min(1e-12)
    if anchor_x.numel() > 0:
        anchor_delta_before = torch.einsum("ga,ab,gb->g", anchor_x, m0, anchor_v)
        anchor_delta_after = torch.einsum("ga,ab,gb->g", anchor_x, theta, anchor_v)
        shadow_drop_before = torch.relu(-anchor_delta_before).mean()
        shadow_drop_after = torch.relu(-anchor_delta_after).mean()
    else:
        shadow_drop_before = torch.tensor(0.0)
        shadow_drop_after = torch.tensor(0.0)

    diagnostics.update(
        {
            "sharp_anchor_rows": float(anchor_idx.numel()),
            "sharp_anchor_fallback_level": float(anchor_fallback_level),
            "sharp_signal_top_k": float(usable_top_k),
            "sharp_fit_rmse_before": float(fit_rmse_before.item()),
            "sharp_fit_rmse_after": float(fit_rmse_after.item()),
            "sharp_update_fro_after": float(after_norm.item()),
            "sharp_removed_update_ratio": float(
                torch.linalg.vector_norm(update_f - purified).item() / max(float(before_norm.item()), 1e-12)
            ),
            "sharp_signal_retention": float((signal_after_sum / signal_before_sum).item()),
            "sharp_shadow_drop_before": float(shadow_drop_before.item()),
            "sharp_shadow_drop_after": float(shadow_drop_after.item()),
            "sharp_shadow_drop_ratio": float(
                (shadow_drop_after / shadow_drop_before.clamp_min(1e-12)).item()
                if float(shadow_drop_before.item()) > 0
                else 0.0
            ),
            "sharp_atom_diag_mean": float(atom_diag.mean().item()),
            "sharp_atom_diag_max": float(atom_diag.max().item()),
            "sharp_signal_atom_mean": float(signal.mean().item()),
            "sharp_shadow_atom_mean": float(shadow_atom.mean().item()),
            "sharp_eta": float(eta_sharp),
            "sharp_shadow_weight": float(shadow_weight),
            "sharp_karp_kappa": float(karp_kappa),
            "sharp_solve_mode": 0.0 if solve_mode == "ridge" else 1.0,
            "sharp_anchor_margin_mean": float(margins[anchor_idx].mean().item()) if anchor_idx.numel() else 0.0,
            "sharp_anchor_candidate_drop_mean": float(torch.relu(-delta0).mean().item()) if delta0.numel() else 0.0,
        }
    )
    return SharpKarpPurificationResult(update=purified.contiguous(), diagnostics=diagnostics)


def _future_transport_capsules(
    *,
    layer_idx: int,
    token_indices: torch.Tensor,
    relation_keys: torch.Tensor,
    current_keys: torch.Tensor,
    current_states: torch.Tensor,
    future_states_by_layer: dict[int, torch.Tensor],
    row_scores: torch.Tensor,
    layer_horizon: int,
    token_top_k: int,
    layer_decay: float,
    token_decay: float,
    relation_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-pass future integrated state targets for STAR.

    Each selected relation key is assigned the weighted residual movement from
    its token/layer state into nearby future layers and high-surprise later
    tokens. This is a single-forward self-distillation target, not a teacher or
    null-prompt contrast.
    """

    tokens = current_states.shape[0]
    dim = current_states.shape[1]
    if relation_keys.shape[0] != token_indices.shape[0]:
        raise ValueError("relation_keys and token_indices must have the same row count")
    if tokens == 0:
        return torch.empty(0, dim), torch.empty(0)
    states = current_states.detach().float().cpu()
    keys_f = current_keys.detach().float().cpu()
    rel = relation_keys.detach().float().cpu()
    scores = row_scores.detach().float().cpu().clamp_min(0.0)
    score_scale = scores / scores.mean().clamp_min(1e-12)
    future_token_count = max(1, min(int(token_top_k), max(tokens - 1, 1)))
    candidate_tokens = torch.topk(score_scale, k=future_token_count, dim=0).indices.sort().values
    future_layers = [
        idx
        for idx in sorted(future_states_by_layer)
        if idx > layer_idx and idx <= layer_idx + max(1, int(layer_horizon))
    ]

    target_rows: list[torch.Tensor] = []
    coherence_rows: list[torch.Tensor] = []
    for row_idx, token_tensor in enumerate(token_indices.detach().long().cpu()):
        token_idx = int(token_tensor.item())
        token_idx = max(0, min(token_idx, tokens - 1))
        base = states[token_idx]
        components: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []

        def add_component(state: torch.Tensor, layer_distance: int, token_distance: int, token_score: torch.Tensor):
            delta = state - base
            if float(torch.linalg.vector_norm(delta).item()) <= 1e-12:
                return
            rel_cos = torch.tensor(1.0)
            if token_distance > 0 and keys_f.numel() > 0:
                key_u = keys_f[max(0, min(token_idx + token_distance, tokens - 1))]
                rel_cos = torch.dot(rel[row_idx], key_u) / (
                    torch.linalg.vector_norm(rel[row_idx]).clamp_min(1e-12)
                    * torch.linalg.vector_norm(key_u).clamp_min(1e-12)
                )
                rel_cos = rel_cos.clamp_min(0.0).pow(float(relation_power))
            layer_weight = torch.exp(torch.tensor(-float(layer_distance) / max(float(layer_decay), 1e-6)))
            token_weight = torch.exp(torch.tensor(-float(token_distance) / max(float(token_decay), 1e-6)))
            weight = layer_weight * token_weight * torch.log1p(token_score.clamp_min(0.0)) * rel_cos.clamp_min(1e-6)
            if float(weight.item()) <= 1e-12:
                return
            components.append(delta)
            weights.append(weight)

        for future_layer in future_layers:
            future = future_states_by_layer[future_layer].detach().float().cpu()
            if future.shape == states.shape:
                add_component(
                    future[token_idx],
                    future_layer - layer_idx,
                    0,
                    score_scale[token_idx],
                )
        for future_token in candidate_tokens.tolist():
            if future_token <= token_idx:
                continue
            add_component(
                states[future_token],
                0,
                future_token - token_idx,
                score_scale[future_token],
            )
            for future_layer in future_layers:
                future = future_states_by_layer[future_layer].detach().float().cpu()
                if future.shape == states.shape:
                    add_component(
                        future[future_token],
                        future_layer - layer_idx,
                        future_token - token_idx,
                        score_scale[future_token],
                    )

        if not components:
            target_rows.append(torch.zeros(dim, dtype=torch.float32))
            coherence_rows.append(torch.tensor(0.0))
            continue
        comp = torch.stack(components, dim=0)
        w = torch.stack(weights, dim=0).clamp_min(1e-12)
        target = (comp * w.unsqueeze(1)).sum(dim=0) / w.sum().clamp_min(1e-12)
        comp_norm = F.normalize(comp, dim=1)
        target_norm = F.normalize(target.unsqueeze(0), dim=1).squeeze(0)
        coherence = (comp_norm @ target_norm).clamp_min(0.0).mean()
        target_rows.append(target)
        coherence_rows.append(coherence)
    return torch.stack(target_rows, dim=0).contiguous(), torch.stack(coherence_rows, dim=0).contiguous()


def _project_rows_to_value_manifold(
    rows: torch.Tensor,
    down_weight: torch.Tensor,
    feature_scores: torch.Tensor,
    *,
    features: int,
    ridge: float,
) -> torch.Tensor:
    if features <= 0 or rows.numel() == 0:
        return rows
    down = down_weight.detach().float().cpu()
    rows_f = rows.detach().float().cpu()
    feature_mass = feature_scores.detach().float().cpu().clamp_min(0.0)
    if feature_mass.ndim == 2:
        feature_mass = feature_mass.mean(dim=0)
    keep = max(1, min(int(features), down.shape[1], feature_mass.numel()))
    selected = torch.topk(feature_mass, k=keep, dim=0).indices
    values = down[:, selected]
    gram = values.T @ values + float(ridge) * torch.eye(keep, dtype=values.dtype)
    coeff = (rows_f @ values) @ torch.linalg.pinv(gram, hermitian=True)
    return (coeff @ values.T).contiguous()


def _subtract_nuisance_explained_posture(
    rows: torch.Tensor,
    nuisance: torch.Tensor,
    weights: torch.Tensor,
    output_basis: torch.Tensor | None,
    *,
    ridge: float,
) -> torch.Tensor:
    if rows.numel() == 0 or output_basis is None or output_basis.numel() == 0 or nuisance.numel() == 0:
        return rows
    rows_f = rows.detach().float().cpu()
    basis = output_basis.detach().float().cpu()
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    if basis.ndim != 2 or basis.shape[1] != rows_f.shape[1]:
        return rows_f
    q, _r = torch.linalg.qr(basis.T, mode="reduced")
    basis = q.T.contiguous()
    posture_coeff = rows_f @ basis.T
    explained = posture_coeff - _weighted_residualize_rows(
        posture_coeff,
        nuisance,
        weights,
        ridge=ridge,
    )
    return (rows_f - explained @ basis).contiguous()


def select_intrinsic_schur_transport_actuator_write(
    mlp_inputs: torch.Tensor,
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    layer_idx: int,
    future_mlp_inputs_by_layer: dict[int, torch.Tensor],
    feature_top_k: int = 128,
    relation_rank: int = 16,
    beta: float = 3.0,
    edge_top_k: int = 0,
    edge_attention_scale: float = 0.5,
    sinkhorn_steps: int = 0,
    target_scale: float = 1.0,
    object_summary_gain: float = 0.5,
    future_layer_horizon: int = 4,
    future_token_top_k: int = 8,
    future_layer_decay: float = 2.0,
    future_token_decay: float = 64.0,
    future_relation_power: float = 1.0,
    ordinary_key_rank: int = 32,
    value_projection_features: int = 128,
    value_projection_ridge: float = 1e-2,
    schur_ridge: float = 1e-3,
    map_ridge: float = 1e-3,
    posture_negative_scale: float = 1.0,
    min_coherence: float = 0.0,
    shuffle_future_targets: bool = False,
    shuffle_keys: bool = False,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    attention_probs: torch.Tensor | None = None,
    surprise_weight_mode: str = "exponential",
    surprise_weight_temperature: float = 2.0,
    surprise_weight_cap: float = 20.0,
) -> IntrinsicSurpriseSelection:
    """STAR: Schur-Transport Actuator Residual rows.

    CORI supplies purified dense object/relation keys. STAR replaces CORI's
    local relation-value target with the component of same-pass future hidden
    computation attributable to those keys after Schur-residualizing position,
    key magnitude, ordinary low-surprise key directions, and readout/posture
    projections. The posture component of the same selected keys is returned as
    zero-target negative keys for the closed-form write.
    """

    inputs_f = mlp_inputs.detach().float().cpu()
    keys_f = keys.detach().float().cpu()
    down = down_weight.detach().float().cpu()
    if inputs_f.ndim != 2 or keys_f.ndim != 2 or inputs_f.shape[0] != keys_f.shape[0]:
        raise ValueError(f"Expected mlp_inputs [T,d] and keys [T,m], got {tuple(inputs_f.shape)} {tuple(keys_f.shape)}")
    if down.ndim != 2 or down.shape[1] != keys_f.shape[1] or down.shape[0] != inputs_f.shape[1]:
        raise ValueError(f"down_weight must be [{inputs_f.shape[1]}, {keys_f.shape[1]}], got {tuple(down.shape)}")

    cori = select_intrinsic_conditional_relation_innovation_write(
        inputs_f,
        keys_f,
        layer,
        down,
        feature_top_k=feature_top_k,
        relation_rank=relation_rank,
        beta=beta,
        edge_top_k=edge_top_k,
        edge_attention_scale=edge_attention_scale,
        sinkhorn_steps=sinkhorn_steps,
        target_mode="svd_value",
        target_scale=1.0,
        persistence_power=persistence_power,
        persistence_threshold_fraction=persistence_threshold_fraction,
        persistence_min_tokens=persistence_min_tokens,
        feature_weights=feature_weights,
        target_projection_basis=target_projection_basis,
        attention_probs=attention_probs,
        surprise_weight_mode=surprise_weight_mode,
        surprise_weight_temperature=surprise_weight_temperature,
        surprise_weight_cap=surprise_weight_cap,
    )
    if cori.keys.shape[0] == 0:
        raise ValueError("STAR received no CORI relation keys")

    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)

    base_weights = cori.weights.detach().float().cpu().clamp_min(1e-12)
    object_summary = (base_weights.unsqueeze(1) * cori.keys).sum(dim=0)
    object_summary = object_summary / torch.linalg.vector_norm(object_summary).clamp_min(1e-12)
    augmented_keys = cori.keys + float(object_summary_gain) * base_weights.sqrt().unsqueeze(1) * object_summary.unsqueeze(0)

    future_targets, coherence = _future_transport_capsules(
        layer_idx=layer_idx,
        token_indices=cori.token_indices,
        relation_keys=cori.keys,
        current_keys=keys_f,
        current_states=inputs_f,
        future_states_by_layer=future_mlp_inputs_by_layer,
        row_scores=row_scores,
        layer_horizon=future_layer_horizon,
        token_top_k=future_token_top_k,
        layer_decay=future_layer_decay,
        token_decay=future_token_decay,
        relation_power=future_relation_power,
    )
    keep_mask = coherence >= float(min_coherence)
    if not torch.any(keep_mask):
        raise ValueError("STAR future transport coherence rejected every row")
    augmented_keys = augmented_keys[keep_mask]
    future_targets = future_targets[keep_mask]
    base_weights = base_weights[keep_mask] * coherence[keep_mask].clamp_min(1e-6)
    token_indices = cori.token_indices[keep_mask]
    row_score_values = cori.row_scores[keep_mask]
    selected_feature_scores = cori.feature_scores[keep_mask]
    target_keys = cori.target_keys[keep_mask]
    feature_indices = cori.feature_indices[keep_mask] if cori.feature_indices is not None else None

    if augmented_keys.shape[0] > 1 and shuffle_keys:
        augmented_keys = torch.roll(augmented_keys, shifts=1, dims=0)
    if future_targets.shape[0] > 1 and shuffle_future_targets:
        future_targets = torch.roll(future_targets, shifts=1, dims=0)

    pos = token_indices.detach().float().cpu() / max(float(keys_f.shape[0] - 1), 1.0)
    nuisance_parts = [
        torch.ones(pos.shape[0], 1, dtype=torch.float32),
        pos.unsqueeze(1),
        pos.square().unsqueeze(1),
        pos.pow(3).unsqueeze(1),
        torch.linalg.vector_norm(augmented_keys, dim=1, keepdim=True),
        torch.linalg.vector_norm(keys_f[token_indices], dim=1, keepdim=True),
    ]
    ordinary_basis = _safe_row_pcs_from_scores(keys_f, row_scores, ordinary_key_rank, low=True)
    if ordinary_basis.numel() > 0:
        nuisance_parts.append(augmented_keys @ ordinary_basis.T)
    output_basis = target_projection_basis.detach().float().cpu() if target_projection_basis is not None else None
    if output_basis is not None and output_basis.numel() > 0:
        if output_basis.ndim == 1:
            output_basis = output_basis.unsqueeze(0)
        if output_basis.shape[1] == down.shape[0]:
            readout_key_basis = output_basis @ down
            if readout_key_basis.numel() > 0:
                readout_key_basis = _orthonormal_basis([readout_key_basis], down.shape[1])
                nuisance_parts.append(augmented_keys @ readout_key_basis.T)
    nuisance = torch.cat(nuisance_parts, dim=1)

    semantic_keys = _weighted_residualize_rows(augmented_keys, nuisance, base_weights, ridge=schur_ridge)
    posture_keys = augmented_keys - semantic_keys
    semantic_future = _weighted_residualize_rows(future_targets, nuisance, base_weights, ridge=schur_ridge)
    if float(torch.linalg.vector_norm(semantic_keys).item()) <= 1e-12:
        raise ValueError("STAR Schur semantic keys vanished")
    if float(torch.linalg.vector_norm(semantic_future).item()) <= 1e-12:
        raise ValueError("STAR Schur future targets vanished")

    row_scale = base_weights.clamp_min(0.0).sqrt().unsqueeze(1)
    k_weighted = semantic_keys * row_scale
    y_weighted = semantic_future * row_scale
    kernel = k_weighted @ k_weighted.T + float(map_ridge) * torch.eye(k_weighted.shape[0], dtype=k_weighted.dtype)
    alpha = torch.linalg.pinv(kernel, hermitian=True) @ y_weighted
    transport_map = k_weighted.T @ alpha
    targets = semantic_keys @ transport_map
    targets = _project_rows_to_value_manifold(
        targets,
        down,
        feature_scores,
        features=value_projection_features,
        ridge=value_projection_ridge,
    )
    targets = _subtract_nuisance_explained_posture(
        targets,
        nuisance,
        base_weights,
        output_basis,
        ridge=schur_ridge,
    )
    targets = float(target_scale) * targets
    keep_nonzero = torch.linalg.vector_norm(targets, dim=1) > 1e-12
    if not torch.any(keep_nonzero):
        raise ValueError("STAR targets vanished after value projection")
    semantic_keys = semantic_keys[keep_nonzero]
    posture_keys = posture_keys[keep_nonzero]
    targets = targets[keep_nonzero]
    base_weights = base_weights[keep_nonzero]
    token_indices = token_indices[keep_nonzero]
    row_score_values = row_score_values[keep_nonzero]
    selected_feature_scores = selected_feature_scores[keep_nonzero]
    target_keys = target_keys[keep_nonzero]
    if feature_indices is not None:
        feature_indices = feature_indices[keep_nonzero]

    weights = shape_surprise_weights(
        base_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    negative_keys = posture_keys * float(posture_negative_scale) if posture_negative_scale > 0 else None
    explained = torch.linalg.vector_norm(targets).square() / torch.linalg.vector_norm(semantic_future).square().clamp_min(1e-12)
    diagnostics = {
        "star_rows_pre_filter": float(cori.keys.shape[0]),
        "star_rows": float(semantic_keys.shape[0]),
        "star_future_coherence_mean": float(coherence.mean().item()),
        "star_future_coherence_kept_mean": float(base_weights.mean().item()),
        "star_explained_ratio": float(explained.item()),
        "star_shuffle_future_targets": float(bool(shuffle_future_targets)),
        "star_shuffle_keys": float(bool(shuffle_keys)),
        "star_semantic_key_fro": float(torch.linalg.vector_norm(semantic_keys).item()),
        "star_posture_key_fro": float(torch.linalg.vector_norm(posture_keys).item()),
        "star_target_fro": float(torch.linalg.vector_norm(targets).item()),
    }
    return IntrinsicSurpriseSelection(
        keys=semantic_keys.contiguous(),
        targets=targets.contiguous(),
        weights=weights.contiguous(),
        token_indices=token_indices.contiguous(),
        row_scores=row_score_values.contiguous(),
        feature_scores=selected_feature_scores.contiguous(),
        target_keys=target_keys.contiguous(),
        feature_indices=feature_indices.contiguous() if feature_indices is not None else None,
        negative_keys=negative_keys.contiguous() if negative_keys is not None else None,
        diagnostics=diagnostics,
    )


def select_intrinsic_surprise_write(
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    target_feature_top_k: int = 32,
    target_scale: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    surprise_weight_mode: str = "linear",
    surprise_weight_temperature: float = 1.0,
    surprise_weight_cap: float = 100.0,
) -> IntrinsicSurpriseSelection:
    """Build one-pass Hebbian write rows from intrinsically surprising features.

    The solve keys and targets are both derived from the same lesson forward:

    - token eligibility is the mean of the top weight-relative feature scores;
    - target keys keep only each selected token's most surprising features;
    - targets are the current down-projection output contributed by those
      surprising features.
    """

    keys_f = keys.detach().float().cpu()
    if keys_f.ndim != 2 or keys_f.shape[0] == 0:
        raise ValueError(f"keys must be non-empty [tokens, features], got {tuple(keys_f.shape)}")
    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    feature_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=feature_k, dim=1).values.mean(dim=1)

    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=feature_k,
    )

    selected_keys = keys_f[token_indices]
    selected_feature_scores = feature_scores[token_indices]
    target_k = max(1, min(int(target_feature_top_k), selected_keys.shape[1]))
    feature_indices = torch.topk(selected_feature_scores, k=target_k, dim=1).indices
    mask = torch.zeros_like(selected_keys)
    mask.scatter_(1, feature_indices, 1.0)
    target_keys = selected_keys * mask
    targets = float(target_scale) * (target_keys @ down_weight.detach().float().cpu().T)
    if target_projection_basis is not None and target_projection_basis.numel() > 0:
        targets = project_rows_away_from_basis(targets, target_projection_basis)

    selected_scores = row_scores[token_indices]
    normed_weights = shape_surprise_weights(
        selected_scores,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=target_keys.contiguous(),
        targets=targets.contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=token_indices.contiguous(),
        row_scores=selected_scores.contiguous(),
        feature_scores=selected_feature_scores.contiguous(),
        target_keys=target_keys.contiguous(),
        feature_indices=None,
    )


def select_intrinsic_associative_binding_write(
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    key_feature_top_k: int = 8,
    value_feature_top_k: int = 32,
    target_scale: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    surprise_weight_mode: str = "linear",
    surprise_weight_temperature: float = 1.0,
    surprise_weight_cap: float = 100.0,
) -> IntrinsicSurpriseSelection:
    """Closed-form associative binding from one forward pass.

    If several weight-surprising MLP features coactivate at a lesson position,
    this creates sparse key rows for each surprising feature and targets the
    down-projection value of the other coactive surprising features. Future
    activation of one feature can then inject the bound context of the others,
    without an external object router or a second prompt pass.
    """

    keys_f = keys.detach().float().cpu()
    if keys_f.ndim != 2 or keys_f.shape[0] == 0:
        raise ValueError(f"keys must be non-empty [tokens, features], got {tuple(keys_f.shape)}")
    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)

    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )

    examples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    example_tokens: list[torch.Tensor] = []
    example_features: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_target_keys: list[torch.Tensor] = []
    down = down_weight.detach().float().cpu()

    key_k = max(1, min(int(key_feature_top_k), keys_f.shape[1]))
    value_k = max(1, min(int(value_feature_top_k), keys_f.shape[1]))
    for token_idx in token_indices.tolist():
        token_key = keys_f[token_idx]
        token_scores = feature_scores[token_idx]
        key_features = torch.topk(token_scores, k=key_k, dim=0).indices
        value_features = torch.topk(token_scores, k=value_k, dim=0).indices
        value_mask = torch.zeros_like(token_key)
        value_mask[value_features] = 1.0
        base_value_key = token_key * value_mask
        for feature_idx in key_features.tolist():
            activation = token_key[feature_idx]
            if float(activation.abs().item()) <= 1e-12:
                continue
            sparse_key = torch.zeros_like(token_key)
            sparse_key[feature_idx] = activation
            context_key = base_value_key.clone()
            context_key[feature_idx] = 0.0
            if float(torch.linalg.vector_norm(context_key).item()) <= 1e-12:
                continue
            examples.append(sparse_key)
            target = float(target_scale) * (context_key @ down.T)
            if target_projection_basis is not None and target_projection_basis.numel() > 0:
                target = project_rows_away_from_basis(target.unsqueeze(0), target_projection_basis).squeeze(0)
            targets.append(target)
            weights.append(token_scores[feature_idx].clamp_min(1e-12))
            example_tokens.append(torch.tensor(token_idx, dtype=torch.long))
            example_features.append(torch.tensor(feature_idx, dtype=torch.long))
            selected_scores.append(row_scores[token_idx])
            selected_feature_scores.append(token_scores)
            selected_target_keys.append(context_key)

    if not examples:
        raise ValueError("No nonzero associative binding examples were selected")
    example_keys = torch.stack(examples, dim=0)
    target_rows = torch.stack(targets, dim=0)
    raw_weights = torch.stack(weights, dim=0)
    normed_weights = shape_surprise_weights(
        raw_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=example_keys.contiguous(),
        targets=target_rows.contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=torch.stack(example_tokens, dim=0).contiguous(),
        row_scores=torch.stack(selected_scores, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(selected_target_keys, dim=0).contiguous(),
        feature_indices=torch.stack(example_features, dim=0).contiguous(),
    )


def _prediction_history(keys: torch.Tensor, token_idx: int, min_rows: int) -> torch.Tensor:
    before = keys[:token_idx]
    if before.shape[0] >= min_rows:
        return before
    if keys.shape[0] <= 1:
        return keys
    mask = torch.ones(keys.shape[0], dtype=torch.bool)
    mask[token_idx] = False
    return keys[mask]


def select_intrinsic_predictive_residual_write(
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    key_feature_top_k: int = 8,
    value_feature_top_k: int = 32,
    target_scale: float = 1.0,
    prediction_ridge: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    surprise_weight_mode: str = "linear",
    surprise_weight_temperature: float = 1.0,
    surprise_weight_cap: float = 100.0,
) -> IntrinsicSurpriseSelection:
    """Write context causes into locally unexplained feature residuals.

    This is a cheap predictive-coding analogue over native MLP channels. For a
    selected token, stable/high-score channels are treated as causes. Other
    active candidate channels are predicted from those causes using a tiny
    within-lesson ridge predictor. The write target is the down-projection value
    of the prediction residual, not the raw coactive feature value.
    """

    keys_f = keys.detach().float().cpu()
    if keys_f.ndim != 2 or keys_f.shape[0] == 0:
        raise ValueError(f"keys must be non-empty [tokens, features], got {tuple(keys_f.shape)}")
    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )

    examples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    example_tokens: list[torch.Tensor] = []
    example_features: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_target_keys: list[torch.Tensor] = []
    down = down_weight.detach().float().cpu()

    candidate_k = max(2, min(int(feature_top_k), keys_f.shape[1]))
    cause_k = max(1, min(int(key_feature_top_k), candidate_k - 1))
    residual_k = max(1, min(int(value_feature_top_k), candidate_k - cause_k))
    for token_idx in token_indices.tolist():
        token_key = keys_f[token_idx]
        token_scores = feature_scores[token_idx]
        candidates = torch.topk(token_scores, k=candidate_k, dim=0).indices
        cause_features = candidates[:cause_k]
        residual_candidates = candidates[cause_k:]
        if residual_candidates.numel() == 0:
            continue

        history = _prediction_history(keys_f, token_idx, min_rows=cause_k + 2)
        x_c_hist = history[:, cause_features]
        x_e_hist = history[:, residual_candidates]
        ones = torch.ones(x_c_hist.shape[0], 1, dtype=x_c_hist.dtype)
        x_aug = torch.cat([x_c_hist, ones], dim=1)
        ridge_diag = torch.eye(x_aug.shape[1], dtype=x_aug.dtype)
        ridge_diag[-1, -1] = 0.0
        system = x_aug.T @ x_aug + float(prediction_ridge) * ridge_diag
        coeff = _solve_symmetric_psd(system, x_aug.T @ x_e_hist)
        token_aug = torch.cat([token_key[cause_features], torch.ones(1, dtype=token_key.dtype)])
        predicted = token_aug @ coeff
        residual = token_key[residual_candidates] - predicted
        variance = x_e_hist.var(dim=0, unbiased=False).clamp_min(1e-6)
        residual_scores = residual.square() / variance
        residual_scores = residual_scores * token_scores[residual_candidates].clamp_min(1e-12).sqrt()
        keep_residual = torch.topk(
            residual_scores,
            k=min(residual_k, residual_scores.shape[0]),
            dim=0,
        ).indices
        residual_features = residual_candidates[keep_residual]
        residual_values = residual[keep_residual]
        if float(torch.linalg.vector_norm(residual_values).item()) <= 1e-12:
            continue

        residual_key = torch.zeros_like(token_key)
        residual_key[residual_features] = residual_values
        target = float(target_scale) * (residual_key @ down.T)
        if target_projection_basis is not None and target_projection_basis.numel() > 0:
            target = project_rows_away_from_basis(target.unsqueeze(0), target_projection_basis).squeeze(0)
        residual_strength = residual_scores[keep_residual].mean().clamp_min(1e-12)
        for feature_idx in cause_features.tolist():
            activation = token_key[feature_idx]
            if float(activation.abs().item()) <= 1e-12:
                continue
            sparse_key = torch.zeros_like(token_key)
            sparse_key[feature_idx] = activation
            examples.append(sparse_key)
            targets.append(target)
            weights.append((token_scores[feature_idx].clamp_min(1e-12).sqrt() * residual_strength).clamp_min(1e-12))
            example_tokens.append(torch.tensor(token_idx, dtype=torch.long))
            example_features.append(torch.tensor(feature_idx, dtype=torch.long))
            selected_scores.append(residual_strength)
            selected_feature_scores.append(token_scores)
            selected_target_keys.append(residual_key)

    if not examples:
        raise ValueError("No nonzero predictive residual examples were selected")
    example_keys = torch.stack(examples, dim=0)
    target_rows = torch.stack(targets, dim=0)
    raw_weights = torch.stack(weights, dim=0)
    normed_weights = shape_surprise_weights(
        raw_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=example_keys.contiguous(),
        targets=target_rows.contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=torch.stack(example_tokens, dim=0).contiguous(),
        row_scores=torch.stack(selected_scores, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(selected_target_keys, dim=0).contiguous(),
        feature_indices=torch.stack(example_features, dim=0).contiguous(),
    )


def select_intrinsic_relational_residual_write(
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    key_feature_top_k: int = 8,
    value_feature_top_k: int = 32,
    pair_top_k: int = 16,
    bidirectional_pairs: bool = False,
    relation_value_mode: str = "residual",
    target_scale: float = 1.0,
    prediction_ridge: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    surprise_weight_mode: str = "linear",
    surprise_weight_temperature: float = 1.0,
    surprise_weight_cap: float = 100.0,
) -> IntrinsicSurpriseSelection:
    """Write native-channel bindings whose coactivation is locally unexpected.

    This keeps the single-forward/no-SAE constraint, but moves the surprise
    coordinate from individual MLP channels to relations between channels. For
    candidate feature pairs at a selected lesson token, it predicts the pair
    product from the pair's marginal activations over the lesson history:

        a_i a_j ~= f(a_i, a_j)

    A pair is eligible only when the current product has more unexplained
    magnitude than that local predictor expects. The write then uses the
    surprising pair to choose sparse associative rows: activation of one member
    retrieves the other's down-projection value.
    """

    keys_f = keys.detach().float().cpu()
    if keys_f.ndim != 2 or keys_f.shape[0] == 0:
        raise ValueError(f"keys must be non-empty [tokens, features], got {tuple(keys_f.shape)}")
    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )

    examples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    example_tokens: list[torch.Tensor] = []
    example_features: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_target_keys: list[torch.Tensor] = []
    down = down_weight.detach().float().cpu()
    if relation_value_mode not in {"residual", "full", "context"}:
        raise ValueError(f"Unknown relation_value_mode {relation_value_mode!r}")

    scale = mlp_weight_prior_scale(layer, keys_f.shape[1]).to(keys_f.device).clamp_min(1e-12)
    z_keys = keys_f / scale.unsqueeze(0)
    candidate_k = max(2, min(int(feature_top_k), keys_f.shape[1]))
    cause_k = max(1, min(int(key_feature_top_k), candidate_k - 1))
    value_k = max(1, min(int(value_feature_top_k), candidate_k - 1))
    max_pairs = max(1, int(pair_top_k))

    for token_idx in token_indices.tolist():
        token_key = keys_f[token_idx]
        token_z = z_keys[token_idx]
        token_scores = feature_scores[token_idx]
        candidates = torch.topk(token_scores, k=candidate_k, dim=0).indices
        cause_features = candidates[:cause_k]
        value_features = candidates[: max(cause_k + 1, min(candidate_k, value_k + cause_k))]

        history = _prediction_history(z_keys, token_idx, min_rows=4)
        pair_rows: list[tuple[float, int, int, float]] = []
        for cause_idx in cause_features.tolist():
            for value_idx in value_features.tolist():
                if value_idx == cause_idx:
                    continue
                current_product = token_z[cause_idx] * token_z[value_idx]
                if float(current_product.abs().item()) <= 1e-12:
                    continue
                hist_i = history[:, cause_idx]
                hist_j = history[:, value_idx]
                product_hist = hist_i * hist_j
                x_aug = torch.stack(
                    [
                        hist_i,
                        hist_j,
                        torch.ones_like(hist_i),
                    ],
                    dim=1,
                )
                ridge_diag = torch.eye(x_aug.shape[1], dtype=x_aug.dtype)
                ridge_diag[-1, -1] = 0.0
                system = x_aug.T @ x_aug + float(prediction_ridge) * ridge_diag
                coeff = _solve_symmetric_psd(system, (x_aug.T @ product_hist).unsqueeze(1)).squeeze(1)
                token_aug = torch.tensor(
                    [token_z[cause_idx], token_z[value_idx], 1.0],
                    dtype=x_aug.dtype,
                )
                predicted = token_aug @ coeff
                residual = current_product - predicted
                excess = (current_product.abs() - predicted.abs()).clamp_min(0.0)
                if float(excess.item()) <= 1e-12:
                    continue
                fit_hist = x_aug @ coeff
                residual_var = (product_hist - fit_hist).var(unbiased=False).clamp_min(1e-6)
                feature_pair_score = torch.sqrt(
                    token_scores[cause_idx].clamp_min(1e-12)
                    * token_scores[value_idx].clamp_min(1e-12)
                )
                pair_score = (excess.square() / residual_var) * feature_pair_score
                if not torch.isfinite(pair_score):
                    continue
                gain = (excess / current_product.abs().clamp_min(1e-6)).clamp(max=2.0)
                pair_rows.append((float(pair_score.item()), cause_idx, value_idx, float(gain.item())))

        if not pair_rows:
            continue
        pair_rows.sort(key=lambda item: item[0], reverse=True)
        for pair_score_float, cause_idx, value_idx, gain_float in pair_rows[:max_pairs]:
            activation = token_key[cause_idx]
            value_gain = gain_float if relation_value_mode == "residual" else 1.0
            value_activation = token_key[value_idx] * value_gain
            if float(activation.abs().item()) <= 1e-12 or float(value_activation.abs().item()) <= 1e-12:
                continue

            rows = [(cause_idx, value_idx, activation, value_activation)]
            if bidirectional_pairs:
                reverse_activation = token_key[value_idx]
                reverse_value_activation = token_key[cause_idx] * value_gain
                if (
                    float(reverse_activation.abs().item()) > 1e-12
                    and float(reverse_value_activation.abs().item()) > 1e-12
                ):
                    rows.append((value_idx, cause_idx, reverse_activation, reverse_value_activation))
            for key_idx, target_idx, key_activation, target_activation in rows:
                sparse_key = torch.zeros_like(token_key)
                sparse_key[key_idx] = key_activation
                target_key = torch.zeros_like(token_key)
                target_key[target_idx] = target_activation
                target = float(target_scale) * (target_key @ down.T)
                if target_projection_basis is not None and target_projection_basis.numel() > 0:
                    target = project_rows_away_from_basis(target.unsqueeze(0), target_projection_basis).squeeze(0)
                examples.append(sparse_key)
                targets.append(target)
                weights.append(torch.tensor(pair_score_float, dtype=torch.float32).clamp_min(1e-12))
                example_tokens.append(torch.tensor(token_idx, dtype=torch.long))
                example_features.append(torch.tensor(key_idx, dtype=torch.long))
                selected_scores.append(torch.tensor(pair_score_float, dtype=torch.float32))
                selected_feature_scores.append(token_scores)
                selected_target_keys.append(target_key)

    if not examples:
        raise ValueError("No nonzero relational residual examples were selected")
    example_keys = torch.stack(examples, dim=0)
    target_rows = torch.stack(targets, dim=0)
    raw_weights = torch.stack(weights, dim=0)
    normed_weights = shape_surprise_weights(
        raw_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=example_keys.contiguous(),
        targets=target_rows.contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=torch.stack(example_tokens, dim=0).contiguous(),
        row_scores=torch.stack(selected_scores, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(selected_target_keys, dim=0).contiguous(),
        feature_indices=torch.stack(example_features, dim=0).contiguous(),
    )


def select_intrinsic_relational_aggregate_write(
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    scoring_keys: torch.Tensor | None = None,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    key_feature_top_k: int = 8,
    value_feature_top_k: int = 32,
    pair_top_k: int = 64,
    bidirectional_pairs: bool = False,
    relation_value_mode: str = "residual",
    target_scale: float = 1.0,
    prediction_ridge: float = 1.0,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    surprise_weight_mode: str = "linear",
    surprise_weight_temperature: float = 1.0,
    surprise_weight_cap: float = 100.0,
) -> IntrinsicSurpriseSelection:
    """Aggregate surprising native-channel relations into richer value rows.

    Pair-row relational writes are very safe but atomized: each sparse trigger
    retrieves one paired value channel. This variant keeps the same
    single-forward relational surprise test, then groups selected pairs by
    trigger feature so one sparse trigger retrieves a weighted mixture of all
    surprising paired value channels at that token.
    """

    keys_f = keys.detach().float().cpu()
    if keys_f.ndim != 2 or keys_f.shape[0] == 0:
        raise ValueError(f"keys must be non-empty [tokens, features], got {tuple(keys_f.shape)}")
    score_keys_f = keys_f
    if scoring_keys is not None:
        score_keys_f = scoring_keys.detach().float().cpu()
        if score_keys_f.shape != keys_f.shape:
            raise ValueError(
                f"scoring_keys must match keys shape {tuple(keys_f.shape)}, got {tuple(score_keys_f.shape)}"
            )
    feature_scores = intrinsic_feature_scores(score_keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )

    examples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    example_tokens: list[torch.Tensor] = []
    example_features: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_target_keys: list[torch.Tensor] = []
    down = down_weight.detach().float().cpu()
    if relation_value_mode not in {"residual", "full", "context"}:
        raise ValueError(f"Unknown relation_value_mode {relation_value_mode!r}")

    scale = mlp_weight_prior_scale(layer, keys_f.shape[1]).to(keys_f.device).clamp_min(1e-12)
    z_keys = score_keys_f / scale.unsqueeze(0)
    candidate_k = max(2, min(int(feature_top_k), keys_f.shape[1]))
    cause_k = max(1, min(int(key_feature_top_k), candidate_k - 1))
    value_k = max(1, min(int(value_feature_top_k), candidate_k - 1))
    max_pairs = max(1, int(pair_top_k))

    for token_idx in token_indices.tolist():
        token_key = keys_f[token_idx]
        token_z = z_keys[token_idx]
        token_scores = feature_scores[token_idx]
        candidates = torch.topk(token_scores, k=candidate_k, dim=0).indices
        cause_features = candidates[:cause_k]
        value_features = candidates[: max(cause_k + 1, min(candidate_k, value_k + cause_k))]

        history = _prediction_history(z_keys, token_idx, min_rows=4)
        pair_rows: list[tuple[float, int, int, float]] = []
        for cause_idx in cause_features.tolist():
            for value_idx in value_features.tolist():
                if value_idx == cause_idx:
                    continue
                current_product = token_z[cause_idx] * token_z[value_idx]
                if float(current_product.abs().item()) <= 1e-12:
                    continue
                hist_i = history[:, cause_idx]
                hist_j = history[:, value_idx]
                product_hist = hist_i * hist_j
                x_aug = torch.stack([hist_i, hist_j, torch.ones_like(hist_i)], dim=1)
                ridge_diag = torch.eye(x_aug.shape[1], dtype=x_aug.dtype)
                ridge_diag[-1, -1] = 0.0
                system = x_aug.T @ x_aug + float(prediction_ridge) * ridge_diag
                coeff = _solve_symmetric_psd(system, (x_aug.T @ product_hist).unsqueeze(1)).squeeze(1)
                token_aug = torch.tensor([token_z[cause_idx], token_z[value_idx], 1.0], dtype=x_aug.dtype)
                predicted = token_aug @ coeff
                excess = (current_product.abs() - predicted.abs()).clamp_min(0.0)
                if float(excess.item()) <= 1e-12:
                    continue
                fit_hist = x_aug @ coeff
                residual_var = (product_hist - fit_hist).var(unbiased=False).clamp_min(1e-6)
                feature_pair_score = torch.sqrt(
                    token_scores[cause_idx].clamp_min(1e-12)
                    * token_scores[value_idx].clamp_min(1e-12)
                )
                pair_score = (excess.square() / residual_var) * feature_pair_score
                if not torch.isfinite(pair_score):
                    continue
                gain = (excess / current_product.abs().clamp_min(1e-6)).clamp(max=2.0)
                pair_rows.append((float(pair_score.item()), cause_idx, value_idx, float(gain.item())))

        if not pair_rows:
            continue
        pair_rows.sort(key=lambda item: item[0], reverse=True)
        grouped: dict[int, list[tuple[float, int, float]]] = {}
        for pair_score_float, cause_idx, value_idx, gain_float in pair_rows[:max_pairs]:
            grouped.setdefault(cause_idx, []).append((pair_score_float, value_idx, gain_float))
            if bidirectional_pairs:
                grouped.setdefault(value_idx, []).append((pair_score_float, cause_idx, gain_float))

        for key_idx, values_for_key in grouped.items():
            key_activation = token_key[key_idx]
            if float(key_activation.abs().item()) <= 1e-12:
                continue
            score_values = torch.tensor([row[0] for row in values_for_key], dtype=torch.float32).clamp_min(1e-12)
            target_key = torch.zeros_like(token_key)
            if relation_value_mode == "context":
                context_features = value_features[value_features != key_idx]
                target_key[context_features] = token_key[context_features]
            else:
                norm_weights = torch.sqrt(score_values / score_values.mean().clamp_min(1e-12)).clamp(max=4.0)
                for norm_weight, (_pair_score, target_idx, gain_float) in zip(norm_weights, values_for_key, strict=False):
                    value_gain = gain_float if relation_value_mode == "residual" else 1.0
                    target_activation = token_key[target_idx] * value_gain * float(norm_weight.item())
                    target_key[target_idx] += target_activation
            if float(torch.linalg.vector_norm(target_key).item()) <= 1e-12:
                continue
            sparse_key = torch.zeros_like(token_key)
            sparse_key[key_idx] = key_activation
            target = float(target_scale) * (target_key @ down.T)
            if target_projection_basis is not None and target_projection_basis.numel() > 0:
                target = project_rows_away_from_basis(target.unsqueeze(0), target_projection_basis).squeeze(0)
            examples.append(sparse_key)
            targets.append(target)
            weights.append(score_values.mean().clamp_min(1e-12))
            example_tokens.append(torch.tensor(token_idx, dtype=torch.long))
            example_features.append(torch.tensor(key_idx, dtype=torch.long))
            selected_scores.append(score_values.mean().clamp_min(1e-12))
            selected_feature_scores.append(token_scores)
            selected_target_keys.append(target_key)

    if not examples:
        raise ValueError("No nonzero relational aggregate examples were selected")
    example_keys = torch.stack(examples, dim=0)
    target_rows = torch.stack(targets, dim=0)
    raw_weights = torch.stack(weights, dim=0)
    normed_weights = shape_surprise_weights(
        raw_weights,
        mode=surprise_weight_mode,
        temperature=surprise_weight_temperature,
        max_weight=surprise_weight_cap,
    )
    return IntrinsicSurpriseSelection(
        keys=example_keys.contiguous(),
        targets=target_rows.contiguous(),
        weights=normed_weights.contiguous(),
        token_indices=torch.stack(example_tokens, dim=0).contiguous(),
        row_scores=torch.stack(selected_scores, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(selected_target_keys, dim=0).contiguous(),
        feature_indices=torch.stack(example_features, dim=0).contiguous(),
    )


def select_intrinsic_feature_birth_update(
    mlp_inputs: torch.Tensor,
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    value_feature_top_k: int = 32,
    target_scale: float = 1.0,
    trigger_scale: float = 4.0,
    trigger_ridge: float = 1e-3,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    current_down_weight: torch.Tensor | None = None,
    avoid_neurons: torch.Tensor | None = None,
) -> IntrinsicFeatureBirthUpdate:
    """Create paired up/gate/down deltas from one lesson forward.

    Down-only writes attach new values to existing MLP features. That is broad:
    if the same feature appears in an unrelated prompt, the write fires there
    too. This update instead repurposes low-impact MLP neurons into conjunction
    detectors for the selected lesson states, then assigns their down columns to
    the surprising value mixture present at those same states.

    No external gate, labels, null pass, or prediction loss are used. The
    detector rows are the minimum-norm closed-form solution that fires on the
    selected lesson states.
    """

    inputs_f = mlp_inputs.detach().float().cpu()
    keys_f = keys.detach().float().cpu()
    if inputs_f.ndim != 2 or inputs_f.shape[0] == 0:
        raise ValueError(f"mlp_inputs must be non-empty [tokens, width], got {tuple(inputs_f.shape)}")
    if keys_f.ndim != 2 or keys_f.shape[0] != inputs_f.shape[0]:
        raise ValueError(
            f"keys must be [tokens, features] with same token count as mlp_inputs, "
            f"got inputs={tuple(inputs_f.shape)} keys={tuple(keys_f.shape)}"
        )
    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )
    selected_inputs = inputs_f[token_indices]
    selected_keys = keys_f[token_indices]
    selected_scores = feature_scores[token_indices]

    current_down = (
        current_down_weight.detach().float().cpu()
        if current_down_weight is not None
        else down_weight.detach().float().cpu()
    )
    neuron_indices = choose_low_impact_neurons(
        layer,
        current_down,
        keys_f,
        token_indices.numel(),
        avoid=avoid_neurons,
    )
    trigger_rows = _closed_form_trigger_rows(
        selected_inputs,
        trigger_scale=trigger_scale,
        ridge=trigger_ridge,
    )
    up_module = _mlp_up_module(layer)
    if up_module is None:
        raise AttributeError(f"Layer {layer.__class__.__name__} has no supported MLP up projection.")
    up_weight = _linear_weight(up_module)
    if up_weight is None or up_weight.shape[0] != keys_f.shape[1]:
        raise ValueError("Could not resolve an up-projection weight compatible with MLP keys.")
    gate_module = _mlp_gate_module(layer)
    gate_weight = _linear_weight(gate_module)
    old_up_rows = up_weight[neuron_indices].detach().float().cpu()
    up_row_delta = trigger_rows - old_up_rows
    if gate_weight is not None and gate_weight.shape[0] == keys_f.shape[1]:
        old_gate_rows = gate_weight[neuron_indices].detach().float().cpu()
        gate_row_delta = trigger_rows - old_gate_rows
        trigger_response = F.silu(torch.full((token_indices.numel(),), float(trigger_scale))) * float(trigger_scale)
    else:
        gate_row_delta = None
        trigger_response = torch.full((token_indices.numel(),), float(trigger_scale))

    value_k = max(1, min(int(value_feature_top_k), keys_f.shape[1]))
    value_features = torch.topk(selected_scores, k=value_k, dim=1).indices
    target_keys = torch.zeros_like(selected_keys)
    target_keys.scatter_(1, value_features, selected_keys.gather(1, value_features))
    for row_idx, neuron_idx in enumerate(neuron_indices.tolist()):
        target_keys[row_idx, neuron_idx] = 0.0

    down = down_weight.detach().float().cpu()
    targets = float(target_scale) * (target_keys @ down.T)
    if target_projection_basis is not None and target_projection_basis.numel() > 0:
        targets = project_rows_away_from_basis(targets, target_projection_basis)
    desired_cols = targets.T / trigger_response.clamp_min(1e-6).unsqueeze(0)
    old_cols = current_down[:, neuron_indices]
    down_col_delta = desired_cols - old_cols

    return IntrinsicFeatureBirthUpdate(
        neuron_indices=neuron_indices.contiguous(),
        token_indices=token_indices.contiguous(),
        up_row_delta=up_row_delta.contiguous(),
        gate_row_delta=gate_row_delta.contiguous() if gate_row_delta is not None else None,
        down_col_delta=down_col_delta.contiguous(),
        row_scores=row_scores[token_indices].contiguous(),
        feature_scores=selected_scores.contiguous(),
        target_keys=target_keys.contiguous(),
        targets=targets.contiguous(),
        trigger_rows=trigger_rows.contiguous(),
        trigger_response=trigger_response.contiguous(),
        feature_indices=value_features.contiguous(),
    )


def _top_feature_pairs(scores: torch.Tensor, features: torch.Tensor, pair_count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    feature_list = [int(idx) for idx in features.tolist()]
    for left_pos, left in enumerate(feature_list):
        for right in feature_list[left_pos + 1 :]:
            value = float((scores[left] * scores[right]).item())
            candidates.append((value, left, right))
    candidates.sort(reverse=True, key=lambda row: row[0])
    return [(left, right) for _value, left, right in candidates[: max(1, int(pair_count))]]


def select_intrinsic_conjunctive_feature_birth_update(
    mlp_inputs: torch.Tensor,
    keys: torch.Tensor,
    layer: nn.Module,
    down_weight: torch.Tensor,
    *,
    token_mode: str = "last",
    top_tokens: int = 16,
    feature_top_k: int = 32,
    key_feature_top_k: int = 8,
    value_feature_top_k: int = 32,
    pair_count: int = 4,
    target_scale: float = 1.0,
    min_response: float = 1e-4,
    persistence_power: float = 0.0,
    persistence_threshold_fraction: float = 0.25,
    persistence_min_tokens: int = 2,
    feature_weights: torch.Tensor | None = None,
    target_projection_basis: torch.Tensor | None = None,
    current_down_weight: torch.Tensor | None = None,
    avoid_neurons: torch.Tensor | None = None,
) -> IntrinsicFeatureBirthUpdate:
    """Create new SwiGLU conjunction features from surprising existing features.

    For gated MLPs, a hidden feature is already a product
    ``silu(gate_i(x)) * up_i(x)``. This rule creates a new low-impact neuron
    whose gate row is copied from one surprising feature and whose up row is
    copied from another. The resulting neuron fires on their co-activation,
    which is closer to a latent-object binding than either a full-state
    prototype or an individual-feature value write.
    """

    inputs_f = mlp_inputs.detach().float().cpu()
    keys_f = keys.detach().float().cpu()
    if inputs_f.ndim != 2 or inputs_f.shape[0] == 0:
        raise ValueError(f"mlp_inputs must be non-empty [tokens, width], got {tuple(inputs_f.shape)}")
    if keys_f.ndim != 2 or keys_f.shape[0] != inputs_f.shape[0]:
        raise ValueError(
            f"keys must be [tokens, features] with same token count as mlp_inputs, "
            f"got inputs={tuple(inputs_f.shape)} keys={tuple(keys_f.shape)}"
        )
    up_weight = _linear_weight(_mlp_up_module(layer))
    gate_weight = _linear_weight(_mlp_gate_module(layer))
    if up_weight is None or gate_weight is None:
        raise AttributeError("Conjunctive feature birth requires gated MLP up_proj and gate_proj weights.")
    if up_weight.shape != gate_weight.shape or up_weight.shape[0] != keys_f.shape[1]:
        raise ValueError("MLP up/gate weights are not compatible with captured down-input keys.")

    feature_scores = intrinsic_feature_scores(keys_f, layer)
    if persistence_power > 0.0:
        persistence = lesson_persistence_weights(
            feature_scores,
            threshold_fraction=persistence_threshold_fraction,
            min_tokens=persistence_min_tokens,
        )
        feature_scores = feature_scores * persistence.clamp_min(0.0).pow(float(persistence_power)).unsqueeze(0)
    if feature_weights is not None:
        feature_scores = feature_scores * feature_weights.detach().float().cpu().clamp_min(0.0).unsqueeze(0)
    score_k = max(1, min(int(feature_top_k), feature_scores.shape[1]))
    row_scores = torch.topk(feature_scores, k=score_k, dim=1).values.mean(dim=1)
    token_indices = _select_tokens_from_feature_scores(
        feature_scores,
        row_scores,
        token_mode=token_mode,
        top_tokens=top_tokens,
        feature_top_k=score_k,
    )

    planned: list[tuple[int, int, int, int, int, float]] = []
    key_k = max(2, min(int(key_feature_top_k), keys_f.shape[1]))
    for token_idx in token_indices.tolist():
        token_scores = feature_scores[token_idx]
        features = torch.topk(token_scores, k=key_k, dim=0).indices
        for left, right in _top_feature_pairs(token_scores, features, pair_count):
            x = inputs_f[token_idx]
            left_gate = torch.dot(gate_weight[left], x)
            left_up = torch.dot(up_weight[left], x)
            right_gate = torch.dot(gate_weight[right], x)
            right_up = torch.dot(up_weight[right], x)
            response_lr = float((F.silu(left_gate) * right_up).item())
            response_rl = float((F.silu(right_gate) * left_up).item())
            if abs(response_rl) > abs(response_lr):
                gate_feature, up_feature, response = right, left, response_rl
            else:
                gate_feature, up_feature, response = left, right, response_lr
            if abs(response) < float(min_response):
                continue
            planned.append((token_idx, gate_feature, up_feature, left, right, response))

    if not planned:
        raise ValueError("No nonzero conjunctive feature-birth pairs were selected")

    current_down = (
        current_down_weight.detach().float().cpu()
        if current_down_weight is not None
        else down_weight.detach().float().cpu()
    )
    neuron_indices = choose_low_impact_neurons(
        layer,
        current_down,
        keys_f,
        len(planned),
        avoid=avoid_neurons,
    )
    down = down_weight.detach().float().cpu()

    up_rows: list[torch.Tensor] = []
    gate_rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_keys: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    row_score_values: list[torch.Tensor] = []
    selected_feature_scores: list[torch.Tensor] = []
    selected_feature_pairs: list[torch.Tensor] = []
    selected_tokens: list[torch.Tensor] = []
    value_k = max(1, min(int(value_feature_top_k), keys_f.shape[1]))
    for token_idx, gate_feature, up_feature, left, right, response in planned:
        token_key = keys_f[token_idx]
        token_scores = feature_scores[token_idx]
        value_features = torch.topk(token_scores, k=value_k, dim=0).indices
        context_key = torch.zeros_like(token_key)
        context_key[value_features] = token_key[value_features]
        context_key[left] = 0.0
        context_key[right] = 0.0
        target = float(target_scale) * (context_key @ down.T)
        if target_projection_basis is not None and target_projection_basis.numel() > 0:
            target = project_rows_away_from_basis(target.unsqueeze(0), target_projection_basis).squeeze(0)
        up_rows.append(up_weight[up_feature])
        gate_rows.append(gate_weight[gate_feature])
        targets.append(target)
        target_keys.append(context_key)
        responses.append(torch.tensor(response, dtype=torch.float32))
        row_score_values.append(row_scores[token_idx])
        selected_feature_scores.append(token_scores)
        selected_feature_pairs.append(torch.tensor([left, right], dtype=torch.long))
        selected_tokens.append(torch.tensor(token_idx, dtype=torch.long))

    new_up_rows = torch.stack(up_rows, dim=0)
    new_gate_rows = torch.stack(gate_rows, dim=0)
    old_up_rows = up_weight[neuron_indices].detach().float().cpu()
    old_gate_rows = gate_weight[neuron_indices].detach().float().cpu()
    target_rows = torch.stack(targets, dim=0)
    trigger_response = torch.stack(responses, dim=0)
    desired_cols = target_rows.T / trigger_response.abs().clamp_min(1e-6).unsqueeze(0)
    desired_cols = desired_cols * trigger_response.sign().unsqueeze(0)
    old_cols = current_down[:, neuron_indices]

    return IntrinsicFeatureBirthUpdate(
        neuron_indices=neuron_indices.contiguous(),
        token_indices=torch.stack(selected_tokens, dim=0).contiguous(),
        up_row_delta=(new_up_rows - old_up_rows).contiguous(),
        gate_row_delta=(new_gate_rows - old_gate_rows).contiguous(),
        down_col_delta=(desired_cols - old_cols).contiguous(),
        row_scores=torch.stack(row_score_values, dim=0).contiguous(),
        feature_scores=torch.stack(selected_feature_scores, dim=0).contiguous(),
        target_keys=torch.stack(target_keys, dim=0).contiguous(),
        targets=target_rows.contiguous(),
        trigger_rows=new_up_rows.contiguous(),
        trigger_response=trigger_response.contiguous(),
        feature_indices=torch.stack(selected_feature_pairs, dim=0).contiguous(),
    )


@torch.no_grad()
def apply_intrinsic_feature_birth_update_(
    layer: nn.Module,
    down_wrapper: AdditiveMemoryLinear,
    update: IntrinsicFeatureBirthUpdate,
    *,
    eta: float = 1.0,
    max_down_update_norm: float | None = None,
) -> dict[str, float]:
    """Apply a sparse feature-birth update in-place."""

    up_module = _mlp_up_module(layer)
    if up_module is None or not hasattr(up_module, "weight"):
        raise AttributeError(f"Layer {layer.__class__.__name__} has no editable MLP up projection.")
    indices = update.neuron_indices.to(device=up_module.weight.device)
    up_delta = update.up_row_delta.to(device=up_module.weight.device, dtype=up_module.weight.dtype)
    up_module.weight[indices].add_(up_delta)

    gate_norm = 0.0
    gate_module = _mlp_gate_module(layer)
    if update.gate_row_delta is not None:
        if gate_module is None or not hasattr(gate_module, "weight"):
            raise AttributeError("Feature-birth update has gate deltas, but no editable gate projection was found.")
        gate_delta = update.gate_row_delta.to(device=gate_module.weight.device, dtype=gate_module.weight.dtype)
        gate_module.weight[indices.to(device=gate_module.weight.device)].add_(gate_delta)
        gate_norm = float(torch.linalg.vector_norm(update.gate_row_delta.float()).item())

    down_delta = torch.zeros_like(down_wrapper.memory)
    cols = update.down_col_delta.float()
    if max_down_update_norm is not None:
        norm = torch.linalg.vector_norm(cols)
        limit = float(max_down_update_norm)
        if limit > 0 and float(norm.item()) > limit:
            cols = cols * (limit / float(norm.item()))
    cols = float(eta) * cols
    down_delta[:, update.neuron_indices] = cols.to(device=down_delta.device, dtype=down_delta.dtype)
    down_wrapper.add_memory_(down_delta)
    return {
        "feature_birth_neurons": int(update.neuron_indices.numel()),
        "feature_birth_up_delta_fro": float(torch.linalg.vector_norm(update.up_row_delta.float()).item()),
        "feature_birth_gate_delta_fro": gate_norm,
        "feature_birth_down_delta_fro": float(torch.linalg.vector_norm(cols.float()).item()),
        "feature_birth_target_fro": float(torch.linalg.vector_norm(update.targets.float()).item()),
        "feature_birth_target_key_fro": float(torch.linalg.vector_norm(update.target_keys.float()).item()),
        "feature_birth_trigger_row_fro": float(torch.linalg.vector_norm(update.trigger_rows.float()).item()),
        "feature_birth_trigger_response_mean": float(update.trigger_response.float().mean().item()),
    }
