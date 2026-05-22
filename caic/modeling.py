"""Model loading, MLP memory wrappers, and activation capture."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F

from caic.contrastive_gate import DensityRatioGateParams, sequence_gate


class ForwardCounter:
    """Lightweight model-forward and token counter."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.calls = 0
        self.tokens = 0
        self._original_forward = model.forward

    def install(self) -> "ForwardCounter":
        def counted_forward(*args: Any, **kwargs: Any) -> Any:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            self.calls += 1
            if isinstance(input_ids, torch.Tensor):
                self.tokens += int(input_ids.numel())
            return self._original_forward(*args, **kwargs)

        self.model.forward = counted_forward  # type: ignore[method-assign]
        return self

    def snapshot(self) -> tuple[int, int]:
        return self.calls, self.tokens

    def delta_since(self, snapshot: tuple[int, int]) -> dict[str, int]:
        calls, tokens = snapshot
        return {
            "forward_calls": self.calls - calls,
            "forward_tokens": self.tokens - tokens,
        }

    def uninstall(self) -> None:
        self.model.forward = self._original_forward  # type: ignore[method-assign]


class AdditiveMemoryLinear(nn.Module):
    """A frozen linear layer plus an additive persistent memory matrix."""

    def __init__(self, base: nn.Linear, memory_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "memory",
            torch.zeros(
                base.out_features,
                base.in_features,
                device=base.weight.device,
                dtype=memory_dtype,
            ),
        )
        self.register_buffer(
            "gate_keys",
            torch.empty(0, base.in_features, device=base.weight.device, dtype=memory_dtype),
        )
        self.register_buffer(
            "object_gate_keys",
            torch.empty(0, base.in_features, device=base.weight.device, dtype=memory_dtype),
        )
        self.memory_scale = 1.0
        self.gate_threshold = 0.95
        self.gate_temperature = 80.0
        self.gate_last_token_only = False
        self.object_gate_threshold = 0.90
        self.object_gate_temperature = 40.0
        self.object_gate_floor = 0.0
        self.object_density_gates: list[DensityRatioGateParams] = []
        self.slot_memories: list[torch.Tensor] = []
        self.slot_gate_keys: list[torch.Tensor] = []
        self.slot_terms: list[tuple[str, ...]] = []
        self._active_slot_weights: torch.Tensor | None = None
        self._active_object_gate: torch.Tensor | None = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.memory_scale == 0.0:
            return out
        mem_input = x.to(self.memory.dtype)
        mem = F.linear(mem_input, self.memory)
        object_gate = self._active_or_internal_object_gate(mem_input)
        if object_gate is not None:
            mem = mem * object_gate
        if self.gate_keys.numel() > 0:
            mem = mem * self._behavior_gate(mem_input, self.gate_keys)
        if self.slot_memories and self._active_slot_weights is not None:
            if self._active_slot_weights.shape[0] != mem_input.shape[0]:
                raise ValueError(
                    "Active slot weights batch size "
                    f"{self._active_slot_weights.shape[0]} != input batch size {mem_input.shape[0]}"
                )
            active = self._active_slot_weights.to(device=mem_input.device, dtype=self.memory.dtype)
            for slot_idx, slot_memory in enumerate(self.slot_memories):
                slot_weight = active[:, slot_idx].reshape(-1, 1, 1)
                if torch.count_nonzero(slot_weight).item() == 0:
                    continue
                slot_mem = F.linear(mem_input, slot_memory.to(device=mem_input.device, dtype=self.memory.dtype))
                if object_gate is not None:
                    slot_mem = slot_mem * object_gate
                slot_keys = self.slot_gate_keys[slot_idx]
                if slot_keys.numel() > 0:
                    slot_mem = slot_mem * self._behavior_gate(mem_input, slot_keys)
                mem = mem + slot_mem * slot_weight
        mem = mem.to(out.dtype)
        return out + self.memory_scale * mem

    def _behavior_gate(self, mem_input: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(mem_input, dim=-1)
        key_norm = F.normalize(keys.to(device=mem_input.device, dtype=mem_input.dtype), dim=-1)
        similarity = torch.matmul(x_norm, key_norm.T).amax(dim=-1, keepdim=True)
        gate = torch.sigmoid((similarity - self.gate_threshold) * self.gate_temperature)
        if self.gate_last_token_only and gate.ndim == 3 and gate.shape[1] > 1:
            mask = torch.zeros_like(gate)
            mask[:, -1:, :] = 1.0
            gate = gate * mask
        return gate

    def _object_gate(self, mem_input: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Sequence-level object gate.

        Behavior gates are token-local: they decide where the memory should
        fire. This gate decides whether the learned object is present anywhere
        in the current sequence, then applies one scalar per batch item to all
        token positions. It is intentionally activation-derived rather than
        string-routed.
        """

        x_norm = F.normalize(mem_input, dim=-1)
        key_norm = F.normalize(keys.to(device=mem_input.device, dtype=mem_input.dtype), dim=-1)
        token_similarity = torch.matmul(x_norm, key_norm.T).amax(dim=-1)
        if token_similarity.ndim == 1:
            sequence_similarity = token_similarity.unsqueeze(-1)
        else:
            sequence_similarity = token_similarity.amax(dim=1, keepdim=True)
            while sequence_similarity.ndim < mem_input.ndim:
                sequence_similarity = sequence_similarity.unsqueeze(-1)
        gate = torch.sigmoid(
            (sequence_similarity - self.object_gate_threshold) * self.object_gate_temperature
        )
        floor = float(self.object_gate_floor)
        if floor <= 0.0:
            return gate
        if floor >= 1.0:
            return torch.ones_like(gate)
        return floor + (1.0 - floor) * gate

    def _density_object_gate(self, mem_input: torch.Tensor) -> torch.Tensor:
        gates = [sequence_gate(mem_input, params) for params in self.object_density_gates]
        gate = torch.stack(gates, dim=0).amax(dim=0)
        floor = float(self.object_gate_floor)
        if floor >= 1.0:
            gate = torch.ones_like(gate)
        elif floor > 0.0:
            gate = floor + (1.0 - floor) * gate
        while gate.ndim < mem_input.ndim:
            gate = gate.unsqueeze(-1)
        return gate

    def _active_or_internal_object_gate(self, mem_input: torch.Tensor) -> torch.Tensor | None:
        if self._active_object_gate is not None:
            gate = self._active_object_gate.to(device=mem_input.device, dtype=mem_input.dtype)
            if gate.ndim == 0:
                gate = gate.reshape(1)
            if gate.shape[0] != mem_input.shape[0]:
                raise ValueError(
                    f"Active object gate batch size {gate.shape[0]} != input batch size {mem_input.shape[0]}"
                )
            while gate.ndim < mem_input.ndim:
                gate = gate.unsqueeze(-1)
            return gate
        gate: torch.Tensor | None = None
        if self.object_gate_keys.numel() > 0:
            gate = self._object_gate(mem_input, self.object_gate_keys)
        if self.object_density_gates:
            density_gate = self._density_object_gate(mem_input)
            gate = density_gate if gate is None else gate * density_gate
        return gate

    @torch.no_grad()
    def add_memory_(self, delta: torch.Tensor, slot_id: int | None = None) -> None:
        delta = delta.to(device=self.memory.device, dtype=self.memory.dtype)
        if slot_id is None:
            self.memory.add_(delta)
            return
        self.slot_memories[slot_id].add_(delta)

    @torch.no_grad()
    def copy_memory_(self, value: torch.Tensor) -> None:
        self.memory.copy_(value.to(device=self.memory.device, dtype=self.memory.dtype))

    def memory_for_slot(self, slot_id: int | None = None) -> torch.Tensor:
        if slot_id is None:
            return self.memory
        return self.slot_memories[slot_id]

    @torch.no_grad()
    def add_slot_(self, terms: list[str] | tuple[str, ...] = ()) -> int:
        slot = torch.zeros_like(self.memory)
        self.slot_memories.append(slot)
        self.slot_gate_keys.append(torch.empty(0, self.in_features, device=self.memory.device, dtype=self.memory.dtype))
        self.slot_terms.append(tuple(term.lower() for term in terms if term))
        return len(self.slot_memories) - 1

    @torch.no_grad()
    def set_slot_terms_(self, slot_id: int, terms: list[str] | tuple[str, ...]) -> None:
        self.slot_terms[slot_id] = tuple(term.lower() for term in terms if term)

    @torch.no_grad()
    def set_slot_gate_keys_(
        self,
        slot_id: int,
        keys: torch.Tensor,
        threshold: float = 0.95,
        temperature: float = 80.0,
    ) -> None:
        keys = keys.detach().to(device=self.memory.device, dtype=self.memory.dtype)
        if keys.ndim != 2 or keys.shape[1] != self.in_features:
            raise ValueError(f"Expected slot gate keys with shape [n, {self.in_features}], got {tuple(keys.shape)}")
        self.slot_gate_keys[slot_id] = keys.contiguous()
        self.gate_threshold = threshold
        self.gate_temperature = temperature

    @torch.no_grad()
    def set_active_slot_weights_(self, weights: torch.Tensor | None) -> None:
        if weights is None:
            self._active_slot_weights = None
            return
        if weights.ndim != 2 or weights.shape[1] != len(self.slot_memories):
            raise ValueError(
                f"Expected active slot weights [batch, {len(self.slot_memories)}], got {tuple(weights.shape)}"
            )
        self._active_slot_weights = weights.detach().to(device=self.memory.device, dtype=self.memory.dtype)

    @torch.no_grad()
    def set_active_object_gate_(self, weights: torch.Tensor | None) -> None:
        if weights is None:
            self._active_object_gate = None
            return
        if weights.ndim not in {1, 2, 3}:
            raise ValueError(
                f"Expected active object gate [batch] or a broadcastable batch tensor, got {tuple(weights.shape)}"
            )
        if weights.ndim > 1 and any(dim != 1 for dim in weights.shape[1:]):
            raise ValueError(f"Expected active object gate trailing dimensions of size 1, got {tuple(weights.shape)}")
        self._active_object_gate = weights.detach().to(device=self.memory.device, dtype=self.memory.dtype)

    @torch.no_grad()
    def set_gate_keys_(
        self,
        keys: torch.Tensor,
        threshold: float = 0.95,
        temperature: float = 80.0,
        append: bool = False,
    ) -> None:
        keys = keys.detach().to(device=self.memory.device, dtype=self.memory.dtype)
        if keys.ndim != 2 or keys.shape[1] != self.in_features:
            raise ValueError(f"Expected gate keys with shape [n, {self.in_features}], got {tuple(keys.shape)}")
        if append and self.gate_keys.numel() > 0:
            keys = torch.cat([self.gate_keys, keys], dim=0)
        self.gate_keys = keys.contiguous()
        self.gate_threshold = threshold
        self.gate_temperature = temperature

    @torch.no_grad()
    def set_gate_last_token_only_(self, value: bool) -> None:
        self.gate_last_token_only = bool(value)

    @torch.no_grad()
    def set_object_gate_keys_(
        self,
        keys: torch.Tensor,
        threshold: float = 0.90,
        temperature: float = 40.0,
        floor: float = 0.0,
        append: bool = False,
    ) -> None:
        keys = keys.detach().to(device=self.memory.device, dtype=self.memory.dtype)
        if keys.ndim != 2 or keys.shape[1] != self.in_features:
            raise ValueError(f"Expected object gate keys with shape [n, {self.in_features}], got {tuple(keys.shape)}")
        if append and self.object_gate_keys.numel() > 0:
            keys = torch.cat([self.object_gate_keys, keys], dim=0)
        self.object_gate_keys = keys.contiguous()
        self.object_gate_threshold = threshold
        self.object_gate_temperature = temperature
        self.object_gate_floor = float(floor)

    @torch.no_grad()
    def add_density_object_gate_(self, params: DensityRatioGateParams, append: bool = True) -> None:
        params = params.to(device=self.memory.device, dtype=self.memory.dtype)
        if append:
            self.object_density_gates.append(params)
        else:
            self.object_density_gates = [params]


@dataclass
class LayerCapture:
    keys: torch.Tensor
    outputs: torch.Tensor


@dataclass
class BlockCapture:
    inputs: torch.Tensor
    outputs: torch.Tensor


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype != "auto":
        raise ValueError(f"Unknown dtype: {dtype}")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(resolved_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer, resolved_device


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    ]
    for root_name, layer_name in candidates:
        root = getattr(model, root_name, None)
        layers = getattr(root, layer_name, None) if root is not None else None
        if layers is not None:
            return list(layers)
    if hasattr(model, "layers"):
        return list(getattr(model, "layers"))
    raise AttributeError("Could not find decoder layers on model.")


def get_mlp_down_module(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise AttributeError(f"Layer {layer.__class__.__name__} has no .mlp module.")
    for name in ("down_proj", "c_proj", "fc2", "dense_4h_to_h"):
        module = getattr(mlp, name, None)
        if module is not None:
            return module
    raise AttributeError(f"Could not find MLP down projection on {mlp.__class__.__name__}.")


def get_attention_o_module(layer: nn.Module) -> nn.Module:
    return get_attention_projection_module(layer, "o")


def get_attention_projection_module(layer: nn.Module, projection: str) -> nn.Module:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
    if attn is None:
        raise AttributeError(f"Layer {layer.__class__.__name__} has no attention module.")
    names_by_projection = {
        "q": ("q_proj", "query", "c_attn_q"),
        "k": ("k_proj", "key", "c_attn_k"),
        "v": ("v_proj", "value", "c_attn_v"),
        "o": ("o_proj", "out_proj", "c_proj", "dense"),
    }
    if projection not in names_by_projection:
        raise ValueError(f"Unknown attention projection {projection!r}; expected one of q/k/v/o.")
    for name in names_by_projection[projection]:
        module = getattr(attn, name, None)
        if module is not None:
            return module
    raise AttributeError(f"Could not find attention {projection} projection on {attn.__class__.__name__}.")


def set_mlp_down_module(layer: nn.Module, module: nn.Module) -> None:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise AttributeError(f"Layer {layer.__class__.__name__} has no .mlp module.")
    for name in ("down_proj", "c_proj", "fc2", "dense_4h_to_h"):
        if hasattr(mlp, name):
            setattr(mlp, name, module)
            return
    raise AttributeError(f"Could not set MLP down projection on {mlp.__class__.__name__}.")


def set_attention_o_module(layer: nn.Module, module: nn.Module) -> None:
    set_attention_projection_module(layer, "o", module)


def set_attention_projection_module(layer: nn.Module, projection: str, module: nn.Module) -> None:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
    if attn is None:
        raise AttributeError(f"Layer {layer.__class__.__name__} has no attention module.")
    names_by_projection = {
        "q": ("q_proj", "query", "c_attn_q"),
        "k": ("k_proj", "key", "c_attn_k"),
        "v": ("v_proj", "value", "c_attn_v"),
        "o": ("o_proj", "out_proj", "c_proj", "dense"),
    }
    if projection not in names_by_projection:
        raise ValueError(f"Unknown attention projection {projection!r}; expected one of q/k/v/o.")
    for name in names_by_projection[projection]:
        if hasattr(attn, name):
            setattr(attn, name, module)
            return
    raise AttributeError(f"Could not set attention {projection} projection on {attn.__class__.__name__}.")


def install_additive_memory(
    model: nn.Module,
    layer_indices: list[int],
    memory_dtype: torch.dtype = torch.float32,
) -> dict[int, AdditiveMemoryLinear]:
    layers = get_decoder_layers(model)
    wrappers: dict[int, AdditiveMemoryLinear] = {}
    for raw_idx in layer_indices:
        idx = raw_idx if raw_idx >= 0 else len(layers) + raw_idx
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {raw_idx} resolved to {idx}, but model has {len(layers)} layers.")
        down = get_mlp_down_module(layers[idx])
        if isinstance(down, AdditiveMemoryLinear):
            wrapper = down
        else:
            if not isinstance(down, nn.Linear):
                raise TypeError(f"Expected nn.Linear down projection, got {type(down)!r}.")
            wrapper = AdditiveMemoryLinear(down, memory_dtype=memory_dtype)
            set_mlp_down_module(layers[idx], wrapper)
        wrappers[idx] = wrapper
    return wrappers


def install_additive_attention_memory(
    model: nn.Module,
    layer_indices: list[int],
    memory_dtype: torch.dtype = torch.float32,
) -> dict[int, AdditiveMemoryLinear]:
    layers = get_decoder_layers(model)
    wrappers: dict[int, AdditiveMemoryLinear] = {}
    for raw_idx in layer_indices:
        idx = raw_idx if raw_idx >= 0 else len(layers) + raw_idx
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {raw_idx} resolved to {idx}, but model has {len(layers)} layers.")
        out_proj = get_attention_o_module(layers[idx])
        if isinstance(out_proj, AdditiveMemoryLinear):
            wrapper = out_proj
        else:
            if not isinstance(out_proj, nn.Linear):
                raise TypeError(f"Expected nn.Linear attention output projection, got {type(out_proj)!r}.")
            wrapper = AdditiveMemoryLinear(out_proj, memory_dtype=memory_dtype)
            set_attention_o_module(layers[idx], wrapper)
        wrappers[idx] = wrapper
    return wrappers


def install_additive_attention_projection_memory(
    model: nn.Module,
    layer_indices: list[int],
    projection: str,
    memory_dtype: torch.dtype = torch.float32,
) -> dict[int, AdditiveMemoryLinear]:
    layers = get_decoder_layers(model)
    wrappers: dict[int, AdditiveMemoryLinear] = {}
    for raw_idx in layer_indices:
        idx = raw_idx if raw_idx >= 0 else len(layers) + raw_idx
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {raw_idx} resolved to {idx}, but model has {len(layers)} layers.")
        proj = get_attention_projection_module(layers[idx], projection)
        if isinstance(proj, AdditiveMemoryLinear):
            wrapper = proj
        else:
            if not isinstance(proj, nn.Linear):
                raise TypeError(f"Expected nn.Linear attention {projection} projection, got {type(proj)!r}.")
            wrapper = AdditiveMemoryLinear(proj, memory_dtype=memory_dtype)
            set_attention_projection_module(layers[idx], projection, wrapper)
        wrappers[idx] = wrapper
    return wrappers


def additive_memory_wrappers(model: nn.Module) -> list[AdditiveMemoryLinear]:
    return [module for module in model.modules() if isinstance(module, AdditiveMemoryLinear)]


def set_active_slot_weights_for_prompts(model: nn.Module, prompts: list[str]) -> None:
    wrappers = additive_memory_wrappers(model)
    if not wrappers:
        return
    object_router = getattr(model, "_caic_object_gate_router", None)
    object_weights = object_router(prompts) if object_router is not None else None
    if object_weights is None:
        for wrapper in wrappers:
            wrapper.set_active_object_gate_(None)
    else:
        for wrapper in wrappers:
            weights = object_weights
            if isinstance(object_weights, dict):
                weights = object_weights.get(wrapper, object_weights.get(id(wrapper)))
            wrapper.set_active_object_gate_(weights)
    router = getattr(model, "_caic_activation_slot_router", None)
    if router is not None and any(wrapper.slot_memories for wrapper in wrappers):
        weights = router(prompts)
        for wrapper in wrappers:
            if not wrapper.slot_memories:
                wrapper.set_active_slot_weights_(None)
                continue
            wrapper.set_active_slot_weights_(weights)
        return
    lowered_prompts = [prompt.lower() for prompt in prompts]
    for wrapper in wrappers:
        if not wrapper.slot_memories:
            wrapper.set_active_slot_weights_(None)
            continue
        rows: list[list[float]] = []
        for prompt in lowered_prompts:
            row = []
            for terms in wrapper.slot_terms:
                row.append(1.0 if terms and any(prompt_has_slot_term(prompt, term) for term in terms) else 0.0)
            rows.append(row)
        wrapper.set_active_slot_weights_(torch.tensor(rows, dtype=torch.float32))


def prompt_has_slot_term(prompt: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", prompt) is not None


def clear_active_slot_weights(model: nn.Module) -> None:
    for wrapper in additive_memory_wrappers(model):
        wrapper.set_active_slot_weights_(None)
        wrapper.set_active_object_gate_(None)


@torch.no_grad()
def _capture_module_io_at_token_indices(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    token_indices: list[int],
    layer_indices: list[int],
    module_getter,
    device: torch.device,
    max_length: int = 2048,
    capture_window: int = 1,
) -> dict[int, LayerCapture]:
    """Capture module IO around explicit token indices in each prompt.

    This is intentionally single-prompt per forward. It is used for diagnostic
    gates where the target token is not the suffix, such as source/content
    tokens inside a chat-formatted prompt.
    """

    if len(prompts) != len(token_indices):
        raise ValueError(f"prompts length {len(prompts)} != token_indices length {len(token_indices)}")
    if capture_window <= 0:
        raise ValueError("capture_window must be positive.")

    layers = get_decoder_layers(model)
    layer_indices = [idx if idx >= 0 else len(layers) + idx for idx in layer_indices]
    stores: dict[int, dict[str, list[torch.Tensor]]] = {
        idx: {"keys": [], "outputs": []} for idx in layer_indices
    }
    active_index = 0

    def make_hook(layer_idx: int):
        def hook(_module: nn.Module, module_inputs: tuple[torch.Tensor, ...], module_output: torch.Tensor):
            hidden_len = module_inputs[0].shape[1]
            center = min(max(active_index, 0), hidden_len - 1)
            start = max(0, center - capture_window + 1)
            end = center + 1
            key = module_inputs[0][:, start:end, :].detach().float().cpu()
            out = module_output[:, start:end, :].detach().float().cpu()
            stores[layer_idx]["keys"].append(key.reshape(-1, key.shape[-1]))
            stores[layer_idx]["outputs"].append(out.reshape(-1, out.shape[-1]))

        return hook

    try:
        for prompt, token_index in zip(prompts, token_indices, strict=True):
            full_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not full_ids:
                continue
            trim = max(0, len(full_ids) - max_length)
            trimmed_ids = full_ids[trim:]
            active_index = token_index - trim
            active_index = min(max(active_index, 0), len(trimmed_ids) - 1)
            input_ids = torch.tensor([trimmed_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            set_active_slot_weights_for_prompts(model, [prompt])
            handles = [
                module_getter(layers[idx]).register_forward_hook(make_hook(idx))
                for idx in layer_indices
            ]
            try:
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    finally:
        clear_active_slot_weights(model)

    captures: dict[int, LayerCapture] = {}
    for idx, store in stores.items():
        captures[idx] = LayerCapture(
            keys=torch.cat(store["keys"], dim=0),
            outputs=torch.cat(store["outputs"], dim=0),
        )
    return captures


def capture_layer_io_at_token_indices(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    token_indices: list[int],
    layer_indices: list[int],
    device: torch.device,
    max_length: int = 2048,
    capture_window: int = 1,
) -> dict[int, LayerCapture]:
    return _capture_module_io_at_token_indices(
        model,
        tokenizer,
        prompts,
        token_indices,
        layer_indices,
        get_mlp_down_module,
        device,
        max_length=max_length,
        capture_window=capture_window,
    )


def capture_attention_io_at_token_indices(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    token_indices: list[int],
    layer_indices: list[int],
    device: torch.device,
    max_length: int = 2048,
    capture_window: int = 1,
) -> dict[int, LayerCapture]:
    return _capture_module_io_at_token_indices(
        model,
        tokenizer,
        prompts,
        token_indices,
        layer_indices,
        get_attention_o_module,
        device,
        max_length=max_length,
        capture_window=capture_window,
    )


def capture_attention_projection_io_at_token_indices(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    token_indices: list[int],
    layer_indices: list[int],
    projection: str,
    device: torch.device,
    max_length: int = 2048,
    capture_window: int = 1,
) -> dict[int, LayerCapture]:
    return _capture_module_io_at_token_indices(
        model,
        tokenizer,
        prompts,
        token_indices,
        layer_indices,
        lambda layer: get_attention_projection_module(layer, projection),
        device,
        max_length=max_length,
        capture_window=capture_window,
    )


def get_memory_wrappers(model: nn.Module, layer_indices: list[int]) -> dict[int, AdditiveMemoryLinear]:
    layers = get_decoder_layers(model)
    wrappers: dict[int, AdditiveMemoryLinear] = {}
    for idx in layer_indices:
        resolved = idx if idx >= 0 else len(layers) + idx
        down = get_mlp_down_module(layers[resolved])
        if not isinstance(down, AdditiveMemoryLinear):
            raise TypeError(f"Layer {resolved} does not have an AdditiveMemoryLinear installed.")
        wrappers[resolved] = down
    return wrappers


@torch.no_grad()
def capture_layer_io(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 2048,
    capture_last_tokens: int = 1,
) -> dict[int, LayerCapture]:
    """Capture down-projection inputs and outputs over a suffix token window."""

    if capture_last_tokens <= 0:
        raise ValueError("capture_last_tokens must be positive.")

    layers = get_decoder_layers(model)
    layer_indices = [idx if idx >= 0 else len(layers) + idx for idx in layer_indices]
    stores: dict[int, dict[str, list[torch.Tensor]]] = {
        idx: {"keys": [], "outputs": []} for idx in layer_indices
    }

    def make_hook(layer_idx: int):
        def hook(_module: nn.Module, module_inputs: tuple[torch.Tensor, ...], module_output: torch.Tensor):
            token_count = min(capture_last_tokens, module_inputs[0].shape[1])
            key = module_inputs[0][:, -token_count:, :].detach().float().cpu()
            out = module_output[:, -token_count:, :].detach().float().cpu()
            stores[layer_idx]["keys"].append(key.reshape(-1, key.shape[-1]))
            stores[layer_idx]["outputs"].append(out.reshape(-1, out.shape[-1]))

        return hook

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            set_active_slot_weights_for_prompts(model, batch)
            handles = [
                get_mlp_down_module(layers[idx]).register_forward_hook(make_hook(idx))
                for idx in layer_indices
            ]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            try:
                model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    finally:
        clear_active_slot_weights(model)

    captures: dict[int, LayerCapture] = {}
    for idx, store in stores.items():
        captures[idx] = LayerCapture(
            keys=torch.cat(store["keys"], dim=0),
            outputs=torch.cat(store["outputs"], dim=0),
        )
    return captures


@torch.no_grad()
def capture_attention_io(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 2048,
    capture_last_tokens: int = 1,
) -> dict[int, LayerCapture]:
    """Capture attention output-projection inputs and outputs over a suffix window."""

    if capture_last_tokens <= 0:
        raise ValueError("capture_last_tokens must be positive.")

    layers = get_decoder_layers(model)
    layer_indices = [idx if idx >= 0 else len(layers) + idx for idx in layer_indices]
    stores: dict[int, dict[str, list[torch.Tensor]]] = {
        idx: {"keys": [], "outputs": []} for idx in layer_indices
    }

    def make_hook(layer_idx: int):
        def hook(_module: nn.Module, module_inputs: tuple[torch.Tensor, ...], module_output: torch.Tensor):
            token_count = min(capture_last_tokens, module_inputs[0].shape[1])
            key = module_inputs[0][:, -token_count:, :].detach().float().cpu()
            out = module_output[:, -token_count:, :].detach().float().cpu()
            stores[layer_idx]["keys"].append(key.reshape(-1, key.shape[-1]))
            stores[layer_idx]["outputs"].append(out.reshape(-1, out.shape[-1]))

        return hook

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            set_active_slot_weights_for_prompts(model, batch)
            handles = [
                get_attention_o_module(layers[idx]).register_forward_hook(make_hook(idx))
                for idx in layer_indices
            ]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            try:
                model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    finally:
        clear_active_slot_weights(model)

    captures: dict[int, LayerCapture] = {}
    for idx, store in stores.items():
        captures[idx] = LayerCapture(
            keys=torch.cat(store["keys"], dim=0),
            outputs=torch.cat(store["outputs"], dim=0),
        )
    return captures


@torch.no_grad()
def capture_attention_projection_io(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    projection: str,
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 2048,
    capture_last_tokens: int = 1,
) -> dict[int, LayerCapture]:
    """Capture attention projection inputs and outputs over a suffix window."""

    if capture_last_tokens <= 0:
        raise ValueError("capture_last_tokens must be positive.")

    layers = get_decoder_layers(model)
    layer_indices = [idx if idx >= 0 else len(layers) + idx for idx in layer_indices]
    stores: dict[int, dict[str, list[torch.Tensor]]] = {
        idx: {"keys": [], "outputs": []} for idx in layer_indices
    }

    def make_hook(layer_idx: int):
        def hook(_module: nn.Module, module_inputs: tuple[torch.Tensor, ...], module_output: torch.Tensor):
            token_count = min(capture_last_tokens, module_inputs[0].shape[1])
            key = module_inputs[0][:, -token_count:, :].detach().float().cpu()
            out = module_output[:, -token_count:, :].detach().float().cpu()
            stores[layer_idx]["keys"].append(key.reshape(-1, key.shape[-1]))
            stores[layer_idx]["outputs"].append(out.reshape(-1, out.shape[-1]))

        return hook

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            set_active_slot_weights_for_prompts(model, batch)
            handles = [
                get_attention_projection_module(layers[idx], projection).register_forward_hook(make_hook(idx))
                for idx in layer_indices
            ]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            try:
                model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    finally:
        clear_active_slot_weights(model)

    captures: dict[int, LayerCapture] = {}
    for idx, store in stores.items():
        captures[idx] = LayerCapture(
            keys=torch.cat(store["keys"], dim=0),
            outputs=torch.cat(store["outputs"], dim=0),
        )
    return captures


@torch.no_grad()
def capture_block_io(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 2048,
    capture_last_tokens: int = 1,
) -> dict[int, BlockCapture]:
    """Capture residual stream inputs and outputs for decoder blocks."""

    if capture_last_tokens <= 0:
        raise ValueError("capture_last_tokens must be positive.")

    layers = get_decoder_layers(model)
    layer_indices = [idx if idx >= 0 else len(layers) + idx for idx in layer_indices]
    stores: dict[int, dict[str, list[torch.Tensor]]] = {
        idx: {"inputs": [], "outputs": []} for idx in layer_indices
    }

    def layer_output_hidden(module_output: Any) -> torch.Tensor:
        if isinstance(module_output, torch.Tensor):
            return module_output
        if isinstance(module_output, (tuple, list)) and module_output:
            first = module_output[0]
            if isinstance(first, torch.Tensor):
                return first
        raise TypeError(f"Could not resolve hidden-state tensor from {type(module_output)!r}.")

    def make_hook(layer_idx: int):
        def hook(_module: nn.Module, module_inputs: tuple[Any, ...], module_output: Any):
            if not module_inputs or not isinstance(module_inputs[0], torch.Tensor):
                raise TypeError("Expected decoder block first input to be a hidden-state tensor.")
            hidden_in = module_inputs[0]
            hidden_out = layer_output_hidden(module_output)
            token_count = min(capture_last_tokens, hidden_in.shape[1])
            inp = hidden_in[:, -token_count:, :].detach().float().cpu()
            out = hidden_out[:, -token_count:, :].detach().float().cpu()
            stores[layer_idx]["inputs"].append(inp.reshape(-1, inp.shape[-1]))
            stores[layer_idx]["outputs"].append(out.reshape(-1, out.shape[-1]))

        return hook

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            set_active_slot_weights_for_prompts(model, batch)
            handles = [layers[idx].register_forward_hook(make_hook(idx)) for idx in layer_indices]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {name: value.to(device) for name, value in tokens.items()}
            try:
                model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
    finally:
        clear_active_slot_weights(model)

    captures: dict[int, BlockCapture] = {}
    for idx, store in stores.items():
        captures[idx] = BlockCapture(
            inputs=torch.cat(store["inputs"], dim=0),
            outputs=torch.cat(store["outputs"], dim=0),
        )
    return captures


@contextmanager
def patched_down_output(
    model: nn.Module,
    layer_idx: int,
    replacement: torch.Tensor,
    device: torch.device,
) -> Iterator[None]:
    """Temporarily replace final-token down-projection output for one layer."""

    layers = get_decoder_layers(model)
    resolved = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
    module = get_mlp_down_module(layers[resolved])

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        patched = output.clone()
        repl = replacement.to(device=device, dtype=patched.dtype)
        patched[:, -1, :] = repl
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def memory_norms(wrappers: dict[int, AdditiveMemoryLinear]) -> dict[str, float]:
    rows: dict[str, float] = {}
    for idx, wrapper in wrappers.items():
        rows[f"layer_{idx}_memory_fro"] = float(torch.linalg.vector_norm(wrapper.memory.detach().float()).cpu())
        if wrapper.slot_memories:
            slot_norms = [
                float(torch.linalg.vector_norm(slot.detach().float()).cpu())
                for slot in wrapper.slot_memories
            ]
            rows[f"layer_{idx}_slot_count"] = float(len(slot_norms))
            rows[f"layer_{idx}_slot_memory_fro_sum"] = sum(slot_norms)
            rows[f"layer_{idx}_slot_memory_fro_max"] = max(slot_norms)
    return rows
