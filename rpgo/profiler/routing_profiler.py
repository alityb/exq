"""RoutingProfiler: PyTorch forward-hook based instrumentation for MoE routing.

This is the Python boundary layer that interacts with PyTorch models.
It collects routing decisions via forward hooks and populates the Rust-core
RoutingProfile data structure for downstream compilation.

Supports multiple MoE architectures:
  - Qwen3 MoE (Qwen3MoeSparseMoeBlock)
  - Mixtral (MixtralSparseMoeBlock)
  - DeepSeek-V2 (DeepseekV2MoE)
  - OLMoE (OlmoeSparseMoeBlock)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from rpgo._core import LayerProfile, RoutingProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture-specific gate detection
# ---------------------------------------------------------------------------

# Map of (module_class_name_substring -> attribute_path_to_gate)
# These are checked in order; first match wins.
_GATE_PATTERNS: list[tuple[str, str]] = [
    ("Qwen3Moe", "gate"),       # Qwen3MoeSparseMoeBlock.gate
    ("Qwen2Moe", "gate"),       # Qwen2MoE variants
    ("Mixtral", "gate"),         # MixtralSparseMoeBlock.gate
    ("Deepseek", "gate"),       # DeepseekV2MoE.gate
    ("Olmoe", "gate"),          # OlmoeSparseMoeBlock.gate
    ("Switch", "gate"),         # SwitchTransformersSparseMLP.gate (T5-based)
]


def _find_moe_layers(model: nn.Module) -> list[tuple[int, nn.Module, nn.Module, int]]:
    """Find all MoE layers in a model.

    Returns list of (layer_idx, moe_block, gate_module, n_experts).
    """
    moe_layers = []

    # Walk the model's transformer layers
    layers_container = None
    for attr in ("model.layers", "transformer.layers", "encoder.layers", "decoder.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            layers_container = obj
            break
        except AttributeError:
            continue

    if layers_container is None:
        logger.warning("Could not find transformer layers in model")
        return []

    for layer_idx, layer in enumerate(layers_container):
        # Find MLP/MoE block
        mlp = getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
        if mlp is None:
            continue

        # Check if it's an MoE block by looking for a gate
        gate = None
        class_name = type(mlp).__name__
        for pattern, gate_attr in _GATE_PATTERNS:
            if pattern.lower() in class_name.lower():
                gate = getattr(mlp, gate_attr, None)
                break

        # Fallback: check for 'gate' attribute directly
        if gate is None:
            gate = getattr(mlp, "gate", None)

        if gate is None:
            continue

        # Determine n_experts
        n_experts = _infer_n_experts(mlp, gate)
        if n_experts is None:
            logger.warning(f"Layer {layer_idx}: found gate but could not determine n_experts")
            continue

        moe_layers.append((layer_idx, mlp, gate, n_experts))

    return moe_layers


def _infer_n_experts(mlp: nn.Module, gate: nn.Module) -> int | None:
    """Infer number of experts from the MoE block or gate."""
    # From gate output features (gate is typically Linear(hidden, n_experts))
    if hasattr(gate, "out_features"):
        return gate.out_features
    if hasattr(gate, "weight"):
        return gate.weight.shape[0]

    # From experts list
    if hasattr(mlp, "experts"):
        return len(mlp.experts)
    if hasattr(mlp, "num_experts"):
        return mlp.num_experts

    return None


def _infer_top_k(mlp: nn.Module) -> int:
    """Infer top-k from the MoE block config."""
    for attr in ("num_experts_per_tok", "top_k", "num_selected_experts",
                 "experts_per_token", "n_experts_per_tok"):
        val = getattr(mlp, attr, None)
        if val is not None:
            return int(val)

    # Check model config if available
    config = getattr(mlp, "config", None)
    if config is not None:
        for attr in ("num_experts_per_tok", "top_k"):
            val = getattr(config, attr, None)
            if val is not None:
                return int(val)

    # Default: top-2 (most common)
    return 2


class RoutingProfiler:
    """Instruments a MoE model's routing decisions during calibration.

    Collects per-layer expert activation counts and cross-layer co-activation
    statistics via PyTorch forward hooks. Populates the Rust-core
    RoutingProfile for downstream compiler passes.

    Usage:
        profiler = RoutingProfiler(model)
        profiler.start()
        # run calibration forward passes
        profiler.stop()
        profile = profiler.build_profile()
        profile.save("routing_profile.json")
    """

    def __init__(self, model: nn.Module, model_id: str = "unknown"):
        self.model = model
        self.model_id = model_id
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # Discover MoE layers
        self._moe_layers = _find_moe_layers(model)
        if not self._moe_layers:
            raise ValueError(
                "No MoE layers found in model. Supported architectures: "
                "Qwen3, Mixtral, DeepSeek-V2, OLMoE"
            )

        # State
        self._layer_profiles: dict[int, LayerProfile] = {}
        self._prev_layer_experts: dict[int, list[int]] = {}  # batch_idx -> experts
        self._total_tokens = 0
        self._started = False

        # Initialize LayerProfiles via Rust core
        for layer_idx, mlp, gate, n_experts in self._moe_layers:
            top_k = _infer_top_k(mlp)
            lp = LayerProfile(layer_idx, n_experts, top_k)
            self._layer_profiles[layer_idx] = lp

        # Order MoE layers for co-activation tracking
        self._moe_layer_order = sorted(self._layer_profiles.keys())
        self._layer_to_order_idx = {
            l: i for i, l in enumerate(self._moe_layer_order)
        }

        logger.info(
            f"RoutingProfiler: found {len(self._moe_layers)} MoE layers "
            f"in '{model_id}'"
        )

    @property
    def n_moe_layers(self) -> int:
        return len(self._moe_layers)

    def start(self) -> None:
        """Register forward hooks and begin collecting routing data."""
        if self._started:
            return
        for layer_idx, mlp, gate, n_experts in self._moe_layers:
            hook = gate.register_forward_hook(self._make_hook(layer_idx, n_experts))
            self._hooks.append(hook)
        self._started = True

    def stop(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._started = False

    def _make_hook(self, layer_idx: int, n_experts: int):
        """Create a forward hook for a specific MoE gate."""
        lp = self._layer_profiles[layer_idx]
        top_k = lp.top_k
        order_idx = self._layer_to_order_idx[layer_idx]
        prev_layer_idx = self._moe_layer_order[order_idx - 1] if order_idx > 0 else None

        def hook(module: nn.Module, input: Any, output: torch.Tensor):
            # output: router logits [batch, seq_len, n_experts] or [batch*seq, n_experts]
            with torch.no_grad():
                logits = output
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.size(-1))  # [tokens, n_experts]
                elif logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # [1, n_experts]

                n_tokens = logits.size(0)
                topk_indices = logits.topk(min(top_k, logits.size(-1)), dim=-1).indices  # [tokens, top_k]

                # Update per-expert activation counts
                for token_idx in range(n_tokens):
                    experts = topk_indices[token_idx].tolist()
                    for e in experts:
                        lp.increment_expert(e)

                    # Co-activation tracking: compare with previous layer's selections
                    if prev_layer_idx is not None and token_idx in self._prev_layer_experts:
                        prev_experts = self._prev_layer_experts[token_idx]
                        prev_lp = self._layer_profiles[prev_layer_idx]
                        for src_e in prev_experts:
                            for dst_e in experts:
                                prev_lp.add_co_activation(src_e, dst_e)

                # Store current layer's selections for next layer's co-activation
                self._prev_layer_experts = {}
                for token_idx in range(n_tokens):
                    self._prev_layer_experts[token_idx] = topk_indices[token_idx].tolist()

                self._total_tokens += n_tokens

        return hook

    def build_profile(self, calibration_samples: int = 0) -> RoutingProfile:
        """Finalize and return the RoutingProfile.

        Normalizes counts to frequencies, computes entropy, and normalizes
        co-activation matrices.
        """
        profile = RoutingProfile(self.model_id, calibration_samples)
        profile.calibration_tokens = self._total_tokens

        for layer_idx in self._moe_layer_order:
            lp = self._layer_profiles[layer_idx]
            lp.finalize()  # normalize freqs + entropy + co-activations
            profile.add_layer(lp)

        return profile

    def reset(self) -> None:
        """Reset all collected statistics."""
        for layer_idx, mlp, gate, n_experts in self._moe_layers:
            top_k = _infer_top_k(mlp)
            self._layer_profiles[layer_idx] = LayerProfile(layer_idx, n_experts, top_k)
        self._prev_layer_experts = {}
        self._total_tokens = 0
