"""AttentionProfiler: per-head contribution profiling for dense transformers."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from exq.profiler.dense_profile import DenseProfile, HeadLayerProfile

logger = logging.getLogger(__name__)


_ATTN_PATTERNS: list[tuple[str, str]] = [
    ("LlamaAttention", "config.num_attention_heads"),
    ("MistralAttention", "config.num_attention_heads"),
    ("PhiAttention", "config.num_attention_heads"),
    ("GemmaAttention", "config.num_attention_heads"),
    ("Qwen2Attention", "config.num_attention_heads"),
    ("GPTNeoXAttention", "config.num_attention_heads"),
    ("FalconAttention", "config.n_head"),
    ("GPT2Attention", "config.n_head"),
    ("OlmoAttention", "config.num_attention_heads"),
]


def _resolve_attr_path(root: Any, path: str) -> Any:
    """Resolve a dotted attribute path from an object."""
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _find_transformer_layers(model: nn.Module):
    """Find the primary transformer layer container for a model."""
    for attr in (
        "model.layers",
        "transformer.h",
        "model.model.layers",
        "transformer.layers",
        "encoder.layers",
    ):
        try:
            return _resolve_attr_path(model, attr)
        except AttributeError:
            continue
    return None


def _find_attention_layers(model: nn.Module) -> list[tuple[int, nn.Module, int]]:
    """Find all attention layers and return `(layer_idx, module, n_heads)` tuples."""
    layers_container = _find_transformer_layers(model)
    if layers_container is None:
        logger.warning("Could not find transformer layers")
        return []

    results: list[tuple[int, nn.Module, int]] = []
    for layer_idx, layer in enumerate(layers_container):
        attn = (
            getattr(layer, "self_attn", None)
            or getattr(layer, "attention", None)
            or getattr(layer, "attn", None)
        )
        if attn is None:
            continue

        n_heads = None
        class_name = type(attn).__name__
        for pattern, config_path in _ATTN_PATTERNS:
            if pattern.lower() not in class_name.lower():
                continue
            try:
                n_heads = int(_resolve_attr_path(model, config_path))
                break
            except AttributeError:
                continue

        if n_heads is None:
            for attr_name in ("num_heads", "n_head", "num_attention_heads", "num_key_value_groups"):
                if hasattr(attn, attr_name):
                    n_heads = int(getattr(attn, attr_name))
                    break

        q_proj = (
            getattr(attn, "q_proj", None)
            or getattr(attn, "query", None)
            or getattr(attn, "c_attn", None)
        )
        if n_heads is None and hasattr(q_proj, "weight"):
            q_rows = q_proj.weight.shape[0]
            candidate = getattr(model, "config", None)
            if candidate is not None and hasattr(candidate, "hidden_size") and candidate.hidden_size > 0:
                hidden_size = int(candidate.hidden_size)
                if q_rows and q_rows <= hidden_size and hidden_size % q_rows == 0:
                    n_heads = q_rows

        if n_heads is None:
            logger.warning("Layer %s: could not determine n_heads, skipping", layer_idx)
            continue

        results.append((layer_idx, attn, n_heads))

    logger.info("Found %s attention layers", len(results))
    return results


class AttentionProfiler:
    """Profiles per-head output magnitude in a dense transformer."""

    def __init__(self, model: nn.Module, model_id: str):
        self.model = model
        self.model_id = model_id
        self._attn_layers = _find_attention_layers(model)
        if not self._attn_layers:
            raise ValueError("No attention layers found")

        self._head_norm_sums: dict[int, list[float]] = {}
        self._head_norm_counts: dict[int, list[int]] = {}
        self._hooks: list[Any] = []
        self._total_tokens = 0
        self._token_count_layer_idx = self._attn_layers[0][0]

        for layer_idx, _, n_heads in self._attn_layers:
            self._head_norm_sums[layer_idx] = [0.0] * n_heads
            self._head_norm_counts[layer_idx] = [0] * n_heads

        self._register_hooks()
        logger.info(
            "AttentionProfiler: %s layers, up to %s heads/layer",
            len(self._attn_layers),
            max((n_heads for _, _, n_heads in self._attn_layers), default=0),
        )

    def _register_hooks(self) -> None:
        for layer_idx, attn, n_heads in self._attn_layers:
            hook = self._make_hook(layer_idx, n_heads)
            target = (
                getattr(attn, "o_proj", None)
                or getattr(attn, "out_proj", None)
                or getattr(attn, "dense", None)
                or getattr(attn, "c_proj", None)
            )
            if target is None:
                logger.warning("Layer %s: no output projection found, skipping hook", layer_idx)
                continue
            self._hooks.append(target.register_forward_hook(hook))

    def _make_hook(self, layer_idx: int, n_heads: int):
        def hook(module, inputs, output):
            if isinstance(inputs, tuple):
                tensor = inputs[0]
            else:
                tensor = inputs

            if tensor is None or tensor.dim() < 3:
                return

            batch_size, seq_len, hidden = tensor.shape
            if hidden % n_heads != 0:
                return

            head_dim = hidden // n_heads
            heads = tensor.reshape(batch_size, seq_len, n_heads, head_dim)
            norms = heads.float().norm(dim=-1).mean(dim=(0, 1))

            for head_idx in range(n_heads):
                self._head_norm_sums[layer_idx][head_idx] += float(norms[head_idx])
                self._head_norm_counts[layer_idx][head_idx] += 1

            if layer_idx == self._token_count_layer_idx:
                self._total_tokens += batch_size * seq_len

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def build_profile(self, calibration_samples: int = 0) -> DenseProfile:
        """Build a dense profile from collected per-head norm statistics."""
        self.remove_hooks()
        layers: dict[int, HeadLayerProfile] = {}
        for layer_idx, _, n_heads in self._attn_layers:
            avg_norms = []
            for total_norm, count in zip(
                self._head_norm_sums[layer_idx],
                self._head_norm_counts[layer_idx],
            ):
                avg_norms.append(total_norm / count if count > 0 else 0.0)

            layers[layer_idx] = HeadLayerProfile(
                layer_idx=layer_idx,
                n_heads=n_heads,
                avg_head_norms=avg_norms,
            )

        return DenseProfile(
            model_id=self.model_id,
            calibration_samples=calibration_samples,
            calibration_tokens=self._total_tokens,
            layers=layers,
        )
