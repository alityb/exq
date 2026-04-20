"""Per-expert quantization runtime shim.

Applies an R-PGO quantization plan to a loaded MoE model by quantizing
individual expert weights in-place according to their frequency tier:
  - BF16: keep as-is (fp16 on GPU)
  - INT8: dynamic quantization (torch int8 weight-only)
  - INT4: simulate via RTN 4-bit quantization (round-to-nearest)

This is the runtime component that enables perplexity comparison between:
  1. Uniform INT4 (all experts quantized the same)
  2. R-PGO stratified (hot=fp16, warm=int8, cold=int4)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def quantize_tensor_int8(weight: torch.Tensor) -> torch.Tensor:
    """Simulate INT8 weight-only quantization via per-channel RTN.

    Quantize to int8 range then dequantize back to fp16.
    This simulates the quality impact without needing a custom kernel.
    """
    # Per-channel (output dim) quantization
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)
    quantized = (weight / scale).round().clamp(-128, 127)
    return (quantized * scale).to(weight.dtype)


def quantize_tensor_int4(weight: torch.Tensor) -> torch.Tensor:
    """Simulate INT4 weight-only quantization via per-group RTN.

    Uses group_size=128 (standard for GPTQ/AWQ-style INT4).
    Quantize to [-8, 7] range then dequantize back to fp16.
    """
    group_size = 128
    orig_shape = weight.shape

    # Pad to multiple of group_size
    if weight.shape[-1] % group_size != 0:
        pad = group_size - (weight.shape[-1] % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))

    # Reshape into groups
    w = weight.reshape(-1, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-8)
    quantized = (w / scale).round().clamp(-8, 7)
    dequantized = (quantized * scale).reshape(weight.shape)

    # Remove padding
    if dequantized.shape != orig_shape:
        dequantized = dequantized[..., : orig_shape[-1]]

    return dequantized.to(torch.float16)


def apply_quant_plan_to_model(
    model: nn.Module,
    quant_plan: dict[tuple[int, int], str],
) -> dict[str, int]:
    """Apply R-PGO quantization plan to expert weights in-place.

    For each MoE layer, quantizes expert weight matrices according to
    the plan's precision assignment.

    Args:
        model: The loaded MoE model (fp16).
        quant_plan: {(layer_idx, expert_idx): "BF16"|"INT8"|"INT4"}

    Returns:
        Stats dict: {"bf16": n, "int8": n, "int4": n, "total": n}
    """
    stats = {"bf16": 0, "int8": 0, "int4": 0, "total": 0}

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue

        for expert_idx, expert in enumerate(mlp.experts):
            key = (layer_idx, expert_idx)
            precision = quant_plan.get(key, "INT4")  # default to INT4 for unmapped
            stats["total"] += 1

            if precision == "BF16":
                stats["bf16"] += 1
                continue  # keep as-is

            # Find all weight tensors in this expert
            for name, param in expert.named_parameters():
                if "weight" not in name:
                    continue
                with torch.no_grad():
                    if precision == "INT8":
                        param.data = quantize_tensor_int8(param.data)
                        stats["int8"] += 1
                    elif precision == "INT4":
                        param.data = quantize_tensor_int4(param.data)
                        stats["int4"] += 1

            # Only count precision once per expert (not per weight tensor)
            if precision == "INT8":
                stats["int8"] = stats["int8"] - len([p for n, p in expert.named_parameters() if "weight" in n]) + 1
            elif precision == "INT4":
                stats["int4"] = stats["int4"] - len([p for n, p in expert.named_parameters() if "weight" in n]) + 1

    logger.info(f"Applied quant plan: {stats}")
    return stats


def apply_uniform_int4(model: nn.Module) -> dict[str, int]:
    """Apply uniform INT4 quantization to ALL expert weights.

    This is the baseline: what llama.cpp / bitsandbytes does uniformly.
    """
    n_experts = 0
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        for expert in mlp.experts:
            for name, param in expert.named_parameters():
                if "weight" not in name:
                    continue
                with torch.no_grad():
                    param.data = quantize_tensor_int4(param.data)
            n_experts += 1

    logger.info(f"Applied uniform INT4 to {n_experts} experts")
    return {"int4": n_experts, "total": n_experts}


def apply_uniform_int8(model: nn.Module) -> dict[str, int]:
    """Apply uniform INT8 quantization to ALL expert weights."""
    n_experts = 0
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        for expert in mlp.experts:
            for name, param in expert.named_parameters():
                if "weight" not in name:
                    continue
                with torch.no_grad():
                    param.data = quantize_tensor_int8(param.data)
            n_experts += 1

    logger.info(f"Applied uniform INT8 to {n_experts} experts")
    return {"int8": n_experts, "total": n_experts}
