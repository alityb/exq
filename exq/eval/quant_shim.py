"""Per-expert quantization shim: applies ExQ quant plans to MoE models."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _quantize_parameter_data(param: nn.Parameter, precision: str, expert_idx: int | None = None) -> None:
    """Quantize a parameter or one expert slice of a fused expert tensor."""
    with torch.no_grad():
        if expert_idx is None:
            weight = param.data
            if precision == "INT8":
                param.data = quantize_tensor_int8(weight)
            elif precision == "INT4":
                param.data = quantize_tensor_int4(weight)
            return

        if precision == "INT8":
            param.data[expert_idx] = quantize_tensor_int8(param.data[expert_idx])
        elif precision == "INT4":
            param.data[expert_idx] = quantize_tensor_int4(param.data[expert_idx])


def _apply_quant_plan_to_fused_glm_experts(
    mlp: nn.Module,
    layer_idx: int,
    quant_plan: dict[tuple[int, int], str],
    stats: dict[str, int],
) -> None:
    """Apply a quant plan to GLM-style fused expert tensors."""
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return

    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        return

    n_experts = gate_up_proj.shape[0]
    for expert_idx in range(n_experts):
        precision = quant_plan.get((layer_idx, expert_idx), "INT4")
        stats["total"] += 1

        if precision == "BF16":
            stats["bf16"] += 1
            continue

        _quantize_parameter_data(gate_up_proj, precision, expert_idx)
        _quantize_parameter_data(down_proj, precision, expert_idx)
        if precision == "INT8":
            stats["int8"] += 1
        elif precision == "INT4":
            stats["int4"] += 1


def _apply_uniform_quant_to_fused_glm_experts(
    mlp: nn.Module,
    precision: str,
) -> int:
    """Apply uniform quantization to GLM-style fused expert tensors."""
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return 0

    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        return 0

    n_experts = gate_up_proj.shape[0]
    for expert_idx in range(n_experts):
        _quantize_parameter_data(gate_up_proj, precision, expert_idx)
        _quantize_parameter_data(down_proj, precision, expert_idx)
    return n_experts


def quantize_tensor_int8(weight: torch.Tensor) -> torch.Tensor:
    """Simulate INT8 weight-only quantization via per-channel RTN.

    Quantize to int8 range then dequantize back to fp16.
    This simulates the quality impact without needing a custom kernel.
    """
    weight_fp32 = weight.float()

    # Per-channel (output dim) quantization in fp32 to avoid fp16 underflow.
    scale = weight_fp32.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=torch.finfo(torch.float32).eps)
    quantized = (weight_fp32 / scale).round().clamp(-128, 127)
    return (quantized * scale).to(weight.dtype)


def quantize_tensor_int4(weight: torch.Tensor) -> torch.Tensor:
    """Simulate INT4 weight-only quantization via per-group RTN.

    Uses group_size=128 (standard for GPTQ/AWQ-style INT4).
    Quantize to [-8, 7] range then dequantize back to fp16.
    """
    group_size = 128
    orig_shape = weight.shape
    weight_fp32 = weight.float()

    # Pad to multiple of group_size
    if weight_fp32.shape[-1] % group_size != 0:
        pad = group_size - (weight_fp32.shape[-1] % group_size)
        weight_fp32 = torch.nn.functional.pad(weight_fp32, (0, pad))

    # Reshape into groups
    w = weight_fp32.reshape(-1, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=torch.finfo(torch.float32).eps)
    quantized = (w / scale).round().clamp(-8, 7)
    dequantized = (quantized * scale).reshape(weight_fp32.shape)

    # Remove padding
    if dequantized.shape != orig_shape:
        dequantized = dequantized[..., : orig_shape[-1]]

    return dequantized.to(weight.dtype)


def apply_quant_plan_to_model(
    model: nn.Module,
    quant_plan: dict[tuple[int, int], str],
) -> dict[str, int]:
    """Apply ExQ quantization plan to expert weights in-place.

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

        if hasattr(mlp.experts, "gate_up_proj") and hasattr(mlp.experts, "down_proj"):
            _apply_quant_plan_to_fused_glm_experts(mlp, layer_idx, quant_plan, stats)
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


def apply_uniform_quant(model: nn.Module, precision: str = "INT4") -> dict[str, int]:
    """Apply uniform quantization to ALL expert weights at the given precision."""
    quantize_fn = quantize_tensor_int4 if precision == "INT4" else quantize_tensor_int8
    n_experts = 0
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        if hasattr(mlp.experts, "gate_up_proj") and hasattr(mlp.experts, "down_proj"):
            n_experts += _apply_uniform_quant_to_fused_glm_experts(mlp, precision)
            continue
        for expert in mlp.experts:
            for name, param in expert.named_parameters():
                if "weight" not in name:
                    continue
                with torch.no_grad():
                    param.data = quantize_fn(param.data)
            n_experts += 1

    logger.info(f"Applied uniform {precision} to {n_experts} experts")
    return {precision.lower(): n_experts, "total": n_experts}


def apply_uniform_int4(model: nn.Module) -> dict[str, int]:
    return apply_uniform_quant(model, "INT4")


def apply_uniform_int8(model: nn.Module) -> dict[str, int]:
    return apply_uniform_quant(model, "INT8")
