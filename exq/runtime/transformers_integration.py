"""ExQ runtime integration for HuggingFace transformers.

Patches a loaded model to use an ExQ compiled artifact so that
model.generate() runs with the compiler's decisions baked in.

Usage:
    model = exq_patch(model, "artifacts/model.json")
    # or:
    model, tokenizer = load_exq_model(model_id, artifact_path)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch

from exq.compiler.dense_quant_planner import DenseQuantPlan
from exq.eval.dense_quant_apply import apply_dense_quant
from exq.eval.quant_shim import apply_quant_plan_to_model
from exq.model_utils import (
    find_transformer_layers,
    fix_tokenizer,
    load_artifact,
    load_model_and_tokenizer,
    parse_moe_assignments,
)

logger = logging.getLogger(__name__)


def exq_patch(model, artifact_path: str) -> object:
    """Patch a loaded model to use an ExQ compiled artifact.

    Applies quantization plan and registers prefetch hooks.
    Returns the patched model.
    """
    artifact = load_artifact(artifact_path)
    artifact_type = artifact.get("type", "moe_expert_quant")

    if artifact_type == "dense_head_quant":
        plan = DenseQuantPlan.from_artifact(artifact)
        model = apply_dense_quant(model, plan)
        quant_stats = plan.summary
        logger.info("ExQ dense patch: %d layers, %s", len(plan.layer_plans), quant_stats)
    else:
        assignments = parse_moe_assignments(artifact)
        quant_stats = apply_quant_plan_to_model(model, assignments)
        logger.info("ExQ MoE patch: %d experts, %s", len(assignments), quant_stats)

    prefetch_entries = artifact.get("prefetch_schedule", [])
    if not prefetch_entries and artifact_type != "dense_head_quant":
        prefetch_entries = _build_simple_prefetch_schedule(artifact)

    if prefetch_entries:
        n_hooks = _register_prefetch_hooks(model, prefetch_entries)
        logger.info("Prefetch schedule: %d entries, %d hooks", len(prefetch_entries), n_hooks)

    model._exq_artifact = artifact_path
    model._exq_type = artifact_type
    model._exq_quant_stats = quant_stats
    return model


def _build_simple_prefetch_schedule(artifact: dict) -> list[list[int]]:
    """Build a conservative prefetch schedule from quant assignments."""
    quant = artifact.get("quant_assignments") or artifact.get("quant_plan", {})
    layers: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key, prec in quant.items():
        layer, expert = map(int, key.split(":"))
        layers[layer].append((expert, prec))

    schedule = []
    sorted_layers = sorted(layers.keys())
    for i in range(len(sorted_layers) - 1):
        src_layer = sorted_layers[i]
        dst_layer = sorted_layers[i + 1]
        src_hot = [e for e, p in layers[src_layer] if p in ("BF16", "INT8")]
        dst_hot = [e for e, p in layers[dst_layer] if p in ("BF16", "INT8")]
        for src_e in src_hot[:4]:
            for dst_e in dst_hot[:4]:
                priority = 0 if quant.get(f"{dst_layer}:{dst_e}") == "BF16" else 1
                schedule.append([src_layer, src_e, dst_layer, dst_e, priority])
    return schedule


def _register_prefetch_hooks(model, schedule: list) -> int:
    """Register async prefetch forward hooks from the compiled schedule."""
    prefetch_map: dict[int, list] = defaultdict(list)
    for entry in schedule:
        prefetch_map[entry[0]].append(entry)

    if not torch.cuda.is_available():
        return 0

    stream = torch.cuda.Stream()
    layers = find_transformer_layers(model)
    if layers is None:
        logger.warning("Could not find transformer layers for prefetch hooks")
        return 0

    hooks_registered = 0
    for layer_idx, layer in enumerate(layers):
        if layer_idx not in prefetch_map:
            continue
        entries = prefetch_map[layer_idx]

        def _make_hook(layer_entries, layer_list):
            def hook(module, input, output):
                with torch.cuda.stream(stream):
                    for entry in layer_entries:
                        dst_layer_idx, dst_expert_idx = entry[2], entry[3]
                        try:
                            target_layer = layer_list[dst_layer_idx]
                            mlp = getattr(target_layer, "mlp", None) or getattr(target_layer, "ffn", None)
                            if mlp is None:
                                continue
                            experts = getattr(mlp, "experts", None)
                            if experts is None:
                                continue
                            if hasattr(experts, "__getitem__") and not hasattr(experts, "gate_up_proj"):
                                for param in experts[dst_expert_idx].parameters():
                                    if param.device.type == "cpu":
                                        param.data = param.data.cuda(non_blocking=True)
                        except (IndexError, AttributeError):
                            pass
            return hook

        layer.register_forward_hook(_make_hook(entries, layers))
        hooks_registered += 1
    return hooks_registered


def load_exq_model(
    model_id: str,
    artifact_path: str,
    torch_dtype=torch.float16,
    device: str = "auto",
    **model_kwargs,
):
    """Load a model and apply an ExQ artifact in one call."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from exq.hf_compat import patch_transformers_remote_code_compat

    patch_transformers_remote_code_compat()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    fix_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="cpu",
        trust_remote_code=True, **model_kwargs,
    )
    model = exq_patch(model, artifact_path)

    if device == "auto":
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        model = model.to(device)

    model.eval()
    return model, tokenizer
