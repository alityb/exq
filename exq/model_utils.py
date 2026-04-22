"""Shared model utilities for ExQ.

Single source of truth for layer discovery, model loading, tokenizer setup,
and artifact key serialization. Every module that needs these should import
from here instead of re-implementing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Attribute paths to search for the transformer layer container.
# Ordered by frequency of occurrence across model architectures.
_LAYER_ATTR_PATHS = (
    "model.layers",
    "model.model.layers",
    "transformer.h",
    "transformer.layers",
    "encoder.layers",
)


def find_transformer_layers(model: nn.Module) -> nn.ModuleList | None:
    """Return the transformer layer container (nn.ModuleList), or None."""
    for attr_path in _LAYER_ATTR_PATHS:
        try:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def find_attention_module(layer: nn.Module) -> nn.Module | None:
    """Return the attention submodule of a transformer layer, or None."""
    return (
        getattr(layer, "self_attn", None)
        or getattr(layer, "attention", None)
        or getattr(layer, "attn", None)
    )


def fix_tokenizer(tokenizer) -> None:
    """Set pad_token to eos_token if missing. Mutates in place."""
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def model_slug(model_id: str) -> str:
    """Convert a HuggingFace model id to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def resolve_offload_folder(offload_folder: str | None = None) -> str:
    """Resolve the accelerate offload directory for model loading."""
    resolved = (
        offload_folder
        or os.environ.get("ExQ_OFFLOAD_DIR")
        or str(Path(tempfile.gettempdir()) / "rpgo_offload")
    )
    Path(resolved).mkdir(parents=True, exist_ok=True)
    return resolved


def load_model_and_tokenizer(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    dtype: Any = torch.float16,
    device_map: str = "auto",
    offload_folder: str | None = None,
):
    """Load a causal LM and tokenizer, ready for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from exq.hf_compat import patch_transformers_remote_code_compat
    patch_transformers_remote_code_compat()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    fix_tokenizer(tokenizer)

    resolved_offload = resolve_offload_folder(offload_folder)
    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "offload_folder": resolved_offload,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        load_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


# -- Artifact key helpers --

def parse_quant_key(key: str) -> tuple[int, int]:
    """Parse 'layer:index' string to (int, int) tuple."""
    parts = key.split(":")
    return int(parts[0]), int(parts[1])


def format_quant_key(layer: int, index: int) -> str:
    """Format (layer, index) tuple as 'layer:index' string."""
    return f"{layer}:{index}"


def load_artifact(path: str) -> dict:
    """Load a JSON artifact file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_moe_assignments(artifact: dict) -> dict[tuple[int, int], str]:
    """Parse MoE quant assignments from an artifact dict.

    Handles both 'quant_assignments' (standard) and 'quant_plan' (legacy).
    """
    raw = artifact.get("quant_assignments") or artifact.get("quant_plan", {})
    return {parse_quant_key(k): v for k, v in raw.items()}
