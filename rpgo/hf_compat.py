"""Compatibility helpers for mixed remote-code model environments."""

from __future__ import annotations


def patch_transformers_remote_code_compat() -> None:
    """Patch missing `transformers` symbols expected by older remote code."""
    from transformers.utils import import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import torch

        def is_torch_fx_available() -> bool:
            return hasattr(torch, "fx")

        import_utils.is_torch_fx_available = is_torch_fx_available
