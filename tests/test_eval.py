"""Tests for evaluation-side Python helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rpgo._core import LayerProfile, RoutingProfile
from rpgo.hf_compat import patch_transformers_remote_code_compat
from rpgo.eval.modeling import apply_precision_to_model, compile_quant_plan, model_slug, resolve_offload_folder
from rpgo.eval.quant_shim import apply_quant_plan_to_model, apply_uniform_int4, quantize_tensor_int4, quantize_tensor_int8
from rpgo.eval.quality import append_eval_result, resolve_benchmark
from scripts.make_results_table import parse_eval_log


def _make_profile(path: Path) -> None:
    """Create a tiny routing profile fixture on disk."""
    profile = RoutingProfile("test/model", 4)
    lp = LayerProfile(0, 4, 2)
    lp.set_expert_count(0, 50)
    lp.set_expert_count(1, 30)
    lp.set_expert_count(2, 15)
    lp.set_expert_count(3, 5)
    lp.finalize()
    profile.add_layer(lp)
    profile.save(str(path))


def test_resolve_benchmark_known_values():
    spec = resolve_benchmark("wikitext2")
    assert spec["dataset_name"] == "wikitext"
    assert spec["dataset_config"] == "wikitext-2-raw-v1"


def test_resolve_benchmark_c4_uses_streaming():
    spec = resolve_benchmark("c4")
    assert spec["dataset_name"] == "allenai/c4"
    assert spec["streaming"] is True


def test_resolve_benchmark_unknown_value():
    with pytest.raises(ValueError, match="unsupported benchmark"):
        resolve_benchmark("unknown")


def test_append_eval_result(tmp_path: Path):
    log_path = tmp_path / "results" / "eval_log.txt"
    append_eval_result(log_path, "model/a", "fp16", "wikitext2", 12.34)
    assert log_path.read_text(encoding="utf-8") == "model/a\tfp16\twikitext2\t12.34\n"


def test_model_slug():
    assert model_slug("zai-org/GLM-4.7-Flash") == "zai-org-glm-4-7-flash"


def test_resolve_offload_folder_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    offload_dir = tmp_path / "offload"
    monkeypatch.setenv("RPGO_OFFLOAD_DIR", str(offload_dir))
    resolved = resolve_offload_folder()
    assert resolved == str(offload_dir)
    assert offload_dir.is_dir()


def test_patch_transformers_remote_code_compat():
    patch_transformers_remote_code_compat()
    from transformers.utils import import_utils

    assert hasattr(import_utils, "is_torch_fx_available")


def test_compile_quant_plan(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    _make_profile(profile_path)
    quant_plan = compile_quant_plan(profile_path)
    assert len(quant_plan) == 4
    assert quant_plan[(0, 0)] in {"BF16", "INT8"}
    assert quant_plan[(0, 3)] == "INT4"


def test_apply_precision_to_model_fp16():
    class DummyModel:
        pass

    stats = apply_precision_to_model(DummyModel(), "fp16")
    assert stats["fp16"] == 1


def test_apply_precision_to_model_rpgo_requires_profile():
    class DummyModel:
        pass

    with pytest.raises(ValueError, match="profile_path"):
        apply_precision_to_model(DummyModel(), "rpgo")


class _DummyFusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 8, 4, dtype=torch.float16))
        self.down_proj = nn.Parameter(torch.randn(4, 4, 8, dtype=torch.float16))


class _DummySharedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False, dtype=torch.float16)


class _DummyGlmMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _DummyFusedExperts()
        self.shared_experts = _DummySharedExperts()


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _DummyGlmMlp()


class _DummyGlmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DummyLayer()])


def test_apply_quant_plan_to_fused_glm_experts():
    model = _DummyGlmModel()
    shared_before = model.model.layers[0].mlp.shared_experts.gate_proj.weight.detach().clone()
    stats = apply_quant_plan_to_model(
        model,
        {(0, 0): "BF16", (0, 1): "INT8", (0, 2): "INT4", (0, 3): "INT4"},
    )
    assert stats == {"bf16": 1, "int8": 1, "int4": 2, "total": 4}
    assert torch.equal(
        model.model.layers[0].mlp.shared_experts.gate_proj.weight,
        shared_before,
    )


def test_apply_uniform_int4_to_fused_glm_experts_counts_experts():
    model = _DummyGlmModel()
    stats = apply_uniform_int4(model)
    assert stats == {"int4": 4, "total": 4}


def test_quantize_tensor_int4_zero_rows_stay_finite():
    weight = torch.zeros((4, 128), dtype=torch.float16)
    quantized = quantize_tensor_int4(weight)
    assert torch.isfinite(quantized).all()
    assert torch.equal(quantized, weight)


def test_quantize_tensor_int8_zero_rows_stay_finite():
    weight = torch.zeros((4, 128), dtype=torch.float16)
    quantized = quantize_tensor_int8(weight)
    assert torch.isfinite(quantized).all()
    assert torch.equal(quantized, weight)


def test_make_results_table_roundtrip(tmp_path: Path):
    log_path = tmp_path / "eval_log.txt"
    log_path.write_text(
        "model/a\tfp16\twikitext2\t8.1\n"
        "model/a\tfp16\tc4\t9.2\n"
        "model/a\tint4\twikitext2\t8.5\n",
        encoding="utf-8",
    )
    records = parse_eval_log(log_path)
    assert "model/a" in records
    assert records["model/a"]["fp16"]["wikitext2"] == 8.1
    assert records["model/a"]["fp16"]["c4"] == 9.2
    assert records["model/a"]["int4"]["wikitext2"] == 8.5
