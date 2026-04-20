"""Tests for dense quantization planning."""

from __future__ import annotations

from rpgo.compiler.dense_quant_planner import compute_thresholds, plan_dense_quant
from rpgo.profiler.dense_profile import DenseProfile, HeadLayerProfile


def _make_dense_profile() -> DenseProfile:
    return DenseProfile(
        model_id="mock-dense",
        calibration_samples=10,
        calibration_tokens=1000,
        layers={
            0: HeadLayerProfile(0, 4, [10.0, 5.0, 2.0, 1.0]),
            1: HeadLayerProfile(1, 4, [8.0, 4.0, 2.0, 1.0]),
        },
    )


def test_compute_thresholds_orders_values():
    hot, warm, cold = compute_thresholds(_make_dense_profile())
    assert hot > warm > cold >= 0.0


def test_plan_dense_quant_assigns_expected_tiers():
    profile = _make_dense_profile()
    plan = plan_dense_quant(profile, hot_threshold=7.0, warm_threshold=3.0, cold_threshold=1.5)
    assert plan.layer_plans[0].assignments[0] == "BF16"
    assert plan.layer_plans[0].assignments[1] == "INT8"
    assert plan.layer_plans[0].assignments[2] == "INT4"
    assert plan.summary["total_heads"] == 8
