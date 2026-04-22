"""Dense quantization planner: frequency-stratified precision per attention head."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from exq.profiler.dense_profile import DenseProfile


@dataclass
class HeadQuantPlan:
    layer_idx: int
    assignments: dict[int, str]
    estimated_memory_ratio: float


@dataclass
class DenseQuantPlan:
    model_id: str
    layer_plans: dict[int, HeadQuantPlan]

    @classmethod
    def from_artifact(cls, path_or_dict: str | Path | dict, model_id: str = "") -> DenseQuantPlan:
        """Construct from an artifact JSON file or already-loaded dict."""
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, encoding="utf-8") as f:
                artifact = json.load(f)
        else:
            artifact = path_or_dict

        raw = artifact.get("quant_assignments", {})
        layer_heads: dict[int, dict[int, str]] = defaultdict(dict)
        for key, precision in raw.items():
            layer_idx, head_idx = map(int, key.split(":"))
            layer_heads[layer_idx][head_idx] = precision

        return cls(
            model_id=artifact.get("model_id", model_id),
            layer_plans={
                idx: HeadQuantPlan(layer_idx=idx, assignments=heads, estimated_memory_ratio=1.0)
                for idx, heads in layer_heads.items()
            },
        )

    @property
    def summary(self) -> dict[str, float | int]:
        all_assignments = [
            precision
            for layer_plan in self.layer_plans.values()
            for precision in layer_plan.assignments.values()
        ]
        counts = Counter(all_assignments)
        total = len(all_assignments)
        return {
            "total_heads": total,
            "BF16": counts.get("BF16", 0),
            "INT8": counts.get("INT8", 0),
            "INT4": counts.get("INT4", 0),
            "BF16_fraction": counts.get("BF16", 0) / total if total else 0.0,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "type": "dense_head_quant",
            "quant_assignments": {
                f"{layer_idx}:{head_idx}": precision
                for layer_idx, layer_plan in self.layer_plans.items()
                for head_idx, precision in layer_plan.assignments.items()
            },
        }


def compute_thresholds(profile: DenseProfile) -> tuple[float, float, float]:
    """Auto-compute HOT/WARM/COLD thresholds from the head-norm distribution."""
    all_norms = [
        norm
        for layer_profile in profile.layers.values()
        for norm in layer_profile.avg_head_norms
        if norm > 0
    ]
    if not all_norms:
        return 1.0, 0.5, 0.1

    mean = statistics.mean(all_norms)
    std = statistics.stdev(all_norms) if len(all_norms) > 1 else mean * 0.3
    hot_threshold = mean + 1.5 * std
    warm_threshold = mean
    cold_threshold = max(mean - 0.5 * std, 0.0)
    return hot_threshold, warm_threshold, cold_threshold


def plan_dense_quant(
    profile: DenseProfile,
    hot_threshold: float | None = None,
    warm_threshold: float | None = None,
    cold_threshold: float | None = None,
) -> DenseQuantPlan:
    """Assign precision to each attention head based on its average output norm."""
    if any(threshold is None for threshold in (hot_threshold, warm_threshold, cold_threshold)):
        hot_threshold, warm_threshold, cold_threshold = compute_thresholds(profile)

    bytes_ratio = {"BF16": 1.0, "INT8": 0.5, "INT4": 0.25}
    layer_plans: dict[int, HeadQuantPlan] = {}

    for layer_idx, layer_profile in profile.layers.items():
        assignments: dict[int, str] = {}
        total_mem = 0.0
        bf16_mem = 0.0

        for head_idx, norm in enumerate(layer_profile.avg_head_norms):
            if norm >= hot_threshold:
                precision = "BF16"
            elif norm >= warm_threshold:
                precision = "INT8"
            elif norm >= cold_threshold:
                precision = "INT4"
            else:
                precision = "INT4"

            assignments[head_idx] = precision
            total_mem += bytes_ratio[precision]
            bf16_mem += bytes_ratio["BF16"]

        layer_plans[layer_idx] = HeadQuantPlan(
            layer_idx=layer_idx,
            assignments=assignments,
            estimated_memory_ratio=total_mem / bf16_mem if bf16_mem else 1.0,
        )

    return DenseQuantPlan(model_id=profile.model_id, layer_plans=layer_plans)
