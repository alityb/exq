"""DenseProfile: data structures for attention head profiling results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HeadLayerProfile:
    """Profile for attention heads in a single transformer layer."""

    layer_idx: int
    n_heads: int
    avg_head_norms: list[float]

    @property
    def head_frequencies(self) -> list[float]:
        """Normalize per-head norms into a probability distribution."""
        total = sum(self.avg_head_norms)
        if total == 0:
            return [1.0 / self.n_heads] * self.n_heads
        return [norm / total for norm in self.avg_head_norms]

    @property
    def entropy(self) -> float:
        """Return Shannon entropy of the head-norm distribution in nats."""
        entropy = 0.0
        for prob in self.head_frequencies:
            if prob > 0:
                entropy -= prob * math.log(prob)
        return entropy

    @property
    def max_entropy(self) -> float:
        """Return the maximum entropy for this number of heads."""
        return math.log(self.n_heads) if self.n_heads > 1 else 1.0

    @property
    def normalized_entropy(self) -> float:
        """Return entropy normalized into [0, 1]."""
        if self.max_entropy <= 0:
            return 0.0
        return self.entropy / self.max_entropy


@dataclass
class DenseProfile:
    """Full attention-head profile across all layers of a dense model."""

    model_id: str
    calibration_samples: int
    calibration_tokens: int = 0
    layers: dict[int, HeadLayerProfile] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Persist the dense profile as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": self.model_id,
            "calibration_samples": self.calibration_samples,
            "calibration_tokens": self.calibration_tokens,
            "type": "dense_attention",
            "layers": {
                str(layer_idx): {
                    "layer_idx": layer_profile.layer_idx,
                    "n_heads": layer_profile.n_heads,
                    "avg_head_norms": layer_profile.avg_head_norms,
                }
                for layer_idx, layer_profile in self.layers.items()
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DenseProfile":
        """Load a dense profile from JSON."""
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        layers = {
            int(layer_idx): HeadLayerProfile(
                layer_idx=layer_data["layer_idx"],
                n_heads=layer_data["n_heads"],
                avg_head_norms=layer_data["avg_head_norms"],
            )
            for layer_idx, layer_data in data["layers"].items()
        }
        return cls(
            model_id=data["model_id"],
            calibration_samples=data["calibration_samples"],
            calibration_tokens=data.get("calibration_tokens", 0),
            layers=layers,
        )

    def summary(self) -> dict[str, float | int | str]:
        """Return a compact summary for logging or inspection."""
        entropies = [layer.normalized_entropy for layer in self.layers.values()]
        return {
            "model_id": self.model_id,
            "n_layers": len(self.layers),
            "total_heads": sum(layer.n_heads for layer in self.layers.values()),
            "avg_normalized_entropy": (
                sum(entropies) / len(entropies) if entropies else 0.0
            ),
            "calibration_tokens": self.calibration_tokens,
        }

    def validate(self) -> list[str]:
        """Validate internal consistency and return warning strings."""
        warnings: list[str] = []
        for layer_idx, layer_profile in self.layers.items():
            if len(layer_profile.avg_head_norms) != layer_profile.n_heads:
                warnings.append(
                    f"Layer {layer_idx}: norm list length {len(layer_profile.avg_head_norms)} "
                    f"!= n_heads {layer_profile.n_heads}"
                )
        return warnings
