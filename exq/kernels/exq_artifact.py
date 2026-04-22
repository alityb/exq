"""
ExQ artifact reader: loads compiled artifact + routing profile into
kernel-ready tensors for the frequency-aware Triton kernels.

Actual OLMoE-1B-7B data (from profiles/olmoe-1b-7b-0924-256.json):
  - 64 experts per layer, top-2 routing
  - Max activation freq: 0.094 (no expert reaches 0.10 threshold)
  - Frequency range: [0.0006, 0.094], mean = 0.0156 (= 1/64, uniform)
  - Quant tier distribution: 585 INT4, 438 INT8, 1 BF16 across 1024 expert-layer pairs

The tile sizing schedule is calibrated to actual OLMoE frequencies,
not the idealised thresholds in the prompt spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PRECISION_TO_INT = {"BF16": 0, "INT8": 1, "INT4": 2}
INT_TO_PRECISION = {0: "BF16", 1: "INT8", 2: "INT4"}

# Tile-M schedule based on quant tier (primary signal) and frequency (secondary).
# In the memory-bound GEMV regime that dominates decode inference, tile-M
# affects padding waste and occupancy more than bandwidth.
# Calibrated for OLMoE-1B-7B at batch=8, seqlen=64 (avg ~16 tokens per expert).
_TIER_TO_BLOCK_M = {
    0: 128,   # BF16 (hot): large tiles, max occupancy
    1: 64,    # INT8 (warm): medium tiles
    2: 32,    # INT4 (cold): small tiles, minimal padding waste
}

# Secondary frequency-based override within each tier
def freq_to_block_m(freq: float, tier: int) -> int:
    """Map (activation_freq, quant_tier) -> recommended BLOCK_M."""
    base = _TIER_TO_BLOCK_M[tier]
    # Within INT8 tier, further split by frequency
    if tier == 1:
        return 64 if freq >= 0.03 else 32
    return base


@dataclass
class ExpertProfile:
    """Per-layer expert metadata in kernel-ready form."""

    n_layers: int
    n_experts_per_layer: int
    # All tensors are [n_layers, n_experts_per_layer] on CUDA
    tiers: torch.Tensor      # int8:  0=BF16, 1=INT8, 2=INT4
    freqs: torch.Tensor      # float32: activation frequency
    block_m: torch.Tensor    # int32:  recommended BLOCK_M per expert
    hot_mask: torch.Tensor   # bool:  True = BF16 tier

    @property
    def n_experts_total(self) -> int:
        return self.n_layers * self.n_experts_per_layer

    def layer_profile(self, layer_idx: int) -> dict:
        """Return per-expert tensors for one layer (still on CUDA)."""
        return {
            "tiers":   self.tiers[layer_idx],
            "freqs":   self.freqs[layer_idx],
            "block_m": self.block_m[layer_idx],
            "hot_mask": self.hot_mask[layer_idx],
        }

    def hot_expert_count(self, layer_idx: int) -> int:
        return int(self.hot_mask[layer_idx].sum().item())

    def expected_memory_ratio(self) -> float:
        """Expected memory footprint vs all-BF16."""
        tier_ratios = torch.tensor([1.0, 0.5, 0.25], device=self.tiers.device)
        return float(tier_ratios[self.tiers.long()].mean().item())

    def precision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t_int, name in INT_TO_PRECISION.items():
            counts[name] = int((self.tiers == t_int).sum().item())
        return counts

    def block_m_distribution(self) -> dict[int, int]:
        """Count experts per BLOCK_M value (for diagnostics)."""
        dist: dict[int, int] = {}
        for bm in self.block_m.flatten().tolist():
            bm_int = int(bm)
            dist[bm_int] = dist.get(bm_int, 0) + 1
        return dict(sorted(dist.items()))


def load_exq_artifact(
    artifact_path: str,
    profile_path: str | None = None,
    device: str = "cuda",
) -> ExpertProfile:
    """
    Load an ExQ compiled artifact (and optionally a routing profile)
    into kernel-ready tensors.

    Args:
        artifact_path:  Path to a compiled artifact JSON
                        (e.g. artifacts/olmoe-1b-7b-0924-256.json).
        profile_path:   Path to the routing profile JSON
                        (e.g. profiles/olmoe-1b-7b-0924-256.json).
                        If None, tries standard path conventions, then falls
                        back to tier-derived frequency estimates.
        device:         CUDA device string (default "cuda").

    Returns:
        ExpertProfile with tensors on `device`.
    """
    with open(artifact_path, encoding="utf-8") as f:
        artifact = json.load(f)

    assignments: dict[str, str] = artifact.get("quant_assignments", {})
    if not assignments:
        raise ValueError(f"Artifact at {artifact_path} has no quant_assignments")

    # Determine grid dimensions from the key set
    layer_ids, expert_ids = set(), set()
    for key in assignments:
        l_str, e_str = key.split(":")
        layer_ids.add(int(l_str))
        expert_ids.add(int(e_str))

    n_layers  = max(layer_ids) + 1
    n_experts = max(expert_ids) + 1

    # Fill tier array (default INT4 for missing entries)
    tiers_np = np.full((n_layers, n_experts), PRECISION_TO_INT["INT4"], dtype=np.int8)
    for key, prec in assignments.items():
        l, e = map(int, key.split(":"))
        tiers_np[l, e] = PRECISION_TO_INT.get(prec, PRECISION_TO_INT["INT4"])

    # Load activation frequencies from profile
    freqs_np = np.zeros((n_layers, n_experts), dtype=np.float32)
    profile_loaded = False

    # Try explicit path first, then standard conventions
    candidate_paths: list[str | None] = [profile_path]
    if profile_path is None:
        artifact_stem = Path(artifact_path).stem
        candidate_paths += [
            str(Path("profiles") / f"{artifact_stem}.json"),
            str(Path("profiles") / "olmoe-1b-7b-0924-256.json"),
        ]

    for cpath in candidate_paths:
        if cpath and Path(cpath).exists():
            with open(cpath, encoding="utf-8") as f:
                profile_data = json.load(f)
            for layer_str, layer_data in profile_data.get("layers", {}).items():
                l = int(layer_str)
                if l >= n_layers:
                    continue
                for stats in layer_data.get("expert_stats", []):
                    e = stats.get("expert_id", 0)
                    if e < n_experts:
                        freqs_np[l, e] = float(stats.get("activation_freq", 0.0))
            profile_loaded = True
            break

    if not profile_loaded:
        # Tier-derived fallback: BF16→high, INT8→medium, INT4→low
        tier_to_freq = {0: 0.15, 1: 0.05, 2: 0.01}
        for l in range(n_layers):
            for e in range(n_experts):
                freqs_np[l, e] = tier_to_freq[int(tiers_np[l, e])]

    # Compute per-expert BLOCK_M
    block_m_np = np.empty((n_layers, n_experts), dtype=np.int32)
    for l in range(n_layers):
        for e in range(n_experts):
            block_m_np[l, e] = freq_to_block_m(
                float(freqs_np[l, e]), int(tiers_np[l, e])
            )

    hot_mask_np = (tiers_np == PRECISION_TO_INT["BF16"])

    return ExpertProfile(
        n_layers=n_layers,
        n_experts_per_layer=n_experts,
        tiers=torch.from_numpy(tiers_np).to(device),
        freqs=torch.from_numpy(freqs_np).to(device),
        block_m=torch.from_numpy(block_m_np).to(device),
        hot_mask=torch.from_numpy(hot_mask_np).to(device),
    )


def print_profile_summary(profile: ExpertProfile) -> None:
    """Print a human-readable summary of the expert profile."""
    print(f"Expert profile: {profile.n_layers} layers × "
          f"{profile.n_experts_per_layer} experts "
          f"({profile.n_experts_total} total)")

    counts = profile.precision_counts()
    print(f"Precision: {counts.get('BF16',0)} BF16  "
          f"{counts.get('INT8',0)} INT8  "
          f"{counts.get('INT4',0)} INT4")
    print(f"Memory vs all-BF16: {profile.expected_memory_ratio():.1%}")

    bm_dist = profile.block_m_distribution()
    print(f"BLOCK_M distribution: {bm_dist}")

    print("Per-layer breakdown (first 4 layers):")
    for l in range(min(4, profile.n_layers)):
        lp = profile.layer_profile(l)
        tiers_l = lp["tiers"]
        freqs_l = lp["freqs"]
        print(
            f"  Layer {l:2d}: "
            f"BF16={int((tiers_l==0).sum()):3d}  "
            f"INT8={int((tiers_l==1).sum()):3d}  "
            f"INT4={int((tiers_l==2).sum()):3d}  "
            f"max_freq={float(freqs_l.max()):.4f}  "
            f"mean_freq={float(freqs_l.mean()):.4f}"
        )
    if profile.n_layers > 4:
        print(f"  ... ({profile.n_layers - 4} more layers, same pattern)")
