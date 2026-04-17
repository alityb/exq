#!/usr/bin/env python3
"""CLI: Inspect routing graph statistics from a profile.

Usage:
    python scripts/inspect_routing_graph.py --profile routing_profile.json
"""

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Inspect routing graph")
    parser.add_argument("--profile", type=str, required=True, help="Path to routing_profile.json")
    args = parser.parse_args()

    from rpgo._core import RoutingProfile

    profile = RoutingProfile.load(args.profile)
    print(f"Model: {profile.model_id}")
    print(f"Calibration samples: {profile.calibration_samples}")
    print(f"Calibration tokens: {profile.calibration_tokens}")
    print(f"MoE layers: {profile.n_layers}")
    print()

    warnings = profile.validate()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
        print()

    for layer_idx in profile.moe_layer_indices():
        lp = profile.get_layer(layer_idx)
        freqs = lp.get_activation_freqs()
        counts = lp.get_activation_counts()
        total = sum(counts)

        # Classify experts
        hot = sum(1 for f in freqs if f >= 0.10)
        warm = sum(1 for f in freqs if 0.03 <= f < 0.10)
        cold = sum(1 for f in freqs if 0.005 <= f < 0.03)
        frozen = sum(1 for f in freqs if f < 0.005)

        print(f"Layer {layer_idx}: "
              f"entropy={lp.routing_entropy:.4f}, "
              f"experts={lp.n_experts}, top_k={lp.top_k}, "
              f"activations={total}")
        print(f"  HOT={hot}, WARM={warm}, COLD={cold}, FROZEN={frozen}")

        # Top-5 experts
        indexed_freqs = sorted(enumerate(freqs), key=lambda x: -x[1])[:5]
        top5_str = ", ".join(f"E{i}={f:.3f}" for i, f in indexed_freqs)
        print(f"  Top-5: {top5_str}")
        print()


if __name__ == "__main__":
    main()
