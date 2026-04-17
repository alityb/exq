#!/usr/bin/env python3
"""CLI: Inspect routing graph statistics from a profile.

Usage:
    python scripts/inspect_routing_graph.py --profile routing_profile.json
    python scripts/inspect_routing_graph.py --profile routing_profile.json --kernels
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Inspect routing graph")
    parser.add_argument("--profile", type=str, required=True, help="Path to routing_profile.json")
    parser.add_argument("--kernels", action="store_true", help="Show pseudo-kernel prefetch schedules")
    args = parser.parse_args()

    from rpgo._core import (
        RoutingProfile,
        CompilerPipeline,
        py_build_routing_graph,
        py_graph_summary,
    )

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

    # Per-layer stats from profile
    for layer_idx in profile.moe_layer_indices():
        lp = profile.get_layer(layer_idx)
        freqs = lp.get_activation_freqs()
        counts = lp.get_activation_counts()
        total = sum(counts)

        hot = sum(1 for f in freqs if f >= 0.10)
        warm = sum(1 for f in freqs if 0.03 <= f < 0.10)
        cold = sum(1 for f in freqs if 0.005 <= f < 0.03)
        frozen = sum(1 for f in freqs if f < 0.005)

        print(f"Layer {layer_idx}: "
              f"entropy={lp.routing_entropy:.4f}, "
              f"experts={lp.n_experts}, top_k={lp.top_k}, "
              f"activations={total}")
        print(f"  HOT={hot}, WARM={warm}, COLD={cold}, FROZEN={frozen}")

        indexed_freqs = sorted(enumerate(freqs), key=lambda x: -x[1])[:5]
        top5_str = ", ".join(f"E{i}={f:.3f}" for i, f in indexed_freqs)
        print(f"  Top-5: {top5_str}")
        print()

    # Graph-level summary
    graph = py_build_routing_graph(profile)
    summary = py_graph_summary(graph)
    print("=== Graph Summary ===")
    print(f"  Nodes: {summary['total_nodes']}, Edges: {summary['total_edges']}")
    print(f"  HOT: {summary['total_hot']}, WARM: {summary['total_warm']}, "
          f"COLD: {summary['total_cold']}, FROZEN: {summary['total_frozen']}")
    print(f"  Entropy: avg={summary['avg_entropy']:.4f}, "
          f"min={summary['min_entropy']:.4f}, max={summary['max_entropy']:.4f}")
    print(f"  Low-entropy layers: {summary['low_entropy_layer_count']}")
    print(f"  High-prob edges (P>=0.60): {summary['high_prob_edge_count']}")
    print(f"  Prefetch coverage @0.60: {summary['prefetch_coverage_at_60']:.4f}")
    print(f"  Prefetch coverage @0.35: {summary['prefetch_coverage_at_35']:.4f}")
    print()

    # Optional: compile and show pseudo-kernel prefetch schedules
    if args.kernels:
        print("=== Pseudo-Kernel Prefetch Schedules ===")
        pipe = CompilerPipeline()
        pipe.run(graph)
        schedule = pipe.get_prefetch_schedule()

        # Group by (src_layer, src_expert)
        from collections import defaultdict
        grouped = defaultdict(list)
        for src_l, src_e, dst_l, dst_e, priority, size in schedule:
            grouped[(src_l, src_e)].append((dst_l, dst_e, priority, size))

        for (src_l, src_e), entries in sorted(grouped.items()):
            print(f"# Kernel for layer {src_l}, expert {src_e}")
            for dst_l, dst_e, priority, size in sorted(entries, key=lambda x: x[2]):
                print(f"  async_prefetch(L{dst_l}:E{dst_e}, "
                      f"size={size}, priority={priority})")
            print()


if __name__ == "__main__":
    main()
