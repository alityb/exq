#!/usr/bin/env python3
"""Solve the ExQ joint scheduler over a routing graph."""

from __future__ import annotations

import argparse
import json
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description="ExQ joint scheduler (CP-SAT)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--min-prefetch-prob", type=float, default=0.35)
    parser.add_argument("--memory-budget-units", type=int, default=None)
    parser.add_argument("--max-prefetch-per-layer", type=int, default=16)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from exq._core import RoutingProfile, py_build_routing_graph
    from exq.compiler import solve_joint_schedule

    profile = RoutingProfile.load(args.profile)
    graph = py_build_routing_graph(profile)
    result = solve_joint_schedule(
        graph,
        memory_budget_units=args.memory_budget_units,
        min_prefetch_prob=args.min_prefetch_prob,
        max_prefetch_per_layer=args.max_prefetch_per_layer,
        max_time_seconds=args.time_limit,
    )

    counts = Counter(result.quant_assignments.values())
    payload = {
        "model_id": profile.model_id,
        "status": result.status,
        "objective_value": result.objective_value,
        "precision_counts": dict(counts),
        "prefetch_edges": len(result.prefetch_edges),
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    **payload,
                    "quant_assignments": {f"{k[0]}:{k[1]}": v for k, v in result.quant_assignments.items()},
                    "prefetch_edges_detail": result.prefetch_edges,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
