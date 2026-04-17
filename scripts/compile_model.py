#!/usr/bin/env python3
"""CLI: Run the R-PGO compilation pipeline.

Usage:
    python scripts/compile_model.py --profile routing_profile.json
    python scripts/compile_model.py --profile routing_profile.json --passes quant_only
"""

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Compile model from routing profile")
    parser.add_argument("--profile", type=str, required=True, help="Path to routing_profile.json")
    parser.add_argument("--output", type=str, default="compiled_artifact.json", help="Output path")
    parser.add_argument(
        "--passes",
        type=str,
        default="all",
        choices=["all", "quant_only", "prefetch_only", "layout_only", "no_prefetch"],
        help="Which passes to run",
    )
    args = parser.parse_args()

    from rpgo._core import (
        RoutingProfile,
        CompilerPipeline,
        py_build_routing_graph,
        py_graph_summary,
    )

    logging.info(f"Loading profile: {args.profile}")
    profile = RoutingProfile.load(args.profile)
    logging.info(f"Profile: {profile.model_id}, {profile.n_layers} layers")

    # Build routing graph via Rust core
    graph = py_build_routing_graph(profile)
    logging.info(f"Routing graph: {graph.n_nodes} nodes, {graph.n_edges} edges")

    # Print summary
    summary = py_graph_summary(graph)
    logging.info(
        f"Graph summary: HOT={summary['total_hot']}, WARM={summary['total_warm']}, "
        f"COLD={summary['total_cold']}, FROZEN={summary['total_frozen']}"
    )

    # Run compiler pipeline
    pipeline = CompilerPipeline()

    if args.passes == "all":
        pipeline.run(graph)
    elif args.passes == "quant_only":
        pipeline.run_selective(graph, layout=False, quant=True, specialize=False, prefetch=False)
    elif args.passes == "prefetch_only":
        pipeline.run_selective(graph, layout=False, quant=False, specialize=False, prefetch=True)
    elif args.passes == "layout_only":
        pipeline.run_selective(graph, layout=True, quant=False, specialize=False, prefetch=False)
    elif args.passes == "no_prefetch":
        pipeline.run_selective(graph, layout=True, quant=True, specialize=True, prefetch=False)

    logging.info(f"Pipeline result: {pipeline}")

    # Save results
    quant_plan = pipeline.get_quant_plan()
    layout_plan = pipeline.get_layout_plan()
    spec_plan = pipeline.get_specialization_plan()
    prefetch_count = pipeline.get_prefetch_entry_count()

    artifact = {
        "model_id": profile.model_id,
        "quant_assignments": {f"{k[0]}:{k[1]}": v for k, v in quant_plan.items()},
        "layout_placements": {f"{k[0]}:{k[1]}": v for k, v in layout_plan.items()},
        "specialization_decisions": spec_plan,
        "prefetch_entry_count": prefetch_count,
    }

    with open(args.output, "w") as f:
        json.dump(artifact, f, indent=2)
    logging.info(f"Artifact saved to {args.output}")


if __name__ == "__main__":
    main()
