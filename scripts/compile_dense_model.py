#!/usr/bin/env python3
"""CLI: Compile a dense model from an attention head profile."""

from __future__ import annotations

import argparse
import json
import logging

from exq.compiler.dense_quant_planner import plan_dense_quant
from exq.profiler.dense_profile import DenseProfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExQ: Compile dense model from attention profile"
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", default="artifacts/dense_artifact.json")
    parser.add_argument("--hot-threshold", type=float, default=None)
    parser.add_argument("--warm-threshold", type=float, default=None)
    parser.add_argument("--cold-threshold", type=float, default=None)
    args = parser.parse_args()

    profile = DenseProfile.load(args.profile)
    logging.info(
        "Profile: %s, %s layers, %s tokens",
        profile.model_id,
        len(profile.layers),
        f"{profile.calibration_tokens:,}",
    )

    plan = plan_dense_quant(
        profile,
        hot_threshold=args.hot_threshold,
        warm_threshold=args.warm_threshold,
        cold_threshold=args.cold_threshold,
    )
    summary = plan.summary
    logging.info(
        "Quant plan: BF16=%s, INT8=%s, INT4=%s (BF16 fraction: %.1f%%)",
        summary["BF16"],
        summary["INT8"],
        summary["INT4"],
        summary["BF16_fraction"] * 100,
    )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2)
    logging.info("Artifact saved: %s", args.output)


if __name__ == "__main__":
    main()
