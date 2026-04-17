#!/usr/bin/env python3
"""CLI: Collect routing profile from an MoE model.

Usage:
    python scripts/profile_model.py --model Qwen/Qwen3-30B-A3B --samples 2048
    python scripts/profile_model.py --config configs/qwen3-30b.yaml
"""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Collect routing profile")
    parser.add_argument("--model", type=str, help="HuggingFace model ID")
    parser.add_argument("--samples", type=int, default=2048, help="Calibration samples")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--output", type=str, default="routing_profile.json", help="Output path")
    parser.add_argument("--config", type=str, help="YAML config file (overrides other args)")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        from rpgo.profiler.calibration_runner import run_calibration_from_config
        profile = run_calibration_from_config(config)
    elif args.model:
        from rpgo.profiler.calibration_runner import CalibrationRunner
        runner = CalibrationRunner(
            model_id=args.model,
            n_samples=args.samples,
            max_length=args.max_length,
            batch_size=args.batch_size,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
        )
        profile = runner.run(output_path=args.output)
    else:
        parser.error("Either --model or --config is required")
        sys.exit(1)

    warnings = profile.validate()
    if warnings:
        for w in warnings:
            logging.warning(w)
    else:
        logging.info("Profile validation passed (all frequencies sum to 1.0)")

    logging.info(f"Profile: {profile.n_layers} MoE layers, "
                 f"{profile.calibration_tokens} total token activations")
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
