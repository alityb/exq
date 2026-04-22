#!/usr/bin/env python3
"""CLI: Save an evaluation model checkpoint for a given precision mode."""

import argparse
import json
import logging
from pathlib import Path

from exq.eval import apply_precision_to_model, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="ExQ: Save precision-adjusted model")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--precision", required=True, choices=["fp16", "rpgo", "int4"])
    parser.add_argument("--output-dir", required=True, help="Directory to save the model into")
    parser.add_argument("--profile", help="Routing profile path required for rpgo")
    parser.add_argument("--safe-serialization", action="store_true", help="Use safetensors when saving")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model, tokenizer = load_model_and_tokenizer(args.model)
    quant_stats = apply_precision_to_model(model, args.precision, profile_path=args.profile)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(
        output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size="4GB",
    )
    metadata = {
        "model_id": args.model,
        "precision": args.precision,
        "profile": args.profile,
        "quant_stats": quant_stats,
    }
    with (output_dir / "rpgo_eval_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logging.info("Saved %s precision model to %s", args.precision, output_dir)


if __name__ == "__main__":
    main()
