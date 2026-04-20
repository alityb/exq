#!/usr/bin/env python3
"""CLI: Render the benchmark log as a compact Markdown table."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_eval_log(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Parse a tab-separated eval log into a nested mapping."""
    records: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            model_id, precision, benchmark, value = line.split("\t")
            records[model_id][precision][benchmark] = float(value)
    return {model_id: {p: dict(v) for p, v in precisions.items()} for model_id, precisions in records.items()}


def render_results_table(records: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render benchmark records as a Markdown table."""
    headers = ["Model", "Precision", "WikiText2", "C4", "GSM8K"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | ---: | ---: | ---: |"]
    for model_id in sorted(records):
        for precision in sorted(records[model_id]):
            row = records[model_id][precision]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_id,
                        precision,
                        f"{row['wikitext2']:.4f}" if "wikitext2" in row else "-",
                        f"{row['c4']:.4f}" if "c4" in row else "-",
                        f"{row['gsm8k']:.4f}" if "gsm8k" in row else "-",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Render evaluation results table")
    parser.add_argument("--log-path", default="results/eval_log.txt", help="Benchmark log file")
    args = parser.parse_args()

    records = parse_eval_log(args.log_path)
    print(render_results_table(records))


if __name__ == "__main__":
    main()
