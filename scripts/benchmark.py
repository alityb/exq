#!/usr/bin/env python3
"""CLI: Benchmark compiled vs baseline model.

Stub for Milestone 6 -- measures tokens/sec, TTFT, TPOT.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Benchmark")
    parser.add_argument("--artifact", type=str, help="Path to compiled artifact")
    parser.add_argument("--baseline", type=str, help="Baseline model ID for comparison")
    parser.add_argument("--n-tokens", type=int, default=128, help="Tokens to generate")
    parser.parse_args()

    logging.info("Benchmark stub -- not yet implemented (Milestone 6)")
    logging.info("Will measure: TPOT, TTFT, tokens/sec, memory usage")


if __name__ == "__main__":
    main()
