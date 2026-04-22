"""Shared benchmark and evaluation utilities for ExQ scripts."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def measure_tpot(
    model,
    tokenizer,
    prompt: str = "The following is a detailed analysis of",
    n_tokens: int = 64,
    n_runs: int = 20,
    warmup: int = 5,
) -> dict:
    """Measure time-per-output-token over n_runs generations.

    Returns dict with tpot_p50, tpot_p95, tpot_p99, tpot_mean, tpot_std,
    tpot_ci95, memory_gb, n_runs.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model.generate(**inputs, max_new_tokens=8, do_sample=False)

    tpots: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                **inputs, max_new_tokens=n_tokens, do_sample=False,
                return_dict_in_generate=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_generated = out.sequences.shape[1] - inputs.input_ids.shape[1]
            if n_generated > 0:
                tpots.append(elapsed / n_generated * 1000)

    if not tpots:
        return {"error": "no tokens generated"}

    return summarize_latencies(tpots)


def summarize_latencies(times: list[float]) -> dict:
    """Compute p50/p95/p99/mean/std/ci95 from a list of latency measurements."""
    if not times:
        return {}
    times = sorted(times)
    n = len(times)
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {
        "tpot_p50": times[n // 2],
        "tpot_p95": times[int(0.95 * n)],
        "tpot_p99": times[min(int(0.99 * n), n - 1)],
        "tpot_mean": statistics.mean(times),
        "tpot_std": statistics.stdev(times) if n > 1 else 0.0,
        "tpot_ci95": 1.96 * statistics.stdev(times) / (n ** 0.5) if n > 1 else 0.0,
        "memory_gb": mem_gb,
        "n_runs": n,
    }


def parse_eval_log(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Parse an eval log TSV into {model: {precision: {dataset: ppl}}}."""
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    p = Path(path)
    if not p.exists():
        return data
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            model_id, precision, dataset, ppl = parts
            data[model_id][precision][dataset] = float(ppl)
    return data


def compute_recovery(fp16_ppl: float, rpgo_ppl: float, int4_ppl: float) -> float | None:
    """Fraction of INT4 degradation that ExQ eliminates.

    Returns a value in [0, 1] where 1.0 means ExQ matches fp16 and 0.0 means
    it is the same as uniform INT4. Returns None if INT4 is not worse than fp16
    (no degradation to recover from).

    To get a percentage, multiply the return value by 100.
    """
    degradation = int4_ppl - fp16_ppl
    if degradation <= 0:
        return None
    return (int4_ppl - rpgo_ppl) / degradation


def compute_recovery_pct(fp16_ppl: float, exq_ppl: float, int4_ppl: float) -> float:
    """Same as compute_recovery but returns a percentage (0–100) and never None.

    Returns 0.0 when INT4 is not worse than fp16.
    This is the form used by scripts and tables.
    """
    degradation = int4_ppl - fp16_ppl
    if not (degradation > 0):
        return 0.0
    return (int4_ppl - exq_ppl) / degradation * 100.0
