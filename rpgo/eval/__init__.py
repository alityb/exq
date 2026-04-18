"""Evaluation utilities: benchmarks, quality metrics, coverage analysis."""

from rpgo.eval.coverage import CoverageAnalyzer
from rpgo.eval.quality import compute_perplexity

__all__ = ["CoverageAnalyzer", "compute_perplexity"]
