"""Online coverage monitor for profile drift and recompilation triggers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class CoverageSnapshot:
    rolling_coverage: float
    observations: int
    threshold: float
    needs_recompile: bool


class OnlineMonitor:
    """Track live routing decisions against the compiled prefetch schedule.

    The monitor is intentionally lightweight: it records whether an activated
    (prev_layer expert -> next layer expert) pair was anticipated by the static
    schedule. When rolling coverage drops below threshold for long enough, it
    signals that recompilation should be considered.
    """

    def __init__(self, prefetch_schedule, *, window: int = 500, threshold: float = 0.75):
        self.threshold = threshold
        self.window = window
        self._hits: deque[float] = deque(maxlen=window)
        self._prefetched = {
            (src_l, src_e, dst_l, dst_e)
            for src_l, src_e, dst_l, dst_e, *_ in prefetch_schedule
        }

    def observe(self, layer: int, experts: list[int], prev_layer: int | None, prev_experts: list[int] | None) -> None:
        if prev_layer is None or prev_experts is None:
            return
        hit = any(
            (prev_layer, pe, layer, ae) in self._prefetched
            for pe in prev_experts
            for ae in experts
        )
        self._hits.append(1.0 if hit else 0.0)

    def rolling_coverage(self) -> float:
        if not self._hits:
            return 1.0
        return sum(self._hits) / len(self._hits)

    def needs_recompile(self) -> bool:
        if len(self._hits) < max(10, self.window // 2):
            return False
        return self.rolling_coverage() < self.threshold

    def snapshot(self) -> CoverageSnapshot:
        cov = self.rolling_coverage()
        return CoverageSnapshot(
            rolling_coverage=cov,
            observations=len(self._hits),
            threshold=self.threshold,
            needs_recompile=self.needs_recompile(),
        )
