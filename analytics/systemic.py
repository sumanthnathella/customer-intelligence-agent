"""
analytics/systemic.py — Evidence-graph view over driver edges + importance tiers.

Two Harness-1-inspired primitives:

1. **Bridges vs singletons.** Harness-1 builds an evidence graph where a
   *bridge* document links many entities and a *singleton* appears once. The
   analogue here: a dimension value that is a *significant* driver across many
   L5 pain points is a **bridge** — a systemic operational problem worth fixing
   once to relieve many pain points. A driver isolated to a single L5 is a
   **singleton** lead.

2. **Importance tiers.** A deterministic warm-start tag (very_high / high /
   fair / low) per L5 from its egregiousness, so the curated set is *refined*
   across runs rather than rebuilt from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from shared.config import (
    BRIDGE_MIN_L5,
    IMPORTANCE_FAIR,
    IMPORTANCE_HIGH,
    IMPORTANCE_VERY_HIGH,
)
from shared.schemas import DriverEdge

IMPORTANCE_ORDER = {"very_high": 3, "high": 2, "fair": 1, "low": 0}


@dataclass
class SystemicDriver:
    """One (dimension, value) rolled up across all the L5s it drives."""

    dimension: str
    value: str
    affected_l5s: list[str] = field(default_factory=list)  # significant only
    n_significant: int = 0
    total_support: int = 0
    mean_lift: float = 0.0
    max_lift: float = 0.0
    is_bridge: bool = False
    is_singleton: bool = False
    systemic_score: float = 0.0


def compute_systemic_drivers(
    edges: list[DriverEdge],
    bridge_min_l5: int = BRIDGE_MIN_L5,
) -> list[SystemicDriver]:
    """Roll up driver edges by (dimension, value) and flag bridges/singletons.

    Only *significant* edges count toward the systemic footprint. Returns the
    list sorted with bridges first, then by systemic_score descending.
    """
    by_dimval: dict[tuple[str, str], list[DriverEdge]] = {}
    for e in edges:
        by_dimval.setdefault((e.dimension, e.value), []).append(e)

    results: list[SystemicDriver] = []
    for (dim, val), group in by_dimval.items():
        sig = [e for e in group if e.significant]
        affected = sorted({e.l5_id for e in sig})
        n_sig = len(affected)
        if n_sig == 0:
            continue
        lifts = [e.lift for e in sig]
        results.append(
            SystemicDriver(
                dimension=dim,
                value=val,
                affected_l5s=affected,
                n_significant=n_sig,
                total_support=sum(e.support for e in sig),
                mean_lift=round(sum(lifts) / len(lifts), 4),
                max_lift=round(max(lifts), 4),
                is_bridge=n_sig >= bridge_min_l5,
                is_singleton=n_sig == 1,
            )
        )

    # Normalize systemic_score = (#L5 reached × mean lift) to [0, 1].
    if results:
        raw = [r.n_significant * r.mean_lift for r in results]
        mx = max(raw) or 1.0
        for r, val in zip(results, raw, strict=True):
            r.systemic_score = round(val / mx, 4)

    results.sort(key=lambda r: (r.is_bridge, r.systemic_score), reverse=True)
    return results


def importance_tier(
    egregiousness: float,
    very_high: float = IMPORTANCE_VERY_HIGH,
    high: float = IMPORTANCE_HIGH,
    fair: float = IMPORTANCE_FAIR,
) -> str:
    """Map an egregiousness score (0..1) to a curated-set importance tier."""
    if egregiousness >= very_high:
        return "very_high"
    if egregiousness >= high:
        return "high"
    if egregiousness >= fair:
        return "fair"
    return "low"


def curation_delta(prev: str | None, current: str) -> str:
    """Describe how an L5's importance moved between runs (curated-set memory)."""
    if prev is None:
        return "new"
    if IMPORTANCE_ORDER.get(current, 0) > IMPORTANCE_ORDER.get(prev, 0):
        return "escalated"
    if IMPORTANCE_ORDER.get(current, 0) < IMPORTANCE_ORDER.get(prev, 0):
        return "de-escalated"
    return "stable"


def zscore_trend(zscore: float, spike: float = 2.0) -> str:
    """Coarse direction label from the latest z-score."""
    if zscore >= spike:
        return "rising"
    if zscore <= -1.0:
        return "falling"
    return "stable"
