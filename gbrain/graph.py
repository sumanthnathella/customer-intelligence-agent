"""
gbrain/graph.py — High-level write helpers that implement the per-run update loop.

Write loop per run:
    analyze → upsert L5 entity → append period_metric → update affects edges
             → refresh exemplars → auto-link → snapshot
"""
from __future__ import annotations

import logging
from typing import Any

from gbrain.store import GBrainStore
from shared.schemas import DriverEdge, EgregScore, WeeklyMetric, ZScoreResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upsert L5 pain_point entity
# ---------------------------------------------------------------------------

def upsert_pain_point(
    store: GBrainStore,
    l5_id: str,
    path: list[str],
    definition: str,
    pack_version: str,
    *,
    total_volume: int,
    severity_avg: float,
    latest_zscore: float,
    latest_egregiousness: float,
    top_drivers: list[dict[str, Any]] | None = None,
    importance: str | None = None,
    trend: str | None = None,
    facets: dict[str, Any] | None = None,
    subthemes: list[dict[str, Any]] | None = None,
    first_seen: str | None = None,
    last_seen: str | None = None,
) -> str:
    existing = store.get_node(l5_id)
    prev_props = existing["props"] if existing else {}
    fs = first_seen or prev_props.get("first_seen")
    # Curated-set memory: remember the prior importance so the report can show
    # week-over-week movement (escalated / de-escalated / new / stable).
    importance_prev = prev_props.get("importance")
    prev_volume = prev_props.get("total_volume")
    props: dict[str, Any] = {
        "l5_id": l5_id,
        "path": path,
        "definition": definition,
        "total_volume": total_volume,
        "severity_avg": round(severity_avg, 3),
        "latest_zscore": round(latest_zscore, 3),
        "latest_egregiousness": round(latest_egregiousness, 4),
        "top_drivers": top_drivers or [],
        "importance": importance,
        "importance_prev": importance_prev,
        "trend": trend,
        "facets": facets or {},
        "subthemes": subthemes or [],
        "volume_prev": prev_volume,
        "first_seen": fs,
        "last_seen": last_seen,
    }
    return store.upsert_node(
        node_type="pain_point",
        node_id=l5_id,
        props=props,
        pack_version=pack_version,
    )


# ---------------------------------------------------------------------------
# Append period_metric (one per L5 per period)
# ---------------------------------------------------------------------------

def upsert_period_metric(
    store: GBrainStore,
    metric: WeeklyMetric,
    zscore_result: ZScoreResult,
    egreg: EgregScore,
    order_value_impact: float = 0.0,
) -> str:
    node_id = f"{metric.l5_id}@{metric.week}"
    props: dict[str, Any] = {
        "l5_id": metric.l5_id,
        "week": metric.week,
        "volume": metric.volume,
        "severity_avg": round(metric.severity_avg, 3),
        "sentiment_avg": round(metric.sentiment_avg, 3),
        "zscore": round(zscore_result.zscore, 3),
        "egregiousness": round(egreg.egregiousness, 4),
        "order_value_impact": round(order_value_impact, 2),
    }
    store.upsert_node(node_type="period_metric", node_id=node_id, props=props)
    store.add_edge(metric.l5_id, "measured_in", node_id)
    if zscore_result.zscore >= 2.0:  # spiked_in
        store.add_edge(metric.l5_id, "spiked_in", node_id)
    return node_id


# ---------------------------------------------------------------------------
# Upsert affects edge (L5 × dimension-value heatmap cell)
# ---------------------------------------------------------------------------

def upsert_affects_edge(store: GBrainStore, driver: DriverEdge) -> str:
    dim_node_id = f"dim:{driver.dimension}:{driver.value}"
    store.upsert_node(
        node_type="dimension",
        node_id=dim_node_id,
        props={"dimension": driver.dimension, "value": driver.value},
    )
    props: dict[str, Any] = {
        "dimension": driver.dimension,
        "value": driver.value,
        "support": driver.support,
        "share": round(driver.share, 4),
        "lift": round(driver.lift, 4),
        "p_value": driver.p_value,
        "significant": driver.significant,
        "excess": round(driver.excess, 2),
        "period": driver.period,
        "history": driver.history,
    }
    return store.add_edge(driver.l5_id, "affects", dim_node_id, props=props)


# ---------------------------------------------------------------------------
# Systemic ("bridge") dimension rollup
# ---------------------------------------------------------------------------

def upsert_dimension_systemic(store: GBrainStore, systemic: Any) -> str:
    """Enrich a dimension node with its systemic (bridge/singleton) footprint.

    ``systemic`` is an ``analytics.systemic.SystemicDriver``. We persist the
    rollup on the dimension node so the agent can read "which operational
    levers are systemic" without re-aggregating.
    """
    dim_node_id = f"dim:{systemic.dimension}:{systemic.value}"
    existing = store.get_node(dim_node_id)
    props: dict[str, Any] = dict(existing["props"]) if existing else {}
    props.update(
        {
            "dimension": systemic.dimension,
            "value": systemic.value,
            "n_significant_l5": systemic.n_significant,
            "affected_l5s": systemic.affected_l5s,
            "total_support": systemic.total_support,
            "mean_lift": round(systemic.mean_lift, 4),
            "max_lift": round(systemic.max_lift, 4),
            "is_bridge": systemic.is_bridge,
            "is_singleton": systemic.is_singleton,
            "systemic_score": round(systemic.systemic_score, 4),
        }
    )
    return store.upsert_node(node_type="dimension", node_id=dim_node_id, props=props)


# ---------------------------------------------------------------------------
# Verification records (claim → evidence → verdict)
# ---------------------------------------------------------------------------

def record_verification(
    store: GBrainStore,
    l5_id: str,
    claim: str,
    verdict: bool,
    detail: dict[str, Any] | None = None,
    evidence_ids: list[str] | None = None,
    run_id: str = "",
) -> str:
    """Persist a claim→evidence→verdict record, mirroring Harness-1's
    verification cache. Links the verification to its L5 and any cited evidence.
    """
    import hashlib

    key = hashlib.sha1(f"{l5_id}|{claim}".encode()).hexdigest()[:12]
    node_id = f"verification:{key}"
    props: dict[str, Any] = {
        "l5_id": l5_id,
        "claim": claim,
        "verdict": "verified" if verdict else "unsupported",
        "detail": detail or {},
        "evidence": evidence_ids or [],
        "run_id": run_id,
    }
    store.upsert_node(node_type="verification", node_id=node_id, props=props)
    store.add_edge(node_id, "verifies", l5_id)
    for eid in (evidence_ids or []):
        store.add_edge(node_id, "cites", eid)
    return node_id


# ---------------------------------------------------------------------------
# Exemplar
# ---------------------------------------------------------------------------

def upsert_exemplar(
    store: GBrainStore,
    conversation_id: str,
    l5_id: str,
    snippet: str,
    severity: float,
    sentiment: str,
    week: str,
    brand: str = "",
) -> str:
    props: dict[str, Any] = {
        "conversation_id": conversation_id,
        "snippet": snippet[:500],
        "severity": round(severity, 2),
        "sentiment": sentiment,
        "week": week,
        "brand": brand,
    }
    node_id = f"exemplar:{conversation_id}"
    store.upsert_node(node_type="exemplar", node_id=node_id, props=props)
    store.add_edge(l5_id, "exemplified_by", node_id)
    return node_id


# ---------------------------------------------------------------------------
# Auto-link helpers
# ---------------------------------------------------------------------------

def auto_link_taxonomy(store: GBrainStore, l5_id: str, path: list[str]) -> None:
    """Add child_of edges up the taxonomy path (L5→L4→L3→L2→L1).

    The leaf level uses the canonical ``l5_id``; intermediate ancestors get
    synthetic ``l{level}:{label}`` ids so rollup nodes are stable across runs.
    """
    # Node id for each level of the path; the deepest level is the real L5 id.
    level_ids = [f"l{i + 1}:{label}" for i, label in enumerate(path)]
    level_ids[-1] = l5_id

    for i in range(len(level_ids) - 1, 0, -1):
        child_id = level_ids[i]
        parent_id = level_ids[i - 1]
        store.upsert_node(
            node_type="pain_point",
            node_id=parent_id,
            props={"l5_id": parent_id, "path": path[:i], "label": path[i - 1]},
        )
        store.add_edge(child_id, "child_of", parent_id)


def auto_link_co_occurs(
    store: GBrainStore,
    co_occurrence_pairs: list[tuple[str, str]],
) -> None:
    """Add co_occurs edges for L5 pairs that appear together in the same conversation."""
    for a, b in co_occurrence_pairs:
        store.add_edge(a, "co_occurs", b)
        store.add_edge(b, "co_occurs", a)
