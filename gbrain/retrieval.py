"""
gbrain/retrieval.py — Hybrid retrieval interface for the agent.

Implements structured, graph, and (optionally) semantic search over the brain.
The agent never re-aggregates — it reads pre-computed props from nodes/edges.
"""
from __future__ import annotations

import logging
from typing import Any

from gbrain.store import GBrainStore, get_store
from shared.config import CURATED_CAP, DRIVER_MIN_SUPPORT
from shared.text import keyword_overlap_score

logger = logging.getLogger(__name__)

_IMPORTANCE_ORDER = {"very_high": 3, "high": 2, "fair": 1, "low": 0}


def _curation_delta(prev: str | None, current: str | None) -> str:
    """How an L5's importance moved between runs (curated-set memory)."""
    if prev is None:
        return "new"
    cur_rank = _IMPORTANCE_ORDER.get(current or "", 0)
    prev_rank = _IMPORTANCE_ORDER.get(prev, 0)
    if cur_rank > prev_rank:
        return "escalated"
    if cur_rank < prev_rank:
        return "de-escalated"
    return "stable"


# ---------------------------------------------------------------------------
# Structured reads
# ---------------------------------------------------------------------------

def get_top_l5(
    store: GBrainStore | None = None,
    by: str = "egregiousness",
    n: int = 10,
) -> list[dict[str, Any]]:
    """
    Return top-N pain_point nodes ranked by a scalar prop.
    by = "egregiousness" | "zscore" | "volume"
    """
    s = store or get_store()
    prop_map = {
        "egregiousness": "latest_egregiousness",
        "zscore": "latest_zscore",
        "volume": "total_volume",
    }
    prop = prop_map.get(by, "latest_egregiousness")
    nodes = s.query(node_type="pain_point", order_by_prop=prop, descending=True, limit=n)
    # filter out non-L5 rollup nodes (those without full path)
    nodes = [n for n in nodes if len(n["props"].get("path", [])) >= 5][:n]
    return nodes


def get_zscore_spikes(
    store: GBrainStore | None = None,
    weeks: int = 2,
    threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Return pain_point nodes that have spiked_in edges to recent period_metric nodes."""
    s = store or get_store()
    spike_nodes = s.query(
        node_type="pain_point",
        order_by_prop="latest_zscore",
        descending=True,
    )
    return [n for n in spike_nodes if n["props"].get("latest_zscore", 0) >= threshold]


def get_drivers(
    l5_id: str,
    store: GBrainStore | None = None,
    significant_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Return affects edges for a given L5, ordered by lift descending.
    Each dict: {dimension, value, support, share, lift, p_value, significant, excess, period, history}
    """
    s = store or get_store()
    edges = s.get_edges(l5_id, edge_type="affects", direction="out")
    drivers = [e["props"] for e in edges if e["props"]]
    if significant_only:
        drivers = [d for d in drivers if d.get("significant")]
    drivers.sort(key=lambda d: d.get("lift", 0), reverse=True)
    return drivers


def get_ops_hotspots(
    dimension: str | None = None,
    store: GBrainStore | None = None,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """
    Ops-first view: for each dimension value (or a specific one), return its top L5s by support.
    Returns {dim_value: [{"l5_id", "support", "lift", "significant", ...}, ...]}
    """
    s = store or get_store()
    if dimension:
        dim_nodes = s.query(node_type="dimension", filters={"dimension": dimension})
    else:
        dim_nodes = s.query(node_type="dimension")

    result: dict[str, list[dict[str, Any]]] = {}
    for dn in dim_nodes:
        dim_val = f"{dn['props']['dimension']}={dn['props']['value']}"
        in_edges = s.get_edges(dn["id"], edge_type="affects", direction="in")
        cells = []
        for edge in in_edges:
            cell = dict(edge["props"])
            cell["l5_id"] = edge["src"]
            cells.append(cell)
        cells.sort(key=lambda c: c.get("support", 0), reverse=True)
        result[dim_val] = cells[:top_k]
    return result


def get_exemplars(
    l5_id: str,
    k: int = 3,
    store: GBrainStore | None = None,
) -> list[dict[str, Any]]:
    """Return k exemplar snippets for a given L5, ordered by severity descending."""
    s = store or get_store()
    edges = s.get_edges(l5_id, edge_type="exemplified_by", direction="out")
    exemplars = []
    for edge in edges:
        node = s.get_node(edge["dst"])
        if node:
            exemplars.append(node["props"])
    exemplars.sort(key=lambda e: e.get("severity", 0), reverse=True)
    return exemplars[:k]


def read_memory(
    l5_id: str,
    store: GBrainStore | None = None,
) -> dict[str, Any]:
    """
    Return prior period metrics, prior insights, and driver trend history for an L5.
    """
    s = store or get_store()
    pain_node = s.get_node(l5_id)

    # period metrics (time-series)
    measured_edges = s.get_edges(l5_id, edge_type="measured_in", direction="out")
    period_ids = [e["dst"] for e in measured_edges]
    period_metrics = []
    for pid in sorted(period_ids):
        node = s.get_node(pid)
        if node:
            period_metrics.append(node["props"])

    # prior insights
    insight_nodes = s.query(node_type="insight")
    prior_insights = [n["props"] for n in insight_nodes if n["props"].get("l5_id") == l5_id]

    # driver trend (from affects edge history prop)
    drivers = get_drivers(l5_id, store=s)
    driver_trends = [
        {
            "dimension": d.get("dimension"),
            "value": d.get("value"),
            "lift": d.get("lift"),
            "history": d.get("history", []),
        }
        for d in drivers
        if d.get("history")
    ]

    return {
        "l5_id": l5_id,
        "pain_point": pain_node["props"] if pain_node else {},
        "period_metrics": period_metrics,
        "prior_insights": prior_insights,
        "driver_trends": driver_trends,
    }


# ---------------------------------------------------------------------------
# Curated set (importance-tagged, capped, with cross-run movement)
# ---------------------------------------------------------------------------

def get_curated_set(
    store: GBrainStore | None = None,
    cap: int = CURATED_CAP,
) -> list[dict[str, Any]]:
    """Return the importance-tagged curated set of pain points (Harness-1 style).

    Warm-started from egregiousness ranking, capped at ``cap``, each entry
    annotated with its importance tier, coarse trend, and how it moved since the
    last run (new / escalated / de-escalated / stable).
    """
    s = store or get_store()
    nodes = s.query(node_type="pain_point", order_by_prop="latest_egregiousness", descending=True)
    nodes = [n for n in nodes if len(n["props"].get("path", [])) >= 5]
    curated: list[dict[str, Any]] = []
    for n in nodes[:cap]:
        p = n["props"]
        vol, vol_prev = p.get("total_volume", 0), p.get("volume_prev")
        vol_delta = (vol - vol_prev) if isinstance(vol_prev, (int, float)) else None
        curated.append(
            {
                "l5_id": p.get("l5_id"),
                "path": p.get("path", []),
                "importance": p.get("importance"),
                "importance_prev": p.get("importance_prev"),
                "movement": _curation_delta(p.get("importance_prev"), p.get("importance")),
                "trend": p.get("trend"),
                "egregiousness": p.get("latest_egregiousness"),
                "zscore": p.get("latest_zscore"),
                "volume": vol,
                "volume_delta": vol_delta,
                "severity_avg": p.get("severity_avg"),
                "top_drivers": p.get("top_drivers", []),
                "facets": p.get("facets", {}),
                "subthemes": p.get("subthemes", []),
            }
        )
    return curated


def get_l5_profile(l5_id: str, store: GBrainStore | None = None) -> dict[str, Any]:
    """Return the deep-dive profile for one pain point.

    Combines the compositional facet breakdown (which operational groups make up
    the pain point) with the emergent sub-themes (the distinctive issues, in
    words, each with a count and a representative quote).
    """
    s = store or get_store()
    node = s.get_node(l5_id)
    if not node:
        return {"l5_id": l5_id, "found": False}
    p = node["props"]
    return {
        "l5_id": l5_id,
        "found": True,
        "path": p.get("path", []),
        "definition": p.get("definition", ""),
        "total_volume": p.get("total_volume"),
        "severity_avg": p.get("severity_avg"),
        "importance": p.get("importance"),
        "facets": p.get("facets", {}),
        "subthemes": p.get("subthemes", []),
        "top_drivers": p.get("top_drivers", []),
    }


# ---------------------------------------------------------------------------
# Systemic ("bridge") drivers
# ---------------------------------------------------------------------------

def get_systemic_drivers(
    store: GBrainStore | None = None,
    only_bridges: bool = False,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return dimension values ranked by systemic footprint (bridges first)."""
    s = store or get_store()
    dims = s.query(node_type="dimension")
    scored = [
        d["props"]
        for d in dims
        if d["props"].get("n_significant_l5", 0) > 0
    ]
    if only_bridges:
        scored = [d for d in scored if d.get("is_bridge")]
    scored.sort(
        key=lambda d: (d.get("is_bridge", False), d.get("systemic_score", 0)),
        reverse=True,
    )
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Verification (claim → evidence → verdict)
# ---------------------------------------------------------------------------

def verify_driver_claim(
    l5_id: str,
    dimension: str,
    value: str,
    store: GBrainStore | None = None,
) -> dict[str, Any]:
    """Check a driver claim against the persisted affects edge.

    Returns the verdict plus the supporting numbers so callers can cite them.
    A claim is *verified* when the edge is significant, over-indexes (lift > 1),
    and meets the minimum support threshold.
    """
    s = store or get_store()
    dim_node_id = f"dim:{dimension}:{value}"
    edge_id = f"{l5_id}:affects:{dim_node_id}"
    row = s.conn.execute("SELECT props FROM edges WHERE id = ?", (edge_id,)).fetchone()
    if row is None:
        return {"verdict": "unsupported", "reason": "no edge", "l5_id": l5_id,
                "dimension": dimension, "value": value}
    import json
    props = json.loads(row["props"] or "{}")
    verdict = bool(
        props.get("significant")
        and props.get("lift", 0) > 1.0
        and props.get("support", 0) >= DRIVER_MIN_SUPPORT
    )
    return {
        "verdict": "verified" if verdict else "unsupported",
        "l5_id": l5_id,
        "dimension": dimension,
        "value": value,
        "lift": props.get("lift"),
        "support": props.get("support"),
        "p_value": props.get("p_value"),
        "significant": props.get("significant"),
    }


# ---------------------------------------------------------------------------
# Full-text transcript search (grep over exemplar evidence)
# ---------------------------------------------------------------------------

def search_transcripts(
    query: str,
    l5_id: str | None = None,
    store: GBrainStore | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Grep exemplar snippets for a keyword query, ranked by term overlap.

    Lets the agent drill into raw evidence (Harness-1's grep_corpus/read_document)
    without recomputing metrics. Scoped to one L5 when ``l5_id`` is given.
    """
    s = store or get_store()
    if l5_id:
        edges = s.get_edges(l5_id, edge_type="exemplified_by", direction="out")
        nodes = [s.get_node(e["dst"]) for e in edges]
        candidates = [n["props"] for n in nodes if n]
    else:
        candidates = [n["props"] for n in s.query(node_type="exemplar")]

    scored: list[tuple[float, dict[str, Any]]] = []
    for ex in candidates:
        score = keyword_overlap_score(ex.get("snippet", ""), query)
        if score > 0:
            scored.append((score, ex))
    scored.sort(key=lambda x: (x[0], x[1].get("severity", 0)), reverse=True)
    return [
        {**ex, "match_score": round(score, 3)}
        for score, ex in scored[:k]
    ]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_insight(
    l5_id: str,
    run_id: str,
    summary: str,
    root_cause: str = "",
    recommended_action: str = "",
    evidence_ids: list[str] | None = None,
    store: GBrainStore | None = None,
) -> str:
    """Persist an insight node and link it to the L5 and any evidence."""
    import uuid
    s = store or get_store()
    insight_id = f"insight:{run_id}:{l5_id}:{uuid.uuid4().hex[:8]}"
    props: dict[str, Any] = {
        "insight_id": insight_id,
        "l5_id": l5_id,
        "summary": summary,
        "root_cause": root_cause,
        "recommended_action": recommended_action,
        "status": "new",
        "run_id": run_id,
        "evidence": evidence_ids or [],
    }
    s.upsert_node(node_type="insight", node_id=insight_id, props=props)
    s.add_edge(insight_id, "explains", l5_id)
    for eid in (evidence_ids or []):
        s.add_edge(insight_id, "cites", eid)
    return insight_id
