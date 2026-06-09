"""
agent/tools.py — Thin wrappers over gbrain retrieval for the ReAct agent.

The agent NEVER computes metrics — it only reads pre-computed aggregates from gbrain.
These tools are used by the Google ADK LlmAgent.
"""
from __future__ import annotations

import logging
from typing import Any

from gbrain.retrieval import (
    get_curated_set,
    get_drivers,
    get_exemplars,
    get_l5_profile,
    get_ops_hotspots,
    get_systemic_drivers,
    get_top_l5,
    get_zscore_spikes,
    read_memory,
    search_transcripts,
    verify_driver_claim,
    write_insight,
)
from gbrain.schema_pack import load_pack
from shared.config import CURATED_CAP, EXEMPLAR_K, TOP_L5_N

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read tools (return plain dicts — ADK serialises to JSON for LLM context)
# ---------------------------------------------------------------------------

def tool_get_taxonomy() -> dict[str, Any]:
    """Return the L1–L5 taxonomy schema pack."""
    pack = load_pack()
    return pack.to_dict()


def tool_get_top_l5(by: str = "egregiousness", n: int = TOP_L5_N) -> dict[str, Any]:
    """
    Return top-N pain points ranked by egregiousness, zscore, or volume.

    Args:
        by: one of "egregiousness" | "zscore" | "volume"
        n: number of results (default 10)
    """
    nodes = get_top_l5(by=by, n=n)
    return {
        "by": by,
        "n": n,
        "results": [
            {
                "l5_id": n["props"]["l5_id"],
                "path": n["props"].get("path", []),
                "egregiousness": n["props"].get("latest_egregiousness"),
                "zscore": n["props"].get("latest_zscore"),
                "volume": n["props"].get("total_volume"),
                "severity_avg": n["props"].get("severity_avg"),
                "top_drivers": n["props"].get("top_drivers", []),
            }
            for n in nodes
        ],
    }


def tool_get_zscore_spikes(weeks: int = 2, threshold: float = 2.0) -> dict[str, Any]:
    """Return pain points with recent z-score spikes above the threshold."""
    nodes = get_zscore_spikes(weeks=weeks, threshold=threshold)
    return {
        "threshold": threshold,
        "count": len(nodes),
        "results": [
            {
                "l5_id": n["props"]["l5_id"],
                "zscore": n["props"].get("latest_zscore"),
                "volume": n["props"].get("total_volume"),
            }
            for n in nodes
        ],
    }


def tool_get_drivers(l5_id: str) -> dict[str, Any]:
    """Return operational drivers (over-indexed dimensions) for a given L5 pain point."""
    drivers = get_drivers(l5_id, significant_only=False)
    sig_drivers = [d for d in drivers if d.get("significant")]
    return {
        "l5_id": l5_id,
        "total_cells": len(drivers),
        "significant_drivers": len(sig_drivers),
        "drivers": [
            {
                "dimension": d["dimension"],
                "value": d["value"],
                "lift": round(d.get("lift", 0), 2),
                "support": d.get("support"),
                "share": round(d.get("share", 0), 4),
                "p_value": round(d.get("p_value", 1), 4),
                "significant": d.get("significant"),
                "excess": round(d.get("excess", 0), 1),
                "history": d.get("history", [])[-3:],  # last 3 periods for trend
            }
            for d in sig_drivers[:10]
        ],
        "heatmap_top": [  # top non-sig cells by support for heatmap context
            {"dimension": d["dimension"], "value": d["value"], "support": d["support"], "share": round(d.get("share", 0), 4)}
            for d in drivers[:15]
        ],
    }


def tool_get_ops_hotspots(dimension: str | None = None, top_k: int = 5) -> dict[str, Any]:
    """
    Ops-first view: top pain points per dimension value.

    Args:
        dimension: specific dimension name (e.g. "vehicle_type"); None = all dimensions
        top_k: top pain points per dimension value
    """
    hotspots = get_ops_hotspots(dimension=dimension, top_k=top_k)
    return {"dimension": dimension, "top_k": top_k, "hotspots": hotspots}


def tool_get_exemplars(l5_id: str, k: int = EXEMPLAR_K) -> dict[str, Any]:
    """Return representative transcript snippets (evidence) for a pain point."""
    exs = get_exemplars(l5_id, k=k)
    return {
        "l5_id": l5_id,
        "count": len(exs),
        "exemplars": [
            {
                "conversation_id": e.get("conversation_id"),
                "snippet": e.get("snippet", ""),
                "severity": e.get("severity"),
                "sentiment": e.get("sentiment"),
                "week": e.get("week"),
            }
            for e in exs
        ],
    }


def tool_read_memory(l5_id: str) -> dict[str, Any]:
    """Return prior periods, insights, and driver trend history for an L5."""
    mem = read_memory(l5_id)
    return {
        "l5_id": l5_id,
        "pain_point": mem["pain_point"],
        "period_count": len(mem["period_metrics"]),
        "latest_period": mem["period_metrics"][-1] if mem["period_metrics"] else {},
        "prior_insights": [
            {
                "summary": i.get("summary"),
                "root_cause": i.get("root_cause"),
                "recommended_action": i.get("recommended_action"),
                "status": i.get("status"),
                "run_id": i.get("run_id"),
            }
            for i in mem["prior_insights"]
        ],
        "driver_trends": mem["driver_trends"][:5],  # top 5 trending drivers
    }


def tool_get_curated_set(cap: int = CURATED_CAP) -> dict[str, Any]:
    """Return the importance-tagged curated set of pain points.

    Each entry carries an importance tier (very_high/high/fair/low), its coarse
    trend, and how it moved since the last run (new/escalated/de-escalated/
    stable). Use this as the backbone of the report — promote/keep the
    very_high and escalated items, summarise the rest.
    """
    curated = get_curated_set(cap=cap)
    by_tier: dict[str, int] = {}
    for c in curated:
        by_tier[c.get("importance") or "low"] = by_tier.get(c.get("importance") or "low", 0) + 1
    return {"cap": cap, "count": len(curated), "by_tier": by_tier, "curated": curated}


def tool_get_systemic_drivers(only_bridges: bool = False, top_k: int = 10) -> dict[str, Any]:
    """Return systemic operational drivers ("bridges") that affect many L5s.

    A bridge is one dimension value (e.g. carrier=DroneX) that is a significant
    driver across multiple pain points — fixing it relieves many issues at once.
    """
    drivers = get_systemic_drivers(only_bridges=only_bridges, top_k=top_k)
    return {
        "only_bridges": only_bridges,
        "count": len(drivers),
        "drivers": [
            {
                "dimension": d.get("dimension"),
                "value": d.get("value"),
                "n_significant_l5": d.get("n_significant_l5"),
                "affected_l5s": d.get("affected_l5s", []),
                "mean_lift": d.get("mean_lift"),
                "systemic_score": d.get("systemic_score"),
                "is_bridge": d.get("is_bridge"),
            }
            for d in drivers
        ],
    }


def tool_get_l5_profile(l5_id: str) -> dict[str, Any]:
    """Return the deep-dive profile for a pain point.

    Use this to explain what is actually happening inside an L5: the
    compositional breakdown (which vendor / product_category / fulfillment /
    region groups make it up, with share and over-index lift) and the emergent
    sub-themes (the distinctive issues in words, each with a count, share, and a
    representative quote). This is the investigator's view.
    """
    return get_l5_profile(l5_id)


def tool_verify_claim(l5_id: str, dimension: str, value: str) -> dict[str, Any]:
    """Verify a driver claim against gbrain before stating it in the report.

    Returns verdict ("verified"/"unsupported") plus the lift, support, and
    p-value you must cite. Always verify a driver before asserting it.
    """
    return verify_driver_claim(l5_id=l5_id, dimension=dimension, value=value)


def tool_search_transcripts(query: str, l5_id: str | None = None, k: int = 5) -> dict[str, Any]:
    """Grep raw transcript evidence (exemplar snippets) for a keyword query.

    Use this to find concrete quotes that support a claim. Scope to one pain
    point by passing its l5_id.
    """
    hits = search_transcripts(query=query, l5_id=l5_id, k=k)
    return {"query": query, "l5_id": l5_id, "count": len(hits), "hits": hits}


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------

def tool_write_insight(
    l5_id: str,
    run_id: str,
    summary: str,
    root_cause: str = "",
    recommended_action: str = "",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist an insight about a pain point into gbrain."""
    insight_id = write_insight(
        l5_id=l5_id,
        run_id=run_id,
        summary=summary,
        root_cause=root_cause,
        recommended_action=recommended_action,
        evidence_ids=evidence_ids or [],
    )
    return {"status": "saved", "insight_id": insight_id, "l5_id": l5_id}
