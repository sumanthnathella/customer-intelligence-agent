"""
agent/report.py — Render a quality report from gbrain without interactive ADK.

This module assembles the report by directly reading gbrain (no LLM calls required).
It produces both Markdown and JSON artifacts.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from gbrain.retrieval import (
    get_curated_set,
    get_ops_hotspots,
    get_systemic_drivers,
    get_zscore_spikes,
    verify_driver_claim,
)
from gbrain.store import get_store
from shared.config import CURATED_CAP, REPORTS_DIR, TOP_L5_N

logger = logging.getLogger(__name__)


def _fmt_path(path: list[str]) -> str:
    return " ▸ ".join(path) if path else "unknown"


_TREND_ARROW = {"rising": "↑", "falling": "↓", "stable": "→"}
_TIER_LABEL = {
    "very_high": "VERY HIGH",
    "high": "HIGH",
    "fair": "FAIR",
    "low": "LOW",
}
_MOVE_LABEL = {
    "new": "NEW",
    "escalated": "▲ ESCALATED",
    "de-escalated": "▼ DE-ESCALATED",
    "stable": "stable",
}


def _fmt_delta(delta: int | float | None) -> str:
    if delta is None:
        return ""
    if delta > 0:
        return f" (+{int(delta)})"
    if delta < 0:
        return f" ({int(delta)})"
    return " (±0)"


# Dimensions most useful for a "what's inside" narrative, in display order.
_FACET_PRIORITY = [
    "product_category", "vendor", "fulfillment_type", "carrier",
    "service_level", "region", "customer_segment", "device_type",
    "payment_method", "season", "vehicle_type",
]


def _render_composition(facets: dict[str, Any]) -> list[str]:
    """One-line-per-dimension composition of dominant, over-indexing groups."""
    if not facets:
        return []
    rows: list[tuple[float, str]] = []
    for dim in _FACET_PRIORITY:
        vals = facets.get(dim)
        if not vals:
            continue
        top = vals[0]
        share, lift = top.get("share", 0), top.get("lift", 1.0)
        # Only surface a facet that is both common and over-indexed.
        if share < 0.15 or lift < 1.3:
            continue
        flag = " *(over-indexed)*" if top.get("significant") else ""
        rows.append(
            (lift, f"`{dim}={top['value']}` — {share*100:.0f}% of contacts, {lift:.1f}× vs baseline{flag}")
        )
    rows.sort(key=lambda r: r[0], reverse=True)
    return [r[1] for r in rows[:5]]


def _render_subthemes(subthemes: list[dict[str, Any]]) -> list[str]:
    """Bulleted sub-themes in words, each with count, share, and a quote."""
    out: list[str] = []
    for t in subthemes[:5]:
        snippet = (t.get("quote", "") or "").replace("\n", " ").strip()
        quote = f' — _"{snippet}"_' if snippet else ""
        out.append(
            f"  - **{t.get('label')}** — {t.get('count')} contacts "
            f"({t.get('share', 0)*100:.0f}%){quote}"
        )
    return out


def _render_report(
    curated: list[dict[str, Any]],
    bridges: list[dict[str, Any]],
    spikes: list[dict[str, Any]],
    ops_hotspots: dict[str, Any],
    period: str,
    store: Any = None,
) -> str:
    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# Customer Intelligence Report — {period}")
    lines.append(f"_Generated {now}_\n")

    # ---- Executive summary -------------------------------------------------
    tier_counts: dict[str, int] = {}
    for c in curated:
        tier_counts[c.get("importance") or "low"] = tier_counts.get(c.get("importance") or "low", 0) + 1
    escalations = [c for c in curated if c.get("movement") in ("escalated", "new")]

    lines.append("## Executive Summary\n")
    lines.append(
        f"Tracking **{len(curated)}** curated pain points — "
        f"{tier_counts.get('very_high', 0)} very-high, {tier_counts.get('high', 0)} high, "
        f"{tier_counts.get('fair', 0)} fair."
    )
    lines.append(
        f"**{len(bridges)}** systemic *bridge* drivers affect multiple pain points; "
        f"**{len(spikes)}** show a significant recent spike (z ≥ 2.0); "
        f"**{len(escalations)}** items are new or escalated since the last run.\n"
    )

    # ---- Systemic drivers (highest leverage — lead with these) -------------
    if bridges:
        lines.append("## Systemic Operational Drivers (Bridges)\n")
        lines.append(
            "_One lever, many pain points. These dimension values over-index as "
            "significant drivers across multiple L5s — fix once, relieve many._\n"
        )
        for b in bridges[:8]:
            affected = b.get("affected_l5s", [])
            leaves = ", ".join(a.split("__")[-1] for a in affected[:6])
            more = f" (+{len(affected) - 6} more)" if len(affected) > 6 else ""
            lines.append(
                f"- **`{b.get('dimension')}={b.get('value')}`** — drives "
                f"**{b.get('n_significant_l5')}** pain points, mean lift "
                f"{b.get('mean_lift', 0):.1f}×, systemic score {b.get('systemic_score', 0):.2f}"
            )
            lines.append(f"  - Affected: {leaves}{more}")
        lines.append("")

    # ---- Curated pain points (importance-tagged, with movement) -----------
    lines.append("## Curated Pain Points\n")
    for rank, c in enumerate(curated[:TOP_L5_N], 1):
        l5_id = c.get("l5_id", "")
        tier = _TIER_LABEL.get(c.get("importance") or "low", "LOW")
        move = _MOVE_LABEL.get(c.get("movement") or "stable", "stable")
        arrow = _TREND_ARROW.get(c.get("trend") or "stable", "→")
        egreg = c.get("egregiousness", 0) or 0
        vol = c.get("volume", 0) or 0
        sev = c.get("severity_avg", 0) or 0
        z = c.get("zscore", 0) or 0

        lines.append(f"### {rank}. [{tier}] {_fmt_path(c.get('path', []))}  {arrow}")
        lines.append(f"- **ID:** `{l5_id}`  ·  _{move}_")
        lines.append(
            f"- **Egregiousness:** {egreg:.4f} "
            f"(volume={vol}{_fmt_delta(c.get('volume_delta'))}, severity={sev:.2f}, z={z:.2f})"
        )

        # What's inside — compositional groups (which vendor/category/fulfillment).
        composition = _render_composition(c.get("facets", {}))
        if composition:
            lines.append("- **What's inside (dominant groups):**")
            for row in composition:
                lines.append(f"  - {row}")

        # Sub-themes — the distinctive issues in words, with quotes.
        subthemes = _render_subthemes(c.get("subthemes", []))
        if subthemes:
            lines.append("- **Issue themes:**")
            lines.extend(subthemes)
        elif vol:
            lines.append("- **Issue themes:** _too few contacts to characterise distinct themes._")

        top_d = c.get("top_drivers", [])
        if top_d:
            lines.append("- **Verified drivers (statistical):**")
            for d in top_d[:5]:
                badge = "✓"
                if store is not None:
                    v = verify_driver_claim(l5_id, d["dimension"], d["value"], store=store)
                    badge = "✓" if v.get("verdict") == "verified" else "✗"
                lines.append(
                    f"  - {badge} `{d['dimension']}={d['value']}` — lift {d['lift']:.1f}×, "
                    f"n={d['support']}, p={d['p_value']:.4f}"
                )
        lines.append("")

    # ---- Spikes ------------------------------------------------------------
    if spikes:
        lines.append("## Recent Spikes (z-score ≥ 2.0)\n")
        for node in spikes[:10]:
            props = node.get("props", {})
            lines.append(
                f"- `{props.get('l5_id')}` — z={props.get('latest_zscore', 0):.2f}, "
                f"volume={props.get('total_volume', 0)}"
            )
        lines.append("")

    # ---- Operations Hotspots (ops-first) ----------------------------------
    lines.append("## Operations Hotspots\n")
    for dim_val, l5s in list(ops_hotspots.get("hotspots", {}).items())[:10]:
        if not l5s:
            continue
        lines.append(f"### {dim_val}")
        for cell in l5s[:5]:
            lines.append(
                f"- `{cell.get('l5_id')}` — support={cell.get('support')}, "
                f"lift={cell.get('lift', 0):.1f}×, significant={'✓' if cell.get('significant') else '✗'}"
            )
        lines.append("")

    # ---- Methodology -------------------------------------------------------
    lines.append("## Methodology\n")
    lines.append(
        "- **Severity:** deterministic rubric over structured signals (churn_intent, financial_harm, safety_legal, repeat_contact, unresolved).\n"
        "- **Egregiousness:** weighted percentile blend of volume (35%), mean severity (25%), recent z-score (25%), and order value at risk (15%).\n"
        "- **Drivers:** over-indexing lift + two-proportion z-test with Benjamini–Hochberg FDR correction; every cited driver is verified against gbrain (✓/✗).\n"
        "- **What's inside:** compositional breakdown of each pain point by operational group (vendor, product_category, fulfillment, etc.) with share + over-index lift.\n"
        "- **Issue themes:** distinctive sub-themes mined from the transcripts via TF-IDF (this L5 vs the rest of the corpus), each with a contact count and representative quote.\n"
        "- **Bridges:** dimension values that are significant drivers across many L5s (systemic levers).\n"
        "- **Curated set:** importance-tiered and carried across runs, so movement (new / escalated / de-escalated) reflects real change.\n"
        "- **Evidence:** transcript snippets compressed to the most relevant sentences (BM25).\n"
        "- **All metrics persisted to gbrain** as durable L5 entities with rolling history.\n"
    )

    return "\n".join(lines)


def render(run_id: str | None = None) -> dict[str, Any]:
    """
    Read gbrain and render the report.

    Returns {"markdown": str, "json": dict, "paths": [str, str]}
    """
    logger.info("Rendering report from gbrain...")

    store = get_store()
    curated = get_curated_set(store=store, cap=CURATED_CAP)
    bridges = get_systemic_drivers(store=store, only_bridges=True, top_k=10)
    spikes = get_zscore_spikes(store=store, threshold=2.0)
    ops = get_ops_hotspots(store=store, dimension=None, top_k=5)

    # Determine period from the top curated pain point.
    period = curated[0].get("last_seen") if curated else None
    if not period and curated:
        # last_seen isn't in the curated projection; read from the node directly.
        node = store.get_node(curated[0]["l5_id"]) if curated else None
        period = (node["props"].get("last_seen") if node else None) or "unknown"
    period = period or "unknown"

    md = _render_report(curated, bridges, spikes, ops, period=period, store=store)

    # JSON artifact
    json_payload = {
        "period": period,
        "generated_at": datetime.now(UTC).isoformat(),
        "curated": curated[:TOP_L5_N],
        "bridges": bridges,
        "spikes": [
            {"l5_id": n["props"].get("l5_id"), "zscore": n["props"].get("latest_zscore")}
            for n in spikes[:10]
        ],
        "ops_hotspots": ops.get("hotspots", {}),
    }

    run_id = run_id or period
    md_path = REPORTS_DIR / f"report_{run_id}.md"
    json_path = REPORTS_DIR / f"report_{run_id}.json"

    md_path.write_text(md)
    json_path.write_text(json.dumps(json_payload, indent=2, default=str))

    logger.info("Report saved:\n  %s\n  %s", md_path, json_path)
    return {"markdown": md, "json": json_payload, "paths": [str(md_path), str(json_path)]}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Render report from gbrain")
    parser.add_argument("--run-id", default=None, help="Run ID for report naming")
    args = parser.parse_args()
    result = render(run_id=args.run_id)
    print(f"Report rendered:\n  {result['paths'][0]}\n  {result['paths'][1]}")


if __name__ == "__main__":
    main()
