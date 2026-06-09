"""
analytics/build_brain.py — Orchestrate one analytics run → gbrain update.

Called by the top-level build.py; reads Table A + Table B parquets, runs the
full analytics pipeline, and writes into gbrain.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.drivers import compute_drivers, facet_breakdown, top_drivers_summary
from analytics.egregiousness import compute_egregiousness
from analytics.metrics import build_weekly_metrics, compute_zscores, latest_period_stats
from analytics.severity import add_severity_column
from analytics.subthemes import compute_subthemes
from analytics.systemic import (
    compute_systemic_drivers,
    importance_tier,
    zscore_trend,
)
from gbrain.graph import (
    auto_link_taxonomy,
    record_verification,
    upsert_affects_edge,
    upsert_dimension_systemic,
    upsert_exemplar,
    upsert_pain_point,
    upsert_period_metric,
)
from gbrain.schema_pack import load_pack
from gbrain.store import GBrainStore, get_store
from shared.config import (
    DRIVER_MIN_SUPPORT,
    EXEMPLAR_COMPRESS_SENTENCES,
    EXEMPLAR_K,
    PACK_VERSION,
    TAXONOMY_PATH,
    ZSCORE_SPIKE_THRESHOLD,
)
from shared.contract import validate
from shared.text import compress_to_sentences

logger = logging.getLogger(__name__)


def run(
    table_a_path: str | Path,
    table_b_path: str | Path,
    taxonomy_path: str | Path = TAXONOMY_PATH,
    dimension_cols: list[str] | None = None,
    order_value_col: str | None = None,
    run_id: str | None = None,
    store: GBrainStore | None = None,
) -> dict[str, Any]:
    """
    Execute one full analytics run and update gbrain.

    Parameters
    ----------
    table_a_path: parquet with tagged transcripts (Table A).
    table_b_path: parquet with operational dimensions (Table B).
    taxonomy_path: path to taxonomy.json schema pack.
    dimension_cols: list of categorical dimension column names in Table B.
                    If None, auto-inferred from object/category dtypes.
    order_value_col: numeric column in Table B for order value (optional).
    run_id: unique run identifier; auto-generated if None.
    store: GBrainStore instance; uses module singleton if None.

    Returns summary dict.
    """
    run_id = run_id or f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    s = store or get_store()

    # ------------------------------------------------------------------
    # 1. Load + validate
    # ------------------------------------------------------------------
    logger.info("[%s] Loading Table A: %s", run_id, table_a_path)
    df_a = pd.read_parquet(table_a_path)
    logger.info("[%s] Loading Table B: %s", run_id, table_b_path)
    df_b = pd.read_parquet(table_b_path)

    df_a, df_b = validate(df_a, df_b, dimension_cols=dimension_cols)

    # ------------------------------------------------------------------
    # 2. Auto-detect dimension cols if not specified
    # ------------------------------------------------------------------
    if dimension_cols is None:
        dimension_cols = [
            c for c in df_b.columns
            if c != "conversation_id" and (
                pd.api.types.is_object_dtype(df_b[c])
                or isinstance(df_b[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(df_b[c])
                or str(df_b[c].dtype).startswith("category")
            )
        ]
        logger.info("[%s] Auto-detected dimension columns: %s", run_id, dimension_cols)

    # ------------------------------------------------------------------
    # 3. Severity
    # ------------------------------------------------------------------
    logger.info("[%s] Computing severity...", run_id)
    df_a = add_severity_column(df_a)

    # ------------------------------------------------------------------
    # 4. Weekly metrics + z-scores
    # ------------------------------------------------------------------
    logger.info("[%s] Building weekly metrics...", run_id)
    df_weekly = build_weekly_metrics(df_a, order_value_col=order_value_col)
    zscores = compute_zscores(df_weekly)

    # Add sentiment_num for latest_period_stats
    sent_map = {"very_neg": 1, "neg": 2, "neutral": 3, "pos": 4}
    df_a["sentiment_num"] = df_a["sentiment"].map(sent_map).fillna(2.5)
    period_stats = latest_period_stats(df_a)

    # ------------------------------------------------------------------
    # 5. Egregiousness
    # ------------------------------------------------------------------
    logger.info("[%s] Computing egregiousness...", run_id)
    egreg_scores = compute_egregiousness(period_stats, zscores)
    egreg_map = {e.l5_id: e for e in egreg_scores}

    # ------------------------------------------------------------------
    # 6. Merge with dimensions for driver analysis
    # ------------------------------------------------------------------
    logger.info("[%s] Merging Table A + B for driver analysis...", run_id)
    df_merged = df_a.merge(df_b, on="conversation_id", how="inner")

    # determine analysis period (latest ISO week)
    latest_week = df_weekly["week"].max()

    # Load existing affects edge history for trend accumulation
    existing_history: dict[str, list] = {}
    for l5_id in df_a["l5_id"].unique():
        edges = s.get_edges(str(l5_id), edge_type="affects", direction="out")
        for edge in edges:
            key = f"{l5_id}:affects:{edge['dst']}"
            existing_history[key] = edge["props"].get("history", [])

    # ------------------------------------------------------------------
    # 7. Driver analysis
    # ------------------------------------------------------------------
    logger.info("[%s] Running driver analysis across %d dimensions...", run_id, len(dimension_cols))
    driver_edges = compute_drivers(
        df_merged,
        dimension_cols=dimension_cols,
        period=latest_week,
        existing_history=existing_history,
    )
    # Group by l5_id
    drivers_by_l5: dict[str, list] = {}
    for edge in driver_edges:
        drivers_by_l5.setdefault(edge.l5_id, []).append(edge)

    # ------------------------------------------------------------------
    # 8. Load taxonomy schema pack
    # ------------------------------------------------------------------
    try:
        pack = load_pack(taxonomy_path)
        pack_version = pack.version
    except FileNotFoundError:
        logger.warning("[%s] Taxonomy not found at %s; L5 path/definition will be empty.", run_id, taxonomy_path)
        pack = None
        pack_version = PACK_VERSION

    # ------------------------------------------------------------------
    # 9. Write to gbrain (one L5 at a time — durable upsert)
    # ------------------------------------------------------------------
    logger.info("[%s] Writing to gbrain...", run_id)

    # Emergent sub-themes (in words) per L5 — distinctive issues within each
    # pain point, mined once over the full window.
    logger.info("[%s] Mining sub-themes from transcripts...", run_id)
    subthemes_by_l5 = compute_subthemes(df_a)

    definition_by_l5: dict[str, str] = {}
    n_verifications = 0

    for _, row in period_stats.iterrows():
        l5_id = str(row["l5_id"])
        z_result = zscores.get(l5_id)
        egreg = egreg_map.get(l5_id)
        if z_result is None or egreg is None:
            continue

        # Schema pack lookups
        path: list[str] = []
        definition: str = ""
        if pack:
            node = pack.get_l5(l5_id)
            if node:
                path = node.get("path", [])
                definition = node.get("definition", "")
            else:
                path = [l5_id]
        definition_by_l5[l5_id] = definition or l5_id

        l5_drivers = drivers_by_l5.get(l5_id, [])
        top_d = top_drivers_summary(l5_drivers)

        # Curated-set warm-start tag + coarse trend (Harness-1 inspired).
        importance = importance_tier(float(egreg.egregiousness))
        trend = zscore_trend(float(z_result.zscore), spike=ZSCORE_SPIKE_THRESHOLD)

        # Deep-dive profile: compositional facets (which groups) + sub-themes
        # (what issues, in words).
        facets = facet_breakdown(l5_drivers)
        subthemes = subthemes_by_l5.get(l5_id, [])

        upsert_pain_point(
            s,
            l5_id=l5_id,
            path=path,
            definition=definition,
            pack_version=pack_version,
            total_volume=int(row["total_volume"]),
            severity_avg=float(row["severity_avg"]),
            latest_zscore=float(z_result.zscore),
            latest_egregiousness=float(egreg.egregiousness),
            top_drivers=top_d,
            importance=importance,
            trend=trend,
            facets=facets,
            subthemes=subthemes,
            last_seen=latest_week,
        )

        # Verification cache: deterministically check each cited top driver
        # against the data (significant ∧ lift > 1 ∧ support ≥ min) and persist
        # the claim→evidence→verdict record.
        for d in top_d[:3]:
            verdict = bool(d.get("lift", 0) > 1.0 and d.get("support", 0) >= DRIVER_MIN_SUPPORT)
            claim = f"{d['dimension']}={d['value']} over-indexes for {l5_id} (lift {d['lift']}x, n={d['support']})"
            record_verification(
                s,
                l5_id=l5_id,
                claim=claim,
                verdict=verdict,
                detail={k: d.get(k) for k in ("dimension", "value", "lift", "support", "p_value")},
                evidence_ids=[f"dim:{d['dimension']}:{d['value']}"],
                run_id=run_id,
            )
            n_verifications += 1

        # Period metric
        weekly_row = df_weekly[(df_weekly["l5_id"] == l5_id) & (df_weekly["week"] == latest_week)]
        if not weekly_row.empty:
            from shared.schemas import WeeklyMetric
            wm = WeeklyMetric(
                l5_id=l5_id,
                week=latest_week,
                volume=int(weekly_row["volume"].iloc[0]),
                severity_avg=float(weekly_row["severity_avg"].iloc[0]),
                sentiment_avg=float(weekly_row["sentiment_avg"].iloc[0]),
                order_value_sum=float(weekly_row.get("order_value_sum", pd.Series([0])).iloc[0]),
            )
            upsert_period_metric(s, wm, z_result, egreg)

        # Affects edges (full dimension heatmap)
        for driver_edge in l5_drivers:
            upsert_affects_edge(s, driver_edge)

        # Taxonomy auto-link
        if pack and len(path) >= 2:
            auto_link_taxonomy(s, l5_id, path)

    # ------------------------------------------------------------------
    # 9b. Systemic (bridge/singleton) driver rollup across all L5s
    # ------------------------------------------------------------------
    logger.info("[%s] Computing systemic (bridge) drivers...", run_id)
    systemic = compute_systemic_drivers(driver_edges)
    for sd in systemic:
        upsert_dimension_systemic(s, sd)
    n_bridges = sum(1 for sd in systemic if sd.is_bridge)

    # ------------------------------------------------------------------
    # 10. Exemplars (top-severity transcripts per L5, BM25-compressed)
    # ------------------------------------------------------------------
    logger.info("[%s] Adding exemplars...", run_id)
    for l5_id, grp in df_a.sort_values("severity", ascending=False).groupby("l5_id"):
        query = definition_by_l5.get(str(l5_id), str(l5_id))
        for _, row in grp.head(EXEMPLAR_K).iterrows():
            w = df_weekly.loc[
                (df_weekly["l5_id"] == l5_id) & (df_weekly["week"] == latest_week), "week"
            ]
            week = w.iloc[0] if not w.empty else latest_week
            snippet = compress_to_sentences(
                str(row.get("text", "")),
                query=query,
                k=EXEMPLAR_COMPRESS_SENTENCES,
                max_chars=500,
            )
            upsert_exemplar(
                s,
                conversation_id=row["conversation_id"],
                l5_id=str(l5_id),
                snippet=snippet,
                severity=float(row["severity"]),
                sentiment=str(row["sentiment"]),
                week=week,
            )

    # ------------------------------------------------------------------
    # 11. Register run + snapshot
    # ------------------------------------------------------------------
    s.register_run(
        run_id=run_id,
        window=latest_week,
        n=len(df_a),
        pack_version=pack_version,
    )
    snapshot_path = s.snapshot(run_id)

    summary = {
        "run_id": run_id,
        "period": latest_week,
        "n_transcripts": len(df_a),
        "n_l5": len(period_stats),
        "n_driver_edges": len(driver_edges),
        "n_significant_drivers": sum(1 for e in driver_edges if e.significant),
        "n_bridges": n_bridges,
        "n_verifications": n_verifications,
        "top_egregious": [e.l5_id for e in egreg_scores[:5]],
        "top_bridges": [f"{sd.dimension}={sd.value}" for sd in systemic if sd.is_bridge][:5],
        "snapshot": str(snapshot_path),
    }
    logger.info("[%s] Done. %s", run_id, summary)
    return summary
