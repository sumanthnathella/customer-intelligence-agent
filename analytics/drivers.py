"""
analytics/drivers.py — Driver analysis: over-indexing lift + BH-FDR significance.

For each L5 × dimension-value cell:
    lift     = P(d | P) / P(d)
    support  = count(P ∧ d)
    excess   = support − expected
    p_value  = two-proportion z-test (P(P|d) vs P(P|¬d))

Benjamini-Hochberg FDR correction across all tested cells.
Persist ALL cells (not just significant) as affects edges — significant flag marks drivers.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from shared.config import DRIVER_FDR_ALPHA, DRIVER_MIN_SUPPORT, DRIVER_TOP_K_PER_DIM
from shared.schemas import DriverEdge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_drivers(
    df_merged: pd.DataFrame,
    dimension_cols: list[str],
    period: str,
    min_support: int = DRIVER_MIN_SUPPORT,
    fdr_alpha: float = DRIVER_FDR_ALPHA,
    top_k_per_dim: int = DRIVER_TOP_K_PER_DIM,
    existing_history: dict[str, list[dict[str, Any]]] | None = None,
) -> list[DriverEdge]:
    """
    df_merged: joined Table A + Table B, must have l5_id + dimension_cols + conversation_id.
    dimension_cols: categorical columns to analyse as drivers.
    period: ISO week string for the current run.
    existing_history: {edge_key → history_list} loaded from current affects edges.

    Returns list[DriverEdge] (full cube — all cells, significant flag set on real drivers).
    """
    if df_merged.empty:
        return []

    n_total = len(df_merged)
    results: list[dict[str, Any]] = []

    for dim in dimension_cols:
        if dim not in df_merged.columns:
            continue
        col = df_merged[dim].fillna("__missing__").astype(str)
        values = col.unique().tolist()

        for val in values:
            mask_d = col == val
            n_d = int(mask_d.sum())
            if n_d == 0:
                continue

            for l5_id, l5_group in df_merged.groupby("l5_id"):
                n_l5 = len(l5_group)
                p_l5 = n_l5 / n_total

                in_l5_and_d = ((df_merged["l5_id"] == l5_id) & mask_d).sum()
                support = int(in_l5_and_d)

                p_l5_given_d = support / n_d if n_d > 0 else 0.0
                expected = p_l5 * n_d
                lift = p_l5_given_d / p_l5 if p_l5 > 0 else 1.0
                excess = support - expected

                results.append(
                    {
                        "l5_id": str(l5_id),
                        "dimension": dim,
                        "value": val,
                        "support": support,
                        "n_d": n_d,
                        "n_l5": n_l5,
                        "n_total": n_total,
                        "lift": lift,
                        "excess": excess,
                        "p_l5_given_d": p_l5_given_d,
                        "p_l5_given_not_d": (n_l5 - support) / (n_total - n_d) if (n_total - n_d) > 0 else 0.0,
                        "p_value": 1.0,
                        "significant": False,
                    }
                )

    if not results:
        return []

    df_res = pd.DataFrame(results)

    # Filter for z-test (support ≥ min_support)
    testable = df_res["support"] >= min_support
    if testable.any():
        sub = df_res[testable].copy()
        pvals = []
        for _, row in sub.iterrows():
            p1 = row["p_l5_given_d"]
            p2 = row["p_l5_given_not_d"]
            n1 = int(row["n_d"])
            n2 = int(row["n_total"]) - n1
            if n1 < 5 or n2 < 5:
                pvals.append(1.0)
                continue
            # two-proportion z-test
            count = np.array([p1 * n1, p2 * n2])
            nobs = np.array([n1, n2])
            try:
                _, pval = proportions_ztest(count, nobs)
                pvals.append(float(pval) if np.isfinite(pval) else 1.0)
            except Exception:
                pvals.append(1.0)

        sub = sub.copy()
        sub["p_value"] = pvals

        # Benjamini-Hochberg FDR correction
        if len(pvals) > 1:
            _, corrected_pvals, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")
            sub["p_value"] = corrected_pvals

        sub["significant"] = sub["p_value"] < fdr_alpha
        df_res.loc[testable, "p_value"] = sub["p_value"].values
        df_res.loc[testable, "significant"] = sub["significant"].values

    # Share = support / l5 volume
    l5_volumes = df_merged.groupby("l5_id").size().to_dict()
    df_res["share"] = df_res.apply(
        lambda r: r["support"] / l5_volumes.get(r["l5_id"], 1), axis=1
    )

    # Build DriverEdge objects — top_k_per_dim per (l5_id, dimension) + all significant
    edge_list: list[DriverEdge] = []
    existing_history = existing_history or {}

    for (l5_id, dim), grp in df_res.groupby(["l5_id", "dimension"]):
        grp_sorted = grp.sort_values("support", ascending=False)
        keep_idx = set(grp_sorted.head(top_k_per_dim).index) | set(grp[grp["significant"]].index)
        kept = grp[grp.index.isin(keep_idx)]

        for _, row in kept.iterrows():
            edge_key = f"{l5_id}:affects:dim:{dim}:{row['value']}"
            hist = list(existing_history.get(edge_key, []))
            # append current period to history if not already present
            if not hist or hist[-1].get("period") != period:
                hist.append(
                    {
                        "period": period,
                        "support": int(row["support"]),
                        "share": round(float(row["share"]), 4),
                        "lift": round(float(row["lift"]), 4),
                    }
                )

            edge_list.append(
                DriverEdge(
                    l5_id=str(l5_id),
                    dimension=str(dim),
                    value=str(row["value"]),
                    support=int(row["support"]),
                    share=float(row["share"]),
                    lift=float(row["lift"]),
                    p_value=float(row["p_value"]),
                    significant=bool(row["significant"]),
                    excess=float(row["excess"]),
                    period=period,
                    history=hist,
                )
            )

    logger.info(
        "Driver analysis: %d cells (%d significant) across %d dimensions.",
        len(edge_list),
        sum(1 for e in edge_list if e.significant),
        len(dimension_cols),
    )
    return edge_list


# ---------------------------------------------------------------------------
# Top-driver summary for pain_point.top_drivers prop
# ---------------------------------------------------------------------------

def top_drivers_summary(edges: list[DriverEdge], k: int = 5) -> list[dict[str, Any]]:
    """Return top-k significant drivers sorted by lift, as dicts."""
    sig = [e for e in edges if e.significant]
    sig.sort(key=lambda e: e.lift, reverse=True)
    return [
        {
            "dimension": e.dimension,
            "value": e.value,
            "lift": round(e.lift, 2),
            "support": e.support,
            "p_value": round(e.p_value, 4),
        }
        for e in sig[:k]
    ]


def facet_breakdown(
    edges: list[DriverEdge],
    top_values_per_dim: int = 3,
    min_share: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    """Compositional breakdown of an L5: dominant value(s) per dimension.

    Answers "which groups make up this pain point" (e.g. fulfillment, vendor,
    product_category), with each value's share of the L5 plus its over-index
    lift and significance. Built from the already-computed driver cube.
    """
    by_dim: dict[str, list[DriverEdge]] = {}
    for e in edges:
        by_dim.setdefault(e.dimension, []).append(e)

    out: dict[str, list[dict[str, Any]]] = {}
    for dim, es in by_dim.items():
        ranked = sorted(es, key=lambda e: e.share, reverse=True)
        vals = [
            {
                "value": e.value,
                "share": round(e.share, 3),
                "lift": round(e.lift, 2),
                "significant": e.significant,
            }
            for e in ranked[:top_values_per_dim]
            if e.share >= min_share
        ]
        if vals:
            out[dim] = vals
    return out
