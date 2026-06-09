"""
analytics/metrics.py — Weekly volume, z-score, and per-L5 aggregates.

Uses real created_at timestamps (ISO weeks) — no synthetic bucketing here.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from shared.config import (
    ZSCORE_BASELINE_WEEKS,
    ZSCORE_MIN_PERIODS,
    ZSCORE_STD_FLOOR,
)
from shared.schemas import WeeklyMetric, ZScoreResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------------------------

def build_weekly_metrics(
    df: pd.DataFrame,
    order_value_col: str | None = None,
) -> pd.DataFrame:
    """
    Bucket rows by (l5_id, ISO week) and aggregate.

    Returns a DataFrame with columns:
        l5_id, week, volume, severity_avg, sentiment_avg, order_value_sum
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    # ISO year-week label (e.g. "2017-W41"). Using isocalendar avoids the
    # year-boundary mismatch between %Y and %V and the tz-drop warning from
    # Period conversion.
    iso = df["created_at"].dt.isocalendar()
    df["week"] = (
        iso["year"].astype(int).astype(str)
        + "-W"
        + iso["week"].astype(int).astype(str).str.zfill(2)
    )

    # numeric sentiment for averaging
    sent_map = {"very_neg": 1, "neg": 2, "neutral": 3, "pos": 4}
    df["sentiment_num"] = df["sentiment"].map(sent_map).fillna(2.5)

    agg: dict[str, Any] = {
        "volume": ("conversation_id", "count"),
        "severity_avg": ("severity", "mean"),
        "sentiment_avg": ("sentiment_num", "mean"),
    }
    if order_value_col and order_value_col in df.columns:
        agg["order_value_sum"] = (order_value_col, "sum")

    weekly = df.groupby(["l5_id", "week"]).agg(**agg).reset_index()

    if "order_value_sum" not in weekly.columns:
        weekly["order_value_sum"] = 0.0

    return weekly


def weekly_metrics_to_records(df_weekly: pd.DataFrame) -> list[WeeklyMetric]:
    records = []
    for _, row in df_weekly.iterrows():
        records.append(
            WeeklyMetric(
                l5_id=row["l5_id"],
                week=row["week"],
                volume=int(row["volume"]),
                severity_avg=float(row["severity_avg"]),
                sentiment_avg=float(row["sentiment_avg"]),
                order_value_sum=float(row.get("order_value_sum", 0.0)),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Z-score per L5 (latest week vs trailing baseline)
# ---------------------------------------------------------------------------

def compute_zscores(
    df_weekly: pd.DataFrame,
    baseline_weeks: int = ZSCORE_BASELINE_WEEKS,
    min_periods: int = ZSCORE_MIN_PERIODS,
    std_floor: float = ZSCORE_STD_FLOOR,
) -> dict[str, ZScoreResult]:
    """
    Compute z-score for the latest week of each L5 vs its trailing baseline.

    Returns {l5_id: ZScoreResult}.
    """
    results: dict[str, ZScoreResult] = {}
    for l5_id, group in df_weekly.groupby("l5_id"):
        grp = group.sort_values("week")
        weeks = grp["week"].tolist()
        volumes = grp["volume"].tolist()

        if len(weeks) < min_periods:
            zscore = 0.0
            baseline_mean = float(np.mean(volumes))
            baseline_std = 0.0
        else:
            latest_vol = volumes[-1]
            baseline = volumes[-(baseline_weeks + 1):-1]
            if len(baseline) < min_periods:
                zscore = 0.0
                baseline_mean = float(np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0])
                baseline_std = 0.0
            else:
                baseline_mean = float(np.mean(baseline))
                baseline_std = max(float(np.std(baseline, ddof=1)), std_floor)
                zscore = (latest_vol - baseline_mean) / baseline_std

        results[str(l5_id)] = ZScoreResult(
            l5_id=str(l5_id),
            latest_week=weeks[-1],
            latest_volume=int(volumes[-1]),
            zscore=float(zscore),
            baseline_mean=float(baseline_mean),
            baseline_std=float(baseline_std),
        )
    return results


# ---------------------------------------------------------------------------
# Latest-period aggregates per L5 (for egregiousness input)
# ---------------------------------------------------------------------------

def latest_period_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over the FULL window per L5 (not just the latest week).
    Returns: l5_id, total_volume, severity_avg, sentiment_avg, order_value_sum
    """
    agg: dict[str, Any] = {
        "total_volume": ("conversation_id", "count"),
        "severity_avg": ("severity", "mean"),
        "sentiment_avg": ("sentiment_num" if "sentiment_num" in df.columns else "severity", "mean"),
    }
    if "order_value_sum" in df.columns:
        agg["order_value_sum"] = ("order_value_sum", "sum")
    elif "order_total" in df.columns:
        agg["order_value_sum"] = ("order_total", "sum")

    result = df.groupby("l5_id").agg(**agg).reset_index()
    if "order_value_sum" not in result.columns:
        result["order_value_sum"] = 0.0
    return result
