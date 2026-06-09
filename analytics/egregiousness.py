"""
analytics/egregiousness.py — Percentile-blended egregiousness score per L5.

egregiousness(L5) = 0.35·pct(volume)
                  + 0.25·pct(mean_severity)
                  + 0.25·pct(zscore_spike)
                  + 0.15·pct(order_value_at_risk)

If no monetary dimension is available, the value weight redistributes to the others.
Weights configurable in shared/config.py.
"""
from __future__ import annotations

import pandas as pd

from shared.config import (
    EGREG_WEIGHT_SEVERITY,
    EGREG_WEIGHT_SPIKE,
    EGREG_WEIGHT_VALUE,
    EGREG_WEIGHT_VOLUME,
)
from shared.schemas import EgregScore, ZScoreResult


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Rank-normalize a series to [0, 1] (fractional rank / n)."""
    n = len(series)
    if n <= 1:
        return pd.Series([0.5] * n, index=series.index)
    ranks = series.rank(method="average", na_option="bottom")
    return (ranks - 1) / (n - 1)


def compute_egregiousness(
    stats: pd.DataFrame,
    zscores: dict[str, ZScoreResult],
    w_volume: float = EGREG_WEIGHT_VOLUME,
    w_severity: float = EGREG_WEIGHT_SEVERITY,
    w_spike: float = EGREG_WEIGHT_SPIKE,
    w_value: float = EGREG_WEIGHT_VALUE,
) -> list[EgregScore]:
    """
    stats: DataFrame with columns [l5_id, total_volume, severity_avg, order_value_sum]
    zscores: {l5_id: ZScoreResult}

    Returns list[EgregScore] sorted by egregiousness descending.
    """
    df = stats.copy()
    df = df.set_index("l5_id")

    # attach z-scores
    df["zscore"] = df.index.map(lambda lid: zscores.get(lid, None))
    df["zscore_val"] = df["zscore"].apply(
        lambda z: max(z.zscore, 0.0) if z is not None else 0.0
    )

    has_value = "order_value_sum" in df.columns and df["order_value_sum"].sum() > 0

    if not has_value:
        # redistribute value weight equally to volume + severity + spike
        extra = w_value / 3.0
        w_volume = w_volume + extra
        w_severity = w_severity + extra
        w_spike = w_spike + extra
        w_value = 0.0

    # percentile ranks
    pct_vol = _percentile_rank(df["total_volume"])
    pct_sev = _percentile_rank(df["severity_avg"])
    pct_spk = _percentile_rank(df["zscore_val"])
    pct_val = _percentile_rank(df["order_value_sum"]) if has_value else pd.Series(0.0, index=df.index)

    egreg = w_volume * pct_vol + w_severity * pct_sev + w_spike * pct_spk + w_value * pct_val

    results: list[EgregScore] = []
    for l5_id in df.index:
        results.append(
            EgregScore(
                l5_id=str(l5_id),
                egregiousness=float(round(egreg[l5_id], 4)),
                pct_volume=float(round(pct_vol[l5_id], 4)),
                pct_severity=float(round(pct_sev[l5_id], 4)),
                pct_spike=float(round(pct_spk[l5_id], 4)),
                pct_value=float(round(pct_val[l5_id], 4)) if has_value else 0.0,
            )
        )
    results.sort(key=lambda e: e.egregiousness, reverse=True)
    return results
