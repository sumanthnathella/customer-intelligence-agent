"""
analytics/severity.py — Deterministic severity rubric.

Computes severity ∈ [1, 5] from structured signal columns.
No LLM involvement; weights live in shared/config.py.
"""
from __future__ import annotations

import pandas as pd

from shared.config import (
    SEVERITY_BASE,
    SEVERITY_MAX,
    SEVERITY_MIN,
    SEVERITY_SIGNAL_WEIGHTS,
)


def compute_severity(df: pd.DataFrame) -> pd.Series:
    """
    Vectorised severity computation over a DataFrame (Table A).

    Expected columns: sentiment, churn_intent, financial_harm, safety_legal,
                      repeat_contact, unresolved.

    If 'severity' column already exists and is non-null, it is used directly
    (BYO users who provide a pre-computed severity skip the rubric).

    Returns a float Series, index-aligned with df.
    """
    # BYO pass-through
    if "severity" in df.columns and df["severity"].notna().all():
        return df["severity"].astype(float)

    base = df["sentiment"].map(SEVERITY_BASE).fillna(1).astype(int)

    # +1 for churn_intent
    signal = df.get("churn_intent", pd.Series(0, index=df.index)).fillna(0).astype(int) * SEVERITY_SIGNAL_WEIGHTS["churn_intent"]
    # +1 for financial_harm
    signal += df.get("financial_harm", pd.Series(0, index=df.index)).fillna(0).astype(int) * SEVERITY_SIGNAL_WEIGHTS["financial_harm"]
    # +2 for safety_legal
    signal += df.get("safety_legal", pd.Series(0, index=df.index)).fillna(0).astype(int) * SEVERITY_SIGNAL_WEIGHTS["safety_legal"]
    # +1 for (repeat_contact OR unresolved) — at most +1 total
    repeat = df.get("repeat_contact", pd.Series(0, index=df.index)).fillna(0).astype(int)
    unresolved = df.get("unresolved", pd.Series(0, index=df.index)).fillna(0).astype(int)
    signal += ((repeat | unresolved) > 0).astype(int)

    raw = base + signal
    return raw.clip(SEVERITY_MIN, SEVERITY_MAX).astype(float)


def add_severity_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a 'severity' column added (or overwritten if already 0-null)."""
    df = df.copy()
    df["severity"] = compute_severity(df)
    return df
