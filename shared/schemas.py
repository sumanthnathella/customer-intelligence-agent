"""Pydantic data schemas for the input contract (Table A + Table B) and analytics outputs."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Sentiment(StrEnum):
    very_neg = "very_neg"
    neg = "neg"
    neutral = "neutral"
    pos = "pos"


# ---------------------------------------------------------------------------
# Input contract — Table A (tagged transcripts)
# ---------------------------------------------------------------------------

class TaggedTranscript(BaseModel):
    """One row of Table A — produced by tagging/ or provided by BYO user."""

    conversation_id: str
    created_at: datetime
    text: str
    l5_id: str
    sentiment: Sentiment
    churn_intent: int = Field(ge=0, le=1)
    financial_harm: int = Field(ge=0, le=1)
    safety_legal: int = Field(ge=0, le=1)
    repeat_contact: int = Field(ge=0, le=1)
    unresolved: int = Field(ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: float | None = Field(default=None, ge=1.0, le=5.0)

    @field_validator("sentiment", mode="before")
    @classmethod
    def coerce_sentiment(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.lower()
        return v


# ---------------------------------------------------------------------------
# Input contract — Table B (operational dimensions)
# ---------------------------------------------------------------------------

class OperationalRow(BaseModel):
    """One row of Table B — arbitrary dimension columns joined on conversation_id."""

    conversation_id: str
    dimensions: dict[str, str | None] = Field(default_factory=dict)
    facts: dict[str, float | None] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analytics intermediate / output types
# ---------------------------------------------------------------------------

class SeverityScore(BaseModel):
    conversation_id: str
    severity: float = Field(ge=1.0, le=5.0)
    breakdown: dict[str, int | float]


class WeeklyMetric(BaseModel):
    l5_id: str
    week: str
    volume: int
    severity_avg: float
    sentiment_avg: float
    order_value_sum: float = 0.0


class ZScoreResult(BaseModel):
    l5_id: str
    latest_week: str
    latest_volume: int
    zscore: float
    baseline_mean: float
    baseline_std: float


class EgregScore(BaseModel):
    l5_id: str
    egregiousness: float = Field(ge=0.0, le=1.0)
    pct_volume: float
    pct_severity: float
    pct_spike: float
    pct_value: float


class DriverEdge(BaseModel):
    """One (L5 × dimension-value) heatmap cell."""

    l5_id: str
    dimension: str
    value: str
    support: int
    share: float
    lift: float
    p_value: float
    significant: bool
    excess: float
    period: str
    history: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# gbrain node/edge canonical dicts (lightweight — store uses plain dicts)
# ---------------------------------------------------------------------------

class InsightRecord(BaseModel):
    insight_id: str
    l5_id: str
    summary: str
    root_cause: str = ""
    recommended_action: str = ""
    status: str = "new"
    run_id: str
    evidence: list[str] = Field(default_factory=list)
