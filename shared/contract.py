"""Input contract validator — validates Table A + Table B before the core runs."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from shared.schemas import Sentiment

logger = logging.getLogger(__name__)

# Required columns per table
TABLE_A_REQUIRED = {
    "conversation_id",
    "created_at",
    "text",
    "l5_id",
    "sentiment",
    "churn_intent",
    "financial_harm",
    "safety_legal",
    "repeat_contact",
    "unresolved",
}

TABLE_B_REQUIRED = {"conversation_id"}

SIGNAL_COLS = {"churn_intent", "financial_harm", "safety_legal", "repeat_contact", "unresolved"}
VALID_SENTIMENTS = {s.value for s in Sentiment}


class ContractError(ValueError):
    """Raised when input data violates the contract."""


def validate_table_a(df: pd.DataFrame) -> pd.DataFrame:
    """Validate tagged transcripts (Table A). Returns cleaned df or raises ContractError."""
    missing = TABLE_A_REQUIRED - set(df.columns)
    if missing:
        raise ContractError(f"Table A missing required columns: {sorted(missing)}")

    # created_at
    try:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    except Exception as exc:
        raise ContractError(f"Table A: 'created_at' cannot be parsed as datetime: {exc}") from exc

    # sentiment values
    bad_sentiment = ~df["sentiment"].isin(VALID_SENTIMENTS)
    if bad_sentiment.any():
        examples = df.loc[bad_sentiment, "sentiment"].unique()[:5].tolist()
        raise ContractError(
            f"Table A: invalid 'sentiment' values {examples}. "
            f"Expected one of: {sorted(VALID_SENTIMENTS)}"
        )

    # signal columns must be 0/1
    for col in SIGNAL_COLS & set(df.columns):
        if not df[col].isin([0, 1]).all():
            bad = df.loc[~df[col].isin([0, 1]), col].unique()[:5].tolist()
            raise ContractError(f"Table A: '{col}' must be 0 or 1, found: {bad}")

    # l5_id must be non-empty strings
    if df["l5_id"].isna().any() or (df["l5_id"].astype(str).str.strip() == "").any():
        raise ContractError("Table A: 'l5_id' has null or empty values.")

    # severity override — if present must be 1–5
    if "severity" in df.columns and df["severity"].notna().any():
        bad_sev = df.loc[df["severity"].notna() & ~df["severity"].between(1, 5), "severity"]
        if len(bad_sev):
            raise ContractError(f"Table A: 'severity' values outside [1,5]: {bad_sev.unique()[:5].tolist()}")

    logger.info("Table A: %d rows validated.", len(df))
    return df


def validate_table_b(df: pd.DataFrame, dimension_cols: list[str] | None = None) -> pd.DataFrame:
    """Validate operational dimensions (Table B). Returns cleaned df or raises ContractError."""
    missing = TABLE_B_REQUIRED - set(df.columns)
    if missing:
        raise ContractError(f"Table B missing required columns: {sorted(missing)}")

    non_id_cols = [c for c in df.columns if c != "conversation_id"]
    if not non_id_cols:
        raise ContractError("Table B must have at least one dimension or fact column besides 'conversation_id'.")

    if dimension_cols:
        missing_dims = set(dimension_cols) - set(df.columns)
        if missing_dims:
            raise ContractError(f"Table B: declared dimension columns not found: {sorted(missing_dims)}")
    else:
        # auto-detect: object/category/string columns (excluding conversation_id) are dimensions
        inferred = [
            c for c in non_id_cols
            if (
                pd.api.types.is_object_dtype(df[c])
                or isinstance(df[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(df[c])
                or str(df[c].dtype).startswith("category")
            )
        ]
        if not inferred:
            raise ContractError(
                "Table B: no categorical dimension columns detected. "
                "Declare dimension_cols explicitly in config or cast columns to object/category dtype."
            )

    logger.info("Table B: %d rows validated.", len(df))
    return df


def validate_join(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Warn (not fail) if join coverage is low."""
    ids_a = set(df_a["conversation_id"])
    ids_b = set(df_b["conversation_id"])
    overlap = ids_a & ids_b
    if not overlap:
        raise ContractError(
            "Tables A and B share no 'conversation_id' values — "
            "check that both tables use the same identifier."
        )
    pct = len(overlap) / len(ids_a) * 100
    if pct < 50:
        logger.warning(
            "Join coverage is %.1f%% (%d / %d Table-A rows have a Table-B match). "
            "Driver analysis will be limited to matched rows.",
            pct,
            len(overlap),
            len(ids_a),
        )
    else:
        logger.info("Join coverage: %.1f%% (%d rows matched).", pct, len(overlap))


def validate(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    dimension_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full contract validation. Returns (cleaned_a, cleaned_b) or raises ContractError."""
    df_a = validate_table_a(df_a)
    df_b = validate_table_b(df_b, dimension_cols=dimension_cols)
    validate_join(df_a, df_b)
    return df_a, df_b


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Convenience loader with a clear error message."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected parquet file not found: {p}")
    return pd.read_parquet(p)
