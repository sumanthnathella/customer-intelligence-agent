"""
tagging/ingest.py — Load and pre-process raw TWCS (or BYO) transcripts.

Produces a clean DataFrame with columns:
    conversation_id, created_at, text, author_id (optional)

TWCS-specific logic is isolated here. BYO users replace this module or provide
their own parquet that already has the required columns.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from shared.config import DATA_DIR

logger = logging.getLogger(__name__)

# TWCS raw columns
TWCS_ID_COL = "tweet_id"
TWCS_CONV_COL = "in_response_to_tweet_id"
TWCS_CREATED_COL = "created_at"
TWCS_TEXT_COL = "text"
TWCS_AUTHOR_COL = "author_id"


def _clean_text(text: str) -> str:
    """Strip @mentions, URLs, and excessive whitespace from a tweet."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_conversation_text(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct conversation threads from TWCS.
    Each conversation = all tweets sharing the same root tweet_id, concatenated
    in reply order.

    Returns DataFrame: conversation_id, created_at, text
    """
    df = df_raw.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").apply(_clean_text)
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)

    # Find root tweets (no in_response_to)
    is_root = df["in_response_to_tweet_id"] == ""
    root_ids = set(df.loc[is_root, "tweet_id"])

    # Assign conversation_id = root tweet_id via simple traversal
    id_to_root: dict[str, str] = {tid: tid for tid in root_ids}

    # BFS to propagate root IDs
    id_to_parent = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"], strict=False))
    for tweet_id in df["tweet_id"]:
        path = []
        current = tweet_id
        while current and current not in root_ids and current not in path:
            path.append(current)
            current = id_to_parent.get(current, "")
        root = current if current in root_ids else tweet_id
        for tid in path:
            id_to_root[tid] = root
        id_to_root[tweet_id] = root

    df["conversation_id"] = df["tweet_id"].map(id_to_root).fillna(df["tweet_id"])

    # Aggregate to conversation level
    conv = (
        df.groupby("conversation_id")
        .agg(
            created_at=("created_at", "min"),
            text=("text", lambda ts: " | ".join(t for t in ts if t)),
        )
        .reset_index()
    )
    return conv


def load_twcs(
    csv_path: str | Path,
    sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load TWCS CSV and build conversation-level rows.

    Parameters
    ----------
    csv_path: path to twcs/twcs.csv
    sample: if set, randomly sample this many conversations
    seed: random seed for reproducibility

    Returns DataFrame with: conversation_id, created_at, text
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"TWCS CSV not found: {p}\n"
            "Download from Kaggle (twcs dataset) and place at data/twcs/twcs.csv"
        )
    logger.info("Loading TWCS from %s ...", p)
    df_raw = pd.read_csv(p, dtype=str, low_memory=False)
    logger.info("Raw rows: %d", len(df_raw))

    required_cols = {"tweet_id", "in_response_to_tweet_id", "text"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"TWCS CSV missing expected columns: {missing}")

    conversations = _build_conversation_text(df_raw)
    logger.info("Conversations built: %d", len(conversations))

    if sample and sample < len(conversations):
        conversations = conversations.sample(n=sample, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d conversations.", sample)

    return conversations


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a pre-processed conversations parquet (BYO or previously ingested)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Conversations parquet not found: {p}")
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logger.info("Saved %d rows to %s", len(df), p)


def default_conversations_path(split: str = "build") -> Path:
    return DATA_DIR / f"conversations_{split}.parquet"
