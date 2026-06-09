"""
tagging/run.py — CLI entry-point for the tagging producer.

Stages:
    ingest  → conversations parquet
    taxonomy (build only) → taxonomy.json schema pack
    tag     → tagged_transcripts parquet (Table A)

Usage:
    uv run python -m tagging.run --build [--sample 10000]
    uv run python -m tagging.run --test-batch 1
    uv run python -m tagging.run --resume
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from shared.config import DATA_DIR, TAXONOMY_PATH
from tagging.ingest import (
    default_conversations_path,
    load_twcs,
    save_parquet,
)
from tagging.tag import tag_conversations
from tagging.taxonomy import generate_static_taxonomy, induce_taxonomy, load_taxonomy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TWCS_CSV = DATA_DIR / "twcs" / "twcs.csv"
TAGGED_PATH_TEMPLATE = str(DATA_DIR / "tagged_{split}.parquet")
MANIFEST_PATH = DATA_DIR / "tagging_manifest.json"


def _tagged_path(split: str) -> Path:
    return Path(TAGGED_PATH_TEMPLATE.format(split=split))


def _manifest_load() -> dict:
    import json
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"completed_batches": []}


def _manifest_save(manifest: dict) -> None:
    import json
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def run_build(sample: int | None = 10_000, static_taxonomy: bool = True) -> None:
    """Full build: ingest → taxonomy → tag → save Table A."""
    logger.info("=== TAGGING BUILD (sample=%s) ===", sample)

    # 1. Ingest
    conv_path = default_conversations_path("build")
    if not conv_path.exists():
        logger.info("Ingesting raw TWCS...")
        df_conv = load_twcs(TWCS_CSV, sample=sample)
        save_parquet(df_conv, conv_path)
    else:
        logger.info("Loading existing conversations parquet: %s", conv_path)
        df_conv = pd.read_parquet(conv_path)
        if sample and sample < len(df_conv):
            df_conv = df_conv.sample(n=sample, random_state=42).reset_index(drop=True)

    # 2. Taxonomy (build once)
    if not TAXONOMY_PATH.exists():
        if static_taxonomy:
            logger.info("Generating static 30-leaf taxonomy...")
            generate_static_taxonomy(output_path=TAXONOMY_PATH)
        else:
            logger.info("Inducing taxonomy from %d conversations...", len(df_conv))
            induce_taxonomy(
                conversations=df_conv["text"].tolist(),
                output_path=TAXONOMY_PATH,
            )
    else:
        logger.info("Taxonomy already exists: %s", TAXONOMY_PATH)

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    taxonomy_nodes = taxonomy.get("nodes", {})

    # 3. Tag
    out_path = _tagged_path("build")
    if out_path.exists():
        logger.info("Tagged parquet already exists at %s; skipping.", out_path)
        return

    logger.info("Tagging %d conversations...", len(df_conv))
    df_tags = tag_conversations(df_conv, taxonomy_nodes=taxonomy_nodes)

    # Merge created_at from conversations
    df_out = df_conv[["conversation_id", "created_at", "text"]].merge(
        df_tags, on="conversation_id", how="inner"
    )
    save_parquet(df_out, out_path)
    logger.info("Tagged %d conversations → %s", len(df_out), out_path)


def run_test_batch(batch_num: int) -> None:
    """Tag a numbered slice of the reserve set (test batches grow the brain)."""
    logger.info("=== TAGGING TEST-BATCH %d ===", batch_num)
    manifest = _manifest_load()
    if batch_num in manifest["completed_batches"]:
        logger.info("Batch %d already completed.", batch_num)
        return

    conv_path = default_conversations_path(f"test_batch_{batch_num}")
    if not conv_path.exists():
        # Try to slice from reserve if raw CSV exists
        reserve_path = default_conversations_path("reserve")
        if not reserve_path.exists():
            if TWCS_CSV.exists():
                logger.info("Ingesting reserve set...")
                df_all = load_twcs(TWCS_CSV, sample=None)
                save_parquet(df_all, reserve_path)
            else:
                logger.error("No reserve parquet or TWCS CSV found. Cannot run test batch.")
                sys.exit(1)

        df_reserve = pd.read_parquet(reserve_path)
        batch_size = 50_000
        start = (batch_num - 1) * batch_size
        end = start + batch_size
        df_batch = df_reserve.iloc[start:end]
        if df_batch.empty:
            logger.info("Batch %d is empty (reserve exhausted).", batch_num)
            return
        save_parquet(df_batch, conv_path)

    df_conv = pd.read_parquet(conv_path)
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    taxonomy_nodes = taxonomy.get("nodes", {})

    out_path = _tagged_path(f"test_batch_{batch_num}")
    logger.info("Tagging batch %d (%d conversations)...", batch_num, len(df_conv))
    df_tags = tag_conversations(df_conv, taxonomy_nodes=taxonomy_nodes)
    df_out = df_conv[["conversation_id", "created_at", "text"]].merge(
        df_tags, on="conversation_id", how="inner"
    )
    save_parquet(df_out, out_path)

    manifest["completed_batches"].append(batch_num)
    _manifest_save(manifest)
    logger.info("Batch %d done → %s", batch_num, out_path)


def run_resume() -> None:
    """Resume interrupted tagging run using the manifest."""
    manifest = _manifest_load()
    completed = set(manifest["completed_batches"])
    batch_num = 1
    while True:
        if batch_num not in completed:
            run_test_batch(batch_num)
        batch_num += 1
        if batch_num > 100:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Tagging producer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="Full build run (ingest+taxonomy+tag)")
    group.add_argument("--test-batch", type=int, metavar="N", help="Tag test batch N")
    group.add_argument("--resume", action="store_true", help="Resume interrupted tagging")
    parser.add_argument("--sample", type=int, default=10_000, help="Max conversations for build")
    parser.add_argument("--hdbscan-taxonomy", action="store_true", help="Use HDBSCAN-induced taxonomy instead of static")
    args = parser.parse_args()

    if args.build:
        run_build(sample=args.sample, static_taxonomy=not args.hdbscan_taxonomy)
    elif args.test_batch:
        run_test_batch(args.test_batch)
    elif args.resume:
        run_resume()


if __name__ == "__main__":
    main()
