"""
datagen/generate.py — CLI entry-point for synthetic operational dimensions.

Produces Table B (operational_dimensions.parquet) from Table A.
The driver model is documented in driver_model.py for validation.

Usage:
    uv run python -m datagen.generate --input data/tagged_build.parquet --output data/ops_build.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from datagen.dimensions import get_dimension_cols
from datagen.driver_model import plant_dimensions
from shared.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = DATA_DIR / "tagged_build.parquet"
DEFAULT_OUTPUT = DATA_DIR / "ops_build.parquet"


def generate(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> None:
    """Read Table A, plant dimensions, write Table B."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Table A not found: {p}")

    df_a = pd.read_parquet(p)
    logger.info("Loaded Table A: %d rows", len(df_a))

    dimension_cols = get_dimension_cols()
    df_b = plant_dimensions(df_a, dimension_cols=dimension_cols, seed=seed)
    logger.info(
        "Generated Table B: %d rows, %d dimensions, %d facts",
        len(df_b),
        len([c for c in df_b.columns if c != "conversation_id" and df_b[c].dtype == object]),
        len([c for c in df_b.columns if c != "conversation_id" and df_b[c].dtype != object]),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_b.to_parquet(out, index=False)
    logger.info("Saved Table B → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic operational dimensions (Table B)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to tagged transcripts parquet (Table A)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output operational dimensions parquet (Table B)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate(input_path=args.input, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
