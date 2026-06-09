"""
build.py — Top-level orchestrator.

Wires the replaceable producers (tagging/, datagen/) and the dataset-agnostic core
(analytics/, gbrain/, agent/) in order. Modules never import each other; wiring
happens only here.

Pipeline:
    tagging/   → Table A (tagged transcripts)
    datagen/   → Table B (operational dimensions)
    analytics/ → gbrain update + snapshot
    agent/     → report from gbrain

Usage:
    uv run python build.py --demo --sample 10000
    uv run python build.py --test-batch 1
    uv run python build.py --report-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shared.config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TAGGED_BUILD = DATA_DIR / "tagged_build.parquet"
OPS_BUILD = DATA_DIR / "ops_build.parquet"


def _run_tagging(sample: int) -> None:
    """Run tagging producer (Table A)."""
    logger.info("Stage 1: TAGGING")
    from tagging.run import run_build
    run_build(sample=sample)


def _run_datagen() -> None:
    """Run datagen producer (Table B)."""
    logger.info("Stage 2: DATAGEN")
    from datagen.generate import generate
    generate(input_path=TAGGED_BUILD, output_path=OPS_BUILD)


def _require(path: Path, hint: str) -> None:
    """Fail fast with an actionable message if a required artifact is missing."""
    if not path.exists():
        logger.error("Required input not found: %s\n  %s", path, hint)
        sys.exit(1)


def _run_analytics(
    dimension_cols: list[str] | None = None,
    order_value_col: str | None = None,
) -> dict:
    """Run analytics → gbrain."""
    logger.info("Stage 3: ANALYTICS → gbrain")
    _require(TAGGED_BUILD, "Run tagging first: `uv run python -m tagging.run --build`.")
    _require(OPS_BUILD, "Run datagen first or omit --skip-datagen.")
    from analytics.build_brain import run as analytics_run
    return analytics_run(
        table_a_path=TAGGED_BUILD,
        table_b_path=OPS_BUILD,
        dimension_cols=dimension_cols,
        order_value_col=order_value_col,
    )


def _run_report(summary: dict) -> None:
    """Render report from gbrain."""
    logger.info("Stage 4: REPORT")
    from agent.report import render
    result = render(run_id=summary.get("run_id"))
    logger.info("Report:\n  %s\n  %s", result["paths"][0], result["paths"][1])


def run_demo(
    sample: int = 10_000,
    skip_tagging: bool = False,
    skip_datagen: bool = False,
    skip_report: bool = False,
) -> dict:
    """Full demo pipeline: tagging → datagen → analytics → report."""
    if not skip_tagging:
        _run_tagging(sample=sample)
    else:
        logger.info("Skipping tagging (--skip-tagging).")

    if not skip_datagen:
        _run_datagen()
    else:
        logger.info("Skipping datagen (--skip-datagen).")

    summary = _run_analytics(
        dimension_cols=None,  # auto-detect
        order_value_col="order_total",
    )

    if not skip_report:
        _run_report(summary)

    logger.info("Demo complete.\nSummary: %s", summary)
    return summary


def run_test_batch(batch_num: int) -> dict:
    """Tag a test batch and append to gbrain."""
    logger.info("=== TEST BATCH %d ===", batch_num)
    from tagging.run import run_test_batch as _tag_batch
    _tag_batch(batch_num)

    tagged = DATA_DIR / f"tagged_test_batch_{batch_num}.parquet"
    if not tagged.exists():
        logger.error("Tagging batch %d produced no output.", batch_num)
        sys.exit(1)

    # Test batches share the build's operational context. Generate dimensions
    # for this batch with datagen so Table B actually covers the batch's
    # conversation_ids (the build ops_build won't contain them).
    batch_ops = DATA_DIR / f"ops_test_batch_{batch_num}.parquet"
    if not batch_ops.exists():
        logger.info("Generating operational dimensions for batch %d...", batch_num)
        from datagen.generate import generate
        generate(input_path=tagged, output_path=batch_ops)

    from analytics.build_brain import run as analytics_run
    summary = analytics_run(
        table_a_path=tagged,
        table_b_path=batch_ops,
    )
    logger.info("Batch %d complete. Summary: %s", batch_num, summary)
    return summary


def run_report_only() -> None:
    """Render report from current gbrain state."""
    from agent.report import render
    result = render()
    print(f"Report:\n  {result['paths'][0]}\n  {result['paths'][1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Intelligence Agent — Build Orchestrator")
    parser.add_argument("--demo", action="store_true", help="Run full demo pipeline")
    parser.add_argument("--sample", type=int, default=10_000, help="Build sample size")
    parser.add_argument("--test-batch", type=int, metavar="N", help="Run test batch N")
    parser.add_argument("--report-only", action="store_true", help="Render report from current gbrain")
    parser.add_argument("--skip-tagging", action="store_true", help="Skip tagging (use existing tagged_build)")
    parser.add_argument("--skip-datagen", action="store_true", help="Skip datagen (use existing ops_build)")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    args = parser.parse_args()

    if args.demo:
        run_demo(
            sample=args.sample,
            skip_tagging=args.skip_tagging,
            skip_datagen=args.skip_datagen,
            skip_report=args.skip_report,
        )
    elif args.test_batch:
        run_test_batch(args.test_batch)
    elif args.report_only:
        run_report_only()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
