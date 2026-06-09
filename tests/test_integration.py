"""
Integration test — end-to-end pipeline with synthetic fixture data.

Runs: tagging-like fixture → datagen → analytics → gbrain → report.
Validates that the full pipeline produces a gbrain with expected nodes and a report.
"""
from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from analytics.build_brain import run as analytics_run
from datagen.generate import generate
from gbrain.store import GBrainStore


@pytest.fixture
def fixture_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def fixture_table_a(fixture_dir: Path) -> Path:
    """Create a synthetic Table A with 5 L5s over 10 weeks."""
    n_per_week = 100
    weeks = [f"2024-W{i:02d}" for i in range(1, 11)]
    l5s = ["A_late", "B_lost", "C_damaged", "D_wrong", "E_overcharged"]

    records = []
    for week in weeks:
        for l5 in l5s:
            # A_late gets more volume in later weeks (spike)
            mult = 1.5 if (l5 == "A_late" and week >= "2024-W08") else 1.0
            count = int(n_per_week * mult)
            for i in range(count):
                records.append({
                    "conversation_id": f"{week}_{l5}_{i}",
                    "created_at": datetime.strptime(week + "-1", "%Y-W%W-%w").replace(tzinfo=UTC),
                    "text": f"Complaint about {l5}",
                    "l5_id": l5,
                    "sentiment": "very_neg" if i % 3 == 0 else "neg",
                    "churn_intent": 1 if i % 5 == 0 else 0,
                    "financial_harm": 1 if "overcharged" in l5 and i % 2 == 0 else 0,
                    "safety_legal": 0,
                    "repeat_contact": 1 if i % 7 == 0 else 0,
                    "unresolved": 1 if i % 4 == 0 else 0,
                })

    df = pd.DataFrame(records)
    path = fixture_dir / "tagged_fixture.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def fixture_table_b(fixture_dir: Path, fixture_table_a: Path) -> Path:
    """Generate synthetic operational dimensions from Table A."""
    out = fixture_dir / "ops_fixture.parquet"
    generate(input_path=fixture_table_a, output_path=out, seed=42)
    return out


@pytest.fixture
def taxonomy_path(fixture_dir: Path) -> Path:
    import json
    pack = {
        "version": "test_v1",
        "nodes": {
            "A_late": {"l5_id": "A_late", "path": ["post_sale", "delivery", "late", "late_detailed", "A_late"], "definition": "Late delivery"},
            "B_lost": {"l5_id": "B_lost", "path": ["post_sale", "delivery", "lost", "lost_detailed", "B_lost"], "definition": "Lost package"},
            "C_damaged": {"l5_id": "C_damaged", "path": ["post_sale", "delivery", "damaged", "damaged_detailed", "C_damaged"], "definition": "Damaged item"},
            "D_wrong": {"l5_id": "D_wrong", "path": ["post_sale", "delivery", "wrong", "wrong_detailed", "D_wrong"], "definition": "Wrong item"},
            "E_overcharged": {"l5_id": "E_overcharged", "path": ["post_sale", "billing", "overcharged", "overcharged_detailed", "E_overcharged"], "definition": "Overcharged"},
        },
        "l1_anchors": ["pre_sale", "sale", "post_sale"],
        "n_leaves": 5,
    }
    p = fixture_dir / "taxonomy.json"
    p.write_text(json.dumps(pack))
    return p


class TestIntegration:
    def test_full_pipeline(self, fixture_dir, fixture_table_a, fixture_table_b, taxonomy_path):
        db_path = fixture_dir / "test_brain.db"
        store = GBrainStore(db_path)
        store.connect()

        summary = analytics_run(
            table_a_path=fixture_table_a,
            table_b_path=fixture_table_b,
            taxonomy_path=taxonomy_path,
            dimension_cols=None,  # auto-detect
            order_value_col="order_total",
            run_id="integration_test",
            store=store,
        )

        # Asserts
        assert summary["n_transcripts"] > 0
        assert summary["n_l5"] == 5
        assert summary["n_driver_edges"] > 0
        assert summary["snapshot"]

        # gbrain contains pain_points
        pp = store.query("pain_point")
        assert len(pp) >= 5

        # gbrain contains period_metrics
        pm = store.query("period_metric")
        assert len(pm) >= 5

        # A_late should have highest volume
        a_node = store.get_node("A_late")
        assert a_node is not None
        assert a_node["props"]["total_volume"] > 0
        assert a_node["props"]["latest_zscore"] > 0

        store.close()

    def test_report_renders(self, fixture_dir, fixture_table_a, fixture_table_b, taxonomy_path):
        import os

        from agent.report import render
        os.environ["REPORTS_DIR"] = str(fixture_dir / "reports")

        db_path = fixture_dir / "test_brain2.db"
        store = GBrainStore(db_path)
        store.connect()

        analytics_run(
            table_a_path=fixture_table_a,
            table_b_path=fixture_table_b,
            taxonomy_path=taxonomy_path,
            dimension_cols=None,
            store=store,
            run_id="report_test",
        )

        result = render(run_id="report_test")
        assert "markdown" in result
        assert "json" in result
        assert Path(result["paths"][0]).exists()
        assert Path(result["paths"][1]).exists()

        store.close()
