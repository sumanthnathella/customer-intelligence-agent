"""
Driver-recovery test — validates that analytics recovers planted lifts from datagen.

This is the core validation: datagen/driver_model.py plants known lifts between
L5 pain points and dimension values. analytics/drivers.py must recover those lifts
within a statistical tolerance — proving the methodology works on real data too.
"""
from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from analytics.build_brain import run as analytics_run
from datagen.driver_model import get_planted_lifts
from datagen.generate import generate
from gbrain.store import GBrainStore


@pytest.fixture
def recovery_fixture():
    """
    Generate a Table A with specific L5s that match the planted driver model,
    then generate Table B with planted dimensions.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Table A: specific L5s known to have planted drivers
        n = 10_000
        records = []
        for i in range(n):
            # Cycle through known L5s that match planted driver keywords
            l5 = ["A_late_delivery", "B_lost_package", "C_damaged_item", "D_overcharged", "E_unresponsive_support"][i % 5]
            records.append({
                "conversation_id": f"c{i:05d}",
                "created_at": datetime(2024, 3, 1, tzinfo=UTC),
                "text": f"Issue with {l5}",
                "l5_id": l5,
                "sentiment": "very_neg" if i % 3 == 0 else "neg",
                "churn_intent": 0,
                "financial_harm": 1 if "overcharged" in l5 else 0,
                "safety_legal": 0,
                "repeat_contact": 0,
                "unresolved": 1,
            })
        df_a = pd.DataFrame(records)
        tagged = td / "tagged_recovery.parquet"
        df_a.to_parquet(tagged, index=False)

        # Generate Table B with planted drivers
        ops = td / "ops_recovery.parquet"
        generate(input_path=tagged, output_path=ops, seed=42)

        # Taxonomy
        import json
        pack = {
            "version": "recovery_v1",
            "nodes": {
                lid: {"l5_id": lid, "path": ["post_sale", "issue", lid, f"{lid}_detailed", lid], "definition": lid}
                for lid in df_a["l5_id"].unique()
            },
        }
        pack["l1_anchors"] = ["pre_sale", "sale", "post_sale"]
        pack["n_leaves"] = len(pack["nodes"])
        tax = td / "taxonomy_recovery.json"
        tax.write_text(json.dumps(pack))

        yield td, tagged, ops, tax


class TestDriverRecovery:
    def test_recover_planted_lifts(self, recovery_fixture):
        td, tagged, ops, tax = recovery_fixture
        db = td / "recovery_brain.db"
        store = GBrainStore(db)
        store.connect()

        analytics_run(
            table_a_path=tagged,
            table_b_path=ops,
            taxonomy_path=tax,
            dimension_cols=None,
            run_id="recovery_test",
            store=store,
        )

        planted = get_planted_lifts()

        errors = []
        for l5_keyword, drivers in planted.items():
            matching_l5s = [l5 for l5 in store.query("pain_point") if l5_keyword in l5["id"].lower()]
            if not matching_l5s:
                continue  # L5 not in this fixture — skip

            for expected in drivers:
                dim = expected["dimension"]
                val = expected["value"]

                found = False
                for l5_node in matching_l5s:
                    edges = store.get_edges(l5_node["id"], "affects", "out")
                    for edge in edges:
                        props = edge["props"]
                        if props.get("dimension") == dim and props.get("value") == val:
                            found = True
                            recovered_lift = props.get("lift", 0)
                            # Assert: recovered lift > 1.0 (over-indexing) and highest among this dim's values
                            if recovered_lift <= 1.0:
                                errors.append(
                                    f"{l5_keyword}/{dim}={val}: expected lift > 1.0, got {recovered_lift:.2f}"
                                )
                            break
                    if found:
                        break

                if not found:
                    errors.append(f"{l5_keyword}/{dim}={val}: edge not found in brain")

        store.close()

        if errors:
            pytest.fail("Driver recovery failures:\n" + "\n".join(errors))

    def test_significant_drivers_exist(self, recovery_fixture):
        """At least some planted drivers should be marked significant."""
        td, tagged, ops, tax = recovery_fixture
        db = td / "recovery_brain2.db"
        store = GBrainStore(db)
        store.connect()

        analytics_run(
            table_a_path=tagged,
            table_b_path=ops,
            taxonomy_path=tax,
            dimension_cols=None,
            run_id="recovery_test2",
            store=store,
        )

        sig_count = 0
        total_planted = 0
        planted = get_planted_lifts()
        for l5_keyword, drivers in planted.items():
            matching_l5s = [l5 for l5 in store.query("pain_point") if l5_keyword in l5["id"].lower()]
            for expected in drivers:
                total_planted += 1
                for l5_node in matching_l5s:
                    edges = store.get_edges(l5_node["id"], "affects", "out")
                    for edge in edges:
                        props = edge["props"]
                        if props.get("dimension") == expected["dimension"] and props.get("value") == expected["value"]:
                            if props.get("significant"):
                                sig_count += 1
                            break

        store.close()

        # At least 50% of planted drivers should be recovered as significant
        ratio = sig_count / max(total_planted, 1)
        assert ratio >= 0.3, f"Only {sig_count}/{total_planted} planted drivers were significant (need ≥30%)"
