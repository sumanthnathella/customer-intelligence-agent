"""Unit tests for driver analysis (lift, significance, FDR)."""
import pandas as pd

from analytics.drivers import compute_drivers, top_drivers_summary
from shared.schemas import DriverEdge


class TestDriverAnalysis:
    def test_lift_basic(self):
        # 1000 rows: A=400 with dim=X, B=600 with dim=Y
        df = pd.DataFrame({
            "conversation_id": [f"c{i}" for i in range(1000)],
            "l5_id": ["A"] * 400 + ["B"] * 600,
            "dim": ["X"] * 300 + ["Y"] * 100 + ["Y"] * 500 + ["X"] * 100,
        })
        edges = compute_drivers(df, dimension_cols=["dim"], period="W01", min_support=10)
        assert len(edges) > 0
        a_x = [e for e in edges if e.l5_id == "A" and e.dimension == "dim" and e.value == "X"]
        assert len(a_x) == 1
        assert a_x[0].lift > 1.0  # A over-indexes on X

    def test_significance_filter(self):
        # Strong planted driver: 1000 rows, A strongly over-indexes on X
        n = 1000
        df = pd.DataFrame({
            "conversation_id": [f"c{i}" for i in range(n)],
            "l5_id": ["A"] * 300 + ["B"] * 700,
            "dim": ["X"] * 250 + ["Y"] * 50 + ["Y"] * 650 + ["X"] * 50,
        })
        edges = compute_drivers(df, dimension_cols=["dim"], period="W01", min_support=10, fdr_alpha=0.05)
        a_x = [e for e in edges if e.l5_id == "A" and e.dimension == "dim" and e.value == "X"]
        assert len(a_x) == 1
        assert a_x[0].significant is True
        assert a_x[0].lift > 2.0

    def test_top_drivers_summary(self):
        edges = [
            DriverEdge(
                l5_id="A", dimension="dim", value="X", support=100, share=0.5,
                lift=3.0, p_value=0.001, significant=True, excess=50.0, period="W01"
            ),
            DriverEdge(
                l5_id="A", dimension="dim", value="Y", support=50, share=0.25,
                lift=1.5, p_value=0.1, significant=False, excess=10.0, period="W01"
            ),
            DriverEdge(
                l5_id="A", dimension="dim2", value="Z", support=80, share=0.4,
                lift=2.5, p_value=0.005, significant=True, excess=30.0, period="W01"
            ),
        ]
        summary = top_drivers_summary(edges, k=2)
        assert len(summary) == 2
        assert summary[0]["lift"] == 3.0
        assert summary[1]["lift"] == 2.5

    def test_history_accumulation(self):
        existing = {
            "A:affects:dim:dim:X": [{"period": "W00", "support": 90, "share": 0.45, "lift": 2.8}]
        }
        df = pd.DataFrame({
            "conversation_id": ["c1", "c2"],
            "l5_id": ["A", "A"],
            "dim": ["X", "Y"],
        })
        edges = compute_drivers(
            df, dimension_cols=["dim"], period="W01", min_support=1,
            existing_history=existing,
        )
        a_x = [e for e in edges if e.l5_id == "A" and e.value == "X"]
        assert len(a_x) == 1
        assert len(a_x[0].history) == 2
        assert a_x[0].history[0]["period"] == "W00"
        assert a_x[0].history[1]["period"] == "W01"
