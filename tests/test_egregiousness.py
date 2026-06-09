"""Unit tests for egregiousness ranking."""
import pandas as pd

from analytics.egregiousness import compute_egregiousness
from shared.schemas import ZScoreResult


class TestEgregiousness:
    def test_basic_computation(self):
        stats = pd.DataFrame({
            "l5_id": ["A", "B", "C"],
            "total_volume": [100, 200, 50],
            "severity_avg": [3.0, 2.0, 4.0],
            "order_value_sum": [1000.0, 500.0, 2000.0],
        })
        zscores = {
            "A": ZScoreResult(l5_id="A", latest_week="W01", latest_volume=100, zscore=2.0, baseline_mean=90, baseline_std=5),
            "B": ZScoreResult(l5_id="B", latest_week="W01", latest_volume=200, zscore=0.5, baseline_mean=190, baseline_std=5),
            "C": ZScoreResult(l5_id="C", latest_week="W01", latest_volume=50, zscore=3.0, baseline_mean=40, baseline_std=5),
        }
        results = compute_egregiousness(stats, zscores)
        # C has highest severity (4.0), z-score (3.0), and value (2000)
        # → highest egregiousness despite lowest volume
        assert len(results) == 3
        assert results[0].l5_id == "C"

    def test_no_value_redistributes(self):
        stats = pd.DataFrame({
            "l5_id": ["A", "B"],
            "total_volume": [100, 200],
            "severity_avg": [3.0, 2.0],
            "order_value_sum": [0.0, 0.0],
        })
        zscores = {
            "A": ZScoreResult(l5_id="A", latest_week="W01", latest_volume=100, zscore=1.0, baseline_mean=90, baseline_std=5),
            "B": ZScoreResult(l5_id="B", latest_week="W01", latest_volume=200, zscore=0.5, baseline_mean=190, baseline_std=5),
        }
        results = compute_egregiousness(stats, zscores)
        # Should still work; value weight redistributed
        assert len(results) == 2
        assert all(r.pct_value == 0.0 for r in results)
