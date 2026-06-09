"""Unit tests for weekly metrics and z-score computation."""
from datetime import datetime

import pandas as pd

from analytics.metrics import build_weekly_metrics, compute_zscores, latest_period_stats


class TestWeeklyMetrics:
    def test_build_weekly(self):
        df = pd.DataFrame({
            "conversation_id": ["a", "b", "c", "d", "e"],
            "created_at": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 8),
                datetime(2024, 1, 9),
                datetime(2024, 1, 15),
            ],
            "text": ["x"] * 5,
            "l5_id": ["A", "A", "A", "B", "B"],
            "sentiment": ["neg", "neg", "very_neg", "neutral", "pos"],
            "severity": [2.0, 2.0, 3.0, 1.0, 1.0],
        })
        weekly = build_weekly_metrics(df)
        assert "week" in weekly.columns
        assert weekly["week"].nunique() == 3  # Jan 1, 8, 15 (ISO weeks)
        a_rows = weekly[weekly["l5_id"] == "A"]
        assert a_rows["volume"].sum() == 3

    def test_compute_zscores(self):
        df = pd.DataFrame({
            "l5_id": ["A"] * 10,
            "week": [f"2024-W{i:02d}" for i in range(1, 11)],
            "volume": [10, 10, 10, 10, 10, 10, 10, 10, 10, 100],
            "severity_avg": [2.0] * 10,
            "sentiment_avg": [2.5] * 10,
            "order_value_sum": [100.0] * 10,
        })
        zscores = compute_zscores(df, baseline_weeks=8, min_periods=3)
        assert "A" in zscores
        assert zscores["A"].zscore > 5  # huge spike
        assert zscores["A"].latest_volume == 100

    def test_compute_zscores_not_enough_periods(self):
        df = pd.DataFrame({
            "l5_id": ["A"] * 2,
            "week": ["2024-W01", "2024-W02"],
            "volume": [10, 20],
            "severity_avg": [2.0, 2.0],
            "sentiment_avg": [2.5, 2.5],
            "order_value_sum": [100.0, 100.0],
        })
        zscores = compute_zscores(df, baseline_weeks=8, min_periods=3)
        assert zscores["A"].zscore == 0.0  # not enough periods

    def test_latest_period_stats(self):
        df = pd.DataFrame({
            "conversation_id": ["a", "b", "c"],
            "created_at": [datetime(2024, 1, 1)] * 3,
            "text": ["x"] * 3,
            "l5_id": ["A", "A", "B"],
            "sentiment": ["neg", "neg", "very_neg"],
            "severity": [2.0, 3.0, 3.0],
            "sentiment_num": [2.0, 2.0, 1.0],
        })
        stats = latest_period_stats(df)
        assert len(stats) == 2
        a = stats[stats["l5_id"] == "A"].iloc[0]
        assert a["total_volume"] == 2
        assert a["severity_avg"] == 2.5
