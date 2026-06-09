"""Unit tests for the deterministic severity rubric."""
import pandas as pd

from analytics.severity import add_severity_column, compute_severity


class TestSeverityRubric:
    def test_base_sentiment(self):
        df = pd.DataFrame({
            "sentiment": ["very_neg", "neg", "neutral", "pos"],
            "churn_intent": [0, 0, 0, 0],
            "financial_harm": [0, 0, 0, 0],
            "safety_legal": [0, 0, 0, 0],
            "repeat_contact": [0, 0, 0, 0],
            "unresolved": [0, 0, 0, 0],
        })
        sev = compute_severity(df)
        assert list(sev) == [3, 2, 1, 1]

    def test_churn_bumps(self):
        df = pd.DataFrame({
            "sentiment": ["neutral"],
            "churn_intent": [1],
            "financial_harm": [0],
            "safety_legal": [0],
            "repeat_contact": [0],
            "unresolved": [0],
        })
        assert compute_severity(df).iloc[0] == 2

    def test_safety_highest(self):
        df = pd.DataFrame({
            "sentiment": ["neg"],
            "churn_intent": [1],
            "financial_harm": [1],
            "safety_legal": [1],
            "repeat_contact": [1],
            "unresolved": [1],
        })
        # base=2 + churn(1) + financial(1) + safety(2) + repeat|unresolved(1) = 7 → clamp to 5
        assert compute_severity(df).iloc[0] == 5

    def test_repeat_or_unresolved_at_most_one(self):
        df = pd.DataFrame({
            "sentiment": ["neg"],
            "churn_intent": [0],
            "financial_harm": [0],
            "safety_legal": [0],
            "repeat_contact": [1],
            "unresolved": [1],
        })
        # base=2 + repeat|unresolved(1) = 3, not 4
        assert compute_severity(df).iloc[0] == 3

    def test_byo_severity_passthrough(self):
        df = pd.DataFrame({
            "sentiment": ["neg"],
            "churn_intent": [0],
            "financial_harm": [0],
            "safety_legal": [0],
            "repeat_contact": [0],
            "unresolved": [0],
            "severity": [4.5],
        })
        sev = compute_severity(df)
        assert sev.iloc[0] == 4.5

    def test_add_severity_column(self):
        df = pd.DataFrame({
            "sentiment": ["very_neg", "pos"],
            "churn_intent": [0, 0],
            "financial_harm": [0, 0],
            "safety_legal": [0, 0],
            "repeat_contact": [0, 0],
            "unresolved": [0, 0],
        })
        df2 = add_severity_column(df)
        assert "severity" in df2.columns
        assert df2["severity"].tolist() == [3, 1]
