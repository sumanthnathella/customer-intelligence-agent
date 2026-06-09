"""Unit tests for input contract validation."""
from datetime import datetime

import pandas as pd
import pytest

from shared.contract import (
    ContractError,
    validate,
    validate_join,
    validate_table_a,
    validate_table_b,
)


class TestTableA:
    def test_valid_table_a(self):
        df = pd.DataFrame({
            "conversation_id": ["a", "b"],
            "created_at": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "text": ["hello", "world"],
            "l5_id": ["L1", "L2"],
            "sentiment": ["neg", "very_neg"],
            "churn_intent": [0, 1],
            "financial_harm": [0, 0],
            "safety_legal": [0, 0],
            "repeat_contact": [0, 0],
            "unresolved": [0, 1],
        })
        result = validate_table_a(df)
        assert len(result) == 2
        assert "severity" not in result.columns

    def test_missing_columns(self):
        df = pd.DataFrame({"conversation_id": ["a"]})
        with pytest.raises(ContractError, match="missing required columns"):
            validate_table_a(df)

    def test_invalid_sentiment(self):
        df = pd.DataFrame({
            "conversation_id": ["a"],
            "created_at": [datetime(2024, 1, 1)],
            "text": ["x"],
            "l5_id": ["L1"],
            "sentiment": ["angry"],
            "churn_intent": [0],
            "financial_harm": [0],
            "safety_legal": [0],
            "repeat_contact": [0],
            "unresolved": [0],
        })
        with pytest.raises(ContractError, match="invalid 'sentiment' values"):
            validate_table_a(df)

    def test_signal_out_of_range(self):
        df = pd.DataFrame({
            "conversation_id": ["a"],
            "created_at": [datetime(2024, 1, 1)],
            "text": ["x"],
            "l5_id": ["L1"],
            "sentiment": ["neg"],
            "churn_intent": [2],  # invalid
            "financial_harm": [0],
            "safety_legal": [0],
            "repeat_contact": [0],
            "unresolved": [0],
        })
        with pytest.raises(ContractError, match="must be 0 or 1"):
            validate_table_a(df)


class TestTableB:
    def test_valid_with_dimensions(self):
        df = pd.DataFrame({
            "conversation_id": ["a", "b"],
            "carrier": ["FedEx", "UPS"],
            "region": ["NE", "SE"],
        })
        result = validate_table_b(df, dimension_cols=["carrier", "region"])
        assert len(result) == 2

    def test_auto_detect_dimensions(self):
        df = pd.DataFrame({
            "conversation_id": ["a", "b"],
            "carrier": ["FedEx", "UPS"],
            "order_total": [100.0, 200.0],
        })
        result = validate_table_b(df)  # auto-detect
        assert len(result) == 2

    def test_no_dimensions(self):
        df = pd.DataFrame({
            "conversation_id": ["a"],
            "order_total": [100.0],
        })
        with pytest.raises(ContractError, match="no categorical dimension columns detected"):
            validate_table_b(df)

    def test_missing_conversation_id(self):
        df = pd.DataFrame({"carrier": ["FedEx"]})
        with pytest.raises(ContractError, match="missing required columns"):
            validate_table_b(df)


class TestJoin:
    def test_good_join(self):
        df_a = pd.DataFrame({"conversation_id": ["a", "b", "c"]})
        df_b = pd.DataFrame({"conversation_id": ["a", "b"]})
        validate_join(df_a, df_b)  # should not raise

    def test_no_overlap(self):
        df_a = pd.DataFrame({"conversation_id": ["a", "b"]})
        df_b = pd.DataFrame({"conversation_id": ["x", "y"]})
        with pytest.raises(ContractError, match="share no 'conversation_id' values"):
            validate_join(df_a, df_b)

    def test_low_coverage_warning(self, caplog):
        df_a = pd.DataFrame({"conversation_id": ["a", "b", "c", "d"]})
        df_b = pd.DataFrame({"conversation_id": ["a"]})
        validate_join(df_a, df_b)
        assert "Join coverage is 25.0%" in caplog.text


class TestValidate:
    def test_full_validation(self):
        df_a = pd.DataFrame({
            "conversation_id": ["a"],
            "created_at": [datetime(2024, 1, 1)],
            "text": ["x"],
            "l5_id": ["L1"],
            "sentiment": ["neg"],
            "churn_intent": [0],
            "financial_harm": [0],
            "safety_legal": [0],
            "repeat_contact": [0],
            "unresolved": [0],
        })
        df_b = pd.DataFrame({
            "conversation_id": ["a"],
            "carrier": ["FedEx"],
        })
        a, b = validate(df_a, df_b)
        assert len(a) == 1
        assert len(b) == 1
