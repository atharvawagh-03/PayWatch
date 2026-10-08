"""
Unit tests for UPIDataGenerator (FR-1)
"""

import pandas as pd
import pytest
from src.generator.synthetic_generator import UPIDataGenerator


def test_generator_reproducibility():
    gen1 = UPIDataGenerator(num_transactions=500, seed=42)
    df1 = gen1.generate()

    gen2 = UPIDataGenerator(num_transactions=500, seed=42)
    df2 = gen2.generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_generator_schema_and_columns():
    gen = UPIDataGenerator(num_transactions=1000, seed=42)
    df = gen.generate()

    expected_cols = [
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "merchant_category",
        "transaction_type",
        "location",
        "device_type",
        "upi_channel",
        "hour_of_day",
        "day_of_week",
        "historical_spend_mean",
        "historical_spend_std",
        "is_fraud",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) == 1000
    assert df["transaction_id"].nunique() == 1000
    assert not df.isnull().any().any()


def test_fraud_rate_configuration():
    gen = UPIDataGenerator(num_transactions=2000, fraud_rate=0.04, seed=123)
    df = gen.generate()

    fraud_rate = df["is_fraud"].mean()
    assert 0.03 <= fraud_rate <= 0.05
    assert df["amount"].min() > 0
