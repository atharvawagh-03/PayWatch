"""
Unit tests for IQROutlierDetector (FR-4)
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.detectors.iqr_detector import IQROutlierDetector


@pytest.fixture
def sample_training_data():
    data = []
    # User 1: Regular amounts around 500
    for i in range(20):
        data.append({
            "sender_id": "U_REGULAR",
            "amount": 400.0 + (i * 10),
            "tx_count_last_1h": 0,
            "amount_zscore": 0.2,
        })
    # User 2: Only 2 transactions (should fall back to global)
    data.append({
        "sender_id": "U_NEWBIE",
        "amount": 550.0,
        "tx_count_last_1h": 0,
        "amount_zscore": 0.3,
    })
    data.append({
        "sender_id": "U_NEWBIE",
        "amount": 600.0,
        "tx_count_last_1h": 0,
        "amount_zscore": 0.4,
    })
    return pd.DataFrame(data)


def test_iqr_fit_and_predict_normal(sample_training_data):
    detector = IQROutlierDetector(k_factor=1.5, min_samples_per_user=5)
    detector.fit(sample_training_data)

    normal_tx = {
        "sender_id": "U_REGULAR",
        "amount": 500.0,
        "tx_count_last_1h": 0,
        "amount_zscore": 0.1,
    }
    res = detector.predict_single(normal_tx)
    assert not res["is_flagged"]
    assert res["score"] == 0.0
    assert len(res["reasons"]) == 0


def test_iqr_predict_amount_outlier(sample_training_data):
    detector = IQROutlierDetector(k_factor=1.5, min_samples_per_user=5)
    detector.fit(sample_training_data)

    outlier_tx = {
        "sender_id": "U_REGULAR",
        "amount": 15000.0,  # massive outlier
        "tx_count_last_1h": 0,
        "amount_zscore": 8.5,
    }
    res = detector.predict_single(outlier_tx)
    assert res["is_flagged"]
    assert "AMOUNT_OUTLIER" in res["reasons"]
    assert res["score"] > 0.0


def test_iqr_frequency_spike(sample_training_data):
    detector = IQROutlierDetector(k_factor=1.5, min_samples_per_user=5)
    detector.fit(sample_training_data)

    freq_spike_tx = {
        "sender_id": "U_REGULAR",
        "amount": 450.0,
        "tx_count_last_1h": 8,  # spike
        "amount_zscore": 0.0,
    }
    res = detector.predict_single(freq_spike_tx)
    assert res["is_flagged"]
    assert "FREQUENCY_SPIKE" in res["reasons"]


def test_iqr_save_and_load(sample_training_data):
    detector = IQROutlierDetector()
    detector.fit(sample_training_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "iqr_model.joblib"
        detector.save(model_file)
        assert model_file.exists()

        loaded = IQROutlierDetector.load(model_file)
        assert loaded.is_fitted
        assert loaded.global_thresholds == detector.global_thresholds
