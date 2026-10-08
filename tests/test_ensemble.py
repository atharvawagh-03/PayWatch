"""
Unit tests for Ensemble Decision Engine & Golden Transactions (FR-7, FR-9)
"""

from unittest.mock import MagicMock
import pytest
from src.ensemble import EnsembleFraudEngine, DECISION_MATRIX
from src.detectors.iqr_detector import IQROutlierDetector
from src.detectors.iso_forest_detector import IsoForestAnomalyDetector
from src.detectors.timeseries_detector import TimeSeriesAnomalyDetector


@pytest.fixture
def mock_ensemble():
    iqr_mock = MagicMock(spec=IQROutlierDetector)
    iso_mock = MagicMock(spec=IsoForestAnomalyDetector)
    ts_mock = MagicMock(spec=TimeSeriesAnomalyDetector)

    # Defaults: clean inlier
    iqr_mock.predict_single.return_value = {"is_flagged": False, "score": 0.0, "reasons": []}
    iso_mock.predict_single.return_value = {"is_flagged": False, "score": 0.1, "reasons": []}
    ts_mock.evaluate_transaction.return_value = {"is_flagged": False, "score": 0.0, "reasons": []}

    engine = EnsembleFraudEngine(
        iqr_detector=iqr_mock,
        iso_detector=iso_mock,
        timeseries_detector=ts_mock,
    )
    return engine, iqr_mock, iso_mock, ts_mock


def test_ensemble_low_risk_zero_flags(mock_ensemble):
    engine, _, _, _ = mock_ensemble
    tx = {
        "transaction_id": "TXN_GOLDEN_NORMAL",
        "sender_id": "U_1001",
        "receiver_id": "U_1002",
        "amount": 250.0,
        "timestamp": "2026-08-01T12:00:00+05:30",
        "merchant_category": "food",
        "transaction_type": "P2M",
        "location": "Mumbai",
        "device_type": "Android",
        "upi_channel": "QR",
    }
    res = engine.score_single(tx)
    assert res["risk_level"] == "low"
    assert res["decision"] == "allow"
    assert res["fraud_score"] <= 35
    assert len(res["reasons"]) == 0


def test_ensemble_single_flag_medium_risk(mock_ensemble):
    engine, iqr_mock, _, _ = mock_ensemble
    # Flag IQR only
    iqr_mock.predict_single.return_value = {
        "is_flagged": True,
        "score": 0.6,
        "reasons": ["AMOUNT_OUTLIER"],
    }
    tx = {
        "transaction_id": "TXN_SINGLE_FLAG",
        "sender_id": "U_1001",
        "receiver_id": "U_1002",
        "amount": 1500.0,
        "timestamp": "2026-08-01T12:00:00+05:30",
        "merchant_category": "P2P",
        "transaction_type": "P2P",
        "location": "Mumbai",
        "device_type": "Android",
        "upi_channel": "app",
    }
    res = engine.score_single(tx)
    assert res["risk_level"] == "medium"
    assert res["decision"] == "mark_for_review"
    assert 40 <= res["fraud_score"] <= 69
    assert "AMOUNT_OUTLIER" in res["reasons"]


def test_ensemble_all_flags_critical_risk(mock_ensemble):
    engine, iqr_mock, iso_mock, ts_mock = mock_ensemble
    iqr_mock.predict_single.return_value = {"is_flagged": True, "score": 0.9, "reasons": ["AMOUNT_OUTLIER"]}
    iso_mock.predict_single.return_value = {"is_flagged": True, "score": 0.95, "reasons": ["ISO_FOREST_ANOMALY"]}
    ts_mock.evaluate_transaction.return_value = {"is_flagged": True, "score": 0.85, "reasons": ["TIME_SERIES_BURST"]}

    tx = {
        "transaction_id": "TXN_GOLDEN_CRITICAL",
        "sender_id": "U_ATTACKER",
        "receiver_id": "U_DRAIN_99",
        "amount": 48500.0,
        "timestamp": "2026-08-01T02:14:00+05:30",
        "merchant_category": "P2P",
        "transaction_type": "P2P",
        "location": "Nashik",
        "device_type": "web",
        "upi_channel": "intent_link",
    }
    res = engine.score_single(tx)
    assert res["risk_level"] == "critical"
    assert res["decision"] == "escalate_with_priority"
    assert res["fraud_score"] >= 90
    assert "AMOUNT_OUTLIER" in res["reasons"]
    assert "ISO_FOREST_ANOMALY" in res["reasons"]
    assert "TIME_SERIES_BURST" in res["reasons"]
