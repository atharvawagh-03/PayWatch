"""
Unit tests for Feature Engineering & Preprocessor (FR-2, FR-3)
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.features.user_history_tracker import UserHistoryTracker
from src.features.preprocessor import DataPreprocessor, NUMERIC_FEATURE_NAMES


def test_user_history_cold_start():
    tracker = UserHistoryTracker(global_mean_amount=1000.0, global_std_amount=500.0)
    feat = tracker.compute_features(
        sender_id="U_NEW",
        receiver_id="U_99",
        amount=1000.0,
        timestamp=datetime(2026, 8, 1, 10, 0, 0),
        location="Mumbai",
        device_type="Android",
        update_state=True,
    )

    assert feat["user_tx_count_history"] == 0
    assert feat["user_mean_amount"] == 1000.0
    assert feat["amount_ratio_to_mean"] == 1.0
    assert feat["tx_count_last_1h"] == 0
    assert feat["is_new_device"] == 0.0  # Cold start initial is not flagged as new device


def test_velocity_and_burst_tracking():
    tracker = UserHistoryTracker()
    uid = "U_VELOCITY_TEST"

    # Transaction 1
    t1 = datetime(2026, 8, 1, 10, 0, 0)
    feat1 = tracker.compute_features(uid, "U_REC", 200.0, t1, "Mumbai", "Android")
    assert feat1["tx_count_last_1h"] == 0

    # Transaction 2 (20 minutes later)
    t2 = datetime(2026, 8, 1, 10, 20, 0)
    feat2 = tracker.compute_features(uid, "U_REC", 250.0, t2, "Mumbai", "Android")
    assert feat2["tx_count_last_1h"] == 1
    assert feat2["recipient_tx_count_24h"] == 1

    # Transaction 3 (50 minutes after t1)
    t3 = datetime(2026, 8, 1, 10, 50, 0)
    feat3 = tracker.compute_features(uid, "U_REC", 300.0, t3, "Mumbai", "Android")
    assert feat3["tx_count_last_1h"] == 2
    assert feat3["recipient_tx_count_24h"] == 2

    # Transaction 4 (2 hours later) -> older transactions out of 1h window
    t4 = datetime(2026, 8, 1, 12, 10, 0)
    feat4 = tracker.compute_features(uid, "U_REC", 300.0, t4, "Mumbai", "Android")
    assert feat4["tx_count_last_1h"] == 0


def test_device_and_location_change_detection():
    tracker = UserHistoryTracker()
    uid = "U_DEVICE_TEST"

    # Seed initial device and city
    t1 = datetime(2026, 8, 1, 10, 0, 0)
    tracker.compute_features(uid, "U_1", 100.0, t1, "Mumbai", "Android")

    # Second transaction from same device and city
    t2 = datetime(2026, 8, 1, 12, 0, 0)
    feat2 = tracker.compute_features(uid, "U_2", 100.0, t2, "Mumbai", "Android")
    assert feat2["is_new_device"] == 0.0
    assert feat2["is_new_location"] == 0.0

    # Third transaction from strange device and location
    t3 = datetime(2026, 8, 1, 14, 0, 0)
    feat3 = tracker.compute_features(uid, "U_3", 100.0, t3, "Delhi", "web")
    assert feat3["is_new_device"] == 1.0
    assert feat3["is_new_location"] == 1.0


def test_preprocessor_single_transaction_pipeline():
    preprocessor = DataPreprocessor()
    tx_sample = {
        "transaction_id": "TXN_TEST_01",
        "timestamp": "2026-08-01T15:30:00+05:30",
        "sender_id": "U_SINGLE",
        "receiver_id": "U_REC_01",
        "amount": 2500.0,
        "merchant_category": "P2P",
        "transaction_type": "P2P",
        "location": "Pune",
        "device_type": "Android",
        "upi_channel": "app"
    }

    feat_dict, feat_vector = preprocessor.transform_single_transaction(tx_sample)
    assert "amount_zscore" in feat_dict
    assert feat_vector.shape[0] == 1
    assert feat_vector.shape[1] > 10
