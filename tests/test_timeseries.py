"""
Unit tests for TimeSeriesAnomalyDetector (FR-6)
"""

from datetime import datetime, timedelta
import pytest

from src.detectors.timeseries_detector import TimeSeriesAnomalyDetector


def test_normal_spaced_transactions():
    det = TimeSeriesAnomalyDetector(rolling_window_hours=24, burst_count_threshold=3)
    uid = "U_NORMAL_USER"

    # Transactions spaced 4 hours apart
    t1 = datetime(2026, 8, 1, 10, 0, 0)
    res1 = det.evaluate_transaction(uid, 500.0, t1)
    assert not res1["is_flagged"]

    t2 = t1 + timedelta(hours=4)
    res2 = det.evaluate_transaction(uid, 550.0, t2)
    assert not res2["is_flagged"]

    t3 = t2 + timedelta(hours=4)
    res3 = det.evaluate_transaction(uid, 480.0, t3)
    assert not res3["is_flagged"]


def test_rapid_burst_detection():
    det = TimeSeriesAnomalyDetector(burst_interval_seconds=300, burst_count_threshold=3)
    uid = "U_BURST_USER"

    t1 = datetime(2026, 8, 1, 14, 0, 0)
    det.evaluate_transaction(uid, 1000.0, t1)

    t2 = t1 + timedelta(seconds=60)
    det.evaluate_transaction(uid, 1200.0, t2)

    # 3rd transaction within 120 seconds -> triggers burst
    t3 = t1 + timedelta(seconds=120)
    res3 = det.evaluate_transaction(uid, 1500.0, t3)
    assert res3["is_flagged"]
    assert "TIME_SERIES_BURST" in res3["reasons"]
    assert res3["score"] >= 0.65


def test_off_hours_burst_detection():
    det = TimeSeriesAnomalyDetector(off_hours=[1, 2, 3, 4])
    uid = "U_OFF_HOURS"

    # 2:30 AM transaction with new device
    t_odd = datetime(2026, 8, 1, 2, 30, 0)
    res = det.evaluate_transaction(uid, 5000.0, t_odd, is_new_device=1.0)
    assert res["is_flagged"]
    assert "OFF_HOURS_BURST" in res["reasons"]


def test_rolling_amount_spike():
    det = TimeSeriesAnomalyDetector(std_multiplier=2.5)
    uid = "U_ROLLING_USER"

    # Establish baseline of 5 transactions around 200
    base_time = datetime(2026, 8, 1, 12, 0, 0)
    for i in range(5):
        det.evaluate_transaction(uid, 200.0, base_time + timedelta(hours=i))

    # Huge sudden spike to 8000
    spike_time = base_time + timedelta(hours=6)
    res_spike = det.evaluate_transaction(uid, 8000.0, spike_time)
    assert res_spike["is_flagged"]
    assert "ROLLING_AMOUNT_SPIKE" in res_spike["reasons"]
