"""
Unit tests for IsoForestAnomalyDetector (FR-5)
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.detectors.iso_forest_detector import IsoForestAnomalyDetector


@pytest.fixture
def sample_feature_matrix():
    rng = np.random.default_rng(42)
    # 500 normal points drawn from normal distribution
    normal_data = rng.normal(loc=0.0, scale=1.0, size=(500, 10))
    # 15 extreme outliers
    outliers = rng.uniform(low=8.0, high=15.0, size=(15, 10))
    return np.vstack([normal_data, outliers])


def test_iso_forest_fit_and_predict(sample_feature_matrix):
    detector = IsoForestAnomalyDetector(n_estimators=50, contamination=0.03, random_state=42)
    detector.fit(sample_feature_matrix)

    # Inlier prediction
    normal_vec = np.zeros((1, 10))
    res_normal = detector.predict_single(normal_vec)
    assert not res_normal["is_flagged"]
    assert res_normal["score"] < 0.5
    assert len(res_normal["reasons"]) == 0

    # Strong outlier prediction
    outlier_vec = np.ones((1, 10)) * 12.0
    res_outlier = detector.predict_single(outlier_vec)
    assert res_outlier["is_flagged"]
    assert "ISO_FOREST_ANOMALY" in res_outlier["reasons"]
    assert res_outlier["score"] > 0.5


def test_iso_forest_batch(sample_feature_matrix):
    detector = IsoForestAnomalyDetector(n_estimators=50, contamination=0.03, random_state=42)
    detector.fit(sample_feature_matrix)

    batch_res = detector.predict_batch(sample_feature_matrix)
    assert len(batch_res) == len(sample_feature_matrix)
    assert "iso_flag" in batch_res.columns
    assert "iso_score" in batch_res.columns
    assert batch_res["iso_flag"].sum() > 0


def test_iso_forest_persistence(sample_feature_matrix):
    detector = IsoForestAnomalyDetector(n_estimators=30, random_state=42)
    detector.fit(sample_feature_matrix, feature_names=[f"f_{i}" for i in range(10)])

    with tempfile.TemporaryDirectory() as tmp_dir:
        m_path = Path(tmp_dir) / "model.joblib"
        s_path = Path(tmp_dir) / "scaler.joblib"
        f_path = Path(tmp_dir) / "features.joblib"

        detector.save(m_path, s_path, f_path)
        assert m_path.exists() and s_path.exists() and f_path.exists()

        loaded = IsoForestAnomalyDetector.load(m_path, s_path, f_path)
        assert loaded.is_fitted
        assert loaded.feature_names == detector.feature_names

        vec = np.zeros((1, 10))
        assert loaded.predict_single(vec)["score"] == detector.predict_single(vec)["score"]
