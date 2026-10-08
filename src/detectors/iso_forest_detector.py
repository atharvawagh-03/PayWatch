"""
Isolation Forest Multivariate Anomaly Detector for PayWatch (FR-5).
Detects multi-feature anomalies and rare feature combinations unsupervised.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsoForestAnomalyDetector:
    """
    Wrapper for scikit-learn IsolationForest with feature scaling,
    score normalization [0.0, 1.0], and artifact persistence.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.035,
        max_samples: Union[int, float] = 256,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.score_offset_: float = 0.0

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "IsoForestAnomalyDetector":
        """
        Fits StandardScaler and IsolationForest on unlabelled engineered feature matrix X.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        self.is_fitted = True
        self.score_offset_ = float(self.model.offset_)
        return self

    def _normalize_score(self, decision_scores: np.ndarray) -> np.ndarray:
        """
        IsolationForest decision_function outputs negative values for anomalies and positive for inliers.
        We invert and normalize to [0.0, 1.0], where 1.0 means highly anomalous.
        """
        # decision_scores <= 0 corresponds to prediction == -1 (anomaly)
        # Using sigmoid transformation around offset 0
        norm = 1.0 / (1.0 + np.exp(decision_scores * 12.0))
        return np.clip(norm, 0.0, 1.0)

    def predict_single(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Scores a single row feature vector (1, d).
        """
        if not self.is_fitted:
            raise RuntimeError("IsoForestAnomalyDetector must be fitted before predict.")

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        X_scaled = self.scaler.transform(feature_vector)
        pred_label = int(self.model.predict(X_scaled)[0])  # -1 for anomaly, 1 for inlier
        decision_score = float(self.model.decision_function(X_scaled)[0])

        normalized_score = float(self._normalize_score(np.array([decision_score]))[0])
        is_flagged = bool(pred_label == -1)

        reasons = ["ISO_FOREST_ANOMALY"] if is_flagged else []

        return {
            "is_flagged": is_flagged,
            "score": round(normalized_score, 4),
            "decision_score": round(decision_score, 4),
            "reasons": reasons,
        }

    def predict_batch(self, X: np.ndarray) -> pd.DataFrame:
        """
        Batch prediction on matrix X.
        """
        if not self.is_fitted:
            raise RuntimeError("IsoForestAnomalyDetector must be fitted before predict.")

        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)  # -1 or 1
        decision_scores = self.model.decision_function(X_scaled)
        normalized_scores = self._normalize_score(decision_scores)

        is_flagged = (preds == -1).astype(int)
        reasons = [("ISO_FOREST_ANOMALY" if f == 1 else "") for f in is_flagged]

        return pd.DataFrame({
            "iso_flag": is_flagged,
            "iso_score": np.round(normalized_scores, 4),
            "iso_reasons": reasons,
        })

    def save(
        self,
        model_path: Union[str, Path] = "models/isolation_forest.joblib",
        scaler_path: Union[str, Path] = "models/scaler.joblib",
        feature_names_path: Union[str, Path] = "models/feature_names.joblib",
    ):
        """
        Persists trained model and scaler artifacts.
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, feature_names_path)

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = "models/isolation_forest.joblib",
        scaler_path: Union[str, Path] = "models/scaler.joblib",
        feature_names_path: Union[str, Path] = "models/feature_names.joblib",
    ) -> "IsoForestAnomalyDetector":
        """
        Loads saved artifacts into detector instance.
        """
        detector = cls()
        detector.model = joblib.load(model_path)
        detector.scaler = joblib.load(scaler_path)
        detector.feature_names = joblib.load(feature_names_path)
        detector.is_fitted = True
        detector.score_offset_ = float(detector.model.offset_)
        return detector
