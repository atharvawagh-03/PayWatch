"""
Ensemble Fraud Decision Engine for PayWatch (FR-7).
Combines signals from IQR, Isolation Forest, and Time Series anomaly detectors
into an explainable composite risk score, risk level, decision, and reason codes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import get_config
from src.features.preprocessor import DataPreprocessor
from src.detectors.iqr_detector import IQROutlierDetector
from src.detectors.iso_forest_detector import IsoForestAnomalyDetector
from src.detectors.timeseries_detector import TimeSeriesAnomalyDetector


DECISION_MATRIX = {
    0: ("low", "allow"),
    1: ("medium", "mark_for_review"),
    2: ("high", "escalate"),
    3: ("critical", "escalate_with_priority"),
}


class EnsembleFraudEngine:
    """
    Combines three complementary detection layers:
    1. IQR Outlier Detector (Statistical univariate)
    2. Isolation Forest (Multivariate unsupervised)
    3. Time Series Detector (Temporal rolling velocity & burst)
    """

    def __init__(
        self,
        iqr_detector: IQROutlierDetector,
        iso_detector: IsoForestAnomalyDetector,
        timeseries_detector: Optional[TimeSeriesAnomalyDetector] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        weights: Optional[Dict[str, float]] = None,
        model_version: str = "1.0.0",
    ):
        self.iqr_detector = iqr_detector
        self.iso_detector = iso_detector
        self.timeseries_detector = timeseries_detector or TimeSeriesAnomalyDetector()
        self.preprocessor = preprocessor or DataPreprocessor()
        self.weights = weights or {"iqr": 0.30, "isolation_forest": 0.40, "timeseries": 0.30}
        self.model_version = model_version

    @classmethod
    def load_from_artifacts(
        cls,
        iqr_path: str = "models/iqr_detector.joblib",
        iso_model_path: str = "models/isolation_forest.joblib",
        iso_scaler_path: str = "models/scaler.joblib",
        iso_features_path: str = "models/feature_names.joblib",
    ) -> "EnsembleFraudEngine":
        config = get_config()
        iqr_detector = IQROutlierDetector.load(iqr_path)
        iso_detector = IsoForestAnomalyDetector.load(iso_model_path, iso_scaler_path, iso_features_path)
        weights = config.ensemble.get("weights", {"iqr": 0.30, "isolation_forest": 0.40, "timeseries": 0.30})
        version = config.version

        return cls(
            iqr_detector=iqr_detector,
            iso_detector=iso_detector,
            weights=weights,
            model_version=version,
        )

    def score_single(self, transaction: Dict[str, Any], update_state: bool = True) -> Dict[str, Any]:
        """
        Scores a single incoming transaction dict.
        Outputs exact PRD specification:
        transaction_id, fraud_score (0-100), risk_level, decision, reasons, model_version.
        """
        tx_id = str(transaction.get("transaction_id", "TXN_UNKNOWN"))
        sender_id = str(transaction.get("sender_id", "U_UNKNOWN"))
        amount = float(transaction.get("amount", 0.0))
        timestamp = transaction.get("timestamp")

        # 1. Feature Engineering
        feat_dict, feat_vector = self.preprocessor.transform_single_transaction(transaction)
        feat_dict["sender_id"] = sender_id

        # 2. Evaluate Layer 1: IQR Detector
        iqr_res = self.iqr_detector.predict_single(feat_dict, sender_id=sender_id)

        # 3. Evaluate Layer 2: Isolation Forest Detector
        iso_res = self.iso_detector.predict_single(feat_vector)

        # 4. Evaluate Layer 3: Time Series Detector
        ts_res = self.timeseries_detector.evaluate_transaction(
            sender_id=sender_id,
            amount=amount,
            timestamp=timestamp,
            is_new_device=feat_dict.get("is_new_device", 0.0),
            update_state=update_state,
        )

        # Aggregate flags
        flags_count = int(iqr_res["is_flagged"]) + int(iso_res["is_flagged"]) + int(ts_res["is_flagged"])
        risk_level, decision = DECISION_MATRIX.get(flags_count, ("critical", "escalate_with_priority"))

        # Calculate composite 0-100 numeric score
        w_iqr = self.weights.get("iqr", 0.30)
        w_iso = self.weights.get("isolation_forest", 0.40)
        w_ts = self.weights.get("timeseries", 0.30)

        raw_composite = (
            (iqr_res["score"] * w_iqr) +
            (iso_res["score"] * w_iso) +
            (ts_res["score"] * w_ts)
        )

        # Base score calibration according to risk tier:
        # 0 flags -> 0 - 35
        # 1 flag  -> 40 - 69
        # 2 flags -> 70 - 89
        # 3 flags -> 90 - 100
        if flags_count == 0:
            fraud_score = int(np.clip(raw_composite * 45, 0, 35))
        elif flags_count == 1:
            fraud_score = int(np.clip(40 + raw_composite * 35, 40, 69))
        elif flags_count == 2:
            fraud_score = int(np.clip(70 + raw_composite * 20, 70, 89))
        else:
            fraud_score = int(np.clip(90 + raw_composite * 10, 90, 100))

        # Collect and deduplicate reason codes
        all_reasons = []
        all_reasons.extend(iqr_res.get("reasons", []))
        all_reasons.extend(iso_res.get("reasons", []))
        all_reasons.extend(ts_res.get("reasons", []))

        # Additional contextual reasons
        if feat_dict.get("is_new_device", 0) > 0 and flags_count > 0:
            all_reasons.append("NEW_DEVICE")
        if feat_dict.get("is_new_location", 0) > 0 and flags_count > 0:
            all_reasons.append("NEW_LOCATION")
        if feat_dict.get("recipient_tx_count_24h", 0) >= 3 and flags_count > 0:
            all_reasons.append("REPEATED_RECIPIENT_TRANSFER")

        # Deduplicate and sort
        unique_reasons = sorted(list(set(all_reasons)))

        return {
            "transaction_id": tx_id,
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "decision": decision,
            "reasons": unique_reasons,
            "model_version": self.model_version,
            "detector_breakdown": {
                "iqr": {
                    "flagged": iqr_res["is_flagged"],
                    "score": round(iqr_res["score"], 4),
                    "reasons": iqr_res.get("reasons", []),
                },
                "isolation_forest": {
                    "flagged": iso_res["is_flagged"],
                    "score": round(iso_res["score"], 4),
                    "reasons": iso_res.get("reasons", []),
                },
                "timeseries": {
                    "flagged": ts_res["is_flagged"],
                    "score": round(ts_res["score"], 4),
                    "reasons": ts_res.get("reasons", []),
                },
            },
        }

    def score_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scores a list of transactions sequentially.
        """
        return [self.score_single(tx, update_state=True) for tx in transactions]
