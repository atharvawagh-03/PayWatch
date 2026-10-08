"""
IQR (Interquartile Range) Statistical Outlier Detector for PayWatch (FR-4).
Provides fast, explainable statistical bounds on transaction amount, frequency,
and spend deviation with support for global and per-user thresholds.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd


COLUMN_REASON_MAP = {
    "amount": "AMOUNT_OUTLIER",
    "tx_count_last_1h": "FREQUENCY_SPIKE",
    "amount_zscore": "USER_DEVIATION_OUTLIER",
}


class IQROutlierDetector:
    """
    Detects univariate statistical outliers via IQR method:
    Outlier if value > Q3 + k * IQR or value < Q1 - k * IQR.
    """

    def __init__(
        self,
        k_factor: float = 1.5,
        min_samples_per_user: int = 5,
        columns_to_check: Optional[List[str]] = None,
    ):
        self.k_factor = k_factor
        self.min_samples_per_user = min_samples_per_user
        self.columns_to_check = columns_to_check or ["amount", "tx_count_last_1h", "amount_zscore"]

        # Fitted parameters
        self.global_thresholds: Dict[str, Dict[str, float]] = {}
        self.user_thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.is_fitted: bool = False

    def _compute_bounds(self, series: pd.Series, col_name: Optional[str] = None) -> Dict[str, float]:
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {"q1": 0.0, "q3": 0.0, "iqr": 0.0, "lower": 0.0, "upper": 0.0}
        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        raw_iqr = q3 - q1
        # For discrete counts like tx_count_last_1h, IQR may be 0; enforce minimum IQR of 1.0 and minimum upper bound
        if col_name == "tx_count_last_1h":
            iqr = float(max(1.0, raw_iqr))
            upper = float(max(3.0, q3 + self.k_factor * iqr))
        else:
            iqr = float(max(1.0, raw_iqr))
            upper = float(q3 + self.k_factor * iqr)
        lower = float(q1 - self.k_factor * iqr)
        return {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
        }

    def fit(self, df: pd.DataFrame, user_col: str = "sender_id") -> "IQROutlierDetector":
        """
        Fits global and per-user IQR bounds on training data.
        """
        # 1. Compute global thresholds
        for col in self.columns_to_check:
            if col in df.columns:
                self.global_thresholds[col] = self._compute_bounds(df[col], col_name=col)

        # 2. Compute per-user thresholds where history >= min_samples_per_user
        self.user_thresholds = {}
        if user_col in df.columns:
            user_counts = df[user_col].value_counts()
            qualified_users = user_counts[user_counts >= self.min_samples_per_user].index

            for user_id in qualified_users:
                user_df = df[df[user_col] == user_id]
                self.user_thresholds[user_id] = {}
                for col in self.columns_to_check:
                    if col in user_df.columns:
                        self.user_thresholds[user_id][col] = self._compute_bounds(user_df[col], col_name=col)

        self.is_fitted = True
        return self

    def predict_single(self, features: Dict[str, Any], sender_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Predicts whether a single transaction feature set is an IQR statistical outlier.
        Returns:
            is_flagged: bool
            score: float (0.0 to 1.0 severity)
            reasons: list of reason codes
            details: dict of triggered thresholds
        """
        if not self.is_fitted:
            raise RuntimeError("IQROutlierDetector must be fitted before predict.")

        sender = sender_id or features.get("sender_id")
        user_bounds = self.user_thresholds.get(sender) if sender else None

        reasons = []
        details = {}
        max_ratio = 0.0

        for col in self.columns_to_check:
            if col not in features:
                continue
            val = float(features[col])

            # Prefer per-user threshold if available and valid IQR, else global
            bounds = None
            source = "global"
            if user_bounds and col in user_bounds and user_bounds[col]["iqr"] > 1e-4:
                bounds = user_bounds[col]
                source = "per_user"
            else:
                bounds = self.global_thresholds.get(col)

            if not bounds:
                continue

            upper = bounds["upper"]
            lower = bounds["lower"]

            # Outlier check (upper bound anomaly for fraud spikes)
            if val > upper:
                reason = COLUMN_REASON_MAP.get(col, f"{col.upper()}_OUTLIER")
                reasons.append(reason)
                excess_ratio = (val - upper) / max(1.0, bounds["iqr"])
                max_ratio = max(max_ratio, excess_ratio)
                details[col] = {
                    "value": val,
                    "upper_bound": upper,
                    "threshold_source": source,
                }

        is_flagged = len(reasons) > 0
        # Normalize severity score into [0.0, 1.0] using sigmoid-like scaling
        severity_score = float(round(1.0 - (1.0 / (1.0 + max_ratio * 0.4)), 4)) if is_flagged else 0.0

        return {
            "is_flagged": is_flagged,
            "score": severity_score,
            "reasons": sorted(list(set(reasons))),
            "details": details,
        }

    def predict_batch(self, df: pd.DataFrame, user_col: str = "sender_id") -> pd.DataFrame:
        """
        Evaluates IQR outlier status across a DataFrame.
        """
        flags = []
        scores = []
        reasons_list = []

        for _, row in df.iterrows():
            sender = row.get(user_col)
            res = self.predict_single(row.to_dict(), sender_id=sender)
            flags.append(int(res["is_flagged"]))
            scores.append(res["score"])
            reasons_list.append(";".join(res["reasons"]))

        return pd.DataFrame({
            "iqr_flag": flags,
            "iqr_score": scores,
            "iqr_reasons": reasons_list,
        }, index=df.index)

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IQROutlierDetector":
        detector = joblib.load(filepath)
        if not isinstance(detector, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}")
        return detector
