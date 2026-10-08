"""
Data Preprocessor and Feature Pipeline for PayWatch.
Shared between training, offline evaluation, and real-time FastAPI serving to prevent train/serve skew.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.user_history_tracker import UserHistoryTracker


CATEGORICAL_VOCABULARIES = {
    "merchant_category": [
        "groceries", "food", "utilities", "travel", "shopping",
        "entertainment", "electronics", "P2P", "gambling"
    ],
    "transaction_type": ["P2P", "P2M"],
    "device_type": ["Android", "iOS", "web"],
    "upi_channel": ["app", "QR", "intent_link"]
}

NUMERIC_FEATURE_NAMES = [
    "amount",
    "amount_ratio_to_mean",
    "amount_zscore",
    "tx_count_last_1h",
    "time_since_last_tx_sec",
    "recipient_tx_count_24h",
    "is_new_device",
    "is_new_location",
    "hour_of_day",
    "is_weekend"
]


class DataPreprocessor:
    """
    Cleans raw transactions, applies point-in-time behavioral feature extraction,
    and formats feature matrices for anomaly detection models.
    """

    def __init__(self, tracker: Optional[UserHistoryTracker] = None):
        self.tracker = tracker or UserHistoryTracker()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns: List[str] = []

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicates and validates raw dataframe per PRD FR-2.
        """
        cleaned = df.copy()

        # Remove duplicate rows
        cleaned = cleaned.drop_duplicates(subset=["transaction_id"], keep="first")
        cleaned = cleaned.drop_duplicates()

        # Missing value rules per FR-2:
        # Amount: forward fill or median if missing, drop if zero/negative
        if "amount" in cleaned.columns:
            cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
            cleaned = cleaned[cleaned["amount"] > 0]
            cleaned["amount"] = cleaned["amount"].fillna(cleaned["amount"].median())

        # Impute missing categorical defaults
        cleaned["merchant_category"] = cleaned.get("merchant_category", pd.Series()).fillna("P2P")
        cleaned["transaction_type"] = cleaned.get("transaction_type", pd.Series()).fillna("P2P")
        cleaned["device_type"] = cleaned.get("device_type", pd.Series()).fillna("Android")
        cleaned["upi_channel"] = cleaned.get("upi_channel", pd.Series()).fillna("app")
        cleaned["location"] = cleaned.get("location", pd.Series()).fillna("Mumbai")

        # Sort chronologically to maintain temporal integrity
        if "timestamp" in cleaned.columns:
            cleaned["timestamp_dt"] = pd.to_datetime(cleaned["timestamp"])
            cleaned = cleaned.sort_values("timestamp_dt").reset_index(drop=True)
            cleaned = cleaned.drop(columns=["timestamp_dt"])

        return cleaned

    def engineer_features_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates chronologically through dataset, computing point-in-time features via UserHistoryTracker.
        Guarantees zero data leakage.
        """
        cleaned = self.clean_dataframe(df)

        # Pre-calculate global baseline for cold starts
        if len(cleaned) > 0:
            self.tracker.global_mean_amount = float(cleaned["amount"].mean())
            self.tracker.global_std_amount = float(max(10.0, cleaned["amount"].std()))

        engineered_rows: List[Dict] = []

        for _, row in cleaned.iterrows():
            ts_str = str(row["timestamp"]).split("+")[0]
            ts = datetime.fromisoformat(ts_str)

            feat = self.tracker.compute_features(
                sender_id=str(row["sender_id"]),
                receiver_id=str(row["receiver_id"]),
                amount=float(row["amount"]),
                timestamp=ts,
                location=str(row["location"]),
                device_type=str(row["device_type"]),
                update_state=True,
            )

            # Add categorical columns and metadata
            feat["transaction_id"] = row["transaction_id"]
            feat["timestamp"] = row["timestamp"]
            feat["sender_id"] = row["sender_id"]
            feat["receiver_id"] = row["receiver_id"]
            feat["location"] = row["location"]
            feat["merchant_category"] = row["merchant_category"]
            feat["transaction_type"] = row["transaction_type"]
            feat["device_type"] = row["device_type"]
            feat["upi_channel"] = row["upi_channel"]
            if "is_fraud" in row:
                feat["is_fraud"] = int(row["is_fraud"])

            engineered_rows.append(feat)

        return pd.DataFrame(engineered_rows)

    def encode_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Creates one-hot encoded and numeric feature vector for Isolation Forest / ML.
        Deterministic columns regardless of input batch.
        """
        numeric_part = df[NUMERIC_FEATURE_NAMES].copy().values

        encoded_cats = []
        cat_names = []

        for col, vocab in CATEGORICAL_VOCABULARIES.items():
            for val in vocab:
                col_name = f"{col}_{val}"
                cat_names.append(col_name)
                encoded_cats.append((df[col] == val).astype(float).values)

        if encoded_cats:
            cat_array = np.column_stack(encoded_cats)
            feature_matrix = np.hstack([numeric_part, cat_array])
        else:
            feature_matrix = numeric_part

        all_names = NUMERIC_FEATURE_NAMES + cat_names
        self.feature_columns = all_names
        return feature_matrix, all_names

    def transform_single_transaction(self, tx_dict: Dict) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Scores an incoming single streaming transaction for FastAPI / real-time scoring.
        Uses shared UserHistoryTracker point-in-time state.
        """
        ts_raw = str(tx_dict["timestamp"]).split("+")[0]
        ts = datetime.fromisoformat(ts_raw)

        # Extract behavioral features point-in-time
        feat = self.tracker.compute_features(
            sender_id=str(tx_dict["sender_id"]),
            receiver_id=str(tx_dict["receiver_id"]),
            amount=float(tx_dict["amount"]),
            timestamp=ts,
            location=str(tx_dict["location"]),
            device_type=str(tx_dict["device_type"]),
            update_state=True,
        )

        # Build feature vector
        num_vec = [feat[col] for col in NUMERIC_FEATURE_NAMES]
        cat_vec = []
        for col, vocab in CATEGORICAL_VOCABULARIES.items():
            tx_val = tx_dict.get(col, "")
            for val in vocab:
                cat_vec.append(1.0 if tx_val == val else 0.0)

        feature_vector = np.array(num_vec + cat_vec, dtype=np.float32).reshape(1, -1)
        return feat, feature_vector
