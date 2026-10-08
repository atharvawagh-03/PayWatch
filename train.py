"""
Model Training Script for PayWatch Anomaly Detectors.
Trains Isolation Forest and IQR detectors on engineered transactions
and persists versioned artifacts to the models/ directory.
"""

from pathlib import Path
import pandas as pd
from src.config import get_config
from src.features.preprocessor import DataPreprocessor
from src.detectors.iqr_detector import IQROutlierDetector
from src.detectors.iso_forest_detector import IsoForestAnomalyDetector


def train_models():
    config = get_config()
    print("Loading engineered transactions dataset...")
    data_path = Path("data/engineered_transactions.csv")
    if not data_path.exists():
        print("Engineered data not found, generating from raw_transactions.csv...")
        raw_df = pd.read_csv("data/raw_transactions.csv")
        preprocessor = DataPreprocessor()
        df = preprocessor.engineer_features_dataset(raw_df)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} transactions.")

    # 1. Train and persist IQR Outlier Detector
    print("Fitting IQR Outlier Detector...")
    iqr_cfg = config.iqr
    iqr_detector = IQROutlierDetector(
        k_factor=iqr_cfg.get("k_factor", 1.5),
        min_samples_per_user=iqr_cfg.get("min_samples_per_user", 5),
        columns_to_check=iqr_cfg.get("columns_to_check", ["amount", "tx_count_last_1h", "amount_zscore"]),
    )
    iqr_detector.fit(df)
    iqr_path = Path("models/iqr_detector.joblib")
    iqr_detector.save(iqr_path)
    print(f"IQR Detector saved to {iqr_path}.")

    # 2. Train and persist Isolation Forest Anomaly Detector
    print("Preparing feature matrix for Isolation Forest...")
    preprocessor = DataPreprocessor()
    X, feature_names = preprocessor.encode_features(df)
    print(f"Feature matrix shape: {X.shape} with {len(feature_names)} features.")

    iso_cfg = config.isolation_forest
    iso_detector = IsoForestAnomalyDetector(
        n_estimators=iso_cfg.get("n_estimators", 150),
        contamination=iso_cfg.get("contamination", 0.035),
        max_samples=iso_cfg.get("max_samples", 256),
        random_state=iso_cfg.get("random_state", 42),
    )
    iso_detector.fit(X, feature_names=feature_names)

    model_path = Path(iso_cfg.get("model_path", "models/isolation_forest.joblib"))
    scaler_path = Path(iso_cfg.get("scaler_path", "models/scaler.joblib"))
    features_path = Path(iso_cfg.get("feature_names_path", "models/feature_names.joblib"))

    iso_detector.save(model_path, scaler_path, features_path)
    print(f"Isolation Forest model saved to {model_path}.")
    print(f"Feature Scaler saved to {scaler_path}.")
    print(f"Feature Names saved to {features_path}.")
    print("Model training completed successfully!")


if __name__ == "__main__":
    train_models()
