from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, REPORT_DIR
from src.pipeline.run_experiment import _local_contribution, _sequence_summary


def load_bundle(path: str | Path = MODEL_DIR / "turbxplain_bundle.joblib") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run `python -m src.pipeline.run_experiment` first.")
    return joblib.load(path)


def predict_sequence(bundle: dict, sequence: np.ndarray) -> dict:
    sequence = np.asarray(sequence, dtype=float)
    if sequence.ndim == 2:
        sequence = sequence[None, :, :]
    if sequence.ndim != 3:
        raise ValueError("Expected sequence with shape (seq_len, features) or (batch, seq_len, features).")
    seq_features = _sequence_summary(sequence)
    tab_features = sequence[:, -1, :]
    lstm_pred = bundle["lstm_proxy"].predict(seq_features)
    xgb_pred = bundle["xgb_proxy"].predict(tab_features)
    weights = np.asarray(bundle["weights"], dtype=float)
    ensemble_pred = weights[0] * lstm_pred + weights[1] * xgb_pred
    contributions = _local_contribution(bundle["shap_surrogate"], tab_features)
    return {
        "rul_prediction": float(ensemble_pred[0]),
        "lstm_prediction": float(lstm_pred[0]),
        "xgboost_prediction": float(xgb_pred[0]),
        "model_weights": {"lstm": float(weights[0]), "xgboost": float(weights[1])},
        "shap_values": dict(zip(bundle["feature_names"], map(float, contributions[0]))),
    }


def risk_level(rul: float) -> str:
    if rul > 80:
        return "low"
    if rul > 40:
        return "medium"
    if rul > 15:
        return "high"
    return "critical"


def load_report_table(name: str) -> pd.DataFrame:
    path = REPORT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
