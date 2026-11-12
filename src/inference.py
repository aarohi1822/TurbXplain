from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import MODEL_DIR, REPORT_DIR


def load_bundle(path: str | Path = MODEL_DIR / "turbxplain_bundle.joblib") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Run `python -m src.models.trainer` first.")
    return joblib.load(path)


def _as_batch(bundle: dict, sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 2:
        sequence = sequence[None, :, :]
    if sequence.ndim != 3:
        raise ValueError("Expected sequence with shape (seq_len, features) or (batch, seq_len, features).")
    expected = (bundle.get("sequence_length"), bundle.get("num_features"))
    if None not in expected and sequence.shape[1:] != expected:
        raise ValueError(
            f"Expected a window of shape {expected[0]} cycles x {expected[1]} preprocessed features "
            f"(see feature_names in the bundle), got {sequence.shape[1]} x {sequence.shape[2]}."
        )
    return sequence


def _tree_explainer(bundle: dict):
    # Built lazily and cached on the bundle dict; TreeSHAP on the XGBoost component.
    if "_tree_explainer" not in bundle:
        import shap

        bundle["_tree_explainer"] = shap.TreeExplainer(bundle["xgb_model"])
    return bundle["_tree_explainer"]


def predict_sequence(bundle: dict, sequence: np.ndarray) -> dict:
    """Predict RUL for one preprocessed window and return per-feature SHAP values.

    Supports the trained bundle written by ``src.models.trainer`` (PyTorch LSTM +
    XGBoost; SHAP explains the XGBoost component) and the older demo bundle from
    ``src.pipeline.run_experiment`` (sklearn proxies + Ridge surrogate).
    """
    sequence = _as_batch(bundle, sequence)
    tab_features = sequence[:, -1, :]
    weights = np.asarray(bundle["weights"], dtype=float)

    if "lstm_model" in bundle and "xgb_model" in bundle:
        model = bundle["lstm_model"]
        model.eval()
        with torch.no_grad():
            lstm_pred = model(torch.as_tensor(sequence, dtype=torch.float32)).cpu().numpy()
        xgb_pred = bundle["xgb_model"].predict(tab_features)
        contributions = np.asarray(_tree_explainer(bundle).shap_values(tab_features))
        explained_model = "xgboost"
    else:
        from src.pipeline.run_experiment import _local_contribution, _sequence_summary

        lstm_pred = bundle["lstm_proxy"].predict(_sequence_summary(sequence))
        xgb_pred = bundle["xgb_proxy"].predict(tab_features)
        contributions = _local_contribution(bundle["shap_surrogate"], tab_features)
        explained_model = "ridge_surrogate"

    ensemble_pred = weights[0] * lstm_pred + weights[1] * xgb_pred
    return {
        "rul_prediction": float(ensemble_pred[0]),
        "lstm_prediction": float(lstm_pred[0]),
        "xgboost_prediction": float(xgb_pred[0]),
        "model_weights": {"lstm": float(weights[0]), "xgboost": float(weights[1])},
        "shap_values": dict(zip(bundle["feature_names"], map(float, contributions[0]))),
        "explained_model": explained_model,
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
