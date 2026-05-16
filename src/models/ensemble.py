import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.evaluation.metrics import regression_report, rmse


class EnsemblePredictor:
    def __init__(self, lstm_model=None, xgb_model=None, weights: tuple[float, float] = (0.5, 0.5)) -> None:
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.weights = np.asarray(weights, dtype=float)

    def learn_weights(self, val_predictions_lstm, val_predictions_xgb, y_val) -> np.ndarray:
        preds = np.vstack([val_predictions_lstm, val_predictions_xgb])

        def objective(weights):
            return rmse(y_val, weights @ preds)

        result = minimize(
            objective,
            x0=np.asarray([0.5, 0.5]),
            bounds=((0.0, 1.0), (0.0, 1.0)),
            constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
        )
        self.weights = result.x
        return self.weights

    def predict_from_predictions(self, lstm_pred, xgb_pred) -> np.ndarray:
        return self.weights[0] * np.asarray(lstm_pred) + self.weights[1] * np.asarray(xgb_pred)

    def predict(self, X_seq, X_tab) -> np.ndarray:
        if self.lstm_model is None or self.xgb_model is None:
            raise ValueError("Both LSTM and XGBoost models are required for direct prediction.")
        return self.predict_from_predictions(self.lstm_model.predict(X_seq), self.xgb_model.predict(X_tab))

    def ablation_report(self, lstm_pred, xgb_pred, y_true, output_path: str | Path = "reports/ablation_results.json") -> dict:
        simple_avg = 0.5 * np.asarray(lstm_pred) + 0.5 * np.asarray(xgb_pred)
        learned = self.predict_from_predictions(lstm_pred, xgb_pred)
        report = {
            "lstm_only": regression_report(y_true, lstm_pred),
            "xgb_only": regression_report(y_true, xgb_pred),
            "xgboost_only": regression_report(y_true, xgb_pred),
            "simple_average": regression_report(y_true, simple_avg),
            "learned_ensemble": {
                **regression_report(y_true, learned),
                "weights": [float(self.weights[0]), float(self.weights[1])],
            },
            "learned_weighted_ensemble": regression_report(y_true, learned),
            "weights": {"lstm": float(self.weights[0]), "xgboost": float(self.weights[1])},
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        return report
