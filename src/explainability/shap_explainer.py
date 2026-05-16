import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import torch


class SHAPExplainer:
    def __init__(self, xgb_model=None, lstm_model=None, background=None, feature_names: list[str] | None = None) -> None:
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.background = background
        self.feature_names = feature_names
        self.xgb_explainer = shap.TreeExplainer(xgb_model) if xgb_model is not None else None
        self.lstm_explainer = None
        if lstm_model is not None and background is not None:
            bg = torch.as_tensor(background, dtype=torch.float32)
            self.lstm_explainer = shap.GradientExplainer(lstm_model, bg)

    def explain_single(self, input_data, model_type: str = "xgboost") -> np.ndarray:
        values = self.explain_batch(np.asarray(input_data), model_type=model_type)
        return values[0]

    def explain_batch(self, engine_data, model_type: str = "xgboost") -> np.ndarray:
        data = np.asarray(engine_data, dtype=np.float32)
        if model_type == "xgboost":
            if self.xgb_explainer is None:
                raise ValueError("XGBoost explainer is not initialized.")
            if data.ndim == 3:
                data = data[:, -1, :]
            values = self.xgb_explainer.shap_values(data)
            return np.asarray(values)
        if model_type == "lstm":
            if self.lstm_explainer is None:
                raise ValueError("LSTM explainer is not initialized.")
            if data.ndim == 2:
                data = data[None, :, :]
            values = self.lstm_explainer.shap_values(torch.as_tensor(data, dtype=torch.float32))
            return np.asarray(values)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def explain_xgb(self, X) -> pd.DataFrame:
        if self.xgb_explainer is None:
            raise ValueError("XGBoost explainer is not initialized.")
        values = self.xgb_explainer.shap_values(X)
        names = self.feature_names or [f"feature_{idx}" for idx in range(values.shape[1])]
        return pd.DataFrame(values, columns=names)

    def top_drivers(self, shap_values: pd.Series | dict, top_n: int = 10) -> dict[str, float]:
        series = pd.Series(shap_values)
        return series.reindex(series.abs().sort_values(ascending=False).head(top_n).index).to_dict()

    def generate_waterfall(self, shap_values: pd.Series | dict, base_value: float = 0.0) -> go.Figure:
        drivers = self.top_drivers(shap_values, top_n=10)
        fig = go.Figure(
            go.Waterfall(
                x=list(drivers.keys()),
                y=list(drivers.values()),
                measure=["relative"] * len(drivers),
                base=base_value,
            )
        )
        fig.update_layout(title="SHAP Waterfall", xaxis_title="Feature", yaxis_title="Contribution to RUL")
        return fig
