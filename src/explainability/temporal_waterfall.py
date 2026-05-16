import numpy as np
import pandas as pd
import plotly.express as px


class TemporalSHAPWaterfall:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names

    def build_matrix(self, shap_values: np.ndarray, cycles: np.ndarray | None = None) -> pd.DataFrame:
        if cycles is None:
            cycles = np.arange(1, shap_values.shape[0] + 1)
        df = pd.DataFrame(shap_values, columns=self.feature_names)
        df.insert(0, "cycle", cycles)
        return df

    def top_features(self, shap_matrix: pd.DataFrame, top_n: int = 8) -> list[str]:
        feature_cols = [col for col in shap_matrix.columns if col != "cycle"]
        importance = shap_matrix[feature_cols].abs().mean().sort_values(ascending=False)
        return importance.head(top_n).index.tolist()

    def generate_temporal_waterfall(self, shap_matrix: pd.DataFrame, top_n: int = 8):
        features = self.top_features(shap_matrix, top_n)
        long_df = shap_matrix.melt(id_vars="cycle", value_vars=features, var_name="feature", value_name="shap_value")
        fig = px.line(long_df, x="cycle", y="shap_value", color="feature", title="Temporal SHAP Waterfall")
        fig.update_layout(hovermode="x unified", yaxis_title="Contribution to RUL")
        return fig

    def detect_degradation_onset(self, shap_matrix: pd.DataFrame, threshold_quantile: float = 0.9) -> dict:
        features = [col for col in shap_matrix.columns if col != "cycle"]
        dominant = shap_matrix[features].abs().mean().idxmax()
        values = shap_matrix[dominant].abs()
        threshold = max(float(values.mean() * 2.0), float(values.quantile(threshold_quantile)))
        crossed = shap_matrix.loc[values >= threshold, "cycle"]
        onset = int(crossed.iloc[0]) if not crossed.empty else None
        return {"dominant_feature": dominant, "threshold": float(threshold), "onset_cycle": onset}
