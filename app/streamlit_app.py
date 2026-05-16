import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, REPORT_DIR

st.set_page_config(page_title="TurbXplain", page_icon="TX", layout="wide")

PAGES = [
    "RUL Predictor",
    "Explainability Panel",
    "Degradation Fingerprinting",
    "What-If Simulator",
    "Model Performance",
    "Download Reports",
]


def risk_badge(rul: float) -> tuple[str, str]:
    if rul > 80:
        return "Low", "#2e7d32"
    if rul > 40:
        return "Medium", "#f9a825"
    if rul > 15:
        return "High", "#ef6c00"
    return "Critical", "#c62828"


st.sidebar.title("TurbXplain")
page = st.sidebar.radio("Page", PAGES)
bundle_path = MODEL_DIR / "turbxplain_bundle.joblib"
bundle = joblib.load(bundle_path) if bundle_path.exists() else None

if page == "RUL Predictor":
    st.title("RUL Predictor")
    predictions_path = REPORT_DIR / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        engine_id = st.selectbox("Engine", sorted(predictions["engine_id"].unique()))
        engine_rows = predictions[predictions["engine_id"] == engine_id]
        rul = float(engine_rows["ensemble_prediction"].iloc[-1])
        data = engine_rows.reset_index().rename(columns={"index": "cycle"})
    else:
        cycles = np.arange(1, 121)
        data = pd.DataFrame({"cycle": cycles, "sensor_7": np.sin(cycles / 12) + cycles / 100, "sensor_12": cycles / 80})
        rul = max(0, 125 - int(data["cycle"].max() * 0.8))

    label, color = risk_badge(rul)
    left, right = st.columns([1, 2])
    left.metric("Estimated Remaining Useful Life", f"{rul} cycles")
    left.markdown(f"<span style='color:{color};font-weight:700'>{label} risk</span>", unsafe_allow_html=True)
    y_cols = [col for col in data.columns if col not in {"cycle", "engine_id"}][:6]
    right.plotly_chart(px.line(data, x="cycle", y=y_cols), use_container_width=True)

elif page == "Explainability Panel":
    st.title("Explainability Panel")
    shap_path = REPORT_DIR / "shap_contributions.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        engine_id = st.selectbox("Engine", sorted(shap_df["engine_id"].unique()))
        engine = shap_df[shap_df["engine_id"] == engine_id].drop(columns=["engine_id"]).reset_index(drop=True)
        top_features = engine.abs().mean().sort_values(ascending=False).head(8).index.tolist()
        engine.insert(0, "cycle", np.arange(1, len(engine) + 1))
        long_df = engine.melt(id_vars="cycle", value_vars=top_features, var_name="feature", value_name="shap_value")
        st.plotly_chart(px.line(long_df, x="cycle", y="shap_value", color="feature"), use_container_width=True)
    else:
        st.info("Run `python -m src.pipeline.run_experiment` to generate explanations.")

elif page == "Degradation Fingerprinting":
    st.title("Degradation Fingerprinting")
    fingerprints_path = REPORT_DIR / "fingerprints.json"
    if fingerprints_path.exists():
        data = json.loads(fingerprints_path.read_text())
        clusters = pd.DataFrame(data["clusters"])
        st.dataframe(clusters, use_container_width=True)
        st.plotly_chart(px.bar(clusters, x="label", y="engine_count", color="label"), use_container_width=True)
    else:
        st.info("Run `python -m src.pipeline.run_experiment` to generate fingerprint results.")

elif page == "What-If Simulator":
    st.title("What-If Simulator")
    sensor_7 = st.slider("sensor_7", 0.0, 2.0, 1.0)
    sensor_12 = st.slider("sensor_12", 0.0, 2.0, 1.0)
    simulated_rul = max(0, 100 - int((sensor_7 + sensor_12) * 25))
    st.metric("Simulated RUL", f"{simulated_rul} cycles")

elif page == "Model Performance":
    st.title("Model Performance")
    if bundle:
        results = pd.DataFrame(
            {"model": list(bundle["metrics"].keys()), "rmse": [value["rmse"] for value in bundle["metrics"].values()]}
        )
    else:
        results = pd.DataFrame({"model": [], "rmse": []})
    st.plotly_chart(px.bar(results, x="model", y="rmse"), use_container_width=True)

else:
    st.title("Download Reports")
    st.download_button("Download maintenance report", "TurbXplain maintenance report placeholder\n", file_name="maintenance_report.txt")
