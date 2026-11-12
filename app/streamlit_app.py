import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_DIR, PROJECT_ROOT as CONFIG_ROOT, REPORT_DIR

RESULTS_DIR = CONFIG_ROOT / "results"

st.set_page_config(page_title="TurbXplain", page_icon="⚙️", layout="wide")

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


st.sidebar.title("⚙️ TurbXplain")
st.sidebar.caption("Explainable Predictive Maintenance")
page = st.sidebar.radio("Navigate", PAGES)

bundle_path = MODEL_DIR / "turbxplain_bundle.joblib"
bundle = joblib.load(bundle_path) if bundle_path.exists() else None

# Standard FD001 benchmark (official test set, RUL cap 125, 5 seeds); written by
# `python -m src.evaluation.benchmark`. Same source as the README results tables.
benchmark_path = RESULTS_DIR / "metrics.json"
benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
BENCH_NAMES = {
    "lstm": "LSTM only",
    "xgboost": "XGBoost only",
    "simple_average": "Simple average",
    "ensemble": "Learned ensemble",
}


def benchmark_caption(m: dict) -> str:
    p = m["protocol"]
    return (
        f"**{m['dataset']}**, official test set. One prediction per engine from its last {p['sequence_length']}-cycle "
        f"window ({m['label_check']['test_engines_evaluated']} engines), target = min(RUL_{m['dataset']}, {p['rul_cap']}). "
        f"Mean ± std over {len(p['seeds'])} seeds. NASA score = PHM08 scoring function summed over the test engines."
    )


# ── PAGE 1: RUL Predictor ──────────────────────────────────────────────────
if page == "RUL Predictor":
    st.title("🔮 RUL Predictor")
    st.markdown("Select an engine to view its predicted Remaining Useful Life and prediction timeline.")

    predictions_path = REPORT_DIR / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        engine_id = st.selectbox("Select Engine", sorted(predictions["engine_id"].unique()))
        engine_rows = predictions[predictions["engine_id"] == engine_id].reset_index(drop=True)

        rul = float(engine_rows["ensemble_prediction"].iloc[-1])
        label, color = risk_badge(rul)

        col1, col2, col3 = st.columns(3)
        col1.metric("Ensemble RUL", f"{rul:.1f} cycles")
        col2.metric("LSTM Prediction", f"{engine_rows['lstm_prediction'].iloc[-1]:.1f} cycles")
        col3.metric("XGBoost Prediction", f"{engine_rows['xgboost_prediction'].iloc[-1]:.1f} cycles")

        st.markdown(f"**Risk Level:** <span style='color:{color};font-size:1.3em;font-weight:700'>{label}</span>", unsafe_allow_html=True)

        engine_rows["cycle"] = np.arange(1, len(engine_rows) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=engine_rows["cycle"], y=engine_rows["y_true"], name="True RUL", mode="lines", line=dict(color="#2196F3", width=2)))
        fig.add_trace(go.Scatter(x=engine_rows["cycle"], y=engine_rows["ensemble_prediction"], name="Ensemble", mode="lines", line=dict(color="#FF5722", width=2)))
        fig.add_trace(go.Scatter(x=engine_rows["cycle"], y=engine_rows["lstm_prediction"], name="LSTM", mode="lines", line=dict(color="#4CAF50", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=engine_rows["cycle"], y=engine_rows["xgboost_prediction"], name="XGBoost", mode="lines", line=dict(color="#9C27B0", width=1, dash="dot")))
        fig.update_layout(title=f"Engine {engine_id} — RUL Predictions Over Time", xaxis_title="Cycle", yaxis_title="RUL (cycles)", template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No predictions found. Run `python -m src.models.trainer` first.")


# ── PAGE 2: Explainability Panel ───────────────────────────────────────────
elif page == "Explainability Panel":
    st.title("🔍 Explainability Panel")
    st.markdown("Temporal SHAP Waterfall — how each sensor's contribution shifts as degradation develops.")

    shap_path = REPORT_DIR / "shap_contributions.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        engine_id = st.selectbox("Select Engine", sorted(shap_df["engine_id"].unique()))
        engine = shap_df[shap_df["engine_id"] == engine_id].drop(columns=["engine_id"]).reset_index(drop=True)

        top_n = st.slider("Top N features", 3, 15, 8)
        top_features = engine.abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
        engine.insert(0, "cycle", np.arange(1, len(engine) + 1))

        long_df = engine.melt(id_vars="cycle", value_vars=top_features, var_name="feature", value_name="shap_value")
        fig = px.line(long_df, x="cycle", y="shap_value", color="feature", title=f"Engine {engine_id} — Temporal SHAP Waterfall")
        fig.update_layout(template="plotly_white", height=500, xaxis_title="Operating Cycle", yaxis_title="SHAP Value (contribution to RUL)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Global Feature Importance (Top 15)")
        importance_path = REPORT_DIR / "global_feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path, index_col=0)
            fig2 = px.bar(importance.head(15).reset_index(), x="mean_abs_shap", y="index", orientation="h", title="Mean |SHAP| Across All Test Predictions")
            fig2.update_layout(template="plotly_white", yaxis_title="Feature", xaxis_title="Mean |SHAP Value|", height=400)
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No SHAP data found. Run `python -m src.models.trainer` first.")


# ── PAGE 3: Degradation Fingerprinting ─────────────────────────────────────
elif page == "Degradation Fingerprinting":
    st.title("🧬 Degradation Fingerprinting")
    st.markdown("Engines clustered into failure archetypes using SHAP trajectory summaries.")

    fingerprints_path = REPORT_DIR / "fingerprints.json"
    if fingerprints_path.exists():
        data = json.loads(fingerprints_path.read_text())
        clusters = pd.DataFrame(data["clusters"])

        col1, col2 = st.columns(2)
        col1.metric("Overall Silhouette Score", f"{data['overall_silhouette_score']:.3f}")
        col2.metric("Number of Clusters", len(clusters))

        st.dataframe(clusters[["label", "engine_count", "avg_rul_at_detection", "silhouette"]], use_container_width=True)

        fig = px.pie(clusters, names="label", values="engine_count", title="Engine Distribution by Failure Pattern", color="label",
                     color_discrete_map={"fast-burn": "#ef5350", "slow-decay": "#42a5f5", "sudden-failure": "#ffa726"})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Clusters above come from the deployed model (first 2,000 test SHAP rows).")
        if benchmark:
            fp = benchmark["summary"]["fingerprints"]
            impl, full = fp["as_implemented_first_2000_rows"], fp["all_test_engines"]
            st.subheader("Do the archetype labels hold up? (5-seed check)")
            st.dataframe(pd.DataFrame([
                {
                    "Engines clustered": name,
                    "Silhouette (mean ± std)": f"{f['silhouette']['mean']:.3f} ± {f['silhouette']['std']:.3f}",
                    "Seeds where fast-burn has steepest slope": f"{f['seeds_where_fast_burn_has_steepest_slope']}/{f['n_seeds']}",
                } for name, f in [("First 2,000 test rows (as implemented)", impl), ("All 100 test engines", full)]
            ]), use_container_width=True, hide_index=True)
            st.warning(
                "⚠️ Exploratory. Silhouette scores are low (no or weak cluster structure), and the names do not consistently "
                "match a model-independent sensor degradation slope. Treat the clusters as indices, not validated failure "
                "types. Details are in the README results section."
            )
        else:
            st.info(f"⚠️ Silhouette score of {data['overall_silhouette_score']:.3f}. This clustering is exploratory.")
    else:
        st.warning("No fingerprint data found. Run `python -m src.models.trainer` first.")


# ── PAGE 4: What-If Simulator ──────────────────────────────────────────────
elif page == "What-If Simulator":
    st.title("🎛️ What-If Simulator")
    st.markdown("Adjust sensor values to see how RUL predictions change. *(Connected to trained model when available.)*")

    if bundle and "xgb_model" in bundle:
        st.markdown("**Using trained XGBoost model for real-time inference.**")
        feature_names = bundle.get("feature_names", [f"feature_{i}" for i in range(bundle.get("num_features", 10))])
        num_features = bundle["num_features"]

        importance_path = REPORT_DIR / "global_feature_importance.csv"
        if importance_path.exists():
            top_feats = pd.read_csv(importance_path, index_col=0).head(8).index.tolist()
        else:
            top_feats = feature_names[:8]

        st.markdown("Adjust the top features below:")
        base_values = np.full(num_features, 0.5)
        for feat in top_feats:
            if feat in feature_names:
                idx = feature_names.index(feat)
                base_values[idx] = st.slider(feat, 0.0, 1.0, 0.5, 0.01)

        import xgboost
        input_array = base_values.reshape(1, -1)
        prediction = bundle["xgb_model"].predict(input_array)[0]
        prediction = max(0, min(125, prediction))

        label, color = risk_badge(prediction)
        col1, col2 = st.columns(2)
        col1.metric("Predicted RUL", f"{prediction:.1f} cycles")
        col2.markdown(f"**Risk:** <span style='color:{color};font-size:1.2em;font-weight:700'>{label}</span>", unsafe_allow_html=True)
    else:
        st.markdown("*Model not loaded. Showing demo simulator.*")
        sensor_7 = st.slider("sensor_7", 0.0, 2.0, 1.0)
        sensor_12 = st.slider("sensor_12", 0.0, 2.0, 1.0)
        simulated_rul = max(0, 100 - int((sensor_7 + sensor_12) * 25))
        st.metric("Simulated RUL", f"{simulated_rul} cycles")


# ── PAGE 5: Model Performance ──────────────────────────────────────────────
elif page == "Model Performance":
    st.title("📊 Model Performance")

    if benchmark:
        off = benchmark["summary"]["official_test"]
        st.markdown(benchmark_caption(benchmark))
        keys = list(BENCH_NAMES)
        st.subheader("Ablation: RMSE (official test set)")
        fig_abl = go.Figure(data=[go.Bar(
            x=[BENCH_NAMES[k] for k in keys],
            y=[off[k]["rmse"]["mean"] for k in keys],
            error_y=dict(type="data", array=[off[k]["rmse"]["std"] for k in keys]),
            marker_color=["#42a5f5", "#ab47bc", "#78909c", "#ef5350"],
            text=[f"{off[k]['rmse']['mean']:.2f}" for k in keys],
            textposition="auto",
        )])
        fig_abl.update_layout(yaxis_title="RMSE (lower is better)", template="plotly_white", height=400)
        st.plotly_chart(fig_abl, use_container_width=True)

        st.subheader("Full Metrics (mean ± std)")
        st.dataframe(pd.DataFrame([
            {
                "Model": BENCH_NAMES[k],
                "RMSE": f"{off[k]['rmse']['mean']:.2f} ± {off[k]['rmse']['std']:.2f}",
                "NASA Score": f"{off[k]['nasa_score']['mean']:.1f} ± {off[k]['nasa_score']['std']:.1f}",
                "MAE": f"{off[k]['mae']['mean']:.2f} ± {off[k]['mae']['std']:.2f}",
                "R²": f"{off[k]['r2']['mean']:.3f} ± {off[k]['r2']['std']:.3f}",
            } for k in keys
        ]), use_container_width=True, hide_index=True)

        w = benchmark["summary"]["ensemble_weight_lstm"]
        st.subheader("Learned Ensemble Weights (mean ± std)")
        col1, col2 = st.columns(2)
        col1.metric("LSTM Weight", f"{w['mean']:.2f} ± {w['std']:.2f}")
        col2.metric("XGBoost Weight", f"{1 - w['mean']:.2f} ± {w['std']:.2f}")

        seeds = benchmark["protocol"]["seeds"]
        seed = st.selectbox("Seed for actual-vs-predicted plot", seeds)
        seed_preds_path = RESULTS_DIR / "predictions" / f"official_test_seed{seed}.csv"
        if seed_preds_path.exists():
            sp = pd.read_csv(seed_preds_path)
            st.subheader(f"Actual vs Predicted, official test set (ensemble, seed {seed})")
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(x=sp["y_true_capped"], y=sp["ensemble"], mode="markers", text=sp["engine_id"],
                                        marker=dict(color="#FF5722", size=7, opacity=0.7), name="Engines"))
            fig_sp.add_trace(go.Scatter(x=[0, 125], y=[0, 125], mode="lines", line=dict(color="gray", dash="dash"), name="Perfect"))
            fig_sp.update_layout(xaxis_title="True RUL (capped at 125)", yaxis_title="Predicted RUL", template="plotly_white", height=450)
            st.plotly_chart(fig_sp, use_container_width=True)
        st.divider()
        st.caption("Below: training curve and latency of the deployed model bundle.")

    if not benchmark and bundle and "metrics" in bundle:
        metrics = bundle["metrics"]
        models = list(metrics.keys())
        rmse_vals = [metrics[m]["rmse"] for m in models]
        r2_vals = [metrics[m]["r2"] for m in models]
        mae_vals = [metrics[m]["mae"] for m in models]
        nasa_vals = [metrics[m]["nasa_score"] for m in models]

        st.subheader("RMSE Comparison")
        colors = ["#4CAF50", "#9C27B0", "#FF5722"]
        fig_rmse = go.Figure(data=[go.Bar(x=models, y=rmse_vals, marker_color=colors, text=[f"{v:.2f}" for v in rmse_vals], textposition="auto")])
        fig_rmse.update_layout(title="RMSE by Model", yaxis_title="RMSE (lower is better)", template="plotly_white", height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.subheader("Full Metrics")
        metrics_df = pd.DataFrame({
            "Model": [m.upper() for m in models],
            "RMSE": [f"{v:.2f}" for v in rmse_vals],
            "MAE": [f"{v:.2f}" for v in mae_vals],
            "R²": [f"{v:.4f}" for v in r2_vals],
            "NASA Score": [f"{v:,.0f}" for v in nasa_vals],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        weights_path = MODEL_DIR / "ensemble_weights.json"
        if weights_path.exists():
            weights = json.loads(weights_path.read_text())
            st.subheader("Learned Ensemble Weights")
            col1, col2 = st.columns(2)
            col1.metric("LSTM Weight", f"{weights.get('lstm', 0):.3f}")
            col2.metric("XGBoost Weight", f"{weights.get('xgboost', 0):.3f}")

    ablation_path = REPORT_DIR / "ablation_results.json"
    if not benchmark and ablation_path.exists():
        ablation = json.loads(ablation_path.read_text())
        st.subheader("Ablation Study")
        ablation_df = pd.DataFrame([
            {"Configuration": "LSTM only", "RMSE": ablation["lstm_only"]["rmse"]},
            {"Configuration": "XGBoost only", "RMSE": ablation["xgboost_only"]["rmse"]},
            {"Configuration": "Simple average", "RMSE": ablation["simple_average"]["rmse"]},
            {"Configuration": "Learned ensemble", "RMSE": ablation["learned_ensemble"]["rmse"]},
        ])
        fig_abl = px.bar(ablation_df, x="Configuration", y="RMSE", color="Configuration", title="Ablation: Ensemble vs Individual Models",
                         color_discrete_sequence=["#42a5f5", "#ab47bc", "#78909c", "#ef5350"])
        fig_abl.update_layout(template="plotly_white", showlegend=False, height=400)
        st.plotly_chart(fig_abl, use_container_width=True)

    history_path = REPORT_DIR / "lstm_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        st.subheader("LSTM Training Curve")
        epochs = list(range(1, len(history.get("train_rmse", history.get("train_loss", []))) + 1))

        fig_train = go.Figure()
        if "train_rmse" in history:
            fig_train.add_trace(go.Scatter(x=epochs, y=history["train_rmse"], name="Train RMSE", mode="lines", line=dict(color="#2196F3")))
        if "val_rmse" in history:
            fig_train.add_trace(go.Scatter(x=epochs, y=history["val_rmse"], name="Val RMSE", mode="lines", line=dict(color="#FF5722")))
        fig_train.update_layout(title="LSTM RMSE Over Training Epochs", xaxis_title="Epoch", yaxis_title="RMSE", template="plotly_white", height=400)
        st.plotly_chart(fig_train, use_container_width=True)

    predictions_path = REPORT_DIR / "predictions.csv"
    if not benchmark and predictions_path.exists():
        preds = pd.read_csv(predictions_path)
        st.subheader("Actual vs Predicted (Ensemble)")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=preds["y_true"], y=preds["ensemble_prediction"], mode="markers", marker=dict(color="#FF5722", size=3, opacity=0.5), name="Predictions"))
        max_val = max(preds["y_true"].max(), preds["ensemble_prediction"].max())
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="gray", dash="dash"), name="Perfect"))
        fig_scatter.update_layout(title="Actual vs Predicted RUL", xaxis_title="True RUL", yaxis_title="Predicted RUL", template="plotly_white", height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    latency_path = REPORT_DIR / "latency.json"
    if latency_path.exists():
        latency = json.loads(latency_path.read_text())
        st.subheader("Inference Latency")
        lat_df = pd.DataFrame([{"Component": k.replace("_ms", "").replace("_", " ").title(), "Time (ms)": f"{v:.3f}"} for k, v in latency.items()])
        st.dataframe(lat_df, use_container_width=True, hide_index=True)


# ── PAGE 6: Download Reports ──────────────────────────────────────────────
else:
    st.title("📥 Download Reports")
    st.markdown("Download generated artifacts from the latest training run.")

    report_files = {
        "FD001 Benchmark Metrics (5 seeds)": RESULTS_DIR / "metrics.json",
        "Results Summary": REPORT_DIR / "results_summary.md",
        "Ablation Results": REPORT_DIR / "ablation_results.json",
        "Fingerprints": REPORT_DIR / "fingerprints.json",
        "SHAP Contributions": REPORT_DIR / "shap_contributions.csv",
        "Predictions": REPORT_DIR / "predictions.csv",
        "Latency Benchmarks": REPORT_DIR / "latency.json",
        "LSTM Training History": REPORT_DIR / "lstm_history.json",
        "Global Feature Importance": REPORT_DIR / "global_feature_importance.csv",
    }

    for name, path in report_files.items():
        if path.exists():
            content = path.read_bytes()
            st.download_button(f"📄 {name}", content, file_name=path.name, key=name)
        else:
            st.caption(f"⏳ {name} — not generated yet")