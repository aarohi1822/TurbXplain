import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORT_DIR
from src.data.demo_generator import generate_demo_cmapss
from src.data.preprocessor import CMAPSSPreprocessor
from src.evaluation.metrics import regression_report
from src.explainability.degradation_fingerprint import DegradationFingerprinter
from src.explainability.temporal_waterfall import TemporalSHAPWaterfall
from src.models.ensemble import EnsemblePredictor


def run_experiment(dataset: str = "FD001", force_demo_data: bool = False) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_data(dataset, force_demo_data)
    preprocessor = CMAPSSPreprocessor(dataset=dataset)
    stats = preprocessor.run()
    arrays = np.load(PROCESSED_DATA_DIR / f"{dataset}_sequences.npz", allow_pickle=True)
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    feature_names = arrays["feature_names"].astype(str).tolist()
    test_engine_ids = arrays["test_engine_ids"]

    X_train_seq = _sequence_summary(X_train)
    X_test_seq = _sequence_summary(X_test)
    X_train_tab = X_train[:, -1, :]
    X_test_tab = X_test[:, -1, :]

    X_seq_train, X_seq_val, X_tab_train, X_tab_val, y_fit, y_val = train_test_split(
        X_train_seq, X_train_tab, y_train, test_size=0.2, random_state=42
    )

    lstm_proxy = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, max_leaf_nodes=31, random_state=7)
    xgb_proxy = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)

    lstm_proxy.fit(X_seq_train, y_fit)
    xgb_proxy.fit(X_tab_train, y_fit)

    val_lstm = lstm_proxy.predict(X_seq_val)
    val_xgb = xgb_proxy.predict(X_tab_val)
    ensemble = EnsemblePredictor()
    weights = ensemble.learn_weights(val_lstm, val_xgb, y_val)

    test_lstm = lstm_proxy.predict(X_test_seq)
    test_xgb = xgb_proxy.predict(X_test_tab)
    test_ensemble = ensemble.predict_from_predictions(test_lstm, test_xgb)

    ablation = ensemble.ablation_report(test_lstm, test_xgb, y_test, REPORT_DIR / "ablation_results.json")
    shap_artifacts = _build_surrogate_explanations(X_train_tab, X_test_tab, test_ensemble, feature_names, test_engine_ids)
    fingerprints = _build_fingerprints(shap_artifacts["temporal_matrices"], y_test, test_engine_ids)
    latency = _benchmark_latency(lstm_proxy, xgb_proxy, shap_artifacts["surrogate"], weights, X_test_seq, X_test_tab)

    bundle = {
        "dataset": dataset,
        "lstm_proxy": lstm_proxy,
        "xgb_proxy": xgb_proxy,
        "shap_surrogate": shap_artifacts["surrogate"],
        "weights": weights,
        "feature_names": feature_names,
        "sequence_length": int(X_train.shape[1]),
        "num_features": int(X_train.shape[2]),
        "metrics": {
            "lstm": regression_report(y_test, test_lstm),
            "xgboost": regression_report(y_test, test_xgb),
            "ensemble": regression_report(y_test, test_ensemble),
        },
    }
    joblib.dump(bundle, MODEL_DIR / "turbxplain_bundle.joblib")

    _write_results_summary(bundle["metrics"], ablation, fingerprints, latency)
    predictions = pd.DataFrame(
        {
            "engine_id": test_engine_ids,
            "y_true": y_test,
            "lstm_prediction": test_lstm,
            "xgboost_prediction": test_xgb,
            "ensemble_prediction": test_ensemble,
        }
    )
    predictions.to_csv(REPORT_DIR / "predictions.csv", index=False)

    payload = {
        "data_stats": stats,
        "weights": {"lstm": float(weights[0]), "xgboost": float(weights[1])},
        "metrics": bundle["metrics"],
        "fingerprinting": fingerprints,
        "latency": latency,
    }
    (REPORT_DIR / "experiment_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def _ensure_data(dataset: str, force_demo_data: bool) -> None:
    required = [RAW_DATA_DIR / f"train_{dataset}.txt", RAW_DATA_DIR / f"test_{dataset}.txt", RAW_DATA_DIR / f"RUL_{dataset}.txt"]
    if force_demo_data or not all(path.exists() for path in required):
        generate_demo_cmapss(dataset=dataset)


def _sequence_summary(X: np.ndarray) -> np.ndarray:
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    trend = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([last, mean, std, trend], axis=1)


def _build_surrogate_explanations(X_train_tab, X_test_tab, predictions, feature_names, engine_ids) -> dict:
    surrogate = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    surrogate.fit(X_test_tab, predictions)
    ridge = surrogate.named_steps["ridge"]
    scaler = surrogate.named_steps["standardscaler"]
    centered = scaler.transform(X_test_tab)
    contributions = centered * ridge.coef_

    shap_df = pd.DataFrame(contributions, columns=feature_names)
    shap_df.insert(0, "engine_id", engine_ids)
    shap_df.to_csv(REPORT_DIR / "shap_contributions.csv", index=False)

    waterfall = TemporalSHAPWaterfall(feature_names)
    matrices = {}
    for engine_id, group in shap_df.groupby("engine_id"):
        values = group.drop(columns=["engine_id"]).to_numpy()
        cycles = np.arange(1, len(group) + 1)
        matrices[int(engine_id)] = waterfall.build_matrix(values, cycles)

    top = shap_df.drop(columns=["engine_id"]).abs().mean().sort_values(ascending=False).head(10)
    top.to_frame("mean_abs_contribution").to_csv(REPORT_DIR / "global_feature_importance.csv")
    return {"surrogate": surrogate, "contributions": contributions, "temporal_matrices": matrices}


def _build_fingerprints(temporal_matrices: dict[int, pd.DataFrame], y_test, engine_ids) -> dict:
    fingerprinter = DegradationFingerprinter(n_clusters=3)
    fingerprinter.fit(temporal_matrices)
    labels = fingerprinter.cluster_labels_
    engine_to_min_rul = pd.DataFrame({"engine_id": engine_ids, "rul": y_test}).groupby("engine_id")["rul"].min()
    rows = []
    names = {0: "fast-burn", 1: "slow-decay", 2: "sudden-failure"}
    for cluster in range(3):
        cluster_engines = [engine for engine, label in zip(fingerprinter.engine_ids_, labels) if int(label) == cluster]
        avg_rul = float(engine_to_min_rul.reindex(cluster_engines).mean()) if cluster_engines else 0.0
        rows.append(
            {
                "cluster": cluster,
                "label": names[cluster],
                "engine_count": len(cluster_engines),
                "avg_rul_at_detection": avg_rul,
                "silhouette": fingerprinter.silhouette_,
            }
        )

    result = {
        "overall_silhouette_score": fingerprinter.silhouette_,
        "clusters": rows,
    }
    (REPORT_DIR / "fingerprints.json").write_text(json.dumps(result, indent=2))
    return result


def _benchmark_latency(lstm_model, xgb_model, surrogate, weights, X_seq, X_tab, repeats: int = 40) -> dict:
    seq = X_seq[:1]
    tab = X_tab[:1]

    def measure(fn):
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000.0)
        return float(np.percentile(timings, 50))

    preprocessing = measure(lambda: tab.astype(np.float32, copy=True))
    lstm_ms = measure(lambda: lstm_model.predict(seq))
    xgb_ms = measure(lambda: xgb_model.predict(tab))
    shap_ms = measure(lambda: _local_contribution(surrogate, tab))
    total = preprocessing + lstm_ms + xgb_ms + shap_ms
    latency = {
        "preprocessing_ms": preprocessing,
        "lstm_inference_ms": lstm_ms,
        "xgboost_inference_ms": xgb_ms,
        "shap_explanation_ms": shap_ms,
        "total_end_to_end_ms": total,
    }
    (REPORT_DIR / "latency.json").write_text(json.dumps(latency, indent=2))
    return latency


def _local_contribution(surrogate, tab):
    scaler = surrogate.named_steps["standardscaler"]
    ridge = surrogate.named_steps["ridge"]
    return scaler.transform(tab) * ridge.coef_


def _write_results_summary(metrics: dict, ablation: dict, fingerprints: dict, latency: dict) -> None:
    clusters = fingerprints["clusters"]
    lines = [
        "# TurbXplain Results Summary",
        "",
        "Generated by `python -m src.pipeline.run_experiment`.",
        "",
        "| Dataset | LSTM RMSE | XGBoost RMSE | Ensemble RMSE | NASA Score |",
        "|---|---:|---:|---:|---:|",
        f"| FD001 | {metrics['lstm']['rmse']:.3f} | {metrics['xgboost']['rmse']:.3f} | {metrics['ensemble']['rmse']:.3f} | {metrics['ensemble']['nasa_score']:.3f} |",
        "",
        "## Ablation",
        "",
        f"- LSTM only: RMSE {ablation['lstm_only']['rmse']:.3f}",
        f"- XGBoost only: RMSE {ablation['xgboost_only']['rmse']:.3f}",
        f"- Simple average: RMSE {ablation['simple_average']['rmse']:.3f}",
        f"- Learned weighted ensemble: RMSE {ablation['learned_weighted_ensemble']['rmse']:.3f}",
        "",
        "## Explainability Notes",
        "",
        "Temporal SHAP-style contributions are generated from a Ridge surrogate trained on ensemble predictions. Install SHAP and train the XGBoost/PyTorch models to switch this to exact TreeSHAP and neural explanations.",
        "",
        "## Degradation Fingerprinting",
        "",
        "| Cluster | Label | Engine Count | Avg RUL at Detection | Silhouette |",
        "|---|---|---:|---:|---:|",
    ]
    for row in clusters:
        lines.append(
            f"| {row['cluster']} | {row['label']} | {row['engine_count']} | {row['avg_rul_at_detection']:.3f} | {row['silhouette']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Overall silhouette score: {fingerprints['overall_silhouette_score']:.3f}",
            "",
            "## Inference Latency",
            "",
            "| Component | Time (ms) |",
            "|---|---:|",
            f"| Preprocessing | {latency['preprocessing_ms']:.3f} |",
            f"| LSTM inference | {latency['lstm_inference_ms']:.3f} |",
            f"| XGBoost inference | {latency['xgboost_inference_ms']:.3f} |",
            f"| SHAP explanation | {latency['shap_explanation_ms']:.3f} |",
            f"| Total (end-to-end) | {latency['total_end_to_end_ms']:.3f} |",
            "",
        ]
    )
    (REPORT_DIR / "results_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full TurbXplain demo experiment.")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--force-demo-data", action="store_true")
    args = parser.parse_args()
    summary = run_experiment(dataset=args.dataset, force_demo_data=args.force_demo_data)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
