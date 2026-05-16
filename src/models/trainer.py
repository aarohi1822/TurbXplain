import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, REPORT_DIR
from src.data.preprocessor import CMAPSSPreprocessor
from src.evaluation.metrics import nasa_score, regression_report, rmse
from src.explainability.degradation_fingerprint import DegradationFingerprinter
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.temporal_waterfall import TemporalSHAPWaterfall
from src.models.ensemble import EnsemblePredictor
from src.models.lstm_model import LSTMPredictor
from src.models.xgb_model import XGBPredictor


class LSTMTrainer:
    def __init__(
        self,
        model: LSTMPredictor,
        checkpoint_path: str | Path = "models/lstm_best.pt",
        batch_size: int = 128,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        patience: int = 15,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_fn = nn.HuberLoss(delta=10.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_rmse": [], "val_nasa_score": []}

    def fit(self, X_train, y_train, X_val, y_val, epochs: int = 150) -> dict[str, list[float]]:
        train_loader = self._loader(X_train, y_train, shuffle=True)
        best_rmse = float("inf")
        stale_epochs = 0
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            train_rmse = rmse(y_train, train_pred)
            val_rmse_val = rmse(y_val, val_pred)
            val_score = nasa_score(y_val, val_pred)
            self.scheduler.step(epoch)
            self.history["train_loss"].append(train_loss)
            self.history.setdefault("train_rmse", []).append(train_rmse)
            self.history["val_rmse"].append(val_rmse_val)
            self.history["val_nasa_score"].append(val_score)
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs} | Train RMSE: {train_rmse:.2f} | "
                f"Val RMSE: {val_rmse_val:.2f} | NASA: {val_score:.2f} | LR: {lr:.6f}",
                flush=True,
            )

            if val_rmse_val < best_rmse:
                best_rmse = val_rmse_val
                stale_epochs = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  Saved new best LSTM checkpoint: Val RMSE {best_rmse:.2f}", flush=True)
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    print(f"  Early stopping after {epoch + 1} epochs.", flush=True)
                    break

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        return self.history

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X)), shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self.model(xb.to(self.device)).cpu().numpy())
        return np.concatenate(preds)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    def _loader(self, X, y, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


def main() -> None:
    set_seeds(42)
    dataset = "FD001"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sequence_path = PROCESSED_DATA_DIR / f"{dataset}_sequences.npz"
    if not sequence_path.exists():
        print(f"[1/8] Preprocessing {dataset} raw data...", flush=True)
        CMAPSSPreprocessor(dataset=dataset).run()

    print(f"[2/8] Loading processed arrays from {sequence_path}...", flush=True)
    arrays = np.load(sequence_path, allow_pickle=True)
    X_train = arrays["X_train"].astype(np.float32)
    y_train = arrays["y_train"].astype(np.float32)
    X_test = arrays["X_test"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.float32)
    train_engine_ids = arrays["train_engine_ids"]
    test_engine_ids = arrays["test_engine_ids"]
    feature_names = arrays["feature_names"].astype(str).tolist()

    train_idx, val_idx = split_by_engine(train_engine_ids, val_fraction=0.2)
    X_fit, y_fit = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(
        f"[3/8] Training LSTM on {len(X_fit)} windows; validating on {len(X_val)} windows...",
        flush=True,
    )
    lstm = LSTMPredictor(num_features=X_train.shape[2])
    lstm_trainer = LSTMTrainer(lstm, checkpoint_path=MODEL_DIR / "lstm_best.pt")
    history = lstm_trainer.fit(X_fit, y_fit, X_val, y_val, epochs=150)
    (REPORT_DIR / "lstm_history.json").write_text(json.dumps(history, indent=2))

    X_fit_tab = X_fit[:, -1, :]
    X_val_tab = X_val[:, -1, :]
    X_test_tab = X_test[:, -1, :]

    print("[4/8] Running XGBoost RandomizedSearchCV on a reproducible tuning subset...", flush=True)
    xgb = XGBPredictor(model_path=MODEL_DIR / "xgb_best.json")
    search_idx = np.random.default_rng(42).choice(len(X_fit_tab), size=min(3000, len(X_fit_tab)), replace=False)
    xgb.randomized_search(X_fit_tab[search_idx], y_fit[search_idx], n_iter=20, cv=3)
    print("[5/8] Fitting tuned XGBoost on full training split...", flush=True)
    xgb.fit(X_fit_tab, y_fit, X_val_tab, y_val)

    print("[6/8] Learning ensemble weights and generating ablation report...", flush=True)
    val_lstm = lstm_trainer.predict(X_val)
    val_xgb = xgb.predict(X_val_tab)
    ensemble = EnsemblePredictor(weights=(0.5, 0.5))
    weights = ensemble.learn_weights(val_lstm, val_xgb, y_val)
    (MODEL_DIR / "ensemble_weights.json").write_text(json.dumps({"lstm": float(weights[0]), "xgboost": float(weights[1])}, indent=2))

    test_lstm = lstm_trainer.predict(X_test)
    test_xgb = xgb.predict(X_test_tab)
    test_ensemble = ensemble.predict_from_predictions(test_lstm, test_xgb)
    ablation = ensemble.ablation_report(test_lstm, test_xgb, y_test, REPORT_DIR / "ablation_results.json")

    print("[7/8] Computing SHAP explanations and degradation fingerprints...", flush=True)
    shap_payload = build_shap_artifacts(xgb, X_test_tab, feature_names, test_engine_ids)
    fingerprints = build_fingerprints(shap_payload["temporal_matrices"], y_test, test_engine_ids)
    print("[8/8] Benchmarking latency, saving bundle, and writing reports...", flush=True)
    latency = benchmark_latency(lstm_trainer, xgb, ensemble, X_test, X_test_tab, shap_payload["explainer"])

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

    bundle = {
        "kind": "real",
        "dataset": dataset,
        "lstm_model": lstm_trainer.model.cpu(),
        "xgb_model": xgb.model,
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
    write_results_summary(bundle["metrics"], ablation, fingerprints, latency)
    print(json.dumps(bundle["metrics"], indent=2))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_engine(engine_ids: np.ndarray, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(engine_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    val_count = max(1, int(len(unique) * val_fraction))
    val_engines = set(unique[:val_count])
    val_mask = np.asarray([engine in val_engines for engine in engine_ids])
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def build_shap_artifacts(xgb: XGBPredictor, X_test_tab: np.ndarray, feature_names: list[str], engine_ids: np.ndarray) -> dict:
    sample_limit = min(len(X_test_tab), 2000)
    explainer = SHAPExplainer(xgb_model=xgb.model, feature_names=feature_names)
    shap_values = explainer.explain_batch(X_test_tab[:sample_limit], model_type="xgboost")
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "engine_id", engine_ids[:sample_limit])
    shap_df.to_csv(REPORT_DIR / "shap_contributions.csv", index=False)
    shap_df.drop(columns=["engine_id"]).abs().mean().sort_values(ascending=False).head(25).to_frame(
        "mean_abs_shap"
    ).to_csv(REPORT_DIR / "global_feature_importance.csv")

    waterfall = TemporalSHAPWaterfall(feature_names)
    matrices = {}
    for engine_id, group in shap_df.groupby("engine_id"):
        cycles = np.arange(1, len(group) + 1)
        matrices[int(engine_id)] = waterfall.build_matrix(group.drop(columns=["engine_id"]).to_numpy(), cycles)
    return {"explainer": explainer, "temporal_matrices": matrices}


def build_fingerprints(temporal_matrices: dict[int, pd.DataFrame], y_test: np.ndarray, engine_ids: np.ndarray) -> dict:
    fingerprinter = DegradationFingerprinter(n_clusters=3)
    fingerprinter.fit(temporal_matrices)
    labels = fingerprinter.cluster_labels_
    engine_to_min_rul = pd.DataFrame({"engine_id": engine_ids, "rul": y_test}).groupby("engine_id")["rul"].min()
    rows = []
    label_names = {0: "fast-burn", 1: "slow-decay", 2: "sudden-failure"}
    for cluster in range(3):
        cluster_engines = [engine for engine, label in zip(fingerprinter.engine_ids_, labels) if int(label) == cluster]
        rows.append(
            {
                "cluster": cluster,
                "label": label_names[cluster],
                "engine_count": len(cluster_engines),
                "avg_rul_at_detection": float(engine_to_min_rul.reindex(cluster_engines).mean()),
                "silhouette": float(fingerprinter.silhouette_),
            }
        )
    result = {"overall_silhouette_score": float(fingerprinter.silhouette_), "clusters": rows}
    (REPORT_DIR / "fingerprints.json").write_text(json.dumps(result, indent=2))
    return result


def benchmark_latency(lstm_trainer: LSTMTrainer, xgb: XGBPredictor, ensemble: EnsemblePredictor, X_seq, X_tab, explainer, repeats=30) -> dict:
    seq = X_seq[:1]
    tab = X_tab[:1]

    def measure(fn):
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000.0)
        return float(np.median(timings))

    preprocessing = measure(lambda: seq.astype(np.float32, copy=True))
    lstm_ms = measure(lambda: lstm_trainer.predict(seq))
    xgb_ms = measure(lambda: xgb.predict(tab))
    shap_ms = measure(lambda: explainer.explain_batch(tab, model_type="xgboost"))
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


def write_results_summary(metrics: dict, ablation: dict, fingerprints: dict, latency: dict) -> None:
    lines = [
        "# TurbXplain Results Summary",
        "",
        "Generated by `python -m src.models.trainer`.",
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
        f"- Learned weighted ensemble: RMSE {ablation['learned_ensemble']['rmse']:.3f}",
        f"- Learned weights: LSTM {ablation['learned_ensemble'].get('weights', [0,0])[0]:.3f}, XGBoost {ablation['learned_ensemble'].get('weights', [0,0])[1]:.3f}",
        "",
        "## Explainability Notes",
        "",
        "SHAP values are computed with `shap.TreeExplainer` on the trained XGBoost model. Temporal waterfalls use per-cycle SHAP values grouped by engine.",
        "",
        "## Degradation Fingerprinting",
        "",
        "| Cluster | Label | Engine Count | Avg RUL at Detection | Silhouette |",
        "|---|---|---:|---:|---:|",
    ]
    for row in fingerprints["clusters"]:
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
        ]
    )
    (REPORT_DIR / "results_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
