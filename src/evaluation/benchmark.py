"""Standardised, multi-seed C-MAPSS FD001 benchmark.

Protocol
--------
* Subset: FD001 only (official NASA train_FD001 / test_FD001 / RUL_FD001 files).
* Labels: piecewise-linear RUL capped at 125 for training *and* test targets.
* Test set: the official test set, one prediction per engine using the last
  30-cycle window of each test trajectory (100 engines), scored against
  min(RUL_FD001, 125). This is the convention used in the C-MAPSS literature.
* Metrics: RMSE and the NASA (PHM08) scoring function
  s = sum(exp(-d/13) - 1) for d < 0, sum(exp(d/10) - 1) for d >= 0, d = pred - true.
* Ablation: LSTM only, XGBoost only, simple average, learned weighted ensemble.
  Each run uses the same training recipe as ``src.models.trainer`` and is repeated
  for 5 seeds. The seed controls the engine-level train/val split, LSTM init and
  batch order, XGBoost's random_state and its hyper-parameter search.
* Fingerprints: silhouette score plus the mean sensor-based degradation slope
  per cluster, to check whether the fast-burn / slow-decay / sudden-failure
  labels describe what is actually inside each cluster.

Run:  python -m src.evaluation.benchmark
Output: results/metrics.json (+ per-seed predictions in results/predictions/).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
import torch
import xgboost

from src.config import (
    CYCLE_COL,
    ENGINE_COL,
    NEAR_CONSTANT_STD,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RUL_CLIP,
    SENSOR_COLS,
    SEQUENCE_LENGTH,
)
from src.data.loader import load_dataset
from src.data.preprocessor import CMAPSSPreprocessor
from src.evaluation.metrics import mae, nasa_score, r2, rmse
from src.explainability.degradation_fingerprint import DegradationFingerprinter
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.temporal_waterfall import TemporalSHAPWaterfall
from src.models.ensemble import EnsemblePredictor
from src.models.lstm_model import LSTMPredictor
from src.models.trainer import LSTMTrainer, split_by_engine
from src.models.xgb_model import XGBPredictor

DATASET = "FD001"
SEEDS = (0, 1, 2, 3, 4)
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_KEYS = ("lstm", "xgboost", "simple_average", "ensemble")
# Same setting as src.models.trainer.build_shap_artifacts (first 2000 test SHAP rows).
FINGERPRINT_SAMPLE_LIMIT = 2000


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def score(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def last_window_index(engine_ids: np.ndarray) -> np.ndarray:
    return np.asarray([np.flatnonzero(engine_ids == e)[-1] for e in np.unique(engine_ids)])


def load_arrays() -> dict:
    path = PROCESSED_DATA_DIR / f"{DATASET}_sequences.npz"
    if not path.exists():
        CMAPSSPreprocessor(dataset=DATASET).run()
    arrays = np.load(path, allow_pickle=True)
    data = {k: arrays[k] for k in arrays.files}
    for k in ("X_train", "y_train", "X_test", "y_test"):
        data[k] = data[k].astype(np.float32)
    data["feature_names"] = data["feature_names"].astype(str).tolist()
    return data


def verify_official_test_labels(data: dict) -> dict:
    """Check that the last-window labels equal min(RUL_FD001, 125) for every test engine."""
    official = np.loadtxt(RAW_DATA_DIR / f"RUL_{DATASET}.txt")
    last_idx = last_window_index(data["test_engine_ids"])
    engines = np.unique(data["test_engine_ids"])
    expected = np.minimum(official[engines - 1], RUL_CLIP)
    ok = bool(len(engines) == len(official) and np.allclose(data["y_test"][last_idx], expected))
    if not ok:
        raise RuntimeError("Last-window test labels do not match capped official RUL_FD001 values.")
    return {
        "test_engines_evaluated": int(len(engines)),
        "official_test_engines": int(len(official)),
        "engines_with_official_rul_above_cap": int((official > RUL_CLIP).sum()),
        "labels_match_capped_official_rul": ok,
    }


def run_seed(seed: int, data: dict, out_dir: Path) -> dict:
    set_all_seeds(seed)
    ckpt_dir = out_dir / "checkpoints" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    test_ids = data["test_engine_ids"]

    train_idx, val_idx = split_by_engine(data["train_engine_ids"], val_fraction=0.2, seed=seed)
    X_fit, y_fit = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    t0 = time.time()
    lstm_trainer = LSTMTrainer(
        LSTMPredictor(num_features=X_train.shape[2]),
        checkpoint_path=ckpt_dir / "lstm_best.pt",
        device="cpu",
    )
    history = lstm_trainer.fit(X_fit, y_fit, X_val, y_val, epochs=150)
    lstm_seconds = time.time() - t0

    t0 = time.time()
    X_fit_tab, X_val_tab, X_test_tab = X_fit[:, -1, :], X_val[:, -1, :], X_test[:, -1, :]
    xgb = XGBPredictor(model_path=ckpt_dir / "xgb_best.json", random_state=seed)
    search_idx = np.random.default_rng(seed).choice(len(X_fit_tab), size=min(3000, len(X_fit_tab)), replace=False)
    search = xgb.randomized_search(X_fit_tab[search_idx], y_fit[search_idx], n_iter=20, cv=3, random_state=seed, verbose=0)
    xgb.fit(X_fit_tab, y_fit, X_val_tab, y_val)
    xgb_seconds = time.time() - t0

    ensemble = EnsemblePredictor()
    weights = ensemble.learn_weights(lstm_trainer.predict(X_val), xgb.predict(X_val_tab), y_val)

    test_lstm = lstm_trainer.predict(X_test)
    test_xgb = xgb.predict(X_test_tab)
    preds_all = {
        "lstm": test_lstm,
        "xgboost": test_xgb,
        "simple_average": 0.5 * test_lstm + 0.5 * test_xgb,
        "ensemble": ensemble.predict_from_predictions(test_lstm, test_xgb),
    }

    last_idx = last_window_index(test_ids)
    official = {k: score(y_test[last_idx], v[last_idx]) for k, v in preds_all.items()}
    all_windows = {k: score(y_test, v) for k, v in preds_all.items()}

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"engine_id": test_ids[last_idx], "y_true_capped": y_test[last_idx], **{k: v[last_idx] for k, v in preds_all.items()}}
    ).to_csv(pred_dir / f"official_test_seed{seed}.csv", index=False)

    fingerprints = run_fingerprints(xgb, X_test_tab, data, seed)

    return {
        "seed": seed,
        "official_test": official,
        "all_test_windows": all_windows,
        "ensemble_weights": {"lstm": float(weights[0]), "xgboost": float(weights[1])},
        "lstm_epochs_run": len(history["val_rmse"]),
        "lstm_best_val_rmse": float(min(history["val_rmse"])),
        "xgb_best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "xgb_best_iteration": None if getattr(xgb.model, "best_iteration", None) is None else int(xgb.model.best_iteration),
        "train_seconds": {"lstm": lstm_seconds, "xgboost": xgb_seconds},
        "fingerprints": fingerprints,
    }


# ---------------------------------------------------------------------------
# Degradation fingerprints
# ---------------------------------------------------------------------------

def health_indicator_slopes() -> tuple[dict[int, float], dict]:
    """Per test engine: slope (per cycle) of a sensor-based health indicator.

    HI = mean of the informative sensors after min-max scaling on the training
    set, with each sensor's sign flipped so that a higher value means more
    degraded (sign = correlation with elapsed life fraction on training data).
    The slope is an OLS fit of HI vs. cycle over each test engine's observed cycles.
    This is independent of the model and of SHAP.
    """
    train, test, _ = load_dataset(DATASET)
    stds = train[SENSOR_COLS].std()
    sensors = [c for c in SENSOR_COLS if stds[c] >= NEAR_CONSTANT_STD]
    lo, hi = train[sensors].min(), train[sensors].max()
    life_frac = train[CYCLE_COL] / train.groupby(ENGINE_COL)[CYCLE_COL].transform("max")
    signs = np.sign([np.corrcoef(train[c], life_frac)[0, 1] for c in sensors])

    scaled = (test[sensors] - lo) / (hi - lo)
    hi_values = (scaled.to_numpy() * signs).mean(axis=1)
    frame = pd.DataFrame({ENGINE_COL: test[ENGINE_COL], CYCLE_COL: test[CYCLE_COL], "hi": hi_values})
    slopes = {
        int(e): float(np.polyfit(g[CYCLE_COL], g["hi"], 1)[0]) for e, g in frame.groupby(ENGINE_COL)
    }
    meta = {
        "definition": "OLS slope of health indicator vs cycle over each test engine's observed cycles; "
        "HI = mean of min-max scaled informative sensors, sign-oriented so higher = more degraded.",
        "units": "normalised HI units per cycle (x1000 in README)",
        "sensors_used": sensors,
        "sensor_signs": {c: int(s) for c, s in zip(sensors, signs)},
    }
    return slopes, meta


_SLOPES: tuple[dict[int, float], dict] | None = None


def run_fingerprints(xgb: XGBPredictor, X_test_tab: np.ndarray, data: dict, seed: int) -> dict:
    global _SLOPES
    if _SLOPES is None:
        _SLOPES = health_indicator_slopes()
    slopes, _ = _SLOPES

    feature_names = data["feature_names"]
    ids = data["test_engine_ids"]
    explainer = SHAPExplainer(xgb_model=xgb.model, feature_names=feature_names)
    shap_values = explainer.explain_batch(X_test_tab, model_type="xgboost")
    waterfall = TemporalSHAPWaterfall(feature_names)

    out = {}
    configs = {
        # Exactly what src.models.trainer does today: first 2000 test SHAP rows.
        "as_implemented_first_2000_rows": FINGERPRINT_SAMPLE_LIMIT,
        # Same method, all 100 test engines.
        "all_test_engines": len(X_test_tab),
    }
    for name, limit in configs.items():
        sv, eids = shap_values[:limit], ids[:limit]
        matrices = {}
        for e in np.unique(eids):
            rows = sv[eids == e]
            matrices[int(e)] = waterfall.build_matrix(rows, np.arange(1, len(rows) + 1))
        fp = DegradationFingerprinter(n_clusters=3, random_state=42)
        fp.fit(matrices)
        clusters = []
        for c in range(3):
            members = [e for e, lab in zip(fp.engine_ids_, fp.cluster_labels_) if int(lab) == c]
            s = np.asarray([slopes[e] for e in members])
            clusters.append(
                {
                    "cluster": c,
                    "label": DegradationFingerprinter.LABELS[c],
                    "engine_count": len(members),
                    "mean_degradation_slope": float(s.mean()) if len(s) else None,
                    "std_degradation_slope": float(s.std(ddof=1)) if len(s) > 1 else None,
                    "engines": members,
                }
            )
        ordered = [c for c in clusters if c["mean_degradation_slope"] is not None]
        fast = clusters[0]["mean_degradation_slope"]
        slow = clusters[1]["mean_degradation_slope"]
        out[name] = {
            "engines_clustered": len(fp.engine_ids_),
            "silhouette": float(fp.silhouette_),
            "clusters": clusters,
            "fast_burn_has_steepest_slope": bool(fast is not None and fast == max(c["mean_degradation_slope"] for c in ordered)),
            "fast_burn_steeper_than_slow_decay": bool(fast is not None and slow is not None and fast > slow),
        }
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def mean_std(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "values": arr.tolist()}


def aggregate(runs: list[dict]) -> dict:
    summary = {}
    for split in ("official_test", "all_test_windows"):
        summary[split] = {
            m: {metric: mean_std([r[split][m][metric] for r in runs]) for metric in ("rmse", "nasa_score", "mae", "r2")}
            for m in MODEL_KEYS
        }
    summary["ensemble_weight_lstm"] = mean_std([r["ensemble_weights"]["lstm"] for r in runs])
    summary["fingerprints"] = {}
    for cfg in runs[0]["fingerprints"]:
        summary["fingerprints"][cfg] = {
            "silhouette": mean_std([r["fingerprints"][cfg]["silhouette"] for r in runs]),
            "seeds_where_fast_burn_has_steepest_slope": int(sum(r["fingerprints"][cfg]["fast_burn_has_steepest_slope"] for r in runs)),
            "seeds_where_fast_burn_steeper_than_slow_decay": int(sum(r["fingerprints"][cfg]["fast_burn_steeper_than_slow_decay"] for r in runs)),
            "n_seeds": len(runs),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs_dir = args.out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    data = load_arrays()
    label_check = verify_official_test_labels(data)

    runs = []
    for seed in args.seeds:
        run_path = runs_dir / f"seed_{seed}.json"
        if run_path.exists():
            print(f"[seed {seed}] reusing {run_path}", flush=True)
            runs.append(json.loads(run_path.read_text()))
            continue
        print(f"[seed {seed}] training...", flush=True)
        result = run_seed(seed, data, args.out)
        run_path.write_text(json.dumps(result, indent=2))
        print(f"[seed {seed}] official test: " + json.dumps({k: round(v["rmse"], 3) for k, v in result["official_test"].items()}), flush=True)
        runs.append(result)

    global _SLOPES
    if _SLOPES is None:
        _SLOPES = health_indicator_slopes()

    metrics = {
        "dataset": DATASET,
        "protocol": {
            "subset": "FD001 (100 train engines, 100 test engines, 1 operating condition, 1 fault mode)",
            "rul_cap": RUL_CLIP,
            "sequence_length": SEQUENCE_LENGTH,
            "test_evaluation": "official test set; one prediction per engine from its last window; "
            "target = min(RUL_FD001, 125)",
            "nasa_score": "sum(exp(-d/13)-1 | d<0) + sum(exp(d/10)-1 | d>=0), d = pred - true, summed over 100 engines",
            "seeds": list(args.seeds),
            "seed_controls": "engine-level train/val split, LSTM init + batch order, XGBoost random_state + hyper-parameter search",
            "std": "sample standard deviation (ddof=1) across seeds",
            "validation": "20% of training engines held out for LSTM checkpoint selection, XGBoost early stopping and ensemble weights",
            "all_test_windows_note": "secondary: every 30-cycle test window (10,196 windows); this is how the previous README numbers were computed",
        },
        "label_check": label_check,
        "degradation_slope": _SLOPES[1],
        "summary": aggregate(runs),
        "runs": runs,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "device": "cpu",
            "machine": platform.machine(),
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"]["official_test"], indent=2, default=lambda x: None)[:4000])


if __name__ == "__main__":
    main()
