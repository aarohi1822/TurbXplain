"""Build the README figures from saved files only, and fail if any number disagrees.

assets/ablation.png       RMSE mean ± std (5 seeds, official FD001 test protocol)
                          from results/metrics.json.
assets/shap_waterfall.png Temporal SHAP (XGBoost component) for one test engine
                          from reports/shap_contributions.csv.

Run:  python -m src.evaluation.make_figures
"""

from __future__ import annotations

import json
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, REPORT_DIR, RUL_CLIP, SEQUENCE_LENGTH  # noqa: E402

ASSETS = PROJECT_ROOT / "assets"
METRICS = PROJECT_ROOT / "results" / "metrics.json"
README = PROJECT_ROOT / "README.md"

MODELS = [  # key in metrics.json, label in figure, label used in the README table, colour
    ("lstm", "LSTM only", "LSTM only", "#42a5f5"),
    ("xgboost", "XGBoost only", "XGBoost only", "#ab47bc"),
    ("simple_average", "Simple average", "Simple average (0.5 / 0.5)", "#78909c"),
    ("ensemble", "Learned ensemble", "Learned weighted ensemble", "#ef5350"),
]
TOP_N = 8  # same default as the dashboard's Explainability panel


class NumbersDiffer(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise NumbersDiffer(message)


# ---------------------------------------------------------------------------
# 1. Ablation
# ---------------------------------------------------------------------------

def ablation_figure() -> dict:
    m = json.loads(METRICS.read_text())
    check(m["dataset"] == "FD001", f"metrics.json dataset is {m['dataset']}, expected FD001")
    off = m["summary"]["official_test"]
    seeds = m["protocol"]["seeds"]
    runs = {r["seed"]: r for r in m["runs"]}
    check(sorted(runs) == sorted(seeds), "per-seed runs do not match protocol.seeds")

    rows = []
    for key, label, readme_label, colour in MODELS:
        mean, std = off[key]["rmse"]["mean"], off[key]["rmse"]["std"]
        per_seed = np.array([runs[s]["official_test"][key]["rmse"] for s in seeds])
        check(np.allclose(per_seed, off[key]["rmse"]["values"], rtol=0, atol=1e-12),
              f"{key}: summary values differ from per-seed runs")
        check(np.isclose(per_seed.mean(), mean, rtol=0, atol=1e-9), f"{key}: mean differs ({per_seed.mean()} vs {mean})")
        check(np.isclose(per_seed.std(ddof=1), std, rtol=0, atol=1e-9), f"{key}: std differs ({per_seed.std(ddof=1)} vs {std})")
        rows.append((key, label, readme_label, colour, mean, std, per_seed))

    # The README headline table must show exactly these values.
    readme = README.read_text()
    for _, _, readme_label, _, mean, std, _ in rows:
        cell = f"{mean:.2f} ± {std:.2f}"
        pattern = rf"\|\s*\**{re.escape(readme_label)}\**\s*\|\s*\**{re.escape(cell)}\**\s*\|"
        check(re.search(pattern, readme) is not None, f"README table does not show '{readme_label} | {cell}'")

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    x = np.arange(len(rows))
    for i, (_, label, _, colour, mean, std, per_seed) in enumerate(rows):
        ax.bar(i, mean, color=colour, width=0.62, zorder=2)
        ax.errorbar(i, mean, yerr=std, color="#263238", capsize=6, linewidth=1.4, zorder=3)
        jitter = np.linspace(-0.12, 0.12, len(per_seed))
        ax.scatter(i + jitter, per_seed, s=14, color="#263238", alpha=0.55, zorder=4,
                   label="Individual seeds" if i == 0 else None)
        top = max(mean + std, per_seed.max())  # keep the label clear of error bar and seed dots
        ax.text(i, top + 0.3, f"{mean:.2f} ± {std:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x, [r[1] for r in rows], fontsize=10)
    ax.set_ylabel("RMSE (cycles, lower is better)", fontsize=10)
    ax.set_ylim(0, max(max(r[4] + r[5], r[6].max()) for r in rows) + 2.5)
    ax.grid(axis="y", color="#e0e0e0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    n_eng = m["label_check"]["test_engines_evaluated"]
    fig.suptitle("RUL prediction on NASA C-MAPSS FD001: ablation", fontsize=13, fontweight="bold", x=0.02, ha="left")
    ax.set_title(f"Official test set ({n_eng} engines, last window each, RUL capped at {m['protocol']['rul_cap']}); "
                 f"mean ± std over {len(seeds)} seeds", fontsize=9.5, loc="left", color="#455a64")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / "ablation.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return {label: (round(mean, 4), round(std, 4)) for _, label, _, _, mean, std, _ in rows}


# ---------------------------------------------------------------------------
# 2. Temporal SHAP for one test engine
# ---------------------------------------------------------------------------

def shap_figure() -> dict:
    shap_df = pd.read_csv(REPORT_DIR / "shap_contributions.csv")
    preds = pd.read_csv(REPORT_DIR / "predictions.csv")
    n = len(shap_df)
    check((shap_df["engine_id"].to_numpy() == preds["engine_id"].to_numpy()[:n]).all(),
          "shap_contributions.csv rows are not aligned with predictions.csv")

    # Engine choice rule: longest test trajectory among engines whose SHAP values are complete.
    total = preds.groupby("engine_id").size()
    have = shap_df.groupby("engine_id").size()
    complete = [e for e in have.index if have[e] == total[e]]
    engine = max(complete, key=lambda e: (total[e], -e))

    rows = shap_df[shap_df["engine_id"] == engine].drop(columns=["engine_id"]).reset_index(drop=True)
    p = preds.iloc[:n][shap_df["engine_id"].to_numpy() == engine].reset_index(drop=True)
    cycles = np.arange(1, len(rows) + 1) + SEQUENCE_LENGTH - 1
    check(len(rows) == total[engine], "selected engine does not have SHAP values for every window")

    # Check 1: the window -> engine-cycle mapping reproduces the saved true RUL.
    test_path, rul_path = RAW_DATA_DIR / f"test_FD001.txt", RAW_DATA_DIR / "RUL_FD001.txt"
    if test_path.exists() and rul_path.exists():
        test = pd.read_csv(test_path, sep=r"\s+", header=None)
        last_cycle = int(test.loc[test[0] == engine, 1].max())
        final_rul = float(np.loadtxt(rul_path)[engine - 1])
        expected = np.minimum(last_cycle - cycles + final_rul, RUL_CLIP)
        check(np.allclose(p["y_true"].to_numpy(), expected), "engine-cycle mapping does not reproduce saved y_true")
        check(cycles[-1] == last_cycle, f"last window should end at cycle {last_cycle}, got {cycles[-1]}")
    else:
        raise NumbersDiffer("raw FD001 files not found; cannot verify the engine-cycle mapping")

    # Check 2: SHAP additivity against the saved XGBoost predictions of the deployed bundle.
    import shap

    bundle = joblib.load(MODEL_DIR / "turbxplain_bundle.joblib")
    explainer = shap.TreeExplainer(bundle["xgb_model"])
    # expected_value is only exact after shap_values() has run once (before that SHAP reports an estimate).
    first = np.load(PROCESSED_DATA_DIR / "FD001_sequences.npz", allow_pickle=True)["X_test"][:1, -1, :]
    explainer.shap_values(first.astype(np.float32))
    base = float(np.ravel(explainer.expected_value)[0])
    recon = rows.to_numpy().sum(axis=1) + base
    max_err = float(np.abs(recon - p["xgboost_prediction"].to_numpy()).max())
    check(max_err < 1e-3, f"SHAP values + base ({base:.3f}) differ from saved XGBoost predictions by {max_err:.4f}")

    top = rows.abs().mean().sort_values(ascending=False).head(TOP_N).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    colours = plt.get_cmap("tab10").colors
    for i, feat in enumerate(top):
        ax.plot(cycles, rows[feat], label=feat, color=colours[i % 10], linewidth=1.5)
    ax.axhline(0, color="#90a4ae", linewidth=0.8)
    ax.set_xlabel("Engine cycle", fontsize=10)
    ax.set_ylabel("SHAP value (contribution to predicted RUL, cycles)", fontsize=10)
    ax.grid(color="#eeeeee")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=f"Top {TOP_N} features by mean |SHAP|", fontsize=8, title_fontsize=8.5,
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"Temporal SHAP: FD001 test engine {engine}", fontsize=13, fontweight="bold", x=0.02, ha="left")
    ax.set_title(f"XGBoost component only · deployed model · cycles {cycles[0]}–{cycles[-1]}", fontsize=9.5,
                 loc="left", color="#455a64")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / "shap_waterfall.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"engine": int(engine), "windows": int(len(rows)), "cycles": f"{cycles[0]}-{cycles[-1]}",
            "base_value": round(base, 4), "max_additivity_error": max_err, "top_features": top}


def main() -> None:
    print("ablation.png:", ablation_figure())
    print("shap_waterfall.png:", shap_figure())


if __name__ == "__main__":
    main()
