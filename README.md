# TurbXplain: Explainable RUL Prediction on NASA C-MAPSS

[Live demo](https://turbxplain-aarohi.streamlit.app/)

> Don't just predict failure. **Explain why.**

TurbXplain predicts the Remaining Useful Life (RUL) of turbofan engines with an LSTM + XGBoost ensemble, explains predictions with SHAP, and serves everything in a Streamlit dashboard. All reported results are on **C-MAPSS FD001**.

---

## Problem

Engines degrade gradually, and maintenance is cheapest when it is scheduled shortly before failure. Given the sensor history of an engine up to now, the task is to predict how many operating cycles it has left (RUL), and to show which sensors drive that prediction.

The data is NASA's C-MAPSS turbofan simulation. This project uses the **FD001** subset: one operating condition, one fault mode (HPC degradation), 100 run-to-failure training engines and 100 test engines whose histories stop at some point before failure. Each cycle has 3 operational settings and 21 sensor readings.

---

## Approach

1. **Preprocessing** ([preprocessor.py](src/data/preprocessor.py))
   - Drop the 6 near-constant sensors and keep 15.
   - Add rolling mean/std (windows 5, 10, 20), first differences and a 10-cycle trend slope per sensor, for 137 features in total.
   - Apply min-max scaling fitted on the training set.
   - Label RUL as piecewise-linear, capped at **125** cycles.
   - Cut sliding windows of **30 cycles**.
2. **LSTM** ([lstm_model.py](src/models/lstm_model.py)): BatchNorm, then a 2-layer LSTM (hidden size 128), then an MLP head. Trained with Huber loss, AdamW and cosine warm restarts; early stopping on validation RMSE.
3. **XGBoost** ([xgb_model.py](src/models/xgb_model.py)): uses the last row of each window as tabular features. Hyper-parameters are chosen by a 20-candidate randomized search, followed by early stopping.
4. **Ensemble** ([ensemble.py](src/models/ensemble.py)): a convex blend of the two predictions. The weights are learned on held-out validation engines (20% of training engines, split by engine) and never on the test set.
5. **Explainability** ([shap_explainer.py](src/explainability/shap_explainer.py), [temporal_waterfall.py](src/explainability/temporal_waterfall.py)): TreeSHAP on the XGBoost model, plotted per engine over time.
6. **Degradation fingerprinting** ([degradation_fingerprint.py](src/explainability/degradation_fingerprint.py)): K-Means (k=3) on summary statistics of each engine's SHAP trajectory. This part is exploratory; see the results below.
7. **Evaluation** ([benchmark.py](src/evaluation/benchmark.py))
   - Uses the official FD001 test set, with one prediction per engine from its last window, scored against min(`RUL_FD001`, 125).
   - Reports RMSE and the NASA/PHM08 score.
   - Runs 5 seeds, each controlling the train/validation split, LSTM initialisation and batch order, and XGBoost's seed and search.

```
┌─────────────────────────────────────────────────────────┐
│                    NASA C-MAPSS Data                    │
│   FD001 · 21 sensors × N cycles · 100 train/100 test    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Preprocessing Pipeline                     │
│  • Drop constant sensors  • Min-Max normalization       │
│  • Rolling mean/std/diff/trend features                 │
│  • RUL labeling (clipped at 125)                        │
│  • Sequence windowing (30 cycles)                       │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌────────────────────┐      ┌────────────────────────┐
│   LSTM (PyTorch)   │      │   XGBoost (Tabular)    │
│   2-layer, h=128   │      │   Tuned via RandomCV   │
│   + BatchNorm      │      │   Last-row features    │
│   + HuberLoss      │      │                        │
│   (see results)    │      │   (see results)        │
└────────┬───────────┘      └───────────┬────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────┐
│           Learned Weighted Ensemble                     │
│     Weights learned on validation engines (SLSQP)       │
│     FD001 test metrics: see Results table below         │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ SHAP         │ │ Temporal     │ │ Degradation      │
│ TreeExplainer│ │ Waterfall    │ │ Fingerprinting   │
│ (XGBoost)    │ │ (per engine) │ │ (K-Means on SHAP)│
└──────────────┘ └──────────────┘ └──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit Dashboard                        │
│  RUL Predictor | Explainability | Fingerprints          │
│  What-If Simulator | Model Performance | Reports        │
└─────────────────────────────────────────────────────────┘
```

---

<!-- RESULTS:START (generated by `python -m src.evaluation.update_readme` from results/metrics.json; do not edit by hand) -->
## Results on NASA C-MAPSS FD001

All numbers below are for **FD001 only**: the official NASA test set, with RUL capped at 125. Each model makes **one prediction per test engine** from that engine's last 30-cycle window (100 engines). The target is min(`RUL_FD001`, 125). Values are mean ± std (sample std) over 5 seeds (0, 1, 2, 3, 4). The NASA score is the PHM08 asymmetric scoring function summed over the 100 test engines; lower is better.

### Ablation (official test set, 5 seeds)

| Model | RMSE | NASA Score |
|---|---:|---:|
| LSTM only | 14.58 ± 0.65 | 476.6 ± 104.8 |
| XGBoost only | 17.89 ± 0.52 | 978.2 ± 205.2 |
| Simple average (0.5 / 0.5) | 14.81 ± 0.56 | 470.7 ± 62.4 |
| **Learned weighted ensemble** | **14.33 ± 0.63** | **451.3 ± 90.6** |

The ensemble weights are learned on a validation split of the training engines. Mean learned LSTM weight: 0.87 ± 0.03.

<details><summary>Per-seed results and secondary metrics</summary>

| Seed | LSTM RMSE | XGBoost RMSE | Ensemble RMSE | LSTM NASA | XGBoost NASA | Ensemble NASA | LSTM weight |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.19 | 17.17 | 14.02 | 572.4 | 661.7 | 518.8 | 0.87 |
| 1 | 15.70 | 18.51 | 15.40 | 563.5 | 1055.5 | 537.3 | 0.83 |
| 2 | 14.20 | 17.61 | 13.99 | 519.2 | 891.1 | 493.3 | 0.86 |
| 3 | 14.61 | 18.22 | 14.39 | 362.5 | 1114.6 | 352.7 | 0.92 |
| 4 | 14.22 | 17.94 | 13.87 | 365.3 | 1168.1 | 354.3 | 0.85 |

MAE and R² on the official test set:

| Model | MAE | R² |
|---|---:|---:|
| LSTM only | 10.40 ± 0.62 | 0.867 ± 0.012 |
| XGBoost only | 12.67 ± 0.42 | 0.801 ± 0.012 |
| Simple average (0.5 / 0.5) | 10.69 ± 0.41 | 0.863 ± 0.011 |
| Learned weighted ensemble | 10.25 ± 0.57 | 0.872 ± 0.011 |

For reference only (not comparable to the literature): the same models scored on every 30-cycle test window (10,196 windows). Earlier versions of this README reported this protocol. The NASA score is a sum, so it grows with the number of windows.

| Model | RMSE (all windows) | NASA Score (all windows) |
|---|---:|---:|
| LSTM only | 14.77 ± 0.64 | 38864 ± 2358 |
| XGBoost only | 18.04 ± 0.16 | 98616 ± 13427 |
| Simple average (0.5 / 0.5) | 15.08 ± 0.41 | 41239 ± 1857 |
| Learned weighted ensemble | 14.56 ± 0.59 | 36955 ± 1997 |

</details>

### Degradation Fingerprinting (Exploratory)

K-Means (k=3) on summary statistics of each engine's XGBoost SHAP trajectory. To test whether the cluster names mean anything, each cluster is compared with a **model-independent degradation slope**: the per-cycle OLS slope of a sensor health indicator (the mean of min-max-scaled informative sensors, oriented so that higher means more degraded). Slopes are shown ×1000. If the names held up, *fast-burn* would have the steepest slope and *slow-decay* a shallow one.

| Engines clustered | Silhouette (mean ± std) | Seeds where fast-burn has steepest slope | Seeds where fast-burn > slow-decay |
|---|---:|---:|---:|
| As implemented in the trainer (first 2,000 test SHAP rows): 23 | 0.203 ± 0.084 | 0/5 | 0/5 |
| All 100 test engines: 100 | 0.084 ± 0.017 | 4/5 | 4/5 |

Per-cluster breakdown, all 100 test engines (count / mean slope ×1000 ± std):

| Seed | Silhouette | fast-burn | slow-decay | sudden-failure |
|---:|---:|---|---|---|
| 0 | 0.093 | 52 / 0.77 ± 0.42 | 47 / 0.52 ± 0.30 | 1 / 0.73 |
| 1 | 0.059 | 19 / 0.60 ± 0.36 | 42 / 0.78 ± 0.43 | 39 / 0.53 ± 0.30 |
| 2 | 0.076 | 49 / 0.72 ± 0.43 | 50 / 0.59 ± 0.33 | 1 / 0.39 |
| 3 | 0.097 | 56 / 0.76 ± 0.43 | 43 / 0.50 ± 0.27 | 1 / 0.73 |
| 4 | 0.097 | 56 / 0.76 ± 0.43 | 43 / 0.50 ± 0.27 | 1 / 0.73 |

Health-indicator sensors: s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.

**Do the archetype labels hold up?** No, not in their current form:

- The silhouette score is below 0.25 in 8 of 10 clusterings and never above 0.280. Below 0.25 is conventionally read as no substantial structure; 0.25 to 0.5 as weak structure.
- 9 of 10 clusterings contain a single-engine cluster, usually *sudden-failure*.
- In the configuration the trainer uses, *fast-burn* has the steepest degradation slope in 0/5 seeds. On all 100 engines the ordering is mostly as named, but the within-cluster spread of slopes is about as large as the gap between cluster means.

The cluster names are cluster indices with descriptive names attached. They are not validated failure types.
<!-- RESULTS:END -->

### Inference Latency (deployed model)

Median of 30 single-window calls, from `reports/latency.json`. This is a one-off measurement on a laptop CPU and was not re-measured in the 5-seed benchmark.

| Component | Time (ms) |
|---|---:|
| LSTM inference | 1.40 |
| XGBoost inference | 0.06 |
| SHAP explanation (XGBoost) | 0.84 |
| **Total** | **2.30** |

---

## How to Run

Requires Python 3.12 or 3.13. The pinned versions in `requirements.txt` were installed and tested on both.

```bash
git clone https://github.com/aarohi1822/TurbXplain.git
cd TurbXplain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Download [NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and put `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` directly in `data/raw/`.

```bash
python -m src.data.loader --dataset FD001     # preprocess -> data/processed/FD001_sequences.npz
python -m src.models.trainer                  # train the deployed model -> models/, reports/
python -m src.evaluation.benchmark            # 5-seed benchmark (CPU, ~1 h on an M2) -> results/metrics.json
python -m src.evaluation.update_readme        # regenerate the results section above from results/metrics.json
streamlit run app/streamlit_app.py            # dashboard
pytest -q                                     # tests
```

The benchmark caches finished seeds in `results/runs/`. Delete that folder to retrain from scratch.

---

## Limitations

- **FD001 only.** The model has not been trained or evaluated on FD002–FD004, which add multiple operating conditions and fault modes. The results say nothing about those settings.
- **Small ensemble gain.** The learned ensemble beats the LSTM in every seed, but by about 0.25 RMSE. That is smaller than the seed-to-seed spread (about ±0.6).
- **SHAP explains XGBoost, not the ensemble.** The explanations and fingerprints come from the XGBoost component, which carries only about 13% of the ensemble weight. The LSTM, which dominates the prediction, is not explained. A GradientExplainer path exists in code but is unused.
- **Fingerprint labels are not validated.** The silhouette scores show little cluster structure, and the fast-burn / slow-decay / sudden-failure names do not consistently match a sensor-based degradation slope (see above). Treat the clusters as indices.
- **The deployed model is not one of the 5 benchmark seeds.** The dashboard's predictor, explainability and what-if pages use the single model from `src.models.trainer` (seed 42). Its metrics are not what the results table reports.
- **API is out of date.** `src/api/serve.py` (and the Dockerfile that runs it) expects the older demo bundle format. With the current trained bundle, `/health` works but `/predict` and `/explain` return HTTP 500.
- **Bitwise reproducibility not verified.** Seeds are fixed and deterministic algorithms are enabled on CPU, but no seed has been re-run to confirm identical numbers. Results on other hardware or library versions may differ slightly.
- **Test-set truncation.** Test engines are scored at a single point (the end of their history). This matches the literature but gives no information about accuracy earlier in an engine's life.

---

## Project Structure

```
app/streamlit_app.py           Dashboard (6 pages)
src/data/                      C-MAPSS loader, preprocessing
src/models/                    LSTM, XGBoost, ensemble, trainer (deployed model)
src/explainability/            SHAP, temporal waterfall, fingerprinting
src/evaluation/                Metrics, 5-seed benchmark, README generator
src/api/serve.py               FastAPI service (see Limitations)
results/                       Benchmark output: metrics.json, per-seed runs and predictions
reports/                       Artifacts of the deployed model (SHAP, predictions, latency)
models/turbxplain_bundle.joblib  Deployed model bundle
tests/                         pytest
```

---

## Author

**Aarohi Gaurav Sharma**
B.Tech CSE, COER University, Roorkee

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/aarohig-sharma22)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/aarohi1822)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:aarohisharma2922@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-7c6ff7?style=flat&logo=googlechrome&logoColor=white)](https://aarohi1822.github.io/aarohi-portfolio/)

---

## License

MIT
