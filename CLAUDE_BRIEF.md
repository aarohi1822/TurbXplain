# TurbXplain: Verified Project Facts (brief for Claude)

## What this is

This is a verified fact sheet about my ML project **TurbXplain**:
- **Repo:** https://github.com/aarohi1822/TurbXplain
- **Live demo:** https://turbxplain-aarohi.streamlit.app/

It was produced by reading the whole repository (code, saved results, README, git history) at commit `ef22a28` (2026-11-13). Every number was copied from a saved output file, not estimated.

## What was asked when this fact sheet was made

> Read this entire repo (code, notebooks, results files, README) and write PROJECT_FACTS.md. Rules: only state what the code or saved outputs actually show. For every number, cite the file and line/cell it came from. If something is missing or unclear, write "NOT FOUND" — never guess or estimate.
>
> Sections:
> 1. One-line summary: what problem it solves, for whom.
> 2. Data: dataset name, size (rows/images/channels), train/val/test split method, any preprocessing.
> 3. Method: models used, key architecture choices, important hyperparameters.
> 4. Results: every final metric with its eval setup (which split, how averaged, number of seeds). Include baselines and ablations side by side.
> 5. Engineering: deployment (API/dashboard/demo link), tests (count, passing?), latency or throughput if measured, notable design decisions.
> 6. Surprising findings: anything where results contradicted expectations (e.g. a baseline beating the main model).
> 7. Limitations: what the evaluation does NOT prove.
> 8. Tech stack: only libraries actually imported and used.
> 9. Timeline: first and last commit dates.
> 10. Visuals: list existing plots/screenshots with paths, and suggest the 2 best for a LinkedIn post. Also include which C-MAPSS subset was used, the per-model ablation with mean ± std, and the cluster silhouette score.

## How to use these facts

- **Use only the facts below.** Don't add, round differently, or estimate any number. If you need something that isn't here, or is marked **NOT FOUND**, say so instead of filling it in.
- **Keep the caveats.** In particular:
  - the ensemble beats LSTM alone by only about 0.25 RMSE, which is less than the seed-to-seed spread
  - SHAP explains only the XGBoost part (about 13% of the ensemble weight)
  - the fingerprint cluster names (fast-burn / slow-decay / sudden-failure) are **not** validated
  - results are FD001 only
- **Figures.** The repo has two verified figures, `assets/ablation.png` and `assets/shap_waterfall.png` (§10). If they are attached to this conversation, they are those files; otherwise you cannot see them.
- **Citations** like `results/metrics.json:L207` point to files and lines in the repo. You can't see those files; treat the citations as provenance, and don't put them in anything public.
- **"Computed"** marks simple arithmetic on cited values.

## My request

<!-- Replace this line with what you want, for example: "Draft a LinkedIn post (150–250 words) about this project, using the two suggested visuals." -->

---

## 1. One-line summary

TurbXplain predicts the Remaining Useful Life (RUL, in operating cycles) of simulated turbofan engines from their sensor history, using an LSTM + XGBoost ensemble with SHAP explanations of the XGBoost component, so that maintenance can be scheduled shortly before failure (`README.md:L7`, `README.md:L32`).

- **For whom:** NOT FOUND in the current files. An earlier README said "a full Streamlit dashboard for operators and engineers" (`git show 1acb669:README.md`, line 6). That sentence was removed in commit `75c5a76`.

---

## 2. Data

### Dataset

- **Name:** NASA C-MAPSS turbofan engine degradation simulation, subset **FD001** (`src/config.py:L10`; `data/raw/CMaps/readme.txt:L1`).
- **FD001 per the NASA readme:**
  - 100 train trajectories and 100 test trajectories (`data/raw/CMaps/readme.txt:L2-L3`)
  - one operating condition, sea level (`:L4`)
  - one fault mode, HPC degradation (`:L5`)
- **Subsets used:** only FD001.
  - Every pipeline defaults to `"FD001"`: `src/config.py:L10`, `src/models/trainer.py:L118`, `src/evaluation/benchmark.py:L62`.
  - FD002–FD004 files are present in `data/raw/CMaps/` but no code reads them.
- **Raw files used:** `data/raw/train_FD001.txt`, `data/raw/test_FD001.txt`, `data/raw/RUL_FD001.txt` (`src/data/loader.py:L29-L31`).
  - These are byte-identical to the copies in `data/raw/CMaps/`, checked with `cmp`.
  - The raw data is not committed (`.gitignore:L8`, `.gitignore:L16`).

### Size

| Item | Value | Source |
|---|---:|---|
| Train rows (engine-cycles) | 20,631 | `data/raw/train_FD001.txt` (line count, `wc -l`) |
| Test rows | 13,096 | `data/raw/test_FD001.txt` (`wc -l`) |
| RUL labels (one per test engine) | 100 | `data/raw/RUL_FD001.txt` (`wc -l`) |
| Columns per row | 26 = engine id, cycle, 3 operational settings, 21 sensors | `src/config.py:L16-L20` |
| Train engines / cycles per engine | 100 / 128–362 | computed from `data/raw/train_FD001.txt` |
| Test engines / cycles per engine | 100 / 31–303 (median 133.5) | computed from `data/raw/test_FD001.txt` |
| Official test RUL range | 7–145; 11 engines above 125 | computed from `data/raw/RUL_FD001.txt`; the 11 also in `results/metrics.json:L24` |
| Model input features after preprocessing | 137 | `data/processed/FD001_data_stats.json:L4` |
| Train windows (30 cycles) | 17,731 | `data/processed/FD001_data_stats.json:L30` |
| Test windows (30 cycles) | 10,196 | `data/processed/FD001_data_stats.json:L31` |

### Preprocessing (`src/data/preprocessor.py`)

1. **Drop near-constant sensors:** sensors with training std below 0.001 are dropped (`:L42-L45`; threshold `src/config.py:L14`).
   - Dropped: s1, s5, s10, s16, s18, s19 (`data/processed/FD001_data_stats.json:L22-L29`).
   - Kept: 15 sensors (`:L5-L21`).
2. **RUL labels:** piecewise-linear, capped at **125**.
   - Train label: max cycle minus current cycle (`:L51-L52`).
   - Test label: adds the official final RUL (`:L54-L56`).
   - Cap value: `src/config.py:L11`.
3. **Feature engineering:** for each kept sensor (`:L63-L70`):
   - first difference
   - rolling mean and std over windows 5, 10 and 20 (`src/config.py:L13`)
   - 10-cycle OLS trend slope (`:L148-L159`)
4. **Constant-column filter:** drops engineered columns with std ≤ 1e-6 (`:L80`). Operational setting `op3` is absent from the saved feature columns (header of `reports/shap_contributions.csv` starts `engine_id, op1, op2, s2, …`).
5. **Scaling:** min-max scaling fitted on train only, then applied to test (`:L82-L84`).
6. **Windowing:** sliding windows of **30 cycles** (`src/config.py:L12`).
   - Each window's target is the RUL at its last cycle (`:L97-L100`).
   - Engines shorter than 30 cycles are skipped (`:L95-L96`). All 100 test engines are kept (`results/metrics.json:L22`).

### Train / validation / test split

- **Train/validation:** split by engine, not by window (`src/models/trainer.py:L217-L224`).
  - 20% of training engines are held out (`val_fraction=0.2`): `src/models/trainer.py:L138`, `src/evaluation/benchmark.py:L128`.
  - With 100 training engines that gives 80 train and 20 validation engines (computed from `src/models/trainer.py:L221`).
- **Seeds:** the deployed model uses split seed 42 (`src/models/trainer.py:L217`, default). Benchmark runs use seed = run seed (`src/evaluation/benchmark.py:L128`).
- **Uses of the validation set:** LSTM checkpoint selection and early stopping, XGBoost early stopping, and ensemble weight fitting (`results/metrics.json:L18`).
- **Test set:** the official NASA test set.
  - **Primary evaluation:** one prediction per engine from its last window (100 points), against min(`RUL_FD001`, 125) (`src/evaluation/benchmark.py:L86-L87`, `:L161-L162`; `results/metrics.json:L7`).
  - **Label check:** the last-window labels equal the capped official RUL for all 100 engines (`results/metrics.json:L22-L25`; check code `src/evaluation/benchmark.py:L102-L116`).

---

## 3. Method

### LSTM (`src/models/lstm_model.py`)

- **Architecture:**
  - BatchNorm1d over features (`:L8`)
  - 2-layer LSTM with hidden size 128 and inter-layer dropout 0.3 (`:L6`, `:L9-L15`)
  - MLP head 128 → 64 → 32 → 1 with ReLU and dropout 0.3 / 0.2 (`:L16-L24`)
  - The prediction uses the last layer's final hidden state (`:L31-L33`)
- **Training** (`src/models/trainer.py`):
  - batch size 128, learning rate 5e-4, AdamW weight decay 1e-4, early-stopping patience 15 (`:L29-L32`, `:L42`)
  - Huber loss with delta 10.0 (`:L41`)
  - Cosine annealing with warm restarts, T_0=10, T_mult=2 (`:L43`)
  - Gradient clipping at max-norm 1.0 (`:L103`)
  - At most 150 epochs (`:L148`; `src/evaluation/benchmark.py:L138`); the best validation-RMSE checkpoint is restored (`:L71-L82`)
- **Device:** CUDA if available, else CPU (`src/models/trainer.py:L39`). Benchmark runs force CPU (`src/evaluation/benchmark.py:L136`; `results/metrics.json:L1827`).
- **LSTM tuning:** NOT FOUND. The LSTM hyperparameters are fixed values in code, and the repo has no tuning code for them.

### XGBoost (`src/models/xgb_model.py`)

- **Input:** the last row of each window, i.e. 137 tabular features (`src/models/trainer.py:L151-L153`; `src/evaluation/benchmark.py:L142`).
- **Base parameters:** `reg:squarederror`, colsample_bytree 0.8, reg_alpha 0.1, reg_lambda 1.0, n_jobs 1 (`:L12-L24`).
- **Hyperparameter search:** RandomizedSearchCV over the grid at `:L57-L62`
  - 20 candidates × 3-fold CV, scored by negative RMSE (`:L63-L72`)
  - run on a 3,000-window random subset of the training split (`src/models/trainer.py:L157-L158`; `src/evaluation/benchmark.py:L144-L145`)
  - grid: n_estimators {250, 500, 800}, max_depth {3, 4, 6, 8}, learning_rate {0.01, 0.03, 0.05, 0.1}, subsample {0.7, 0.8, 1.0}
- **Final fit:** on the full training split with early stopping after 50 rounds on validation (`:L29-L40`).
- **Deployed model's best parameters:** subsample 0.8, n_estimators 800, max_depth 6, learning_rate 0.05 (`models/xgb_best_params.json:L2-L5`). This file is local only (gitignored by `.gitignore:L11`).

### Ensemble (`src/models/ensemble.py`)

- A convex combination w_lstm·LSTM + w_xgb·XGB (`:L31-L32`).
- Weights are fitted with `scipy.optimize.minimize` to minimise validation RMSE, with bounds [0, 1] and the constraint w_lstm + w_xgb = 1 (`:L16-L29`). The solver method is not named in code.
- Deployed weights: LSTM 0.879, XGBoost 0.121 (`models/ensemble_weights.json:L2-L3`; `reports/ablation_results.json:L43-L44`).
- **Ablation baseline:** "simple average" fixes both weights at 0.5 (`src/evaluation/benchmark.py:L157`; `src/models/ensemble.py:L40`).

### Explainability

- **SHAP:** `shap.TreeExplainer` on the **XGBoost model only** (`src/explainability/shap_explainer.py:L14`; `src/inference.py:L33-L38`).
  - A GradientExplainer for the LSTM exists (`src/explainability/shap_explainer.py:L16-L18`), but no pipeline passes it a background set, so it is never built.
- **Temporal SHAP waterfall:** per-window SHAP values of an engine, plotted over its windows (`src/explainability/temporal_waterfall.py:L10-L27`).
  - The deployed model saves SHAP for the first 2,000 test windows only (`src/models/trainer.py:L228`), which covers test engines 1–23 (computed from `reports/shap_contributions.csv`: 2,000 rows, engine ids 1–23).

### Degradation fingerprinting (`src/explainability/degradation_fingerprint.py`)

- **Features:** for each feature's SHAP trajectory, 5 statistics are computed: max |SHAP|, mean |SHAP|, onset index / length, OLS slope and mean second difference (`:L24-L35`).
- **Clustering:** standardise, then K-Means with k=3 and random_state 42; quality is measured with the silhouette score (`:L15-L19`, `:L37-L44`).
- **Cluster names:** "fast-burn", "slow-decay" and "sudden-failure" are attached to cluster indices 0, 1 and 2 (`:L13`; `src/models/trainer.py:L252`). The dashboard does not show these names; it labels the clusters Cluster A / B / C (`app/streamlit_app.py:L156`). The names are assigned by index, not derived from the data.
- **Validation check** (benchmark only): a model-independent degradation slope per test engine (`src/evaluation/benchmark.py:L191-L220`).
  - Health indicator = mean of the 15 min-max-scaled kept sensors, sign-oriented by their correlation with life fraction (`results/metrics.json:L28-L61`).
  - The slope is an OLS fit of the health indicator against cycle.

### Legacy pipeline (not used for any reported result)

- `src/pipeline/run_experiment.py` is an older demo pipeline. It uses sklearn stand-ins: `HistGradientBoostingRegressor` as the "LSTM proxy" (`:L49`), `ExtraTreesRegressor` as the "XGBoost proxy" (`:L50`), and a Ridge surrogate for "SHAP-style" contributions (`:L124`).
- If the real data files are missing, it can generate synthetic data (`:L109-L112`; `src/data/demo_generator.py:L9-L32`).

---

## 4. Results

### 4a. Main benchmark: official FD001 test set, 5 seeds (source: `results/metrics.json`)

- **Setup:**
  - one prediction per test engine (100 engines) from its last 30-cycle window (`results/metrics.json:L7`, `:L22`)
  - RUL capped at 125 (`:L5`)
  - seeds 0–4 (`:L9-L15`), each controlling the split, LSTM initialisation and batch order, and the XGBoost seed and search (`:L16`)
  - values are mean ± sample std (ddof=1) across seeds (`:L17`)
  - NASA score = PHM08 asymmetric score summed over the 100 engines; lower is better (`:L8`; `src/evaluation/metrics.py:L17-L23`)
- **Environment:** Python 3.12.4, torch 2.7.1, xgboost 3.0.3, scikit-learn 1.6.1, shap 0.51.0, CPU, arm64 (`results/metrics.json:L1821-L1829`).

**Ablation: baselines and main model side by side**

| Model | RMSE | NASA score | MAE | R² |
|---|---:|---:|---:|---:|
| LSTM only | 14.58 ± 0.65 (`L69-L70`) | 476.6 ± 104.8 (`L80-L81`) | 10.40 ± 0.62 (`L91-L92`) | 0.867 ± 0.012 (`L102-L103`) |
| XGBoost only | 17.89 ± 0.52 (`L115-L116`) | 978.2 ± 205.2 (`L126-L127`) | 12.67 ± 0.42 (`L137-L138`) | 0.801 ± 0.012 (`L148-L149`) |
| Simple average (0.5/0.5) | 14.81 ± 0.56 (`L161-L162`) | 470.7 ± 62.4 (`L172-L173`) | 10.69 ± 0.41 (`L183-L184`) | 0.863 ± 0.011 (`L194-L195`) |
| **Learned ensemble** | **14.33 ± 0.63** (`L207-L208`) | **451.3 ± 90.6** (`L218-L219`) | **10.25 ± 0.57** (`L229-L230`) | **0.872 ± 0.011** (`L240-L241`) |

- **Learned LSTM weight:** 0.868 ± 0.035 (`results/metrics.json:L439-L440`).
- **Ensemble gain over LSTM only:** 0.25 RMSE (computed: 14.585 − 14.335; `L69`, `L207`).

**Per-seed, official test set** (all from `results/metrics.json`)

| Seed | LSTM RMSE | XGB RMSE | Avg RMSE | Ens RMSE | LSTM NASA | XGB NASA | Avg NASA | Ens NASA | w_lstm | LSTM epochs | Lines |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 14.19 | 17.17 | 14.50 | 14.02 | 572.4 | 661.7 | 453.4 | 518.8 | 0.870 | 20 | L489–L508, L540, L543 |
| 1 | 15.70 | 18.51 | 15.78 | 15.40 | 563.5 | 1055.5 | 566.1 | 537.3 | 0.833 | 20 | L756–L775, L807, L810 |
| 2 | 14.20 | 17.61 | 14.56 | 13.99 | 519.2 | 891.1 | 496.9 | 493.3 | 0.863 | 19 | L1023–L1042, L1074, L1077 |
| 3 | 14.61 | 18.22 | 14.78 | 14.39 | 362.5 | 1114.6 | 413.4 | 352.7 | 0.925 | 21 | L1290–L1309, L1341, L1344 |
| 4 | 14.22 | 17.94 | 14.41 | 13.87 | 365.3 | 1168.1 | 423.7 | 354.3 | 0.849 | 20 | L1557–L1576, L1608, L1611 |

- **Per-seed XGBoost best parameters:** `results/metrics.json` L546–L549, L813–L816, L1080–L1083, L1347–L1350, L1614–L1617.
- **Training time per seed** (LSTM + XGBoost, CPU): 422 s, 575 s, 726 s, 660 s, 745 s (computed from `train_seconds` at L553–L554, L820–L821, L1087–L1088, L1354–L1355, L1621–L1622).

### 4b. Secondary: every test window (10,196 windows, 5 seeds); not comparable to published results

| Model | RMSE | NASA score (sum over 10,196 windows) | Source lines in `results/metrics.json` |
|---|---:|---:|---|
| LSTM only | 14.77 ± 0.64 | 38,864 ± 2,358 | L255–L256, L266–L267 |
| XGBoost only | 18.04 ± 0.16 | 98,616 ± 13,427 | L301–L302, L312–L313 |
| Simple average | 15.08 ± 0.41 | 41,239 ± 1,857 | L347–L348, L358–L359 |
| Learned ensemble | 14.56 ± 0.59 | 36,955 ± 1,997 | L393–L394, L404–L405 |

### 4c. Deployed model: single run, seed 42, all-windows protocol (`reports/ablation_results.json`)

These numbers are produced by `src/models/trainer.py:L172` (all 10,196 test windows, one seed). They are what the README reported before commit `261e280`.

| Model | RMSE | MAE | R² | NASA score | Lines |
|---|---:|---:|---:|---:|---|
| LSTM only | 15.16 | 11.76 | 0.740 | 39,295 | L3–L6 |
| XGBoost only | 17.80 | 13.33 | 0.641 | 100,588 | L9–L12 |
| Simple average | 15.15 | 12.05 | 0.740 | 41,126 | L21–L24 |
| Learned ensemble (0.879 / 0.121) | 14.89 | 11.69 | 0.749 | 37,274 | L27–L33 |

- **Deployed LSTM training:** 20 epochs run, best validation RMSE 12.59 at epoch 5 (`reports/lstm_history.json:L29`; 20 entries per series).

### 4d. Legacy file: do not use

`reports/experiment_summary.json` reports LSTM RMSE 9.14, XGBoost 13.56 and ensemble 9.63 (`:L40`, `:L46`, `:L52`).

- **Why it can't be trusted:**
  - It is written only by the legacy pipeline (`src/pipeline/run_experiment.py:L105`).
  - Its features don't match the current code: 168 features (`:L5`), 12,009 train / 2,933 test windows (`:L31-L32`), sensor names like `sensor_2` (`:L7`) where the current config uses `s2` (`src/config.py:L19`).
- **Which data and code produced it:** NOT FOUND (real or synthetic data cannot be determined).

### 4e. Degradation fingerprinting (`results/metrics.json`)

| Clustering | Engines | Silhouette (mean ± std, 5 seeds) | Seeds where "fast-burn" has the steepest slope | Seeds where fast-burn > slow-decay |
|---|---:|---:|---:|---:|
| As implemented (first 2,000 test SHAP rows) | 23 (L558) | 0.203 ± 0.084 (L452–L453) | 0/5 (L462) | 0/5 (L463) |
| All 100 test engines | 100 (L616) | 0.084 ± 0.017 (L468–L469) | 4/5 (L478) | 4/5 (L479) |

- **Deployed model's clustering:** 23 engines, silhouette **0.153** (`reports/fingerprints.json:L2`). Cluster sizes 1 / 21 / 1 (`:L7`, `:L14`, `:L21`).
- **Per-seed silhouettes:**
  - as implemented: 0.144, 0.277, 0.090, 0.280, 0.224 (L559, L826, L1093, L1360, L1627)
  - all engines: 0.093, 0.059, 0.076, 0.097, 0.097 (L617, L884, L1151, L1418, L1685)

---

## 5. Engineering

- **Live demo:** https://turbxplain-aarohi.streamlit.app/ (`README.md:L3`).
  - The app's Streamlit health endpoint (`/~/+/_stcore/health`) returned `ok`, HTTP 200, on 2026-11-13 at 11:13 IST. This was a manual check, not a repo artifact.
- **Dashboard:** Streamlit, `app/streamlit_app.py`.
  - **Pages (6):** RUL Predictor, Explainability Panel, Degradation Fingerprinting, What-If Simulator, Model Performance, Download Reports (`:L22-L29`).
  - **Benchmark data:** the Model Performance page reads `results/metrics.json` (`:L51`) and per-seed predictions (`:L272`).
  - **Deployed-model data:** the other pages read `reports/*` (`:L75`, `:L113`, `:L151`).
  - **Timelines:** the RUL and SHAP timelines plot the real engine cycle on an axis titled "Engine cycle". Window *k* is plotted at cycle *k* + 29, so the first prediction is at cycle 30 (`:L92`, `:L98`, `:L127`).
  - **Explainability Panel:** labelled "XGBoost component only" (`:L110`, `:L130`, `:L134`). For engines whose SHAP values are only partly saved (engine 23), it says how many windows are covered (`:L122`).
  - **Degradation Fingerprinting:** clusters are shown as Cluster A / B / C, with the silhouette score and a "Cluster structure is weak" warning (`:L156`, `:L163`). The fast-burn / slow-decay / sudden-failure names do not appear in the dashboard.
  - **What-If Simulator:** labelled "XGBoost component only" (`:L198`, `:L223`). Uses only the XGBoost model; features not set by a slider are fixed at 0.5, and the output is clipped to [0, 125] (`:L210`, `:L218-L219`).
- **REST API:** FastAPI, `src/api/serve.py`.
  - **Endpoints:** `GET /health` (`:L59`), `POST /predict` (`:L64`), `POST /explain` (`:L81`), `GET /engine/{id}/timeline` (`:L100`).
  - **Timeline endpoint:** a stub that returns empty data (`:L102`).
  - **Input:** one window of 30 × 137 preprocessed features; anything else gets HTTP 400 (`src/inference.py:L18-L30`; `src/api/serve.py:L71`, `:L88`).
  - **Confidence interval:** a fixed ±10 cycles, not a model estimate (`src/api/serve.py:L75`).
- **Docker:** `python:3.12-slim` image that runs the API with uvicorn on port 8000 (`Dockerfile:L1`, `:L8-L9`). Building the image: NOT FOUND (no record of a successful build).
- **Tests:** 9 tests in 4 files, all passing (`pytest`, Python 3.12.4 with torch 2.7.1 and xgboost 3.0.3, on 2026-11-13).
  - **Model:** `tests/test_models.py:L6` checks that the ensemble weights sum to 1.
  - **Preprocessing:** `tests/test_preprocessor.py:L8`, `:L14` check the rolling slope and the sequence targets.
  - **Waterfall:** `tests/test_shap.py:L6` checks waterfall onset detection.
  - **API:** `tests/test_api.py:L28`, `:L34`, `:L47-L48` (2 parametrised cases), `:L56`.
  - **API correctness:** `:L56` asserts the API reproduces `reports/predictions.csv` within 1e-3.
  - **Test coverage percentage:** NOT FOUND.
- **Latency:** deployed model, batch of 1 window, median of 30 calls (`src/models/trainer.py:L269-L294`; `reports/latency.json`):

| Component | ms | Line |
|---|---:|---|
| LSTM inference | 1.40 | `reports/latency.json:L3` |
| XGBoost inference | 0.06 | `:L4` |
| SHAP explanation (XGBoost) | 0.84 | `:L5` |
| Total | 2.30 | `:L6` |

  - Hardware for the latency run: NOT FOUND. Throughput: NOT FOUND.
- **Notable design decisions:**
  - Engine-level validation split, which avoids leaking windows from one engine into both sets (`src/models/trainer.py:L217-L224`).
  - Ensemble weights fitted on validation, never on test (`src/evaluation/benchmark.py:L150`).
  - One results file feeds both the README results section (`src/evaluation/update_readme.py`) and the dashboard (`app/streamlit_app.py:L51`).
  - Benchmark runs are resumable per seed (`src/evaluation/benchmark.py:L322-L326`).
  - Deterministic torch algorithms are enabled in the benchmark (`src/evaluation/benchmark.py:L74`). Bitwise reproducibility on a re-run: NOT FOUND (never re-run).
  - Dependencies are pinned to exact versions (`requirements.txt:L3-L18`).
  - README figures are generated by `src/evaluation/make_figures.py`, which fails if any plotted number differs from the saved results (details in §10).
- **Licence:** MIT (`LICENSE:L1`).

---

## 6. Surprising findings

1. **The ensemble's gain is smaller than seed noise.**
   - The learned ensemble improves on LSTM alone by 0.25 RMSE (computed, `results/metrics.json:L69`, `:L207`), while the seed-to-seed std is 0.63–0.65 (`:L70`, `:L208`).
   - It still wins in every seed, by 0.18–0.35 RMSE (computed from the per-seed table, §4a).
2. **The simple average beats LSTM alone on NASA score, despite worse RMSE.**
   - Mean NASA: 470.7 vs 476.6 (`L172` vs `L80`), with the lowest NASA std of all models (62.4, `L173`).
   - In seed 0 the simple average also beats the learned ensemble on NASA score (453.4 vs 518.8, `L502` vs `L508`).
   - The ensemble weights minimise RMSE, not the NASA score (`src/models/ensemble.py:L19-L20`).
3. **XGBoost is the weakest model in every seed** on both RMSE and NASA score (§4a). Its NASA score varies widely across seeds, from 661.7 to 1,168.1 (`L496`, `L1564`).
4. **The LSTM stops very early.**
   - Early stopping triggers after 19–21 of the 150 allowed epochs in all 5 seeds (`L543`, `L810`, `L1077`, `L1344`, `L1611`).
   - With patience 15 (`src/models/trainer.py:L32`), the best checkpoint is at epoch 4–6 (computed).
   - The deployed model's best epoch is 5 (`reports/lstm_history.json:L29`).
5. **The fingerprint names don't match the data.**
   - **Steepest slope:** in the as-implemented clustering, "fast-burn" has the steepest degradation slope in 0 of 5 seeds (`results/metrics.json:L462`).
   - **Negative slope:** in seeds 1 and 3 its mean slope is negative, i.e. the least degrading cluster (`L832`, `L1366`).
   - **"Sudden-failure" is one short engine:** in 8 of 10 clusterings it is a single engine, test engine 1 (computed from the `engines` lists in `results/metrics.json`). That engine has the shortest history in the test set (31 cycles, 2 windows) and a true RUL of 112 (`data/raw/RUL_FD001.txt:L1`), so it is far from failure.
   - **Short histories:** every 1- or 2-engine cluster consists of short-history test engines (engines 1, 9, 14, 22: 31–55 cycles, ranks 1–11 of 100; test median 133.5; computed from `data/raw/test_FD001.txt`).
   - **No real structure:** silhouette scores of 0.08–0.20 (§4e) indicate little cluster structure.
6. **The pre-`261e280` headline metrics used a non-standard protocol.** The README's earlier ensemble NASA score of 37,274 (`reports/ablation_results.json:L30`) is a sum over all 10,196 test windows. The standard one-per-engine score for the same model family is 451.3 ± 90.6 (`results/metrics.json:L218-L219`).

---

## 7. Limitations: what the evaluation does NOT show

- **Other subsets:** nothing about FD002–FD004 (multiple operating conditions or fault modes). Only FD001 is evaluated (§2).
- **Comparison with published work:** the repo has no side-by-side comparison with published C-MAPSS results. NOT FOUND.
- **Statistical significance:** NOT FOUND. With 5 seeds, the ensemble-vs-LSTM gap (0.25) is below one standard deviation (§6.1).
- **Accuracy earlier in an engine's life:** the primary metric scores one point per test engine, at the end of its history (`src/evaluation/benchmark.py:L161-L162`).
- **Explaining the ensemble:** SHAP explains only the XGBoost component, which carries about 13% of the ensemble weight (learned LSTM weight 0.868 ± 0.035, `results/metrics.json:L439-L440`). The LSTM is not explained (§3).
- **Failure archetypes:** the clusters show weak structure, the names are assigned by index, and they don't track a sensor-based degradation slope (§4e, §6.5).
- **Tuning of the LSTM:** none exists (§3).
- **Validation reuse:** one validation split selects the LSTM checkpoint, stops XGBoost, *and* fits the ensemble weights (`results/metrics.json:L18`).
- **Reproducibility:** no seed has been re-run to confirm identical outputs.
- **Deployed model vs benchmark:** the dashboard and API model (trainer, seed 42, `src/models/trainer.py:L117`) is not one of the 5 benchmark seeds.
- **Latency:** a single run on unrecorded hardware (§5).
- **API inputs:** the API takes preprocessed features, not raw sensors, and its confidence interval is a fixed ±10 (`src/api/serve.py:L75`).
- **SHAP coverage:** the deployed model saves SHAP values for the first 2,000 test windows only (`src/models/trainer.py:L228`), covering test engines 1–23 with engine 23 partial. The Explainability and Fingerprinting pages therefore cover 23 of 100 test engines.

---

## 8. Tech stack (imported and used in `src/`, `app/` or `tests/`)

| Library | Used for | Where |
|---|---|---|
| PyTorch | LSTM model, training, inference | `src/models/lstm_model.py`, `src/models/trainer.py`, `src/inference.py`, `src/evaluation/benchmark.py` |
| XGBoost | Gradient-boosted regressor | `src/models/xgb_model.py` |
| SHAP | TreeExplainer on XGBoost | `src/explainability/shap_explainer.py`, `src/inference.py`, `src/evaluation/benchmark.py` |
| scikit-learn | MinMaxScaler, RandomizedSearchCV, KMeans, StandardScaler, silhouette_score, regression metrics | `src/data/preprocessor.py`, `src/models/xgb_model.py`, `src/explainability/degradation_fingerprint.py`, `src/evaluation/metrics.py` |
| SciPy | `optimize.minimize` for the ensemble weights | `src/models/ensemble.py` |
| pandas, NumPy | Data handling | throughout |
| FastAPI, Pydantic | REST API | `src/api/serve.py` |
| Streamlit | Dashboard | `app/streamlit_app.py` |
| Plotly | Dashboard charts | `app/streamlit_app.py`, `src/explainability/*.py` |
| joblib | Model bundle serialisation | `src/models/trainer.py`, `src/inference.py`, `app/streamlit_app.py` |
| pytest | Tests | `tests/` |
| Matplotlib | README figures (not in `requirements.txt`; README How to Run says `pip install matplotlib==3.10.1`) | `src/evaluation/make_figures.py` |

Listed in `requirements.txt` but not imported:
- `uvicorn`, used only as the Docker server command (`Dockerfile:L9`)
- `starlette` and `httpx2`, pulled in through FastAPI's test client (`requirements.txt:L11`, `:L18`)

---

## 9. Timeline

- **First commit:** `fb8d5ff`, 2026-05-16 15:33:29 +0530, "Initial commit: TurbXplain — Explainable Predictive Maintenance" (`git log`).
- **Last commit:** `ef22a28`, 2026-11-13 15:56:37 +0530, "Add verified README figures: ablation RMSE and temporal SHAP" (`git log`).
- **Total:** 13 commits.
  - 8 commits on 2026-05-16 and 2026-05-17
  - none between 2026-05-17 and 2026-11-12
  - 3 commits on 2026-11-12 and 2 on 2026-11-13

---

## 10. Visuals

### Existing image files

| File | What it shows | Source data | Embedded in |
|---|---|---|---|
| `assets/ablation.png` | RMSE mean ± std for LSTM only, XGBoost only, simple average, learned ensemble; official FD001 test set, 5 seeds, individual seeds as dots | `results/metrics.json` | `README.md:L21` |
| `assets/shap_waterfall.png` | Temporal SHAP of the top 8 features (by mean \|SHAP\|) for FD001 test engine 12, cycles 30–217; x-axis "Engine cycle"; subtitle "XGBoost component only · deployed model" | `reports/shap_contributions.csv` | `README.md:L108` |

- **Generated by** `src/evaluation/make_figures.py`. It stops with an error if any of these fail:
  - plotted means and stds equal the `metrics.json` summary, a recomputation from the 5 per-seed runs, and the README table (`:L52`)
  - the chosen engine has SHAP values for every window
  - the window-to-cycle mapping reproduces the saved true RUL (`:L134`)
  - SHAP values plus the base value reproduce the saved XGBoost predictions within 1e-3; measured maximum error 6e-5 (`:L150`)
- **Engine choice rule:** the longest test trajectory whose SHAP values are all saved (`:L120`); test engine 12 has 188 windows.
- **No other images:** no `.jpg`, `.gif`, `.svg` or `.ipynb` files exist, and `notebooks/` is empty.

### Charts rendered by the dashboard (interactive Plotly, `app/streamlit_app.py`)

| Chart | Page | Code |
|---|---|---|
| RUL predictions over engine cycles for one engine (true, ensemble, LSTM, XGBoost) | RUL Predictor | `:L93-L98` |
| Temporal SHAP for one engine over engine cycles (XGBoost component only) | Explainability Panel | `:L130-L131` |
| Mean \|SHAP\| global feature importance (top 15, first 2,000 test windows) | Explainability Panel | `:L138-L139` |
| Engines per cluster, Cluster A / B / C (pie) | Degradation Fingerprinting | `:L173` |
| Ablation RMSE bar chart with ±std error bars (5 seeds) | Model Performance | `:L242-L250` |
| Actual vs predicted RUL, official test set, selectable seed | Model Performance | `:L276-L280` |
| LSTM train/validation RMSE per epoch (deployed model) | Model Performance | `:L338-L343` |

### Suggested 2 visuals for a LinkedIn post

1. **`assets/ablation.png`.** All four models with seed variance on the standard FD001 protocol, in one image. Caption the gain honestly: 14.33 ± 0.63 for the learned ensemble vs 14.58 ± 0.65 for the LSTM alone.
2. **`assets/shap_waterfall.png`.** The project's explainability visual. The caption must say it explains the XGBoost component only, not the LSTM or the ensemble.

### Requested quick-reference facts

- **C-MAPSS subset:** **FD001** only (`src/config.py:L10`; `results/metrics.json:L2`, `:L4`).
- **Per-model ablation, official FD001 test set, mean ± std over 5 seeds** (`results/metrics.json`):

| Model | RMSE | NASA score |
|---|---:|---:|
| LSTM only | 14.58 ± 0.65 | 476.6 ± 104.8 |
| XGBoost only | 17.89 ± 0.52 | 978.2 ± 205.2 |
| Simple average | 14.81 ± 0.56 | 470.7 ± 62.4 |
| Learned ensemble | 14.33 ± 0.63 | 451.3 ± 90.6 |

  Lines: L69–L70, L80–L81, L115–L116, L126–L127, L161–L162, L172–L173, L207–L208, L218–L219.

- **Cluster silhouette score:**
  - deployed model: **0.153** (`reports/fingerprints.json:L2`)
  - 5-seed benchmark, as-implemented 23 engines: **0.203 ± 0.084** (`results/metrics.json:L452-L453`)
  - 5-seed benchmark, all 100 test engines: **0.084 ± 0.017** (`results/metrics.json:L468-L469`)
