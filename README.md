# TurbXplain — Explainable Predictive Maintenance

> Don't just predict failure. **Explain why.**

TurbXplain predicts Remaining Useful Life (RUL) for turbofan engines using a hybrid LSTM + XGBoost ensemble, then explains every prediction with SHAP-based diagnostics. Built on NASA C-MAPSS data with a full Streamlit dashboard for operators and engineers.

---

## Why This Exists

Most predictive maintenance models are black boxes — they tell you *if* something will fail, but not *when*, *why*, or *what type* of failure pattern it matches. TurbXplain answers all three:

- **When** — Remaining Useful Life regression (not binary fail/no-fail)
- **Why** — Per-prediction SHAP explanations showing which sensors drive each alert
- **What type** — Degradation fingerprinting clusters engines into failure archetypes

---

## What's Different

| Feature | Typical PredMaint Projects | TurbXplain |
|---|---|---|
| Prediction | Binary (fail/no-fail) | RUL regression (cycles to failure) |
| Model | Single model | Hybrid LSTM + XGBoost ensemble with learned weights |
| Explainability | None or basic feature importance | Per-prediction SHAP + Temporal SHAP Waterfall |
| Failure typing | None | Degradation fingerprinting via SHAP trajectory clustering |
| Deployment | Notebook only | FastAPI + Streamlit dashboard |

### Key Innovations

- **Hybrid Ensemble with Learned Blending** — LSTM captures temporal degradation patterns, XGBoost captures tabular feature interactions. Weights are learned on validation data via constrained optimization — not manually set.
- **Temporal SHAP Waterfall** — An interactive timeline showing how each sensor's contribution to the failure prediction shifts as degradation develops. Reveals *when* the model first detects trouble and *which sensor* triggers the alert.
- **Degradation Fingerprinting** — Engines are clustered into failure archetypes (fast-burn, slow-decay, sudden-failure) using summary statistics of their SHAP trajectory vectors.

---

## Results on NASA C-MAPSS FD001

| Model | RMSE | MAE | R² | NASA Score |
|---|---:|---:|---:|---:|
| LSTM only | 15.16 | 11.76 | 0.74 | 39,295 |
| XGBoost only | 17.80 | 13.33 | 0.64 | 100,588 |
| **Ensemble** | **14.89** | **11.69** | **0.75** | **37,274** |

### Ablation Study

| Configuration | RMSE |
|---|---:|
| LSTM only | 15.16 |
| XGBoost only | 17.80 |
| Simple average (0.5 / 0.5) | 15.15 |
| **Learned weighted ensemble (0.88 / 0.12)** | **14.89** |

The ensemble learns to weight LSTM at 88% and XGBoost at 12% — the temporal model dominates, but XGBoost's tabular features still improve the final prediction.

### Inference Latency

| Component | Time (ms) |
|---|---:|
| LSTM inference | 1.40 |
| XGBoost inference | 0.06 |
| SHAP explanation | 0.84 |
| **Total end-to-end** | **2.30** |

### Degradation Fingerprinting (Exploratory)

| Cluster | Label | Engine Count | Silhouette |
|---|---|---:|---:|
| 0 | Fast-burn | 1 | 0.153 |
| 1 | Slow-decay | 21 | 0.153 |
| 2 | Sudden-failure | 1 | 0.153 |

Silhouette score of 0.153 indicates limited cluster separability using XGBoost SHAP features alone. This is marked as exploratory — future work includes using LSTM temporal attention weights as alternative clustering features.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NASA C-MAPSS Data                     │
│              (21 sensors × N cycles × 100 engines)       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Preprocessing Pipeline                      │
│  • Drop constant sensors  • Min-Max normalization        │
│  • Rolling mean/std/diff/trend features                  │
│  • RUL labeling (clipped at 125)                         │
│  • Sequence windowing (30 cycles)                        │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌────────────────────┐      ┌────────────────────────┐
│   LSTM (PyTorch)   │      │   XGBoost (Tabular)    │
│   2-layer, h=128   │      │   Tuned via RandomCV   │
│   + BatchNorm      │      │   Last-row features    │
│   + HuberLoss      │      │                        │
│   RMSE: 15.16      │      │   RMSE: 17.80          │
└────────┬───────────┘      └───────────┬────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────┐
│           Learned Weighted Ensemble                      │
│           LSTM: 0.88  |  XGBoost: 0.12                   │
│           Ensemble RMSE: 14.89  |  R²: 0.75             │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ SHAP         │ │ Temporal     │ │ Degradation      │
│ TreeExplainer│ │ Waterfall    │ │ Fingerprinting   │
│ (per pred)   │ │ (per engine) │ │ (K-Means on SHAP)│
└──────────────┘ └──────────────┘ └──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit Dashboard                         │
│  RUL Predictor | Explainability | Fingerprints           │
│  What-If Simulator | Model Performance | Reports         │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Core ML | PyTorch, XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer, GradientExplainer) |
| Data | Pandas, NumPy, SciPy |
| Backend | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Storage | joblib (model bundles) |
| Dev | Git, pytest |

---

## Project Structure

```
TurbXplain/
├── app/
│   └── streamlit_app.py          # Interactive dashboard (6 pages)
├── data/
│   ├── raw/                      # NASA C-MAPSS files (not in repo)
│   ├── processed/                # Preprocessed .npz arrays
│   └── download_data.py          # Data download helper
├── models/                       # Trained model weights (not in repo)
├── reports/
│   ├── results_summary.md        # Metrics, ablation, latency
│   ├── ablation_results.json     # Full ablation data
│   ├── fingerprints.json         # Clustering results
│   ├── shap_contributions.csv    # Per-prediction SHAP values
│   ├── predictions.csv           # All test predictions
│   ├── latency.json              # Inference benchmarks
│   └── lstm_history.json         # Training curves
├── src/
│   ├── api/
│   │   └── serve.py              # FastAPI endpoints
│   ├── data/
│   │   ├── loader.py             # C-MAPSS file parser
│   │   └── preprocessor.py       # Normalization, features, windowing
│   ├── evaluation/
│   │   ├── metrics.py            # RMSE, MAE, R², NASA scoring function
│   │   └── ablation.py           # Ablation study runner
│   ├── explainability/
│   │   ├── shap_explainer.py     # SHAP value computation
│   │   ├── temporal_waterfall.py # Temporal SHAP timeline
│   │   └── degradation_fingerprint.py
│   └── models/
│       ├── lstm_model.py         # PyTorch LSTM architecture
│       ├── xgb_model.py          # XGBoost wrapper
│       ├── ensemble.py           # Learned weighted blending
│       └── trainer.py            # End-to-end training pipeline
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/aarohi1822/TurbXplain.git
cd TurbXplain
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the Data

Download [NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) from Kaggle and place the `.txt` files in `data/raw/CMaps/`.

### 3. Preprocess + Train

```bash
python -m src.data.loader --dataset FD001
python -m src.models.trainer
```

### 4. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 5. Launch API (optional)

```bash
uvicorn src.api.serve:app --reload
```

### 6. Run Tests

```bash
pytest -q
```

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation**

- 4 sub-datasets (FD001–FD004) with varying operating conditions and fault modes
- FD001: 100 train engines, 100 test engines, 1 fault mode, 1 operating condition
- 21 sensor channels + 3 operational settings per engine per cycle
- Source: [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Future Work

- Train on FD002–FD004 (multi-fault, multi-operating-condition)
- LSTM GradientExplainer for richer temporal SHAP trajectories
- Attention-based degradation fingerprinting
- Transfer learning on SECOM semiconductor manufacturing data
- Downloadable PDF maintenance reports from dashboard

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
