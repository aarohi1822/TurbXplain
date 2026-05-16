# TurbXplain — Explainable Predictive Maintenance

> Don't just predict failure. Explain why.

TurbXplain predicts Remaining Useful Life (RUL) for turbofan engines and explains each prediction with SHAP-based diagnostics. It is built around NASA C-MAPSS data and combines temporal modeling, tabular modeling, and operator-facing explanations.

## What's Different

- **Hybrid ensemble**: LSTM for sequence behavior plus XGBoost for engineered tabular features, blended with validation-learned weights.
- **Temporal SHAP Waterfall**: An interactive timeline showing how each sensor's contribution changes as degradation develops.
- **Degradation fingerprinting**: Engines are clustered into failure archetypes such as fast-burn, slow-decay, and sudden-failure using SHAP trajectory summaries.
- **Dashboard-first delivery**: Streamlit pages for RUL prediction, explanations, fingerprints, what-if analysis, model performance, and maintenance reports.

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── download_data.py
├── models/
├── notebooks/
├── reports/
│   └── results_summary.md
├── src/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   └── models/
└── tests/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full runnable build. Uses NASA files if present; otherwise generates a
# C-MAPSS-shaped demo dataset so the project works end to end immediately.
python -m src.pipeline.run_experiment --force-demo-data

# Run API after installing FastAPI dependencies:
uvicorn src.api.serve:app --reload

# Run dashboard:
streamlit run app/streamlit_app.py
```

Generated artifacts:

- `models/turbxplain_bundle.joblib`
- `data/processed/FD001_sequences.npz`
- `reports/results_summary.md`
- `reports/ablation_results.json`
- `reports/fingerprints.json`
- `reports/latency.json`
- `reports/shap_contributions.csv`
- `reports/predictions.csv`

The current local environment does not include PyTorch, XGBoost, or SHAP, so the runnable build uses scikit-learn fallback models and Ridge-surrogate SHAP-style contributions. The PyTorch LSTM, XGBoost, and SHAP modules remain in `src/` for the full dependency environment.

## Expected Metrics

Report RMSE, MAE, R2, NASA scoring function, ensemble ablation, cluster silhouette score, and prediction latency.

## License

MIT
