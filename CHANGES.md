# Changes since PROJECT_FACTS.md (commit d3780ef → 8e7df41)
> Snapshot as of `8e7df41`. Since then `ef22a28` added `assets/ablation.png` and `assets/shap_waterfall.png`, and PROJECT_FACTS.md was updated to `ef22a28`.

## 1. Fixes requested
- Fingerprinting page: Cluster A/B/C, archetype labels removed, silhouette shown with a weak-structure note. **Done** (`app/streamlit_app.py:L156`, `:L163`; no "fast-burn/slow-decay/sudden-failure" left in the file).
- Timeline axis: **Done**, converted to the real engine cycle (window k → cycle k+29), with the axis titled "Engine cycle" rather than "Window index" (`app/streamlit_app.py:L98`, `:L131`).
- "XGBoost component only" labels on the What-If Simulator and SHAP panels. **Done** (`app/streamlit_app.py:L110`, `:L130`, `:L134`, `:L198`).
- README: leads with the 5-seed official-protocol table. **Done** (`README.md:L9-L24`).
- README: Limitations section from PROJECT_FACTS.md. **Done, adapted** (`README.md:L206`). File/line citations were dropped, and the "Dashboard axis" item was replaced by "SHAP coverage" because the axis fix resolved it.

## 2. Numbers in PROJECT_FACTS.md that changed
- Metrics and results: none. `results/`, `reports/`, `models/` and `data/` are unchanged between d3780ef and 8e7df41 (`git diff`). Test count: 9 → 9.
- Line citations that moved: `README.md` L13 → L30; `app/streamlit_app.py` L108 → L113, L140 → L151, L196 → L210, L204-L205 → L218-L219, L258 → L272, L228-L236 → L242-L250, L262-L266 → L276-L280, L324-L329 → L338-L343.
- Line citations whose code was rewritten (no longer valid): `app/streamlit_app.py` L91, L92-L97, L116, L119-L120, L127-L128, L151.

## 3. README identifiers
- Project name: "TurbXplain: Explainable RUL Prediction on NASA C-MAPSS" (`README.md:L1`).
- Repo URL: https://github.com/aarohi1822/TurbXplain (`README.md:L184`).
- Demo URL: https://turbxplain-aarohi.streamlit.app/ (`README.md:L3`).

## 4. New image files
NOT FOUND. The repo contains no .png, .jpg, .jpeg, .gif, .svg or .webp files.

## 5. Tests
Yes, all pass: **9 passed** (pytest, Python 3.12.4, torch 2.7.1, xgboost 3.0.3).
