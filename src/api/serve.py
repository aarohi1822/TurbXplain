from enum import Enum

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("Install API dependencies with `pip install fastapi uvicorn pydantic`.") from exc

from src.inference import load_bundle, predict_sequence


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class SensorPayload(BaseModel):
    engine_id: str
    sensor_data: list[list[float]] = Field(min_length=1)


app = FastAPI(title="TurbXplain API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_BUNDLE = None


@app.on_event("startup")
def load_models() -> None:
    global MODEL_BUNDLE
    try:
        MODEL_BUNDLE = load_bundle()
    except FileNotFoundError:
        MODEL_BUNDLE = None


def risk_level(rul: float) -> RiskLevel:
    if rul > 80:
        return RiskLevel.low
    if rul > 40:
        return RiskLevel.medium
    if rul > 15:
        return RiskLevel.high
    return RiskLevel.critical


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": MODEL_BUNDLE is not None, "version": "1.0.0"}


@app.post("/predict")
def predict(payload: SensorPayload) -> dict:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Run the experiment pipeline first.")
    try:
        result = predict_sequence(MODEL_BUNDLE, np.asarray(payload.sensor_data, dtype=float))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rul = result["rul_prediction"]
    return {
        "rul_prediction": rul,
        "confidence_interval": [max(0.0, rul - 10.0), min(125.0, rul + 10.0)],
        "model_weights": result["model_weights"],
        "risk_level": risk_level(rul),
    }


@app.post("/explain")
def explain(payload: SensorPayload) -> dict:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Run the experiment pipeline first.")
    result = predict_sequence(MODEL_BUNDLE, np.asarray(payload.sensor_data, dtype=float))
    top = sorted(result["shap_values"].items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    return {
        "shap_values": dict(top),
        "top_drivers": [f"{feature} ({value:+.3f})" for feature, value in top[:3]],
        "degradation_cluster": "see reports/fingerprints.json",
        "degradation_confidence": None,
        "onset_cycle_pct": None,
    }


@app.get("/engine/{engine_id}/timeline")
def timeline(engine_id: str) -> dict:
    return {"engine_id": engine_id, "cycles": [], "features": {}, "status": "no timeline loaded"}
