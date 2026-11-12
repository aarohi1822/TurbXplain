import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, REPORT_DIR

BUNDLE = MODEL_DIR / "turbxplain_bundle.joblib"
SEQUENCES = PROCESSED_DATA_DIR / "FD001_sequences.npz"
PREDICTIONS = REPORT_DIR / "predictions.csv"

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="trained model bundle not available")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.serve import app

    with TestClient(app) as c:
        yield c


def _window(n_cycles=30, n_features=137, value=0.5):
    return np.full((n_cycles, n_features), value).tolist()


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_predict_and_explain_return_200(client):
    payload = {"engine_id": "1", "sensor_data": _window()}
    pred = client.post("/predict", json=payload)
    assert pred.status_code == 200
    assert np.isfinite(pred.json()["rul_prediction"])

    expl = client.post("/explain", json=payload)
    assert expl.status_code == 200
    body = expl.json()
    assert body["explained_model"] == "xgboost"
    assert len(body["shap_values"]) == 10


@pytest.mark.parametrize("shape", [(30, 10), (5, 137)])
def test_wrong_window_shape_is_400(client, shape):
    resp = client.post("/predict", json={"engine_id": "1", "sensor_data": _window(*shape)})
    assert resp.status_code == 400
    resp = client.post("/explain", json={"engine_id": "1", "sensor_data": _window(*shape)})
    assert resp.status_code == 400


@pytest.mark.skipif(not (SEQUENCES.exists() and PREDICTIONS.exists()), reason="processed data / reports not available")
def test_api_matches_trainer_predictions(client):
    """The API must reproduce the predictions the trainer saved for the same bundle."""
    X_test = np.load(SEQUENCES, allow_pickle=True)["X_test"]
    saved = pd.read_csv(PREDICTIONS)
    for idx in (0, 500, len(X_test) - 1):
        resp = client.post("/predict", json={"engine_id": "x", "sensor_data": X_test[idx].tolist()})
        assert resp.status_code == 200
        assert resp.json()["rul_prediction"] == pytest.approx(saved["ensemble_prediction"].iloc[idx], abs=1e-3)
