import numpy as np

from src.models.ensemble import EnsemblePredictor


def test_ensemble_weights_sum_to_one():
    y = np.array([10, 20, 30])
    lstm = np.array([11, 19, 31])
    xgb = np.array([13, 18, 28])
    ensemble = EnsemblePredictor()
    weights = ensemble.learn_weights(lstm, xgb, y)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
