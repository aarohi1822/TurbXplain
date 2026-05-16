import numpy as np

from src.explainability.temporal_waterfall import TemporalSHAPWaterfall


def test_temporal_waterfall_detects_dominant_feature():
    shap_values = np.array([[0.0, 0.0], [0.1, 1.0], [0.2, 5.0]])
    tw = TemporalSHAPWaterfall(["sensor_1", "sensor_2"])
    matrix = tw.build_matrix(shap_values)
    onset = tw.detect_degradation_onset(matrix)
    assert onset["dominant_feature"] == "sensor_2"
