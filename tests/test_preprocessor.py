import numpy as np
import pandas as pd

from src.config import CYCLE_COL, ENGINE_COL
from src.data.preprocessor import CMAPSSPreprocessor


def test_rolling_slope_rises_for_increasing_series():
    series = pd.Series([1, 2, 3, 4, 5])
    slopes = CMAPSSPreprocessor._rolling_slope(series, window=3)
    assert slopes.iloc[-1] > 0


def test_create_sequences_uses_last_cycle_target():
    pre = CMAPSSPreprocessor(sequence_length=3)
    pre.feature_cols = ["sensor_1"]
    df = pd.DataFrame(
        {
            ENGINE_COL: [1, 1, 1, 1],
            CYCLE_COL: [1, 2, 3, 4],
            "sensor_1": [0.1, 0.2, 0.3, 0.4],
            "rul": [3, 2, 1, 0],
        }
    )
    X, y, engine_ids = pre.create_sequences(df)
    assert X.shape == (2, 3, 1)
    assert np.array_equal(y, np.array([1, 0], dtype=np.float32))
    assert np.array_equal(engine_ids, np.array([1, 1]))
