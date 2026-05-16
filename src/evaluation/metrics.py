import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def nasa_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_pred - y_true
    early = np.exp(-diff[diff < 0] / 13.0) - 1.0
    late = np.exp(diff[diff >= 0] / 10.0) - 1.0
    return float(np.sum(early) + np.sum(late))


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }
