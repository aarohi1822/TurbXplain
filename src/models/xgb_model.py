from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


class XGBPredictor:
    def __init__(self, model_path: str | Path = "models/xgb_best.json", **params) -> None:
        defaults = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": 1,
        }
        defaults.update(params)
        self.model = XGBRegressor(**defaults)
        self.model_path = Path(model_path)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBPredictor":
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        fit_kwargs = {"verbose": 50}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = 50
        try:
            self.model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(X_train, y_train, **fit_kwargs)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(self.model_path)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        names = feature_names or [f"feature_{idx}" for idx in range(self.model.feature_importances_.shape[0])]
        return (
            pd.DataFrame({"feature": names, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def randomized_search(self, X, y, n_iter: int = 20, cv: int = 3) -> RandomizedSearchCV:
        param_dist = {
            "n_estimators": [250, 500, 800],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
        }
        search = RandomizedSearchCV(
            self.model,
            param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=1,
            verbose=2,
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        (self.model_path.parent / "xgb_best_params.json").write_text(json.dumps(search.best_params_, indent=2))
        return search
