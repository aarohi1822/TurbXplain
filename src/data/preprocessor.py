import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    CYCLE_COL,
    ENGINE_COL,
    NEAR_CONSTANT_STD,
    OP_SETTING_COLS,
    PROCESSED_DATA_DIR,
    ROLLING_WINDOWS,
    RUL_CLIP,
    SENSOR_COLS,
    SEQUENCE_LENGTH,
)
from src.data.loader import load_dataset


class CMAPSSPreprocessor:
    def __init__(
        self,
        dataset: str = "FD001",
        sequence_length: int = SEQUENCE_LENGTH,
        rul_clip: int = RUL_CLIP,
        output_dir: str | Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.rul_clip = rul_clip
        self.output_dir = Path(output_dir)
        self.scaler = MinMaxScaler()
        self.kept_sensors: list[str] = []
        self.dropped_sensors: list[str] = []
        self.feature_cols: list[str] = []

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        return load_dataset(self.dataset)

    def clean(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        stds = train[SENSOR_COLS].std()
        self.dropped_sensors = stds[stds < NEAR_CONSTANT_STD].index.tolist()
        self.kept_sensors = [col for col in SENSOR_COLS if col not in self.dropped_sensors]
        return train.copy(), test.copy()

    def add_rul_labels(self, train: pd.DataFrame, test: pd.DataFrame, test_rul: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = train.copy()
        test = test.copy()
        max_cycle = train.groupby(ENGINE_COL)[CYCLE_COL].transform("max")
        train["rul"] = (max_cycle - train[CYCLE_COL]).clip(upper=self.rul_clip)

        test_max = test.groupby(ENGINE_COL)[CYCLE_COL].transform("max")
        final_rul = test[ENGINE_COL].map(dict(zip(range(1, len(test_rul) + 1), test_rul)))
        test["rul"] = (test_max - test[CYCLE_COL] + final_rul).clip(upper=self.rul_clip)
        return train, test

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([ENGINE_COL, CYCLE_COL]).copy()
        base_cols = [*OP_SETTING_COLS, *self.kept_sensors]
        engineered = {}
        for col in self.kept_sensors:
            grouped = df.groupby(ENGINE_COL)[col]
            engineered[f"{col}_diff_1"] = grouped.diff().fillna(0.0)
            for window in ROLLING_WINDOWS:
                rolling = grouped.rolling(window=window, min_periods=1)
                engineered[f"{col}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
                engineered[f"{col}_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
            engineered[f"{col}_trend_10"] = grouped.transform(lambda s: self._rolling_slope(s, 10))

        if engineered:
            df = pd.concat([df, pd.DataFrame(engineered, index=df.index)], axis=1)
        self.feature_cols = [col for col in df.columns if col in base_cols or col.startswith("s")]
        return df

    def normalize_all(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = train.copy()
        test = test.copy()
        non_constant = [c for c in self.feature_cols if train[c].std() > 1e-6]
        self.feature_cols = non_constant
        self.scaler = MinMaxScaler()
        train[self.feature_cols] = self.scaler.fit_transform(train[self.feature_cols])
        test[self.feature_cols] = self.scaler.transform(test[self.feature_cols])
        return train, test

    def create_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        engine_ids: list[int] = []
        for engine_id, group in df.groupby(ENGINE_COL, sort=True):
            group = group.sort_values(CYCLE_COL)
            values = group[self.feature_cols].to_numpy(dtype=np.float32)
            labels = group["rul"].to_numpy(dtype=np.float32)
            if len(group) < self.sequence_length:
                continue
            for end in range(self.sequence_length, len(group) + 1):
                sequences.append(values[end - self.sequence_length : end])
                targets.append(labels[end - 1])
                engine_ids.append(int(engine_id))
        return np.stack(sequences), np.asarray(targets, dtype=np.float32), np.asarray(engine_ids)

    def run(self) -> dict:
        train, test, test_rul = self.load()
        train, test = self.clean(train, test)
        train, test = self.add_rul_labels(train, test, test_rul)
        train = self.engineer_features(train)
        test = self.engineer_features(test)
        train, test = self.normalize_all(train, test)
        X_train, y_train, train_engine_ids = self.create_sequences(train)
        X_test, y_test, test_engine_ids = self.create_sequences(test)
        stats = self.save(X_train, y_train, X_test, y_test, train_engine_ids, test_engine_ids)
        return stats

    def save(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_engine_ids: np.ndarray,
        test_engine_ids: np.ndarray,
    ) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.output_dir / f"{self.dataset}_sequences.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_engine_ids=train_engine_ids,
            test_engine_ids=test_engine_ids,
            feature_names=np.asarray(self.feature_cols),
        )
        stats = {
            "dataset": self.dataset,
            "sequence_length": self.sequence_length,
            "num_features": len(self.feature_cols),
            "kept_sensors": self.kept_sensors,
            "dropped_sensors": self.dropped_sensors,
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
        }
        (self.output_dir / f"{self.dataset}_data_stats.json").write_text(json.dumps(stats, indent=2))
        return stats

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        values = series.to_numpy(dtype=float)
        slopes = np.zeros_like(values)
        for idx in range(len(values)):
            start = max(0, idx - window + 1)
            y = values[start : idx + 1]
            if len(y) < 2:
                slopes[idx] = 0.0
                continue
            x = np.arange(len(y), dtype=float)
            slopes[idx] = np.polyfit(x, y, 1)[0]
        return pd.Series(slopes, index=series.index)
