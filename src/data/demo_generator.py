from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_COLUMNS, RAW_DATA_DIR, SENSOR_COLS


def generate_demo_cmapss(
    dataset: str = "FD001",
    raw_dir: str | Path = RAW_DATA_DIR,
    train_engines: int = 80,
    test_engines: int = 30,
    seed: int = 42,
) -> dict:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    train = _generate_split(rng, train_engines, include_failure=True)
    test, final_rul = _generate_test_split(rng, test_engines)

    _write_cmapss(train, raw_dir / f"train_{dataset}.txt")
    _write_cmapss(test, raw_dir / f"test_{dataset}.txt")
    pd.Series(final_rul).to_csv(raw_dir / f"RUL_{dataset}.txt", index=False, header=False)
    return {
        "dataset": dataset,
        "train_engines": train_engines,
        "test_engines": test_engines,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }


def _generate_split(rng: np.random.Generator, num_engines: int, include_failure: bool) -> pd.DataFrame:
    rows = []
    patterns = ["fast-burn", "slow-decay", "sudden-failure"]
    for engine_id in range(1, num_engines + 1):
        lifetime = int(rng.integers(145, 230))
        observed_cycles = lifetime if include_failure else int(rng.integers(90, lifetime - 15))
        pattern = patterns[engine_id % len(patterns)]
        rows.extend(_engine_rows(rng, engine_id, observed_cycles, lifetime, pattern))
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _generate_test_split(rng: np.random.Generator, num_engines: int) -> tuple[pd.DataFrame, list[int]]:
    rows = []
    final_rul = []
    patterns = ["fast-burn", "slow-decay", "sudden-failure"]
    for engine_id in range(1, num_engines + 1):
        lifetime = int(rng.integers(145, 230))
        observed_cycles = int(rng.integers(90, lifetime - 15))
        final_rul.append(lifetime - observed_cycles)
        pattern = patterns[engine_id % len(patterns)]
        rows.extend(_engine_rows(rng, engine_id, observed_cycles, lifetime, pattern))
    return pd.DataFrame(rows, columns=RAW_COLUMNS), final_rul


def _engine_rows(
    rng: np.random.Generator,
    engine_id: int,
    observed_cycles: int,
    lifetime: int,
    pattern: str,
) -> list[list[float]]:
    baseline = rng.normal(0.0, 0.04, size=len(SENSOR_COLS))
    rows = []
    for cycle in range(1, observed_cycles + 1):
        pct = cycle / lifetime
        if pattern == "fast-burn":
            degradation = np.clip((pct - 0.35) / 0.65, 0.0, 1.0) ** 1.6
        elif pattern == "slow-decay":
            degradation = np.clip((pct - 0.15) / 0.85, 0.0, 1.0) ** 0.9
        else:
            degradation = np.clip((pct - 0.72) / 0.28, 0.0, 1.0) ** 2.4

        settings = [
            0.45 + 0.05 * np.sin(cycle / 25) + rng.normal(0, 0.01),
            0.60 + 0.04 * np.cos(cycle / 19) + rng.normal(0, 0.01),
            1.00 + rng.normal(0, 0.002),
        ]
        sensors = _sensor_values(rng, baseline, degradation, pct, pattern)
        rows.append([engine_id, cycle, *settings, *sensors])
    return rows


def _sensor_values(
    rng: np.random.Generator,
    baseline: np.ndarray,
    degradation: float,
    pct: float,
    pattern: str,
) -> list[float]:
    noise = rng.normal(0.0, 0.025, size=len(SENSOR_COLS))
    sensors = 0.5 + baseline + noise
    driver_scale = 1.25 if pattern == "fast-burn" else 1.0
    sensors[1] += 0.10 * np.sin(pct * 8) + degradation * 0.25
    sensors[2] -= degradation * 0.18
    sensors[6] += degradation * 0.55 * driver_scale
    sensors[7] += degradation * 0.36
    sensors[10] -= degradation * 0.24
    sensors[11] += degradation * 0.44
    sensors[14] += max(0.0, pct - 0.6) * 0.22
    sensors[19] -= max(0.0, pct - 0.7) * 0.28
    sensors[0] = 0.7
    sensors[4] = 0.4
    sensors[9] = 0.9
    return sensors.round(6).tolist()


def _write_cmapss(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=" ", index=False, header=False, float_format="%.6f")


if __name__ == "__main__":
    print(generate_demo_cmapss())
