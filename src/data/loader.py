from pathlib import Path
import argparse

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_COLUMNS, RAW_DATA_DIR


def load_cmapss_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns in {path}, found {df.shape[1]}.")
    df.columns = RAW_COLUMNS
    return df


def load_rul_file(path: str | Path) -> pd.Series:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]


def load_dataset(dataset: str, raw_dir: str | Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_dir = Path(raw_dir)
    train = load_cmapss_file(raw_dir / f"train_{dataset}.txt")
    test = load_cmapss_file(raw_dir / f"test_{dataset}.txt")
    rul = load_rul_file(raw_dir / f"RUL_{dataset}.txt")
    return train, test, rul


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and preprocess NASA C-MAPSS files.")
    parser.add_argument("--dataset", default="FD001")
    args = parser.parse_args()
    from src.data.preprocessor import CMAPSSPreprocessor

    preprocessor = CMAPSSPreprocessor(dataset=args.dataset, output_dir=PROCESSED_DATA_DIR)
    stats = preprocessor.run()
    print(stats)


if __name__ == "__main__":
    main()
