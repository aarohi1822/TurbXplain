from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Download NASA C-MAPSS files from https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq")
    print(f"Place train_FD001.txt, test_FD001.txt, and RUL_FD001.txt under {RAW_DIR}")


if __name__ == "__main__":
    main()
