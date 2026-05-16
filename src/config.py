from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DATASET = "FD001"
RUL_CLIP = 125
SEQUENCE_LENGTH = 30
ROLLING_WINDOWS = (5, 10, 20)
NEAR_CONSTANT_STD = 0.001

ENGINE_COL = "engine_id"
CYCLE_COL = "cycle"
OP_SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{idx}" for idx in range(1, 22)]
RAW_COLUMNS = [ENGINE_COL, CYCLE_COL, *OP_SETTING_COLS, *SENSOR_COLS]
