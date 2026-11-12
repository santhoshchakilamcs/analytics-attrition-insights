"""Shared data loading for the people-analytics attrition project."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Columns excluded from modelling: identifiers, constants, or leakage-adjacent.
# EmployeeNumber is an ID; EmployeeCount, StandardHours and Over18 are constant
# across all 1,470 rows. Attrition is the target.
DROP_COLS = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
TARGET = "Attrition"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the IBM HR Analytics Employee Attrition CSV and verify its shape."""
    df = pd.read_csv(path)
    if df.shape != (1470, 35):
        raise ValueError(f"Unexpected dataset shape {df.shape}; expected (1470, 35)")
    if TARGET not in df.columns:
        raise ValueError("Attrition column missing from dataset")
    return df
