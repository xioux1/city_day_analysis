from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def dataset_summary(df: pd.DataFrame, target_column: str) -> dict:
    missing = df.isna().sum().to_dict()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values": {k: int(v) for k, v in missing.items() if v > 0},
    }
