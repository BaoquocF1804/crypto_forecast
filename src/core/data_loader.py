from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_ohlcv_csv(filename: str | None = None) -> pd.DataFrame:
    """
    Load OHLCV data from data/processed or data/raw.
    If filename is None, tries common default: btc_usdt_1h.csv
    Columns expected: timestamp, open, high, low, close, volume
    """
    candidates: list[Path] = []
    if filename is None:
        filename = "btc_usdt_1h.csv"
    for sub in ["processed", "raw"]:
        candidates.append(DATA_DIR / sub / filename)
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.sort_values("timestamp").reset_index(drop=True)
            return df
    raise FileNotFoundError(f"Could not find {filename} under data/processed or data/raw")
