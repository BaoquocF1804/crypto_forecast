from __future__ import annotations
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator


def add_basic_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few basic TA features. Requires columns: close."""
    out = df.copy()
    if "close" not in out.columns:
        raise ValueError("DataFrame must contain 'close' column")
    out["rsi_14"] = RSIIndicator(close=out["close"], window=14).rsi()
    out["sma_20"] = SMAIndicator(close=out["close"], window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close=out["close"], window=50).sma_indicator()
    out["ret_1"] = out["close"].pct_change()
    out = out.dropna().reset_index(drop=True)
    return out
