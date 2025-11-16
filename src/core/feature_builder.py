from __future__ import annotations
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD


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


FEATURE_COLS: list[str] = [
    "return_1h", "return_4h", "return_24h",
    "volatility_24h", "ma_10", "ma_50",
    "rsi_14", "macd", "macd_signal",
    "vol_ma_20", "vol_zscore_20",
    "atr_14", "bb_width_20",
    "trend_10_50",
    "hour_sin", "hour_cos",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize to have a 'time' column
    if "time" in df.columns:
        pass
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    elif df.index.name == "time":
        df = df.reset_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "time"})
    else:
        raise ValueError("compute_features expects a 'time' column or datetime index")

    df = df.sort_values("time").reset_index(drop=True)
    df.set_index("time", inplace=True)

    df["return_1h"] = df["close"].pct_change(1)
    df["return_4h"] = df["close"].pct_change(4)
    df["return_24h"] = df["close"].pct_change(24)

    df["volatility_24h"] = df["return_1h"].rolling(24).std()

    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()
    df["vol_zscore_20"] = (df["volume"] - df["vol_ma_20"]) / (vol_std_20 + 1e-9)

    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_width_20"] = (upper - lower) / (ma20 + 1e-9)

    df["trend_10_50"] = (df["ma_10"] > df["ma_50"]).astype(float)

    hours = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    df.reset_index(inplace=True)
    return df


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = compute_features(df)
    feats = feats.dropna(subset=FEATURE_COLS)
    return feats[FEATURE_COLS]
