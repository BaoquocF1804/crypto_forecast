from __future__ import annotations
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
import os
import psycopg2
from dotenv import load_dotenv

from src.core.data_loader import load_ohlcv_csv
from src.core.feature_builder import build_model_features, compute_features, FEATURE_COLS
from src.core.config import settings

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = Path(settings.model_path)
THRESHOLD_FILE = Path(settings.threshold_path) if getattr(settings, "threshold_path", None) else (MODELS_DIR / "threshold_btc_1h.txt")

_model: XGBClassifier | None = None


def get_db_connection():
    load_dotenv()
    db_settings = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS"),
    }
    # psycopg2 will raise if any required field missing
    return psycopg2.connect(**db_settings)


def load_recent_candles_from_db(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
    """Load recent OHLCV candles from DB for the configured symbol/timeframe.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume (ascending by time)
    """
    conn = None
    try:
        conn = get_db_connection()
        query = (
            "SELECT time, open, high, low, close, volume FROM candles "
            "WHERE symbol = %s AND timeframe = %s "
            "ORDER BY time DESC LIMIT %s"
        )
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, timeframe, limit),
        )
        if df.empty:
            raise ValueError(f"No candles returned from DB for {symbol} ({timeframe})")
        # rename and sort ascending
        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    finally:
        if conn is not None:
            conn.close()


def get_model(symbol: str = "BTC/USDT", timeframe: str = "1h") -> XGBClassifier:
    # Construct model path based on symbol and timeframe
    coin = symbol.split("/")[0].lower()
    model_filename = f"xgb_{coin}_{timeframe}_model.json"
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        # Try fallback to old naming convention if 1h (migration support)
        if timeframe == '1h':
             old_path = MODELS_DIR / f"xgb_{coin}_model.json"
             if old_path.exists():
                 clf = XGBClassifier()
                 clf.load_model(str(old_path))
                 return clf
        
        raise FileNotFoundError(f"Model not found for {symbol} {timeframe} at {model_path}")

    clf = XGBClassifier()
    clf.load_model(str(model_path))
    return clf


def load_threshold(symbol: str = "BTC/USDT", timeframe: str = "1h") -> float:
    coin = symbol.split("/")[0].lower()
    threshold_filename = f"threshold_{coin}_{timeframe}.txt"
    threshold_path = MODELS_DIR / threshold_filename
    
    try:
        if threshold_path.exists():
            txt = threshold_path.read_text().strip()
            return float(txt)
        
        # Fallback for 1h old naming
        if timeframe == '1h':
             old_path = MODELS_DIR / f"threshold_{coin}_1h.txt"
             if old_path.exists():
                 return float(old_path.read_text().strip())
                 
    except Exception:
        pass
    # Default fallback
    return 0.5


def predict_from_df(df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1h") -> dict:
    # Use compute_features directly to access 'time' column
    feats = compute_features(df)
    feats = feats.dropna(subset=FEATURE_COLS)
    
    if feats.empty:
        raise ValueError("Not enough data to compute features for prediction")
    
    # Get latest data for report
    latest_row = feats.iloc[-1]
    latest_time = latest_row["time"]
    
    # Extract key indicators if available
    rsi = float(latest_row.get("rsi_14", 0.0))
    macd = float(latest_row.get("macd", 0.0))
    vol_z = float(latest_row.get("vol_zscore_20", 0.0))
    
    current_price = float(latest_row["close"]) if "close" in latest_row else 0.0

    X = feats[FEATURE_COLS]

    try:
        model = get_model(symbol, timeframe)
        # Align columns to model's expected feature names if available
        try:
            booster = model.get_booster()
            model_feats = getattr(booster, "feature_names", None)
        except Exception:
            model_feats = None
        if model_feats:
            missing = [c for c in model_feats if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required features for the loaded model: {missing}"
                )
            X = X[model_feats]
        proba = model.predict_proba(X)[:, 1]
        last_proba = float(proba[-1])
        th = load_threshold(symbol, timeframe)
        label = int(last_proba > th)
    except FileNotFoundError:
        # Model not found, return neutral/unknown
        last_proba = 0.0
        label = -1 # Unknown
        th = 0.0
    
    return {
        "timestamp": str(latest_time),
        "price": current_price,
        "rsi": rsi,
        "macd": macd,
        "vol_zscore": vol_z,
        "last_proba": last_proba,
        "label": label,
        "threshold": th,
        "n_samples": int(len(X)),
    }


def main() -> None:
    try:
        df = load_ohlcv_csv()
        source = "CSV"
    except FileNotFoundError:
        df = load_recent_candles_from_db()
        source = "Database"
        
    result = predict_from_df(df)
    
    # Print formatted report
    print("\n" + "="*40)
    print(f"🚀 CRYPTO FORECAST REPORT ({source})")
    print("="*40)
    print(f"🕒 Time      : {result['timestamp']}")
    print(f"💰 Price     : {result['price']:.2f}")
    print("-" * 40)
    print(f"📊 RSI (14)  : {result['rsi']:.2f}")
    print(f"📈 MACD      : {result['macd']:.4f}")
    print(f"📉 Vol ZScore: {result['vol_zscore']:.2f}")
    print("-" * 40)
    print(f"🎯 Probability: {result['last_proba']:.1%} (Threshold: {result['threshold']:.1%})")
    
    if result['label'] == 1:
        print(f"🟢 SIGNAL     : BUY (Strong Signal)")
    else:
        print(f"🔴 SIGNAL     : WAIT / NEUTRAL")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
