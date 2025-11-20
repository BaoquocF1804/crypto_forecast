from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
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


from sqlalchemy import create_engine

def get_db_engine():
    load_dotenv()
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    
    uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(uri)


def load_recent_candles_from_db(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
    """Load recent OHLCV candles from DB for the configured symbol/timeframe.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume (ascending by time)
    """
    engine = get_db_engine()
    query = (
        "SELECT time, open, high, low, close, volume FROM candles "
        "WHERE symbol = %(symbol)s AND timeframe = %(timeframe)s "
        "ORDER BY time DESC LIMIT %(limit)s"
    )
    
    with engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={"symbol": symbol, "timeframe": timeframe, "limit": limit},
        )
        
    if df.empty:
        raise ValueError(f"No candles returned from DB for {symbol} ({timeframe})")
    # rename and sort ascending
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_model(symbol: str = "BTC/USDT", timeframe: str = "1h", target_return: float = 0.01) -> XGBClassifier:
    # Construct model path based on symbol and timeframe
    coin = symbol.split("/")[0].lower()
    
    # Format target return for filename
    target_return_str = f"{int(target_return*100)}pct" if (target_return*100).is_integer() else f"{target_return*100}pct"
    
    model_filename = f"xgb_{coin}_{timeframe}_{target_return_str}_model.json"
    model_path = MODELS_DIR / model_filename
    
    if not model_path.exists():
        # Try fallback to old naming convention if 1h and target is 1% (migration support)
        if timeframe == '1h' and target_return == 0.01:
             old_path = MODELS_DIR / f"xgb_{coin}_model.json"
             if old_path.exists():
                 clf = XGBClassifier()
                 clf.load_model(str(old_path))
                 return clf
        
        # Try fallback to standard model without target_return suffix (backward compatibility)
        fallback_path = MODELS_DIR / f"xgb_{coin}_{timeframe}_model.json"
        if fallback_path.exists() and target_return == 0.01:
             clf = XGBClassifier()
             clf.load_model(str(fallback_path))
             return clf

        raise FileNotFoundError(f"Model not found for {symbol} {timeframe} {target_return_str} at {model_path}")

    clf = XGBClassifier()
    clf.load_model(str(model_path))
    return clf


def load_threshold(symbol: str = "BTC/USDT", timeframe: str = "1h", target_return: float = 0.01) -> float:
    # Threshold might not be strictly needed for multi-class argmax, but keeping for compatibility
    # or if we want to implement custom thresholds for Buy/Sell classes later.
    return 0.5


def predict_from_df(df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1h", target_return: float = 0.01) -> dict:
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
        model = get_model(symbol, timeframe, target_return)
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
        
        # Predict probabilities
        # For multi-class (3 classes), predict_proba returns (N, 3)
        proba = model.predict_proba(X)
        latest_probs = proba[-1] # [p_neutral, p_buy, p_sell]
        
        # Assuming class mapping: 0=Neutral, 1=Buy, 2=Sell
        # Check if model has 3 classes
        if len(latest_probs) == 3:
            p_neutral, p_buy, p_sell = latest_probs
            label = int(np.argmax(latest_probs))
            # Use the probability of the predicted class as 'last_proba' for backward compatibility/generic usage
            last_proba = float(latest_probs[label])
        else:
            # Fallback for binary model (if loaded old model)
            p_buy = float(latest_probs[1])
            p_sell = 0.0
            p_neutral = float(latest_probs[0])
            label = int(p_buy > 0.5)
            last_proba = p_buy
            
        th = 0.5 # Default threshold for argmax logic
        
    except FileNotFoundError:
        # Model not found, return neutral/unknown
        last_proba = 0.0
        p_buy = 0.0
        p_sell = 0.0
        label = -1 # Unknown
        th = 0.0
    
    return {
        "timestamp": str(latest_time),
        "price": current_price,
        "rsi": rsi,
        "macd": macd,
        "vol_zscore": vol_z,
        "last_proba": last_proba,
        "buy_proba": float(p_buy),
        "sell_proba": float(p_sell),
        "label": label,
        "threshold": th,
        "n_samples": int(len(X)),
        "target_return": target_return
    }


def analyze_risk(df: pd.DataFrame, symbol: str = "BTC/USDT", timeframe: str = "1h", targets: list[float] = [0.01, 0.02, 0.05]) -> list[dict]:
    """
    Run predictions for multiple target returns to analyze risk/reward profile.
    Returns a list of dicts with target_return, buy_proba, sell_proba.
    """
    results = []
    
    # Compute features once
    feats = compute_features(df)
    feats = feats.dropna(subset=FEATURE_COLS)
    
    if feats.empty:
        return []

    X = feats[FEATURE_COLS]
    
    for target in targets:
        try:
            model = get_model(symbol, timeframe, target)
            
            # Align columns
            try:
                booster = model.get_booster()
                model_feats = getattr(booster, "feature_names", None)
            except Exception:
                model_feats = None
                
            X_target = X
            if model_feats:
                missing = [c for c in model_feats if c not in X.columns]
                if missing:
                    continue # Skip this target if features missing
                X_target = X[model_feats]
            
            # Predict
            proba = model.predict_proba(X_target)
            latest_probs = proba[-1]
            
            if len(latest_probs) == 3:
                p_buy = float(latest_probs[1])
                p_sell = float(latest_probs[2])
            else:
                p_buy = float(latest_probs[1])
                p_sell = 0.0
                
            results.append({
                "target_return": target,
                "buy_proba": p_buy,
                "sell_proba": p_sell
            })
            
        except FileNotFoundError:
            continue # Skip if model not found for this target
            
    return results


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
    
    label = result['label']
    buy_prob = result.get('buy_proba', 0.0)
    sell_prob = result.get('sell_proba', 0.0)
    
    print(f"🎯 Signal: {label} (Buy: {buy_prob:.1%}, Sell: {sell_prob:.1%})")
    
    if label == 1:
        print(f"🟢 SIGNAL     : BUY (Prob: {buy_prob:.1%})")
    elif label == 2:
        print(f"🔴 SIGNAL     : SELL (Prob: {sell_prob:.1%})")
    else:
        print(f"⚪ SIGNAL     : NEUTRAL")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
