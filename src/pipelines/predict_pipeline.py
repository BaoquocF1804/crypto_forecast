from __future__ import annotations
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
import os
import psycopg2
from dotenv import load_dotenv

from src.core.data_loader import load_ohlcv_csv
from src.core.feature_builder import build_model_features
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


def load_recent_candles_from_db(limit: int = 300) -> pd.DataFrame:
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
            params=(settings.symbol, settings.timeframe, limit),
        )
        if df.empty:
            raise ValueError("No candles returned from DB")
        # rename and sort ascending
        df = df.rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    finally:
        if conn is not None:
            conn.close()


def get_model() -> XGBClassifier:
    global _model
    if _model is None:
        clf = XGBClassifier()
        clf.load_model(str(MODEL_PATH))
        _model = clf
    return _model


def load_threshold() -> float:
    try:
        if THRESHOLD_FILE.exists():
            txt = THRESHOLD_FILE.read_text().strip()
            return float(txt)
    except Exception:
        pass
    return float(getattr(settings, "threshold", 0.5))


def predict_from_df(df: pd.DataFrame) -> dict:
    X = build_model_features(df)
    if X.empty:
        raise ValueError("Not enough data to compute features for prediction")
    model = get_model()
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
    th = load_threshold()
    label = int(last_proba > th)
    return {
        "last_proba": last_proba,
        "label": label,
        "threshold": th,
        "n_samples": int(len(X)),
    }


def main() -> None:
    try:
        df = load_ohlcv_csv()
    except FileNotFoundError:
        df = load_recent_candles_from_db()
    result = predict_from_df(df)
    print(result)


if __name__ == "__main__":
    main()
