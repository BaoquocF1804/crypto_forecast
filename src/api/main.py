from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd

from src.pipelines.predict_pipeline import predict_from_df, load_recent_candles_from_db
from src.core.data_loader import load_ohlcv_csv

app = FastAPI(title="Crypto Forecast API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok"}


class Candle(BaseModel):
    timestamp: Optional[datetime] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None


class PredictRequest(BaseModel):
    candles: Optional[List[Candle]] = None
    filename: Optional[str] = None
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"


class PredictResponse(BaseModel):
    last_proba: float
    label: int
    threshold: float
    n_samples: int
    last_timestamp: Optional[datetime] = None
    price: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    vol_zscore: Optional[float] = None


@app.get("/history")
def get_history(limit: int = 100, symbol: str = "BTC/USDT", timeframe: str = "1h"):
    """Get recent candles for charting."""
    try:
        df = load_recent_candles_from_db(symbol=symbol, timeframe=timeframe, limit=limit)
        # Convert to list of dicts
        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        last_ts = None
        if req.candles:
            rows = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in req.candles]
            df = pd.DataFrame(rows)
            if "timestamp" in df.columns and not df["timestamp"].isna().all():
                df = df.sort_values("timestamp")
                last_ts = df["timestamp"].iloc[-1]
        elif req.filename:
            df = load_ohlcv_csv(req.filename)
            if "timestamp" in df.columns:
                last_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").iloc[-1]
        else:
            # Fallback: load recent candles from DB
            df = load_recent_candles_from_db(symbol=req.symbol, timeframe=req.timeframe)
            last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None

        result = predict_from_df(df, symbol=req.symbol, timeframe=req.timeframe)
        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict", response_model=PredictResponse)
async def predict_default(symbol: str = "BTC/USDT", timeframe: str = "1h"):
    try:
        df = load_ohlcv_csv()
    except FileNotFoundError:
        df = load_recent_candles_from_db(symbol=symbol, timeframe=timeframe)
    
    result = predict_from_df(df, symbol=symbol, timeframe=timeframe)
    return PredictResponse(**result)
