from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd

from src.pipelines.predict_pipeline import predict_from_df, load_recent_candles_from_db
from src.core.data_loader import load_ohlcv_csv

app = FastAPI(title="Crypto Forecast API")

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


class PredictResponse(BaseModel):
    last_proba: float
    label: int
    threshold: float
    n_samples: int
    last_timestamp: Optional[datetime] = None


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
            df = load_recent_candles_from_db()
            last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None

        result = predict_from_df(df)
        return PredictResponse(**result, last_timestamp=last_ts)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict", response_model=PredictResponse)
async def predict_default():
    try:
        df = load_ohlcv_csv()
    except FileNotFoundError:
        df = load_recent_candles_from_db()
    last_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").iloc[-1] if "timestamp" in df.columns else None
    result = predict_from_df(df)
    return PredictResponse(**result, last_timestamp=last_ts)
