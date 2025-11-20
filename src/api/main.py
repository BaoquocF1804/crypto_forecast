from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import subprocess
import threading
import time
import sys
import logging
from contextlib import asynccontextmanager

from src.pipelines.predict_pipeline import predict_from_df, load_recent_candles_from_db, analyze_risk
from src.core.data_loader import load_ohlcv_csv

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables for Background Tasks ---
data_worker_process = None
training_thread = None
stop_training_event = threading.Event()

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XAUT/USDT']
TIMEFRAMES = ['1h', '4h', '1d']
TARGET_RETURNS = [0.01, 0.02, 0.05]

def run_training_job():
    """Chạy training cho tất cả các cặp coin, timeframe và target return."""
    logger.info("⏳ Bắt đầu chu kỳ huấn luyện định kỳ...")
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            for target_return in TARGET_RETURNS:
                try:
                    logger.info(f"🚀 Đang train {symbol} - {timeframe} - Target {target_return:.1%}...")
                    # Chạy training pipeline như một subprocess để tránh block main thread
                    subprocess.run(
                        [sys.executable, "-m", "src.pipelines.train_pipeline", 
                         "--symbol", symbol, 
                         "--timeframe", timeframe,
                         "--target_return", str(target_return)],
                        check=False
                    )
                    logger.info(f"✅ Train xong {symbol} - {timeframe} - Target {target_return:.1%}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Lỗi khi train {symbol} - {timeframe} - Target {target_return:.1%}: {e.stderr.decode()}")
                except Exception as e:
                    logger.error(f"❌ Lỗi không xác định khi train {symbol} - {timeframe}: {e}")
    logger.info("🏁 Hoàn tất chu kỳ huấn luyện.")

def training_scheduler():
    """Luồng chạy ngầm để kích hoạt training mỗi 30 phút."""
    while not stop_training_event.is_set():
        run_training_job()
        # Chờ 30 phút (4800 giây) hoặc cho đến khi có tín hiệu dừng
        if stop_training_event.wait(4800):
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    global data_worker_process, training_thread
    
    # 1. Khởi động Data Worker (Subprocess)
    logger.info("🛠️ Đang khởi động Data Worker...")
    try:
        data_worker_process = subprocess.Popen([sys.executable, "scripts/data_worker.py"])
        logger.info(f"✅ Data Worker đã chạy (PID: {data_worker_process.pid})")
    except Exception as e:
        logger.error(f"❌ Không thể khởi động Data Worker: {e}")

    # 2. Khởi động Training Scheduler (Background Thread)
    logger.info("⏰ Đang khởi động Training Scheduler...")
    stop_training_event.clear()
    training_thread = threading.Thread(target=training_scheduler, daemon=True)
    training_thread.start()
    logger.info("✅ Training Scheduler đã chạy.")

    yield

    # --- SHUTDOWN ---
    logger.info("🛑 Đang tắt server...")
    
    # 1. Tắt Data Worker
    if data_worker_process:
        logger.info("Killing Data Worker...")
        data_worker_process.terminate()
        try:
            data_worker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            data_worker_process.kill()
        logger.info("✅ Data Worker đã tắt.")

    # 2. Tắt Training Scheduler
    logger.info("Stopping Training Scheduler...")
    stop_training_event.set()
    if training_thread:
        training_thread.join(timeout=5)
    logger.info("✅ Training Scheduler đã dừng.")

app = FastAPI(lifespan=lifespan, title="Crypto Forecast API")

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
    target_return: float = 0.01


class RiskItem(BaseModel):
    target_return: float
    buy_proba: float
    sell_proba: float


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
    target_return: Optional[float] = 0.01
    buy_proba: Optional[float] = None
    sell_proba: Optional[float] = None
    risk_analysis: Optional[List[RiskItem]] = None


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

        result = predict_from_df(df, symbol=req.symbol, timeframe=req.timeframe, target_return=req.target_return)
        
        # Add risk analysis
        risk_data = analyze_risk(df, symbol=req.symbol, timeframe=req.timeframe)
        result["risk_analysis"] = risk_data
        
        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict", response_model=PredictResponse)
async def predict_default(symbol: str = "BTC/USDT", timeframe: str = "1h", target_return: float = 0.01):
    try:
        df = load_ohlcv_csv()
    except FileNotFoundError:
        df = load_recent_candles_from_db(symbol=symbol, timeframe=timeframe)
    
    result = predict_from_df(df, symbol=symbol, timeframe=timeframe, target_return=target_return)
    
    # Add risk analysis
    risk_data = analyze_risk(df, symbol=symbol, timeframe=timeframe)
    result["risk_analysis"] = risk_data
    
    return PredictResponse(**result)
