import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    ccxt_exchange: str = os.getenv("CCXT_EXCHANGE", "binance")
    ccxt_api_key: str | None = os.getenv("CCXT_API_KEY")
    ccxt_api_secret: str | None = os.getenv("CCXT_API_SECRET")
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    model_path: str = os.getenv("MODEL_PATH", "models/xgb_btc_model.json")
    threshold: float = float(os.getenv("THRESHOLD", "0.5"))

settings = Settings()
