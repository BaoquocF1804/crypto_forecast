import os
from typing import Optional
try:
    # Pydantic v2
    from pydantic_settings import BaseSettings  # type: ignore
except Exception:
    try:
        # Pydantic v1
        from pydantic import BaseSettings  # type: ignore
    except Exception:
        BaseSettings = None  # type: ignore

if BaseSettings is not None:
    class Settings(BaseSettings):
        ccxt_exchange: str = os.getenv("CCXT_EXCHANGE", "binance")
        ccxt_api_key: Optional[str] = os.getenv("CCXT_API_KEY")
        ccxt_api_secret: Optional[str] = os.getenv("CCXT_API_SECRET")
        symbol: str = os.getenv("SYMBOL", "BTC/USDT")
        timeframe: str = os.getenv("TIMEFRAME", "1h")
        model_path: str = os.getenv("MODEL_PATH", "models/xgb_btc_model.json")
        threshold: float = float(os.getenv("THRESHOLD", "0.5"))
        threshold_path: Optional[str] = os.getenv("THRESHOLD_PATH")
else:
    class Settings:
        def __init__(self) -> None:
            self.ccxt_exchange: str = os.getenv("CCXT_EXCHANGE", "binance")
            self.ccxt_api_key: Optional[str] = os.getenv("CCXT_API_KEY")
            self.ccxt_api_secret: Optional[str] = os.getenv("CCXT_API_SECRET")
            self.symbol: str = os.getenv("SYMBOL", "BTC/USDT")
            self.timeframe: str = os.getenv("TIMEFRAME", "1h")
            self.model_path: str = os.getenv("MODEL_PATH", "models/xgb_btc_model.json")
            self.threshold: float = float(os.getenv("THRESHOLD", "0.5"))
            self.threshold_path: Optional[str] = os.getenv("THRESHOLD_PATH")

settings = Settings()
