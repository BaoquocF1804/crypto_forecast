from __future__ import annotations
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor

from src.core.data_loader import load_ohlcv_csv
from src.core.feature_builder import add_basic_ta

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "xgb_btc_model.json"


def main() -> None:
    df = load_ohlcv_csv()
    df = add_basic_ta(df)
    X = df.select_dtypes(include=["number"]).copy()

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    preds = model.predict(X)
    last_pred = float(preds[-1])
    print({"last_pred_return": last_pred})


if __name__ == "__main__":
    main()
