from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.core.data_loader import load_ohlcv_csv
from src.core.feature_builder import add_basic_ta

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "xgb_btc_model.json"


def make_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["target"] = df["close"].pct_change().shift(-1)  # predict next return
    df = df.dropna().reset_index(drop=True)
    features = df.drop(columns=["target"])  # keep all features incl. OHLCV
    target = df["target"]
    # Select numeric features only
    X = features.select_dtypes(include=["number"]).copy()
    return X, target


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ohlcv_csv()
    df = add_basic_ta(df)
    X, y = make_supervised(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print({"rmse": float(rmse)})

    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
