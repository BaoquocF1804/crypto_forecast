import pandas as pd
import numpy as np
import psycopg2
import os
import logging
from dotenv import load_dotenv
try:
    from sqlalchemy import create_engine  # type: ignore
    _SQLA = True
except Exception:
    create_engine = None  # type: ignore
    _SQLA = False

# Import các thư viện tính toán và ML
import ta
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from src.core.feature_builder import compute_features, FEATURE_COLS

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Tải biến môi trường (chứa credentials DB) ---
# Đảm bảo file .env ở thư mục gốc của dự án
load_dotenv() 

# --- Cấu hình Pipeline ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
MODEL_SAVE_PATH = "models/xgb_btc_model.json" # Đường dẫn lưu model
THRESHOLD_SAVE_PATH = os.getenv("THRESHOLD_PATH", "models/threshold_btc_1h.txt")

# Cấu hình Database
DB_SETTINGS = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS")
}

def get_db_connection():
    """Tạo và trả về một kết nối database mới."""
    try:
        conn = psycopg2.connect(**DB_SETTINGS)
        logging.info("Kết nối database thành công.")
        return conn
    except Exception as e:
        logging.error(f"Lỗi khi kết nối database: {e}")
        return None

def load_data_from_db(conn, symbol, timeframe):
    """Tải toàn bộ dữ liệu nến từ DB bằng pandas."""
    logging.info(f"Đang tải dữ liệu cho {symbol} ({timeframe}) từ DB...")
    
    # Sắp xếp theo 'time' ASC để đảm bảo đúng thứ tự cho time series
    query = """
    SELECT * FROM candles
    WHERE symbol = %s AND timeframe = %s
    ORDER BY time ASC
    """
    
    try:
        if _SQLA and create_engine is not None:
            user = DB_SETTINGS.get("user") or ""
            pwd = DB_SETTINGS.get("password") or ""
            host = DB_SETTINGS.get("host") or "localhost"
            port = DB_SETTINGS.get("port") or "5432"
            dbname = DB_SETTINGS.get("dbname") or "postgres"
            uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{dbname}"
            engine = create_engine(uri)
            with engine.connect() as eng_conn:
                df = pd.read_sql_query(
                    query,
                    eng_conn,
                    params=(symbol, timeframe),
                    index_col='time'
                )
        else:
            # Fallback: dùng psycopg2 connection (có cảnh báo của pandas)
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, timeframe),
                index_col='time'
            )
        
        if df.empty:
            logging.warning("Không tìm thấy dữ liệu trong DB.")
            return None
            
        logging.info(f"Tải thành công {len(df)} nến.")
        return df
        
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu từ DB: {e}")
        return None

def create_features_and_labels(df):
    """Tạo features và labels từ DataFrame OHLCV (mở rộng)."""
    logging.info("Đang tính toán features và labels...")
    # Bảo đảm có cột 'time'
    df_in = df.copy()
    if 'time' not in df_in.columns:
        df_in = df_in.reset_index()
    # Tính features mở rộng
    feats = compute_features(df_in)  # có cột 'time' và FEATURE_COLS
    feats = feats.dropna(subset=FEATURE_COLS)
    feats = feats.set_index('time')

    # Tạo label trên chuỗi close gốc (theo 4h tới >1%)
    price_df = df_in.copy()
    price_df['time'] = pd.to_datetime(price_df['time'], utc=True, errors='coerce')
    price_df = price_df.set_index('time').sort_index()
    future_close = price_df['close'].shift(-4)
    future_return_4h = (future_close - price_df['close']) / price_df['close']
    labels = (future_return_4h > 0.01).astype(int).to_frame('label')

    # Căn chỉnh theo time (chỉ các hàng có đủ features)
    merged = feats.join(labels, how='inner').dropna()

    logging.info("Hoàn tất tính toán features và labels.")

    X = merged[FEATURE_COLS]
    y = merged['label']
    return X, y

def split_data(X, y, test_size=0.2):
    """Chia dữ liệu train/test theo thứ tự thời gian."""
    logging.info(f"Đang chia dữ liệu: {1-test_size:.0%} train, {test_size:.0%} test")
    
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    return X_train, y_train, X_test, y_test

def tune_hyperparams(X, y, n_splits=5):
    """
    Dùng TimeSeriesSplit + RandomizedSearchCV để tìm bộ hyperparameter tốt.
    """
    logging.info("Bắt đầu hyperparameter tuning với TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    logging.info(f"Best AUC: {search.best_score_:.4f}")
    logging.info(f"Best params: {search.best_params_}")
    return search.best_params_


def find_best_threshold(y_true, y_proba):
    """
    Quét nhiều threshold để tìm ngưỡng tốt nhất theo F1-score.
    """
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.3, 0.7, 41):  # quét từ 0.3 → 0.7
        y_pred = (y_proba > th).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    logging.info(f"Best threshold: {best_th:.3f} với F1 = {best_f1:.4f}")
    return best_th


def train_and_evaluate(X_train, y_train, X_test, y_test, use_tuning=True):
    """Huấn luyện mô hình XGBoost và đánh giá."""
    if use_tuning:
        best_params = tune_hyperparams(X_train, y_train)
    else:
        best_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        }

    logging.info("Bắt đầu huấn luyện mô hình XGBoost với best_params...")
    logging.info(best_params)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )

    model.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")

    # Đánh giá
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    best_th = find_best_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba > best_th).astype(int)

    logging.info("--- Kết quả Đánh giá trên Tập Test ---")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"AUC Score: {auc_score:.4f}")
    logging.info(f"Threshold sử dụng: {best_th:.3f}")

    return model, best_th

def save_model(model, path):
    """Lưu mô hình đã huấn luyện."""
    try:
        # Đảm bảo thư mục 'models/' tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model.save_model(path)
        logging.info(f"Mô hình đã được lưu tại: {path}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu mô hình: {e}")

import argparse

def main():
    """Hàm chính điều phối toàn bộ pipeline."""
    parser = argparse.ArgumentParser(description="Train crypto forecast model")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to train (e.g. BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe to train (e.g. 1h, 4h, 1d)")
    args = parser.parse_args()
    
    symbol = args.symbol
    timeframe = args.timeframe
    coin = symbol.split("/")[0].lower()
    
    # Dynamic paths
    model_save_path = f"models/xgb_{coin}_{timeframe}_model.json"
    threshold_save_path = f"models/threshold_{coin}_{timeframe}.txt"
    
    logging.info(f"Bắt đầu training cho {symbol} ({timeframe})...")
    logging.info(f"Model sẽ được lưu tại: {model_save_path}")

    conn = None
    try:
        # 1. Kết nối DB
        conn = get_db_connection()
        if conn is None:
            return

        # 2. Tải dữ liệu
        df = load_data_from_db(conn, symbol, timeframe)
        if df is None:
            return

        # 3. Tạo features và labels
        X, y = create_features_and_labels(df)
        if X.empty:
            logging.error("Không đủ dữ liệu để tạo features/labels.")
            return

        # 4. Chia data
        X_train, y_train, X_test, y_test = split_data(X, y)

        # 5. Huấn luyện và Đánh giá
        model, best_th = train_and_evaluate(X_train, y_train, X_test, y_test)

        # 6. Lưu mô hình + threshold
        save_model(model, model_save_path)
        os.makedirs(os.path.dirname(threshold_save_path) or ".", exist_ok=True)
        with open(threshold_save_path, "w") as f:
            f.write(str(best_th))

    finally:
        # Đảm bảo đóng kết nối DB dù có lỗi hay không
        if conn:
            conn.close()
            logging.info("Đã đóng kết nối database.")

if __name__ == "__main__":
    main()