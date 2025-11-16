import pandas as pd
import numpy as np
import psycopg2
import os
import logging
from dotenv import load_dotenv

# Import các thư viện tính toán và ML
import ta
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

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
        # Dùng read_sql_query để tải trực tiếp vào DataFrame
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, timeframe),
            index_col='time' # Set cột 'time' làm index
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
    """Tạo features và labels từ DataFrame OHLCV."""
    logging.info("Đang tính toán features và labels...")
    
    # 1. Tạo Features (giống logic ban đầu của bạn)
    df['return_1h'] = df['close'].pct_change()
    df['return_4h'] = df['close'].pct_change(4)
    df['return_24h'] = df['close'].pct_change(24)
    df['volatility_24h'] = df['return_1h'].rolling(24).std()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # 2. Tạo Label
    # Dự đoán: Giá có tăng > 1% trong 4 giờ tới không?
    future_close = df['close'].shift(-4)
    future_return_4h = (future_close - df['close']) / df['close']
    df['label'] = (future_return_4h > 0.01).astype(int)
    
    # 3. Xóa các dòng NaN (do rolling và shifting)
    df.dropna(inplace=True)
    
    logging.info("Hoàn tất tính toán features và labels.")
    
    # 4. Tách X (features) và y (label)
    feature_cols = [
        'return_1h', 'return_4h', 'return_24h',
        'volatility_24h', 'ma_10', 'ma_50',
        'rsi_14', 'macd', 'macd_signal'
    ]
    
    X = df[feature_cols]
    y = df['label']
    
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

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Huấn luyện mô hình XGBoost và đánh giá."""
    logging.info("Bắt đầu huấn luyện mô hình XGBoost...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1 # Sử dụng tất cả các core
    )
    
    model.fit(X_train, y_train)
    logging.info("Huấn luyện hoàn tất.")
    
    # Đánh giá trên tập test
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    logging.info("--- Kết quả Đánh giá trên Tập Test ---")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"AUC Score: {auc_score:.4f}")
    
    # (Tùy chọn) Hiển thị feature importances
    # ...
    
    return model

def save_model(model, path):
    """Lưu mô hình đã huấn luyện."""
    try:
        # Đảm bảo thư mục 'models/' tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model.save_model(path)
        logging.info(f"Mô hình đã được lưu tại: {path}")
    except Exception as e:
        logging.error(f"Lỗi khi lưu mô hình: {e}")

def main():
    """Hàm chính điều phối toàn bộ pipeline."""
    conn = None
    try:
        # 1. Kết nối DB
        conn = get_db_connection()
        if conn is None:
            return

        # 2. Tải dữ liệu
        df = load_data_from_db(conn, SYMBOL, TIMEFRAME)
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
        model = train_and_evaluate(X_train, y_train, X_test, y_test)

        # 6. Lưu mô hình
        save_model(model, MODEL_SAVE_PATH)

    finally:
        # Đảm bảo đóng kết nối DB dù có lỗi hay không
        if conn:
            conn.close()
            logging.info("Đã đóng kết nối database.")

if __name__ == "__main__":
    main()