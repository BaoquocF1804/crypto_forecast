import ccxt
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import time
import os
import logging
from dotenv import load_dotenv
import ta
import numpy as np

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_worker.log"),
        logging.StreamHandler()
    ]
)

# --- Tải biến môi trường từ file .env ---
load_dotenv()

# --- Cấu hình Worker ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
TIMEFRAMES = ['1h', '4h', '1d']
SLEEP_INTERVAL = 60  # Giây
FEATURE_LOOKBACK_PERIOD = 100 

# --- Cấu hình Database (lấy từ .env) ---
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

def get_last_timestamp(conn, symbol, timeframe):
    """Lấy timestamp (dạng millisecond) của nến cuối cùng trong DB."""
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT time FROM candles
                WHERE symbol = %s AND timeframe = %s
                ORDER BY time DESC
                LIMIT 1
                """,
                (symbol, timeframe)
            )
            result = cursor.fetchone()
            if result:
                last_time_dt = result[0]
                last_timestamp_ms = int(last_time_dt.timestamp() * 1000)
                logging.info(f"Timestamp cuối cùng trong DB: {last_time_dt} ({last_timestamp_ms})")
                return last_timestamp_ms
            else:
                logging.info("Không có dữ liệu trong DB, sẽ tải từ đầu.")
                return None
    except Exception as e:
        logging.error(f"Lỗi khi lấy last_timestamp: {e}")
        return None

def fetch_new_candles(exchange, symbol, timeframe, since_timestamp):
    """Tải nến mới từ sàn bằng CCXT."""
    try:
        limit = 1000
        logging.info(f"Đang tải nến mới cho {symbol} từ {since_timestamp}...")
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=since_timestamp,
            limit=limit
        )
        if not ohlcv:
            logging.info("Không có nến mới.")
            return None
        logging.info(f"Đã tải được {len(ohlcv)} nến mới.")
        return ohlcv
    except Exception as e:
        logging.error(f"Lỗi khi fetch_ohlcv: {e}")
        return None

def process_and_insert_candles(conn, ohlcv_data, symbol, timeframe):
    """
    Chuyển đổi dữ liệu, INSERT vào 'candles', và
    TRẢ VỀ (return) một DataFrame của các nến mới.
    """
    
    # 1. Chuyển đổi sang DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        return None

    # 2. Chuyển đổi timestamp (ms) sang datetime UTC và set làm index
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    
    # 3. Thêm các cột metadata
    df['symbol'] = symbol
    df['timeframe'] = timeframe
    
    # 4. Chuẩn bị dữ liệu để insert (dạng list of tuples)
    # Cần reset_index để 'time' trở thành một cột cho việc insert
    df_for_tuples = df.reset_index()
    insert_data = [
        tuple(row) for row in df_for_tuples[
            ['time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
        ].itertuples(index=False)
    ]
    
    # 5. Insert hàng loạt (Bulk Insert)
    try:
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO candles (time, symbol, timeframe, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (time, symbol, timeframe) DO NOTHING
                """,
                insert_data
            )
        conn.commit()
        logging.info(f"Đã insert/update thành công {len(insert_data)} nến.")
        
        # Trả về DataFrame (đã có index 'time') để tính feature
        return df 
        
    except Exception as e:
        conn.rollback() 
        logging.error(f"Lỗi khi insert vào 'candles': {e}")
        return None

def calculate_and_save_features(conn, new_candles_df, symbol, timeframe):
    """
    Tính toán features cho các nến mới, dựa trên dữ liệu lịch sử
    trong DB, và lưu vào bảng 'features'.
    """
    try:
        # 1. Lấy timestamp đầu tiên của nến mới
        first_new_time = new_candles_df.index.min()

        # 2. Query DB để lấy dữ liệu lịch sử (chỉ cần cột 'close')
        # Chúng ta cần 'FEATURE_LOOKBACK_PERIOD' nến TRƯỚC nến mới đầu tiên
        query = """
        SELECT time, close FROM candles
        WHERE symbol = %s AND timeframe = %s AND time < %s
        ORDER BY time DESC
        LIMIT %s
        """
        params = (symbol, timeframe, first_new_time, FEATURE_LOOKBACK_PERIOD)
        
        hist_df = pd.read_sql_query(query, conn, params=params, index_col='time')
        # Sắp xếp lại theo thứ tự thời gian tăng dần
        hist_df.sort_index(ascending=True, inplace=True)

        # 3. Kết hợp dữ liệu lịch sử và dữ liệu mới
        # Chỉ lấy cột 'close' từ nến mới để tính toán
        new_close_df = new_candles_df[['close']]
        
        # full_close_df chứa (lịch sử + nến mới) để tính toán chính xác
        full_close_df = pd.concat([hist_df, new_close_df])

        # 4. Tính toán tất cả features trên DataFrame kết hợp
        features_df = pd.DataFrame(index=full_close_df.index)
        close = full_close_df['close']
        
        features_df['return_1h'] = close.pct_change()
        features_df['return_4h'] = close.pct_change(4)
        features_df['return_24h'] = close.pct_change(24)
        features_df['volatility_24h'] = features_df['return_1h'].rolling(24).std()
        features_df['ma_10'] = close.rolling(10).mean()
        features_df['ma_50'] = close.rolling(50).mean()
        features_df['rsi_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd = ta.trend.MACD(close=close)
        features_df['macd'] = macd.macd()
        features_df['macd_signal'] = macd.macd_signal()
        
        # 5. **QUAN TRỌNG**: Lọc để chỉ lấy features của các nến MỚI
        new_features_df = features_df.loc[new_candles_df.index]
        
        # 6. Chuẩn bị dữ liệu để insert vào bảng 'features'
        new_features_df['symbol'] = symbol
        new_features_df['timeframe'] = timeframe
        
        # Xử lý giá trị Inf (nếu có)
        new_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Reset index để 'time' thành cột
        df_for_insert = new_features_df.reset_index()
        
        # Lấy danh sách cột
        feature_cols = [
            'time', 'symbol', 'timeframe', 'return_1h', 'return_4h', 'return_24h',
            'volatility_24h', 'ma_10', 'ma_50', 'rsi_14', 'macd', 'macd_signal'
        ]
        
        # Chỉ lấy các cột có trong DB và fillna(None)
        df_for_insert = df_for_insert[feature_cols].where(pd.notnull(df_for_insert), None)

        insert_data = [tuple(row) for row in df_for_insert.itertuples(index=False, name=None)]
        
        # 7. Insert vào DB
        if not insert_data:
            logging.info("Không có feature mới để lưu.")
            return

        with conn.cursor() as cursor:
            cols_str = ", ".join(feature_cols)
            insert_query = f"""
            INSERT INTO features ({cols_str})
            VALUES %s
            ON CONFLICT (time, symbol, timeframe) DO NOTHING
            """
            execute_values(cursor, insert_query, insert_data)
            
        conn.commit()
        logging.info(f"Đã tính toán và lưu {len(insert_data)} hàng features mới.")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Lỗi khi tính toán/lưu features: {e}")

def main_worker_loop():
    """Vòng lặp chính của worker."""
    
    conn = get_db_connection()
    if not conn:
        logging.error("Không thể kết nối DB, worker dừng lại.")
        return

    exchange = ccxt.binance()
    logging.info(f"Worker bắt đầu chạy cho {SYMBOLS} với các timeframe {TIMEFRAMES}...")

    while True:
        try:
            for symbol in SYMBOLS:
                for timeframe in TIMEFRAMES:
                    try:
                        logging.info(f"--- Xử lý {symbol} ({timeframe}) ---")
                        # 1. Kiểm tra nến cuối cùng
                        last_ts = get_last_timestamp(conn, symbol, timeframe)
                        since_ts = (last_ts + 1) if last_ts else None 
                        
                        # 2. Tải nến mới
                        new_ohlcv = fetch_new_candles(exchange, symbol, timeframe, since_ts)
                        
                        if new_ohlcv:
                            # 3. Lưu nến MỚI, và lấy về DataFrame của chúng
                            new_candles_df = process_and_insert_candles(
                                conn, new_ohlcv, symbol, timeframe
                            )
                            
                            # 4. TÍCH HỢP: Nếu lưu nến thành công, tính feature
                            if new_candles_df is not None and not new_candles_df.empty:
                                calculate_and_save_features(
                                    conn, new_candles_df, symbol, timeframe
                                )
                        
                        # Nghỉ ngắn giữa các request
                        time.sleep(1) 
                        
                    except Exception as e:
                        logging.error(f"Lỗi khi xử lý {symbol} - {timeframe}: {e}")
                        continue
            
            # 5. Chờ sau khi hết vòng lặp
            logging.info(f"Hoàn thành vòng lặp. Nghỉ {SLEEP_INTERVAL} giây.")
            time.sleep(SLEEP_INTERVAL)

        except psycopg2.OperationalError as db_e:
            logging.error(f"Mất kết nối DB: {db_e}. Thử kết nối lại...")
            conn.close()
            time.sleep(10) 
            conn = get_db_connection()
            if not conn:
                logging.error("Kết nối lại thất bại. Worker dừng.")
                break
        
        except KeyboardInterrupt:
            logging.info("Worker dừng bởi người dùng.")
            break
            
        except Exception as e:
            logging.error(f"Lỗi không xác định trong vòng lặp chính: {e}")
            time.sleep(SLEEP_INTERVAL) 

    if conn:
        conn.close()
    logging.info("Worker đã đóng kết nối DB và dừng hẳn.")

if __name__ == "__main__":
    main_worker_loop()