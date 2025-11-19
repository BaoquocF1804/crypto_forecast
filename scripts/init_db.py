import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_SETTINGS = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS")
}

def init_db():
    try:
        conn = psycopg2.connect(**DB_SETTINGS)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create candles table
        print("Creating 'candles' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                PRIMARY KEY (time, symbol, timeframe)
            );
        """)
        
        # Create hypertable for timescaledb
        try:
            cursor.execute("SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);")
            print("Converted 'candles' to hypertable.")
        except Exception as e:
            print(f"Warning: Could not convert to hypertable (maybe not TimescaleDB?): {e}")

        # Create features table
        print("Creating 'features' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                return_1h DOUBLE PRECISION,
                return_4h DOUBLE PRECISION,
                return_24h DOUBLE PRECISION,
                volatility_24h DOUBLE PRECISION,
                ma_10 DOUBLE PRECISION,
                ma_50 DOUBLE PRECISION,
                rsi_14 DOUBLE PRECISION,
                macd DOUBLE PRECISION,
                macd_signal DOUBLE PRECISION,
                PRIMARY KEY (time, symbol, timeframe)
            );
        """)
        
        # Create hypertable for features
        try:
            cursor.execute("SELECT create_hypertable('features', 'time', if_not_exists => TRUE);")
            print("Converted 'features' to hypertable.")
        except Exception as e:
            print(f"Warning: Could not convert to hypertable: {e}")

        print("Database initialization completed successfully.")
        conn.close()
        
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_db()
