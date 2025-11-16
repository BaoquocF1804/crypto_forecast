from __future__ import annotations
import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv

from src.core.feature_builder import compute_features, FEATURE_COLS

load_dotenv()

DB_SETTINGS = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
}

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")


def main():
    conn = psycopg2.connect(**DB_SETTINGS)
    try:
        candles = pd.read_sql_query(
            """
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = %s AND timeframe = %s
            ORDER BY time ASC
            """,
            conn,
            params=(SYMBOL, TIMEFRAME),
        )

        if candles.empty:
            print("No candles found. Nothing to compute.")
            return

        df_feat = compute_features(candles)
        df_feat["symbol"] = SYMBOL
        df_feat["timeframe"] = TIMEFRAME

        cols = (
            ["time", "symbol", "timeframe"] + FEATURE_COLS
        )
        df_feat = df_feat[cols].dropna()

        rows = [tuple(r) for r in df_feat.itertuples(index=False, name=None)]
        if not rows:
            print("No rows to upsert after dropna().")
            return

        with conn.cursor() as cur:
            insert_cols = ", ".join(cols)
            update_assignments = ",\n                ".join(
                [f"{c} = EXCLUDED.{c}" for c in FEATURE_COLS]
            )
            query = f"""
            INSERT INTO features (
                {insert_cols}
            )
            VALUES %s
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                {update_assignments};
            """
            execute_values(cur, query, rows)
        conn.commit()
        print(f"Upserted {len(rows)} feature rows for {SYMBOL} {TIMEFRAME}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
