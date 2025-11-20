import sys
import subprocess
import logging

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Các cấu hình giống trong src/api/main.py
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XAUT/USDT']
TIMEFRAMES = ['1h', '4h', '1d']
TARGET_RETURNS = [0.01, 0.02, 0.05]

def main():
    logging.info("🚀 Bắt đầu chạy training thủ công cho TOÀN BỘ mô hình...")
    
    total_jobs = len(SYMBOLS) * len(TIMEFRAMES) * len(TARGET_RETURNS)
    current_job = 0
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            for target_return in TARGET_RETURNS:
                current_job += 1
                logging.info(f"[{current_job}/{total_jobs}] Đang train {symbol} - {timeframe} - Target {target_return:.1%}...")
                
                try:
                    # Gọi training pipeline
                    # Mặc định script này sẽ KHÔNG dùng tuning (vì code mới đã default tuning=False)
                    # Nếu muốn tuning, thêm "--tuning" vào danh sách tham số
                    subprocess.run(
                        [sys.executable, "-m", "src.pipelines.train_pipeline", 
                         "--symbol", symbol, 
                         "--timeframe", timeframe,
                         "--target_return", str(target_return),
                         "--tuning"], # BẬT Tuning Mode
                        check=True
                    )
                    logging.info(f"✅ Xong {symbol} - {timeframe} - Target {target_return:.1%}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"❌ Lỗi khi train {symbol} - {timeframe} - Target {target_return:.1%}")
                except KeyboardInterrupt:
                    logging.warning("\n⚠️ Đã dừng bởi người dùng.")
                    return

    logging.info("🏁 Hoàn tất toàn bộ quá trình training.")

if __name__ == "__main__":
    main()
