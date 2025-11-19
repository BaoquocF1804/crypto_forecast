# Crypto Forecast Project 🚀

> **Note**: This project is available in both English and Vietnamese. Scroll down for English version.

## 📌 Giới thiệu
Dự án **Crypto Forecast** là một hệ thống dự đoán giá tiền điện tử toàn diện, sử dụng Machine Learning (XGBoost) để dự báo xu hướng giá (Tăng/Giảm) cho các đồng coin phổ biến.

**Tính năng nổi bật:**
*   **Đa Coin**: Hỗ trợ BTC, ETH, SOL, ADA.
*   **Đa Khung Thời Gian**: Dự báo cho 1H, 4H, và 1D.
*   **Real-time Dashboard**: Giao diện web hiện đại (React + Vite + Tailwind) hiển thị biểu đồ giá và tín hiệu dự báo theo thời gian thực.
*   **Tự động hóa**: Data Worker tự động thu thập dữ liệu từ Binance và cập nhật feature liên tục.

## 🏗️ Cấu trúc dự án
```
crypto_forecast/
├── data/                   # Dữ liệu (nếu dùng file CSV)
├── frontend/               # Source code Frontend (React)
├── models/                 # Lưu các mô hình XGBoost đã huấn luyện
├── scripts/                # Script phụ trợ
│   ├── data_worker.py      # Thu thập dữ liệu & tính feature realtime
│   └── init_db.py          # Khởi tạo database
└── src/
    ├── api/                # Backend API (FastAPI)
    └── pipelines/          # Pipelines (Train/Predict)
```

## 🚀 Cài đặt

### 1. Backend & Database
*   **Yêu cầu**: Python 3.8+, PostgreSQL (TimescaleDB khuyến nghị).
*   **Cài đặt**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
*   **Cấu hình**: Tạo file `.env` với thông tin DB:
    ```env
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=crypto_db
    DB_USER=postgres
    DB_PASS=password
    ```

### 2. Frontend
*   **Yêu cầu**: Node.js 16+.
*   **Cài đặt**:
    ```bash
    cd frontend
    npm install
    ```

## 🏃‍♂️ Hướng dẫn Chạy

Bạn cần mở 3 terminal để chạy toàn bộ hệ thống:

**Terminal 1: Data Worker (Thu thập dữ liệu)**
```bash
source venv/bin/activate
python scripts/data_worker.py
```

**Terminal 2: Backend API**
```bash
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 3: Frontend Dashboard**
```bash
cd frontend
npm run dev
```
Truy cập Dashboard tại: `http://localhost:3000` (hoặc port do Vite cấp).

## 🧠 Huấn luyện Mô hình (Tùy chọn)
Hệ thống cần mô hình đã được huấn luyện để đưa ra dự báo. Bạn có thể tự huấn luyện lại:

```bash
# Cú pháp: python -m src.pipelines.train_pipeline --symbol [SYMBOL] --timeframe [TIMEFRAME]

# Ví dụ: Train BTC khung 1H
python -m src.pipelines.train_pipeline --symbol BTC/USDT --timeframe 1h

# Ví dụ: Train ETH khung 4H
python -m src.pipelines.train_pipeline --symbol ETH/USDT --timeframe 4h
```

---

# Crypto Forecast Project (English) 🌍

## 📌 Introduction
**Crypto Forecast** is a comprehensive cryptocurrency price prediction system leveraging Machine Learning (XGBoost) to forecast price trends.

**Key Features:**
*   **Multi-Coin Support**: BTC, ETH, SOL, ADA.
*   **Multi-Timeframe**: Forecasts for 1H, 4H, and 1D intervals.
*   **Real-time Dashboard**: Modern web UI (React + Vite + Tailwind) displaying live charts and prediction signals.
*   **Automated Pipeline**: Data Worker automatically fetches Binance data and computes features in real-time.

## 🚀 Setup

### 1. Backend & Database
*   **Prerequisites**: Python 3.8+, PostgreSQL.
*   **Install**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
*   **Config**: Create `.env` file with DB credentials.

### 2. Frontend
*   **Prerequisites**: Node.js 16+.
*   **Install**:
    ```bash
    cd frontend
    npm install
    ```

## 🏃‍♂️ How to Run

Open 3 separate terminals:

**Terminal 1: Data Worker**
```bash
source venv/bin/activate
python scripts/data_worker.py
```

**Terminal 2: Backend API**
```bash
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 3: Frontend Dashboard**
```bash
cd frontend
npm run dev
```
Access Dashboard at: `http://localhost:3000`.

## 🧠 Model Training
To retrain models for specific coins/timeframes:

```bash
python -m src.pipelines.train_pipeline --symbol BTC/USDT --timeframe 1h
python -m src.pipelines.train_pipeline --symbol ETH/USDT --timeframe 4h
```
