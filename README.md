# Crypto Forecast Project

> **Note**: This project is available in both English and Vietnamese. Scroll down for English version.

## 📌 Giới thiệu
Dự án xây dựng mô hình dự đoán giá tiền điện tử (Crypto) sử dụng XGBoost, với dữ liệu từ sàn giao dịch và lưu trữ trong PostgreSQL.

## 🏗️ Cấu trúc dự án
```
crypto_forecast/
├── data/                   # Thư mục chứa dữ liệu thô/đã xử lý
├── models/                 # Lưu các mô hình đã huấn luyện
├── scripts/                # Các script phụ trợ (data worker, v.v.)
└── src/
    ├── api/                # FastAPI server (đang phát triển)
    ├── core/               # Core modules
    │   ├── config.py       # Cấu hình ứng dụng
    │   ├── data_loader.py  # Tải dữ liệu từ file CSV
    │   └── feature_builder.py # Xây dựng đặc trưng (features)
    └── pipelines/          # Các pipeline xử lý chính
        ├── train_pipeline.py   # Huấn luyện mô hình
        └── predict_pipeline.py # Dự đoán từ mô hình
```

## 🚀 Cài đặt

1. **Cài đặt thư viện**
```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Trên Windows: .\venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

2. **Cấu hình cơ sở dữ liệu**
Tạo file `.env` trong thư mục gốc với nội dung:
```
# Database
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_username
DB_PASS=your_password

# Model
MODEL_PATH=models/xgb_btc_model.json
```

## 🏃‍♂️ Cách chạy

### 1. Huấn luyện mô hình
```bash
python src/pipelines/train_pipeline.py
```

### 2. Dự đoán từ file CSV
```bash
# Đặt file CSV vào thư mục data/processed/ hoặc data/raw/
python src/pipelines/predict_pipeline.py
```

### 3. Chạy API (nếu cần)
```bash
uvicorn src.api.main:app --reload
```

## 📊 Mô tả Pipeline

### 1. Huấn luyện (`train_pipeline.py`)
- Kết nối PostgreSQL → lấy dữ liệu `candles`
- Tính toán các đặc trưng kỹ thuật (RSI, MACD, MA, v.v.)
- Tạo label: Giá có tăng >1% trong 4h tới không?
- Huấn luyện XGBoost Classifier
- Lưu mô hình vào `models/xgb_btc_model.json`

### 2. Dự đoán (`predict_pipeline.py`)
- Đọc dữ liệu từ file CSV
- Áp dụng các đặc trưng tương tự như khi huấn luyện
- Tải mô hình đã huấn luyện và dự đoán

## 📝 Ghi chú
- Dữ liệu huấn luyện được giả định đã có sẵn trong bảng `candles` của PostgreSQL
- Có thể điều chỉnh tham số mô hình trong `train_pipeline.py`

---

# Crypto Forecast Project (English)

## 📌 Introduction
A cryptocurrency price prediction project using XGBoost, with data from exchanges stored in PostgreSQL.

## 🏗️ Project Structure
```
crypto_forecast/
├── data/                   # Raw/processed data
├── models/                 # Trained models
├── scripts/                # Utility scripts (data worker, etc.)
└── src/
    ├── api/                # FastAPI server (WIP)
    ├── core/               # Core modules
    │   ├── config.py       # App configuration
    │   ├── data_loader.py  # Load data from CSV
    │   └── feature_builder.py # Feature engineering
    └── pipelines/          # Main processing pipelines
        ├── train_pipeline.py   # Model training
        └── predict_pipeline.py # Make predictions
```

## 🚀 Setup

1. **Install dependencies**
```bash
# Create virtual env (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure database**
Create `.env` file in the root directory:
```
# Database
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_username
DB_PASS=your_password

# Model
MODEL_PATH=models/xgb_btc_model.json
```

## 🏃‍♂️ How to Run

### 1. Train the model
```bash
python src/pipelines/train_pipeline.py
```

### 2. Make predictions from CSV
```bash
# Place CSV file in data/processed/ or data/raw/
python src/pipelines/predict_pipeline.py
```

### 3. Run API (if needed)
```bash
uvicorn src.api.main:app --reload
```

## 📊 Pipeline Description

### 1. Training (`train_pipeline.py`)
- Connect to PostgreSQL → fetch `candles` data
- Calculate technical indicators (RSI, MACD, MA, etc.)
- Create binary label: Will price increase >1% in next 4h?
- Train XGBoost Classifier
- Save model to `models/xgb_btc_model.json`

### 2. Prediction (`predict_pipeline.py`)
- Read data from CSV file
- Apply same feature engineering as training
- Load trained model and make predictions

## 📝 Notes
- Training data is assumed to be available in PostgreSQL `candles` table
- Model parameters can be adjusted in `train_pipeline.py`
