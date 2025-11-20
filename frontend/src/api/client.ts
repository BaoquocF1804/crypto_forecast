import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface Candle {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface RiskItem {
    target_return: number;
    buy_proba: number;
    sell_proba: number;
}

export interface PredictionResponse {
    last_proba: number;
    label: number;
    threshold: number;
    n_samples: number;
    last_timestamp?: string;
    price?: number;
    rsi?: number;
    macd?: number;
    vol_zscore?: number;
    target_return?: number;
    buy_proba?: number;
    sell_proba?: number;
    risk_analysis?: RiskItem[];
}

export const getHistory = async (limit = 100, symbol = 'BTC/USDT', timeframe = '1h') => {
    const response = await api.get<Candle[]>(`/history?limit=${limit}&symbol=${symbol}&timeframe=${timeframe}`);
    return response.data;
};

export const getPrediction = async (symbol = 'BTC/USDT', timeframe = '1h', targetReturn = 0.01) => {
    const response = await api.get<PredictionResponse>(`/predict?symbol=${symbol}&timeframe=${timeframe}&target_return=${targetReturn}`);
    return response.data;
};
