import React from 'react';
import { PredictionResponse } from '../api/client';
import { TrendingUp, TrendingDown, Activity, BarChart2, DollarSign } from 'lucide-react';
import clsx from 'clsx';

interface PredictionCardProps {
    prediction: PredictionResponse | null;
    loading: boolean;
    timeframe: string;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, loading, timeframe }) => {
    if (loading) {
        return <div className="animate-pulse bg-gray-800 h-64 rounded-xl"></div>;
    }

    if (!prediction) return null;

    const isBuy = prediction.label === 1;
    const prob = prediction.last_proba;
    const th = prediction.threshold;
    const confidence = (prob * 100).toFixed(1);
    const thresholdPct = (th * 100).toFixed(1);

    // Calculate Horizon Text
    let horizonText = "Next 4 Periods";
    if (timeframe === '1h') horizonText = "Next 4 Hours";
    else if (timeframe === '4h') horizonText = "Next 16 Hours";
    else if (timeframe === '1d') horizonText = "Next 4 Days";

    let signalText = "WAIT / NEUTRAL";
    let signalColor = "text-gray-400";
    let bgColor = "bg-gray-500/10 border-gray-500/50";
    let Icon = Activity;

    if (isBuy) {
        if (prob > 0.7) {
            signalText = "STRONG BUY 🚀";
            signalColor = "text-green-400";
            bgColor = "bg-green-500/20 border-green-500/80 shadow-[0_0_15px_rgba(74,222,128,0.3)]";
            Icon = TrendingUp;
        } else {
            signalText = "BUY SIGNAL";
            signalColor = "text-green-500";
            bgColor = "bg-green-500/10 border-green-500/50";
            Icon = TrendingUp;
        }
    } else {
        if (prob < 0.3) {
            signalText = "STRONG SELL / AVOID 🛑";
            signalColor = "text-red-500";
            bgColor = "bg-red-500/10 border-red-500/50";
            Icon = TrendingDown;
        } else {
            signalText = "WAIT / NEUTRAL ⏳";
            signalColor = "text-yellow-500";
            bgColor = "bg-yellow-500/10 border-yellow-500/50";
            Icon = Activity;
        }
    }

    return (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-xl font-bold text-white">AI Forecast</h2>
                    <p className="text-xs text-blue-400 font-medium mt-0.5">
                        Horizon: {horizonText}
                    </p>
                </div>
                <span className="text-xs text-gray-400">
                    {prediction.last_timestamp ? new Date(prediction.last_timestamp).toLocaleString() : ''}
                </span>
            </div>

            <div className={clsx(
                "flex flex-col items-center justify-center p-6 rounded-lg mb-6 border-2 transition-all duration-300",
                bgColor
            )}>
                <Icon className={clsx("w-12 h-12 mb-2", signalColor)} />
                <span className={clsx("text-2xl font-bold text-center", signalColor)}>
                    {signalText}
                </span>

                <div className="w-full mt-4">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Confidence</span>
                        <span>{confidence}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2.5 relative">
                        <div
                            className={clsx("h-2.5 rounded-full transition-all duration-500", isBuy ? "bg-green-500" : "bg-yellow-500")}
                            style={{ width: `${Math.min(prob * 100, 100)}%` }}
                        ></div>
                        {/* Threshold Marker */}
                        <div
                            className="absolute top-0 bottom-0 w-0.5 bg-white/50"
                            style={{ left: `${thresholdPct}%` }}
                            title={`Threshold: ${thresholdPct}%`}
                        ></div>
                    </div>
                    <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                        <span>0%</span>
                        <span className="text-gray-400">Threshold: {thresholdPct}%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-900/50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-xs">Current Price</span>
                    </div>
                    <div className="text-lg font-semibold text-white">
                        ${prediction.price?.toLocaleString() ?? '---'}
                    </div>
                </div>

                <div className="bg-gray-900/50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-xs">RSI (14)</span>
                    </div>
                    <div className={clsx("text-lg font-semibold",
                        (prediction.rsi || 0) > 70 ? "text-red-400" : (prediction.rsi || 0) < 30 ? "text-green-400" : "text-white"
                    )}>
                        {prediction.rsi?.toFixed(2) ?? '---'}
                    </div>
                </div>

                <div className="bg-gray-900/50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-400 mb-1">
                        <BarChart2 className="w-4 h-4" />
                        <span className="text-xs">MACD</span>
                    </div>
                    <div className={clsx("text-lg font-semibold", (prediction.macd || 0) > 0 ? "text-green-400" : "text-red-400")}>
                        {prediction.macd?.toFixed(4) ?? '---'}
                    </div>
                </div>

                <div className="bg-gray-900/50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-xs">Vol Z-Score</span>
                    </div>
                    <div className="text-lg font-semibold text-white">
                        {prediction.vol_zscore?.toFixed(2) ?? '---'}
                    </div>
                </div>
            </div>
        </div>
    );
};
