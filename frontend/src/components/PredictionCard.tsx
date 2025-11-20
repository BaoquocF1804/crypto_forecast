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

    const label = prediction.label;
    const buyProb = prediction.buy_proba ?? 0;
    const sellProb = prediction.sell_proba ?? 0;

    // Determine primary probability based on label
    let primaryProb = 0;
    if (label === 1) primaryProb = buyProb;
    else if (label === 2) primaryProb = sellProb;
    else primaryProb = prediction.last_proba; // Fallback or Neutral prob if we had it

    const confidence = (primaryProb * 100).toFixed(1);

    // Calculate Horizon Text
    let horizonText = "Next 4 Periods";
    if (timeframe === '1h') horizonText = "Next 4 Hours";
    else if (timeframe === '4h') horizonText = "Next 16 Hours";
    else if (timeframe === '1d') horizonText = "Next 4 Days";

    let signalText = "WAIT / NEUTRAL";
    let signalColor = "text-gray-400";
    let bgColor = "bg-gray-500/10 border-gray-500/50";
    let Icon = Activity;

    if (label === 1) {
        // BUY
        if (buyProb > 0.7) {
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
    } else if (label === 2) {
        // SELL
        if (sellProb > 0.7) {
            signalText = "STRONG SELL 🛑";
            signalColor = "text-red-500";
            bgColor = "bg-red-500/20 border-red-500/80 shadow-[0_0_15px_rgba(239,68,68,0.3)]";
            Icon = TrendingDown;
        } else {
            signalText = "SELL SIGNAL";
            signalColor = "text-red-500";
            bgColor = "bg-red-500/10 border-red-500/50";
            Icon = TrendingDown;
        }
    } else {
        // NEUTRAL
        signalText = "WAIT / NEUTRAL ⏳";
        signalColor = "text-yellow-500";
        bgColor = "bg-yellow-500/10 border-yellow-500/50";
        Icon = Activity;
    }

    return (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-xl font-bold text-white">AI Forecast</h2>
                    <p className="text-xs text-blue-400 font-medium mt-0.5">
                        Horizon: {horizonText} • Target: {((prediction.target_return ?? 0.01) * 100).toFixed(1)}%
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
                            className={clsx("h-2.5 rounded-full transition-all duration-500",
                                label === 1 ? "bg-green-500" : label === 2 ? "bg-red-500" : "bg-yellow-500"
                            )}
                            style={{ width: `${Math.min(primaryProb * 100, 100)}%` }}
                        ></div>
                    </div>
                    <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                        <span>0%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>

            {/* Risk Analysis Section */}
            {prediction.risk_analysis && prediction.risk_analysis.length > 0 && (
                <div className="bg-gray-900/50 p-4 rounded-lg mb-6">
                    <div className="flex items-center gap-2 text-gray-400 mb-3">
                        <Activity className="w-4 h-4" />
                        <span className="text-xs font-semibold uppercase tracking-wider">Risk Analysis</span>
                    </div>
                    <div className="space-y-2">
                        {prediction.risk_analysis.map((item, idx) => (
                            <div key={idx} className="flex items-center text-sm">
                                <span className="text-gray-300 w-24">
                                    Target {item.target_return * 100}%
                                </span>
                                <div className="flex gap-6">
                                    <span className="text-green-400 w-20">Buy: {(item.buy_proba * 100).toFixed(0)}%</span>
                                    <span className="text-red-400 w-20">Sell: {(item.sell_proba * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

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
