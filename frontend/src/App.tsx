import React, { useEffect, useState } from 'react';
import { Candle, PredictionResponse, getHistory, getPrediction } from './api/client';
import { PriceChart } from './components/PriceChart';
import { PredictionCard } from './components/PredictionCard';
import { RefreshCw } from 'lucide-react';

function App() {
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [targetReturn, setTargetReturn] = useState(0.01);
  const [history, setHistory] = useState<Candle[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XAUT/USDT'];
  const TIMEFRAMES = ['1h', '4h', '1d'];
  const TARGET_RETURNS = [
    { value: 0.01, label: '1%' },
    { value: 0.02, label: '2%' },
    { value: 0.05, label: '5%' },
  ];

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [historyData, predictionData] = await Promise.all([
        getHistory(100, symbol, timeframe),
        getPrediction(symbol, timeframe, targetReturn)
      ]);
      setHistory(historyData);
      setPrediction(predictionData);
    } catch (err) {
      setError('Failed to fetch data. Please ensure the backend is running and data is available.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [symbol, timeframe, targetReturn]); // Refetch when symbol, timeframe or targetReturn changes

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Crypto Forecast AI
            </h1>
            <p className="text-gray-400 mt-1">Real-time Prediction Model</p>
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </header>

        <div className="flex flex-col md:flex-row gap-4 mb-8 justify-between items-start md:items-center">
          {/* Symbol Selector */}
          <div className="flex gap-2 overflow-x-auto pb-2">
            {SYMBOLS.map((s) => (
              <button
                key={s}
                onClick={() => setSymbol(s)}
                className={`px-6 py-2 rounded-full font-medium transition-all ${symbol === s
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                  }`}
              >
                {s.split('/')[0]}
              </button>
            ))}
          </div>

          <div className="flex gap-4">
            {/* Timeframe Selector */}
            <div className="flex gap-2 bg-gray-800 p-1 rounded-lg">
              {TIMEFRAMES.map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${timeframe === tf
                    ? 'bg-gray-700 text-white shadow-sm'
                    : 'text-gray-400 hover:text-white'
                    }`}
                >
                  {tf.toUpperCase()}
                </button>
              ))}
            </div>

            {/* Target Return Selector */}
            <div className="flex gap-2 bg-gray-800 p-1 rounded-lg">
              {TARGET_RETURNS.map((tr) => (
                <button
                  key={tr.value}
                  onClick={() => setTargetReturn(tr.value)}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${targetReturn === tr.value
                    ? 'bg-purple-600 text-white shadow-sm'
                    : 'text-gray-400 hover:text-white'
                    }`}
                >
                  {tr.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 text-red-500 p-4 rounded-lg mb-8">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <PriceChart data={history} />
          </div>
          <div>
            <PredictionCard prediction={prediction} loading={loading} timeframe={timeframe} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
