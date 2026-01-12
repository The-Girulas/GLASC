import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Activity, Network, ShieldAlert, Play, TrendingUp } from 'lucide-react';
import NegotiationTerminal from './components/NegotiationTerminal';
import PowerBalanceChart from './components/PowerBalanceChart';
import MarketChart from './components/MarketChart';
import CorporateGraph from './components/CorporateGraph';
import ProbabilityGauge from './components/ProbabilityGauge';
import ChaosControl from './components/ChaosControl';
import axios from 'axios';
import ErrorBoundary from './components/ErrorBoundary';
import GameOverModal from './components/GameOverModal';

function App() {
  const [ticker, setTicker] = useState("EVIL_CORP");
  const [simState, setSimState] = useState("IDLE"); // IDLE, RUNNING, DONE
  const [gameOverData, setGameOverData] = useState(null);

  // Real-time Data State
  const [price, setPrice] = useState(100.0);
  const [priceHistory, setPriceHistory] = useState([]); // Array of { time, price }
  const [probability, setProbability] = useState(0.5);
  const [volatility, setVolatility] = useState(0.2);
  const [powerHistory, setPowerHistory] = useState([]); // Array of { time, attacker, defender }
  const [agentLogs, setAgentLogs] = useState([]);
  const wsRef = useRef(null);

  // Connect to WebSocket on Launch
  useEffect(() => {
    if (simState === "RUNNING") {
      const ws = new WebSocket("ws://localhost:8000/ws/monitor");
      ws.onopen = () => console.log("WS Connected");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "TICK") {
          const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
          setPrice(data.price);
          setProbability(data.probability);
          setVolatility(data.volatility);

          // Update History (Keep last 60 points ~ 1 minute)
          setPriceHistory(prev => {
            const newHistory = [...prev, { time: timestamp, price: data.price }];
            return newHistory.slice(-60);
          });

          // Update Power History (from power_balance)
          if (data.power_balance) {
            setPowerHistory(prev => {
              const newPoint = {
                time: timestamp,
                attacker: data.power_balance.attacker,
                defender: data.power_balance.defender
              };
              return [...prev, newPoint].slice(-60);
            });
          }

        } else if (data.type === "AGENT_LOG") {
          setAgentLogs(prev => [...prev, data]);
        } else if (data.type === "ALERT") {
          console.warn(data.message);
        } else if (data.type === "GAME_OVER") {
          setGameOverData(data);
          setSimState("DONE");
        }
      };
      wsRef.current = ws;
      return () => ws.close();
    };
    wsRef.current = ws;
    return () => ws.close();
  }
  }, [simState]);

const handleLaunch = async () => {
  setSimState("RUNNING");
  try {
    await axios.post('http://localhost:8000/api/sim/launch', {
      ticker: ticker,
      volatility_override: 0.2
    });
  } catch (e) {
    console.error("Launch failed", e);
    setSimState("ERROR");
  }
};

return (
  <div className="h-screen w-screen bg-glasc-bg text-gray-200 p-4 font-sans flex flex-col gap-4 overflow-hidden">

    {/* HEADER */}
    <header className="flex justify-between items-center border-b border-glasc-neon/30 pb-2">
      <div className="flex items-center gap-2">
        <Activity className="text-glasc-neon w-8 h-8 animate-pulse" />
        <div>
          <h1 className="text-2xl font-bold tracking-widest text-glasc-neon">GLASC WAR ROOM</h1>
          <p className="text-xs text-gray-500 uppercase">Global Leverage & Asset Strategy Controller</p>
        </div>
      </div>

      {/* Metric Tickers */}
      {simState === "RUNNING" && (
        <div className="flex gap-6 font-mono text-sm">
          <div className="flex flex-col items-center">
            <span className="text-gray-500 text-[10px]">LIVE PRICE</span>
            <span className="text-white font-bold">${price.toFixed(2)}</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-gray-500 text-[10px]">IMPLIED VOL</span>
            <span className={`font-bold ${volatility > 0.4 ? 'text-red-500 animate-pulse' : 'text-glasc-neon'}`}>
              {(volatility * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      <div className="flex gap-4 items-center">
        <div className="bg-glasc-panel border border-glasc-neon/20 px-4 py-1 rounded backdrop-blur">
          <span className="text-xs text-gray-400">TARGET:</span>
          <input
            className="bg-transparent border-none outline-none text-glasc-warning font-mono ml-2 uppercase w-24"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
          />
        </div>
        <button
          onClick={handleLaunch}
          disabled={simState === "RUNNING"}
          className="bg-glasc-neon/10 hover:bg-glasc-neon/20 text-glasc-neon border border-glasc-neon px-6 py-2 rounded uppercase font-bold tracking-wider flex items-center gap-2 transition-all hover:shadow-[0_0_15px_rgba(0,238,255,0.5)]">
          <Play size={16} />
          {simState === "RUNNING" ? "SYSTEM ACTIVE" : "INITIATE ATTACK"}
        </button>
      </div>
    </header>

    {/* MAIN GRID */}
    <div className="flex-1 grid grid-cols-12 grid-rows-12 gap-4 min-h-0">

      {/* TOP LEFT: MARKET CHART (Span 8x6) */}
      <div className="col-span-8 row-span-6 bg-glasc-panel border border-glasc-neon/20 rounded-lg p-4 relative overflow-hidden backdrop-blur-md flex flex-col">
        <h2 className="text-glasc-neon/80 text-sm font-bold mb-2 flex items-center gap-2">
          <Activity size={14} /> MARKET DYNAMICS (JAX ENGINE)
        </h2>
        <div className="flex-1 min-h-0">
          {/* Live Scrolling Chart */}
          <MarketChart data={priceHistory} />
        </div>
      </div>

      {/* RIGHT COLUMN: INTELLIGENCE NEXUS (Span 4x12) */}
      <div className="col-span-4 row-span-12 flex flex-col gap-4">


        {/* 3D GRAPH (Top Half) */}
        <div className="flex-1 bg-glasc-panel border border-glasc-neon/20 rounded-lg relative overflow-hidden flex flex-col">
          <div className="absolute top-4 left-4 z-10 pointer-events-none">
            <h2 className="text-glasc-neon/80 text-sm font-bold flex items-center gap-2">
              <Network size={14} /> CORPORATE NEXUS
            </h2>
          </div>
          <div className="flex-1">
            <ErrorBoundary>
              <CorporateGraph
                ticker={ticker}
                probability={probability}
                attackerInfluence={powerHistory.length > 0 ? powerHistory[powerHistory.length - 1].attacker : 0}
                latestLog={agentLogs.length > 0 ? agentLogs[agentLogs.length - 1] : null}
              />
            </ErrorBoundary>
          </div>
        </div>

        {/* LIVE METRICS (Bottom Half split) */}
        <div className="h-64 grid grid-cols-2 gap-4">
          {/* PROBABILITY GAUGE */}
          <div className="bg-glasc-panel border border-glasc-neon/20 rounded-lg relative">
            <ProbabilityGauge probability={probability} />
          </div>

          {/* CHAOS CONTROL */}
          <div className="bg-glasc-panel border border-red-500/20 rounded-lg relative">
            <ChaosControl />
          </div>
        </div>

      </div>

      {/* BOTTOM LEFT: AGENT TERMINAL (Span 8x6) */}
      {/* BOTTOM LEFT SPLIT: TERMINAL & POWER CHART */}
      {/* NEGOTIATION TERMINAL (Span 5x6) */}
      <div className="col-span-5 row-span-6 bg-black/40 border border-glasc-neon/20 rounded-lg relative overflow-hidden flex flex-col h-full">
        <div className="absolute top-0 left-0 w-full h-1 bg-glasc-neon/20"></div>
        <NegotiationTerminal
          logs={agentLogs}
          onSendMessage={async (msg) => {
            try {
              await axios.post('http://localhost:8000/api/sim/inject_message', { content: msg });
            } catch (e) {
              console.error("Failed to inject message", e);
            }
          }}
        />
      </div>

      {/* POWER BALANCE CHART (Span 3x6) */}
      <div className="col-span-3 row-span-6 bg-glasc-panel border border-glasc-neon/20 rounded-lg p-4 relative overflow-hidden backdrop-blur-md flex flex-col">
        <h2 className="text-glasc-neon/80 text-sm font-bold mb-2 flex items-center gap-2">
          <TrendingUp size={14} /> POWER STRUGGLE
        </h2>
        <div className="flex-1 min-h-0">
          <PowerBalanceChart data={powerHistory} />
        </div>
      </div>

    </div>
    {gameOverData && (
      <GameOverModal
        result={gameOverData.result}
        stats={gameOverData.stats}
        onRestart={() => {
          setGameOverData(null);
          setSimState("IDLE");
          setPriceHistory([]);
          setPowerHistory([]);
          setAgentLogs([]);
          setPrice(100.0);
        }}
      />
    )}
  </div>
);
}

export default App;
