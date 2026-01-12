import React, { useState } from 'react';
import { Skull, TrendingDown, Zap } from 'lucide-react';
import axios from 'axios';

const ChaosControl = () => {
    const [loading, setLoading] = useState(false);

    const injectChaos = async (event, vol) => {
        setLoading(true);
        try {
            await axios.post('http://localhost:8000/api/sim/chaos', {
                volatility: vol,
                event: event
            });
        } catch (e) {
            console.error("Chaos Error", e);
        }
        setTimeout(() => setLoading(false), 1000);
    };

    return (
        <div className="p-4 flex flex-col gap-4 h-full justify-center">
            <h3 className="text-glasc-warning text-xs font-bold uppercase tracking-widest flex items-center gap-2 border-b border-glasc-warning/30 pb-2">
                <Skull size={14} /> Chaos Injection
            </h3>

            <div className="grid grid-cols-2 gap-3">
                <button
                    onClick={() => injectChaos("SCANDAL", 0.60)}
                    disabled={loading}
                    className="bg-red-900/40 hover:bg-red-500/20 border border-red-500/50 text-red-500 p-3 rounded flex flex-col items-center gap-1 transition-all group"
                >
                    <TrendingDown size={20} className="group-hover:animate-bounce" />
                    <span className="text-xs font-bold">SCANDAL LEAK</span>
                    <span className="text-[9px] text-gray-400">Vol: 60% | Price: -15%</span>
                </button>

                <button
                    onClick={() => injectChaos("RATES_HIKE", 0.35)}
                    disabled={loading}
                    className="bg-orange-900/40 hover:bg-orange-500/20 border border-orange-500/50 text-orange-500 p-3 rounded flex flex-col items-center gap-1 transition-all"
                >
                    <Zap size={20} />
                    <span className="text-xs font-bold">RATES SHOCK</span>
                    <span className="text-[9px] text-gray-400">Vol: 35% | Drift: -20%</span>
                </button>
            </div>

            <div className="mt-2 text-[10px] text-gray-600 text-center italic">
                WARNING: Events are irreversible and trigger immediate quantitative re-pricing.
            </div>
        </div>
    );
};

export default ChaosControl;
