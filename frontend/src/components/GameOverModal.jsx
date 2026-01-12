import React from 'react';
import { Trophy, AlertTriangle, RefreshCw } from 'lucide-react';

const GameOverModal = ({ result, stats, onRestart }) => {
    // result: "VICTORY" | "DEFEAT"
    const isVictory = result === "VICTORY";

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md animate-fadeIn">
            <div className={`
                ${isVictory ? 'border-glasc-neon shadow-[0_0_50px_#00eeff]' : 'border-red-600 shadow-[0_0_50px_#ff0000]'}
                bg-gray-900/90 border-2 rounded-xl p-8 max-w-lg w-full transform scale-100 transition-transform
            `}>
                <div className="text-center mb-6">
                    {isVictory ? (
                        <Trophy className="w-20 h-20 text-glasc-neon mx-auto mb-4 animate-bounce" />
                    ) : (
                        <AlertTriangle className="w-20 h-20 text-red-600 mx-auto mb-4 animate-pulse" />
                    )}

                    <h1 className={`text-4xl font-bold tracking-widest ${isVictory ? 'text-glasc-neon' : 'text-red-600'}`}>
                        {isVictory ? 'HOSTILE TAKEOVER COMPLETE' : 'DEFENSE SUCCESSFUL'}
                    </h1>
                    <p className="text-gray-400 mt-2 uppercase tracking-wider text-sm">
                        {isVictory ? 'Target Acquired. Board Dissolved.' : 'The Board rejected your bid. Attempt Failed.'}
                    </p>
                </div>

                <div className="space-y-4 mb-8 bg-black/50 p-4 rounded border border-gray-800">
                    <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                        <span className="text-gray-500 text-xs uppercase">Final Price</span>
                        <span className="font-mono text-xl text-white">${stats.price?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                        <span className="text-gray-500 text-xs uppercase">Total Duration</span>
                        <span className="font-mono text-base text-white">{stats.duration}s</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-gray-500 text-xs uppercase">Power Balance</span>
                        <span className="font-mono text-base text-white">
                            ATT: {(stats.attackerInfluence * 100).toFixed(0)}% / DEF: {(stats.defenderControl * 100).toFixed(0)}%
                        </span>
                    </div>
                </div>

                <button
                    onClick={onRestart}
                    className={`
                        w-full py-3 rounded text-lg font-bold uppercase tracking-widest transition-all
                        ${isVictory
                            ? 'bg-glasc-neon hover:bg-white text-black hover:shadow-[0_0_20px_#00eeff]'
                            : 'bg-red-600 hover:bg-red-500 text-white hover:shadow-[0_0_20px_#ff0000]'}
                    `}
                >
                    <div className="flex items-center justify-center gap-2">
                        <RefreshCw size={20} />
                        Initialize New Target
                    </div>
                </button>
            </div>
        </div>
    );
};

export default GameOverModal;
