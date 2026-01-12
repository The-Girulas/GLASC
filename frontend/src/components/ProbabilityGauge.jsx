import React from 'react';

const ProbabilityGauge = ({ probability }) => {
    // probability is 0.0 to 1.0
    const pct = Math.min(100, Math.max(0, probability * 100));

    // Color Scale:
    // < 30% : Red (Hard)
    // > 70% : Neon (Guaranteed)
    let color = "#ef4444"; // red-500
    if (pct > 40) color = "#eab308"; // yellow-500
    if (pct > 70) color = "#00eeff"; // glasc-neon

    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (pct / 100) * circumference;

    return (
        <div className="flex flex-col items-center justify-center p-4">
            <div className="relative w-32 h-32 flex items-center justify-center">
                {/* Background Circle */}
                <svg className="w-full h-full transform -rotate-90">
                    <circle
                        cx="64" cy="64" r={radius}
                        stroke="#1f2937"
                        strokeWidth="8"
                        fill="transparent"
                    />
                    {/* Foreground Circle */}
                    <circle
                        cx="64" cy="64" r={radius}
                        stroke={color}
                        strokeWidth="8"
                        fill="transparent"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        className="transition-all duration-500 ease-out"
                        strokeLinecap="round"
                    />
                </svg>
                <div className="absolute flex flex-col items-center">
                    <span className="text-2xl font-bold font-mono text-white">
                        {pct.toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-gray-500 uppercase tracking-widest">Success</span>
                </div>
            </div>

            <div className="mt-2 text-center">
                <div className="text-xs text-gray-400 uppercase">Takeover Likelihood</div>
                <div className="text-[9px] text-gray-600 font-mono mt-1">
                    JAX Engine :: 100k Monte Carlo Paths<br />
                    Vol-Adjusted (Jump Diffusion)
                </div>
            </div>
        </div>
    );
};

export default ProbabilityGauge;
