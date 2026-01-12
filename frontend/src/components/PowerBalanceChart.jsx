import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const PowerBalanceChart = ({ data }) => {
    // data should be array of { time: '10:00', attacker: 40, defender: 60 }

    if (!data || data.length === 0) {
        return (
            <div className="w-full h-full flex items-center justify-center text-gray-600 font-mono text-xs">
                CALCULATING POWER DYNAMICS...
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col">
            <div className="text-[10px] text-gray-400 font-mono text-center mb-1 tracking-widest uppercase">
                Dominance Ratio
            </div>
            <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                        data={data}
                        stackOffset="expand"
                        margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
                    >
                        <Tooltip
                            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', border: '1px solid #333', fontSize: '12px' }}
                            itemStyle={{ padding: 0 }}
                        />
                        <XAxis dataKey="time" hide />
                        <YAxis hide domain={[0, 1]} />
                        <Area
                            type="monotone"
                            dataKey="attacker"
                            stackId="1"
                            stroke="#ef4444"
                            fill="#ef4444"
                            fillOpacity={0.6}
                            name="Attacker"
                            isAnimationActive={false}
                        />
                        <Area
                            type="monotone"
                            dataKey="defender"
                            stackId="1"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.6}
                            name="Defender"
                            isAnimationActive={false}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            <div className="flex justify-between px-2 text-[9px] font-mono mt-1">
                <span className="text-red-400">ATTACKER</span>
                <span className="text-blue-400">DEFENDER</span>
            </div>
        </div>
    );
};

export default PowerBalanceChart;
