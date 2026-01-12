import React from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, CartesianGrid, ReferenceLine } from 'recharts';

const MarketChart = ({ data, targetPrice = 90 }) => {
    // If no data yet, show loading state or empty grid
    if (!data || data.length === 0) {
        return (
            <div className="w-full h-full flex items-center justify-center text-gray-600 font-mono text-xs">
                WAITING FOR MARKET FEED...
            </div>
        );
    }

    // Calcul min/max domain for Y-Axis dynamically (zoom on price)
    const latestPrice = data[data.length - 1].price;
    const minPrice = Math.min(...data.map(d => d.price), targetPrice) * 0.95;
    const maxPrice = Math.max(...data.map(d => d.price), 110) * 1.05;

    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                <XAxis
                    dataKey="time"
                    stroke="#444"
                    tick={{ fontSize: 10, fill: '#666' }}
                    interval="preserveStartEnd"
                />
                <YAxis
                    domain={[minPrice, maxPrice]}
                    stroke="#444"
                    tick={{ fontSize: 10, fill: '#666' }}
                    width={40}
                />
                <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: '#333', fontSize: '12px' }}
                    itemStyle={{ color: '#fff' }}
                    labelStyle={{ display: 'none' }}
                />

                {/* Target Acquisition Price Line */}
                <ReferenceLine y={targetPrice} stroke="#ef4444" strokeDasharray="5 5" label={{ position: 'right', value: 'TARGET', fill: '#ef4444', fontSize: 10 }} />

                <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#00eeff"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false} // Crucial for smooth high-freq updates
                />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default MarketChart;
