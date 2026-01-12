import React, { useRef, useEffect, useState, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { Shield, Users, DollarSign, X, Zap } from 'lucide-react';
import axios from 'axios';

// Consistent Data Gen (Deterministic for demo)
// Consistent Data Gen (Deterministic for demo)
const genData = (ticker) => {
    const nodes = [
        { id: ticker, group: 0, role: "Target", shares: "N/A", val: 100 }, // CENTER NODE
        { id: "BlackRock", group: 2, role: "Institutional" },
        { id: "Vanguard", group: 2, role: "Institutional" },
        { id: "Founder", group: 1, role: "Insider" },
        { id: "Cousin Greg", group: 1, role: "Insider" },
        { id: "Activist Fund", group: 2, role: "Hostile" },
        { id: "Pension Plan", group: 2, role: "Long-term" },
        { id: "Retail Mob", group: 2, role: "Public" },
        { id: "Bank Syndicate", group: 2, role: "Debtholder" },
        { id: "Hedge Fund A", group: 2, role: "Speculator" },
        { id: "Hedge Fund B", group: 2, role: "Speculator" },
        { id: "Insurer X", group: 2, role: "Conservative" },
        { id: "Sovereign Fund", group: 2, role: "Sovereign" }
    ].map(n => ({
        ...n,
        val: n.val || Math.random() * 10 + 5,
        shares: n.shares || (Math.random() * 10).toFixed(1) + "%",
        loyalty: n.loyalty || Math.floor(Math.random() * 100),
        desc: "Key strategic stakeholder."
    }));

    const links = [];
    nodes.forEach((n, i) => {
        if (n.group !== 0) {
            links.push({ source: nodes[0].id, target: n.id }); // Link to Center
            if (Math.random() > 0.7) links.push({ source: n.id, target: nodes[Math.floor(Math.random() * nodes.length)].id });
        }
    });

    return { nodes, links };
};

const CorporateGraph = ({ ticker, probability = 0.5, attackerInfluence = 0, latestLog = null }) => {
    const fgRef = useRef();
    const containerRef = useRef();
    const [dims, setDims] = useState({ width: 400, height: 600 });
    const [data, setData] = useState(genData(ticker));
    const [extraLinks, setExtraLinks] = useState([]); // Transient visual beams

    // Synch Data with Ticker Prop to prevent ID mismatch
    useEffect(() => {
        setData(genData(ticker));
    }, [ticker]);
    const [selectedNode, setSelectedNode] = useState(null);
    const [rotationActive, setRotationActive] = useState(true);

    // ... (Resize and Rotate - existing code assumed same)

    // Handle Visual Beams from Logs
    useEffect(() => {
        if (!latestLog) return;

        // Map Source to Node ID
        let sourceId = null;
        if (latestLog.source === "BANKER") sourceId = "Bank Syndicate";
        else if (latestLog.source === "INSIDER") sourceId = "Founder";
        else if (latestLog.source === "DEFENDER") sourceId = ticker; // Center
        else if (latestLog.source === "DEFENDER_AI") sourceId = ticker;

        if (sourceId) {
            const newLink = {
                source: sourceId,
                target: sourceId === ticker ? "Activist Fund" : ticker, // If Center acts, target the enemy?
                isBeam: true,
                color: latestLog.source.includes("DEFENDER") ? "#3b82f6" : "#ef4444"
            };

            setExtraLinks(prev => [...prev, newLink]);

            // Cleanup beam
            setTimeout(() => {
                setExtraLinks(prev => prev.filter(l => l !== newLink));
            }, 2000);
        }
    }, [latestLog, ticker]);


    // ... (Rest of component)

    const finalData = {
        nodes: data.nodes,
        links: [...data.links, ...extraLinks]
    };

    // Helper Functions
    const getNodeColor = (node) => {
        if (node.id === ticker) {
            // Center Node reacts to probability
            if (probability > 0.70) return '#ef4444'; // High Risk (Red)
            if (probability > 0.40) return '#eab308'; // Medium (Yellow)
            return '#3b82f6'; // Safe (Blue)
        }
        if (node.role === 'Hostile') return '#ef4444';
        if (node.role === 'Institutional') return '#6366f1';
        if (node.role === 'Insider') return '#ec4899';
        if (node.group === 1) return '#10b981'; // Allies
        return '#94a3b8'; // Neutral
    };

    const getLinkColor = () => {
        if (attackerInfluence > 0.6) return '#ef4444'; // High Stress
        if (attackerInfluence > 0.3) return '#eab308';
        return '#334155'; // Calm
    };

    // Events
    const handleNodeClick = useCallback((node) => {
        setSelectedNode(node);
        if (fgRef.current) {
            // Aim at node from outside it
            const distance = 100;
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
            fgRef.current.cameraPosition(
                { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
                node, // lookAt ({ x, y, z })
                3000  // ms transition duration
            );
        }
    }, [fgRef]);

    const handleBackgroundClick = useCallback(() => {
        setSelectedNode(null);
        if (fgRef.current) {
            fgRef.current.cameraPosition({ x: 0, y: 0, z: 200 }, { x: 0, y: 0, z: 0 }, 1000);
        }
    }, [fgRef]);

    // Action Handler (Mock API Call)
    const handleAction = async (actionType) => {
        console.log(`Executing ${actionType} on ${selectedNode.id}`);
        try {
            await axios.post('http://localhost:8000/api/agent/action', {
                target: selectedNode.id,
                action: actionType,
                role: selectedNode.role
            });
            // Visual feedback handled by log update
        } catch (e) {
            console.error("Action Trigger Failed", e);
        }
    };

    return (
        <div ref={containerRef} className="w-full h-full relative group">
            <ForceGraph3D
                ref={fgRef}
                width={dims.width}
                height={dims.height}
                graphData={finalData}
                nodeLabel="id"
                nodeColor={getNodeColor}
                linkColor={link => link.isBeam ? link.color : getLinkColor()}
                linkWidth={link => link.isBeam ? 2 : 1}
                linkDirectionalParticles={link => link.isBeam ? 4 : 0}
                linkDirectionalParticleSpeed={link => link.isBeam ? 0.01 : 0}
                linkDirectionalParticleWidth={4}
                backgroundColor="rgba(0,0,0,0)"
                showNavInfo={false}
                onNodeClick={handleNodeClick}
                onBackgroundClick={handleBackgroundClick}
            />


            {/* LEGEND overlay */}
            {!selectedNode && (
                <div className="absolute bottom-4 right-4 bg-black/80 p-2 rounded border border-gray-700 text-[10px] pointer-events-none">
                    <span className="text-gray-400">REACTIVE NODES ACTIVE</span>
                </div>
            )}

            {/* NODE DETAIL CARD + AGENT MENU */}
            {selectedNode && (
                <div className="absolute top-4 right-4 w-64 bg-black/90 border border-glasc-neon shadow-[0_0_30px_rgba(0,238,255,0.3)] backdrop-blur-xl p-4 rounded-lg text-sm select-none animate-in fade-in slide-in-from-right-10 duration-300">
                    <button onClick={handleBackgroundClick} className="absolute top-2 right-2 text-gray-500 hover:text-white cursor-pointer z-50">
                        <X size={16} />
                    </button>
                    <h3 className="text-glasc-neon text-lg font-bold mb-1">{selectedNode.id}</h3>
                    <div className="text-gray-400 text-xs uppercase mb-4 tracking-widest">{selectedNode.role}</div>

                    <div className="space-y-3">
                        <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                            <span className="text-gray-400 flex items-center gap-2"><DollarSign size={14} /> Stake</span>
                            <span className="font-mono text-white">{selectedNode.shares}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                            <span className="text-gray-400 flex items-center gap-2"><Shield size={14} /> Loyalty</span>
                            <div className="w-16 bg-gray-800 h-1.5 rounded-full overflow-hidden">
                                <div className={`h-full ${selectedNode.loyalty > 50 ? 'bg-glasc-neon' : 'bg-red-500'}`} style={{ width: `${selectedNode.loyalty}%` }}></div>
                            </div>
                        </div>
                    </div>

                    {/* ACTION MENU */}
                    <div className="mt-4 border-t border-gray-800 pt-3">
                        <p className="text-[10px] text-gray-500 uppercase mb-2 font-bold tracking-wider">Operational Levers</p>
                        <div className="grid grid-cols-1 gap-2">
                            {selectedNode.role === 'Debtholder' && (
                                <button onClick={() => handleAction('CUT_CREDIT')} className="bg-red-900/40 border border-red-500/50 text-red-400 text-xs py-2 rounded hover:bg-red-900/80 flex items-center justify-center gap-2 transition-colors">
                                    <Zap size={12} /> TRIGGER LIQUIDITY CRISIS
                                </button>
                            )}
                            {selectedNode.group === 1 && (
                                <button onClick={() => handleAction('BETRAY_CEO')} className="bg-orange-900/40 border border-orange-500/50 text-orange-400 text-xs py-2 rounded hover:bg-orange-900/80 flex items-center justify-center gap-2 transition-colors">
                                    <Users size={12} /> FLIP LOYALTY (BRIBE)
                                </button>
                            )}
                            <button onClick={() => handleAction('LEAK_ALPHA')} className="bg-blue-900/40 border border-blue-500/50 text-blue-400 text-xs py-2 rounded hover:bg-blue-900/80 flex items-center justify-center gap-2 transition-colors">
                                <Shield size={12} /> LEAK ALPHA INTEL
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CorporateGraph;
