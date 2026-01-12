import React, { useEffect, useRef } from 'react';
import { Terminal } from 'lucide-react';

const NegotiationTerminal = ({ logs = [], onSendMessage }) => {
    const bottomRef = useRef(null);
    const [isGodMode, setIsGodMode] = React.useState(false);
    const [input, setInput] = React.useState("");

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!input.trim()) return;
        if (onSendMessage) onSendMessage(input);
        setInput("");
    };

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="flex flex-col h-full bg-black/80 border border-glasc-neon/30 rounded-lg p-2 font-mono text-xs overflow-hidden backdrop-blur-md shadow-[0_0_15px_rgba(0,238,255,0.1)]">
            <div className="flex items-center gap-2 border-b border-glasc-neon/30 pb-2 mb-2 text-glasc-neon animate-pulse">
                <Terminal size={14} />
                <span className="font-bold tracking-widest uppercase text-[10px]">Neural Interface // Society Uplink</span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 scrollbar-hide">
                {logs.length === 0 && (
                    <div className="text-gray-600 italic text-center mt-10">
                        ... Awaiting encrypted transmission ...
                    </div>
                )}
                {logs.map((log, i) => {
                    const isChat = log.type === "NEGOTIATION_EVENT";

                    if (isChat) {
                        return (
                            <div key={i} className="flex flex-col gap-1 w-full animate-fadeIn">
                                <div className="flex items-center gap-2 text-[10px] text-gray-500 mb-0.5 ml-2">
                                    <span>[{log.time}]</span>
                                    <span className="text-glasc-neon font-bold uppercase">{log.sender}</span>
                                    <span>to</span>
                                    <span className="text-gray-400">{log.message?.target || "Unknown"}</span>
                                </div>
                                <div className="bg-gray-900/80 border-l-2 border-glasc-neon/50 p-2 rounded-r-md text-gray-300 ml-2 italic">
                                    "{log.content}"
                                </div>
                            </div>
                        );
                    }

                    // Standard Action Log
                    return (
                        <div key={i} className="flex gap-2 animate-fadeIn opacity-90 hover:opacity-100 transition-opacity bg-red-900/10 p-1 rounded">
                            <span className="text-gray-500">[{log.time}]</span>
                            <span className={`font-bold ${log.source === 'ATTACKER' ? 'text-red-400' :
                                log.source === 'DEFENDER' ? 'text-blue-400' :
                                    'text-yellow-400'
                                }`}>
                                {log.source}
                            </span>
                            <span className="text-gray-300">::</span>
                            <div className="text-gray-200">
                                <span className="text-glasc-neon/80 mr-1">{log.action ? `[ACTION: ${log.action}]` : ''}</span>
                                {log.message}
                            </div>
                        </div>
                    );
                })}
                <div ref={bottomRef} />
            </div>

            <div className={`mt-2 border-t border-gray-800 pt-1 transition-all duration-300 ${isGodMode ? 'opacity-100' : 'opacity-70'}`}>
                <div className="flex items-center justify-between mb-2">
                    <div className="text-[9px] text-gray-600 flex gap-2">
                        <span>MODE: {isGodMode ? "INTERVENTION ACTIVE" : "AUTONOMOUS SOCIETY"}</span>
                        <span>ENCRYPTION: 256-BIT</span>
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer group">
                        <span className={`text-[9px] font-bold ${isGodMode ? 'text-glasc-neon animate-pulse' : 'text-gray-600'}`}>
                            GOD MODE
                        </span>
                        <div className="relative">
                            <input
                                type="checkbox"
                                className="sr-only"
                                checked={isGodMode}
                                onChange={(e) => setIsGodMode(e.target.checked)}
                            />
                            <div className={`block w-8 h-4 rounded-full transition-colors ${isGodMode ? 'bg-glasc-neon/30 border-glasc-neon' : 'bg-gray-800 border-gray-700'} border`}></div>
                            <div className={`absolute left-1 top-0.5 bg-white w-3 h-3 rounded-full transition-transform ${isGodMode ? 'translate-x-4 bg-glasc-neon shadow-[0_0_10px_#00eeff]' : 'bg-gray-500'}`}></div>
                        </div>
                    </label>
                </div>

                {isGodMode && (
                    <form onSubmit={handleSubmit} className="flex gap-2 animate-slideUp">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Inject command into neural stream..."
                            className="flex-1 bg-black/50 border border-glasc-neon/30 rounded px-2 py-1 text-xs text-glasc-neon placeholder-gray-700 outline-none focus:border-glasc-neon focus:shadow-[0_0_10px_rgba(0,238,255,0.2)]"
                        />
                        <button
                            type="submit"
                            className="bg-glasc-neon/10 hover:bg-glasc-neon/20 text-glasc-neon border border-glasc-neon/50 rounded px-3 py-1 text-[10px] font-bold uppercase transition-all"
                        >
                            SEND
                        </button>
                    </form>
                )}
            </div>
        </div>
    );
};

export default NegotiationTerminal;
