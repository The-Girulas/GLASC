import React from 'react';
import { AlertTriangle } from 'lucide-react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("ErrorBoundary caught an error", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="w-full h-full flex flex-col items-center justify-center bg-black/90 text-red-500 border border-red-500/30 rounded p-4">
                    <AlertTriangle size={32} className="mb-2 animate-pulse" />
                    <h2 className="text-lg font-bold mb-1">COMPONENT CRASHED</h2>
                    <p className="text-xs font-mono text-center opacity-80 max-w-xs break-all">
                        {this.state.error && this.state.error.toString()}
                    </p>
                    <button
                        onClick={() => this.setState({ hasError: false })}
                        className="mt-4 px-3 py-1 bg-red-900/40 text-red-300 text-xs rounded hover:bg-red-800/60"
                    >
                        RETRY
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
