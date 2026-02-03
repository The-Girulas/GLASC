# GLASC: Global Leverage & Asset Strategy Controller

![GLASC War Room](https://via.placeholder.com/800x400?text=GLASC+War+Room+Interface)

## 🚀 Overview

**GLASC** is an advanced financial simulation and "War Room" designed to model, detect, and execute hostile takeovers (M&A) using **Agentic AI** and **High-Performance Computing**.

It combines:
- **JAX** for physics-based market simulation (Merton Jump Diffusion) running on GPU.
- **LangGraph** for autonomous agent orchestration (The "Strategist").
- **React/Three.js** for a futuristic, real-time visualization interface.

> **Research Goal**: To demonstrate how AI can move from "Passive Analysis" (Chatbot) to "Active Execution" (Agent) in complex financial environments.

## 🏗️ Architecture

The project is split into three layers:

### 1. The Physics Engine (Core)
*Located in `glasc/core/`*
- **Market Dynamics**: Simulates 100,000+ price paths in parallel using `jax.vmap` and `jax.jit`.
- **Risk Engine**: Calculates Greeks (Delta, Vega) via Automatic Differentiation (`jax.grad`).
- **Corporate Graph**: Models the web of shareholders, debt tranches, and influence using `NetworkX` and `Pydantic`.

### 2. The Brain (AI)
*Located in `glasc/core/predator_brain.py` & `agent_orchestrator.py`*
- **Predator Brain**: A Deep Learning classifier (Flax/Optax) trained on synthetic data to spot undervalued targets.
- **Strategist Agent**: A LangGraph state machine that reasons, uses tools (Calculate VaR, Check Covenants), and formulates an attack plan.

### 3. The War Room (Interface)
*Located in `frontend/` & `api/`*
- **FastAPI**: Serves the simulation state via WebSockets (Real-time).
- **React + Vite**: Renders the "Iron Man" style dashboard.
- **3D Graph**: Interactive visualization of the corporate network using `react-force-graph`.

## 🛠️ Usage

### Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) NVIDIA GPU for JAX acceleration.

### Quick Start

1.  **Start the Backend**:
    ```bash
    cd Projet/Recherche/GLASC
    pip install -r requirements.txt
    python -m uvicorn api.main:app --reload
    ```

2.  **Start the Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

3.  **Access the War Room**:
    Open `http://localhost:5173`.

## 📚 Masterclasses (Documentation)

We provide detailed "Masterclasses" explaining the deep tech behind each component:

- **[Market Dynamics & JAX](docs/masterclasses/market_dynamics_masterclass.md)**: How we simulate millions of paths in milliseconds.
- **[Risk Engine & AD](docs/masterclasses/risk_engine_masterclass.md)**: Using Automatic Differentiation for risk sensibilities.
- **[Corporate Graph](docs/masterclasses/corporate_graph_masterclass.md)**: Modeling the intricate web of ownership.
- **[Predator Brain](docs/masterclasses/predator_brain_masterclass.md)**: Training a Neural Network on synthetic financial data.
- **[Agent Orchestrator](docs/masterclasses/agent_orchestrator_masterclass.md)**: Building the AI Strategist with LangGraph.
- **[War Room Architecture](docs/masterclasses/war_room_v2_architecture.md)**: The "Hollywood-Grade" UI stack.

*(French versions are available as `_FR.md`)*

## 🔮 Future Work

- **Multi-Agent Simulation**: "Simulated Society" where multiple funds compete for the target.
- **Real-Time News Injection**: Connecting the engine to live RSS feeds.
- **LLM-driven Negotiation**: Autonomous negotiation between the Attacker and Board Members.

---

*This project is part of the Advanced Agentic Coding Research.*
