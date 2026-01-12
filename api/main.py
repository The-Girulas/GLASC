import asyncio
import json
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jax
import jax.numpy as jnp

from glasc.core.market_dynamics import MarketParameters, compute_success_probability
from glasc.core.agent_orchestrator import create_orchestrator
from glasc.core.agent_system import GameEngine
from glasc.core.simulation_manager import SimulationManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sim_manager = SimulationManager()

# --- API ---

class LaunchRequest(BaseModel):
    ticker: str
    volatility_override: float = 0.2

@app.post("/api/sim/launch")
async def launch_sim(req: LaunchRequest):
    sim_manager.ticker = req.ticker
    sim_manager.params = sim_manager.params.replace(sigma=req.volatility_override)
    sim_manager.current_price = 100.0
    sim_manager.running = True
    
    # Start loop in background if not running
    # Note: In production use proper background tasks
    asyncio.create_task(sim_manager.run_loop())
    
    return {"status": "LAUNCHED", "ticker": req.ticker}

class ChaosRequest(BaseModel):
    volatility: float
    event: str # "SCANDAL", "RATES_HIKE"

@app.post("/api/sim/chaos")
async def inject_chaos(req: ChaosRequest):
    print(f"INJECTING CHAOS: {req.event} (Vol -> {req.volatility})")
    sim_manager.update_params(volatility=req.volatility)
    
    # Send a dramatic alert to frontend
    await sim_manager.broadcast({
        "type": "ALERT",
        "message": f"CHAOS EVENT: {req.event} TRIGGERED!",
        "severity": "CRITICAL"
    })
    
    # Force a rapid price drop if Scandal
    if req.event == "SCANDAL":
        sim_manager.current_price *= 0.85 # -15% instant gap
        sim_manager.update_params(sentiment="bearish")
        
    return {"status": "CHAOS_INJECTED"}

class AgentInteraction(BaseModel):
    agent_id: str
    action_type: str # e.g. "CUT_CREDIT", "LEAK_ALPHA"

@app.post("/api/agent/action")
async def agent_act(req: AgentInteraction):
    """ Agent System Entry Point """
    # 1. Resolve Action Impact via Game Logic
    role = "Banker" if "Bank" in req.agent_id else ("Insider" if "Founder" in req.agent_id else "Defender")
    if "Target" in req.agent_id or "CEO" in req.agent_id: role = "Defender"
        
    action_result = GameEngine.execute_action(req.action_type, role)

    if action_result.action_type == "ERROR":
        return {"status": "ERROR", "message": "Unknown Action"}

    # 2. Apply to Simulation
    sim_manager.apply_impact(
        action_result.impact_price, 
        action_result.impact_volatility,
        action_result.impact_attacker,
        action_result.impact_defender
    )
    
    # 3. Broadcast Agent Log
    timestamp_str = time.strftime("%H:%M:%S", time.localtime())
    
    # Map role to source type for UI
    source_type = "NEUTRAL"
    if role == "Defender": source_type = "DEFENDER"
    elif role == "Insider" or role == "Banker": source_type = "STAKEHOLDER" # Could be varied
    
    # If action is hostile, label as ATTACKER helper
    if action_result.impact_attacker > 0: source_type = "ATTACKER"
    
    msg = {
        "type": "AGENT_LOG",
        "time": timestamp_str,
        "source": role.upper(), # Or req.agent_id
        "action": req.action_type,
        "message": action_result.description,
        "impact_data": action_result.dict()
    }
    await sim_manager.broadcast(msg)
    
    return {"status": "EXECUTED", "impact": action_result.dict()}

class UserMessageRequest(BaseModel):
    content: str
    target: str = "ALL"

@app.post("/api/sim/inject_message")
async def inject_message(req: UserMessageRequest):
    await sim_manager.inject_user_message(req.content, req.target)
    return {"status": "SENT"}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim_manager.subscribers.append(websocket)
    try:
        while True:
            # Keep alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in sim_manager.subscribers:
            sim_manager.subscribers.remove(websocket)
