"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Agent Orchestrator

Le Cœur du système. Utilise LangGraph pour orchestrer la prise de décision.
Connecte le LLM (Strategist) aux outils JAX/NetworkX/Flax.
"""

import json
from typing import TypedDict, Annotated, List, Dict, Any, Union
from langgraph.graph import StateGraph, END
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from glasc.core.llm_client import LLMClient
from glasc.core.market_dynamics import MarketParameters, simulate_paths
from glasc.core.risk_engine import calculate_var_es
from glasc.core.corporate_graph import CorporateNetwork, Shareholder, DebtTranche
from glasc.core.predator_brain import PredatorModel, RealisticDataGenerator # In reality we'd load weights here
import jax
import jax.numpy as jnp

# --- 1. State Definition ---

class AgentState(TypedDict):
    ticker: str
    messages: Annotated[List[BaseMessage], operator.add]
    market_data: Dict[str, Any]
    risk_report: Dict[str, Any]
    corp_structure: Dict[str, Any]
    brain_score: float
    decision: str  # "WAIT", "ATTACK", "ABORT"

# --- 2. Nodes (Specialists) ---

def data_scout_node(state: AgentState) -> Dict:
    """Simule la récupération de données pour le ticker."""
    # Mock data for now, but structured
    return {
        "market_data": {
            "price": 100.0,
            "volatility": 0.2,
            "net_debt_ebitda": 3.5,
            "ebitda_margin": 0.12
        },
        "messages": [SystemMessage(content=f"DataScout: Retrieved data for {state['ticker']}.")]
    }

def quant_analyst_node(state: AgentState) -> Dict:
    """Lance les simulations JAX."""
    # Utilisation réelle de notre market_dynamics.py
    # On crée des params basés sur market_data
    md = state["market_data"]
    params = MarketParameters(
        mu=0.05,
        sigma=md["volatility"],
        lambda_j=0.1,
        jump_mean=-0.1,
        jump_std=0.1
    )
    key = jax.random.PRNGKey(42)
    # Simulation rapide 10k paths
    paths = simulate_paths(key, params, md["price"], 1.0, 252, 10_000)
    final_prices = paths[:, -1]
    var, es = calculate_var_es(final_prices)
    
    return {
        "risk_report": {"VaR_95": float(var), "ES_95": float(es)},
        "messages": [SystemMessage(content=f"QuantAnalyst: JAX Simulation complete. VaR={var:.2f}")]
    }

def corporate_spy_node(state: AgentState) -> Dict:
    """Construit et analyse le graphe NetworkX."""
    # Création d'un graph simple à la volée pour l'analyse
    corp = CorporateNetwork(state["ticker"])
    # Mock some data based on what we might know
    corp.add_shareholder(Shareholder(name="PensionFund_A", share_count=40000, category="INSTITUTIONAL", cost_basis=80, loyalty_score=0.4))
    corp.add_shareholder(Shareholder(name="Founder_Fam", share_count=30000, category="INSIDER", cost_basis=10, loyalty_score=0.9))
    corp.add_debt(DebtTranche(lender_name="Bank_A", amount=500000, interest_rate=0.05, maturity_years=3, max_leverage_ratio=4.0))
    
    dist = corp.get_shareholder_distribution()
    # Test OPA +20%
    success_prob = corp.simulate_tender_offer(120.0, 100.0)
    
    return {
        "corp_structure": {"shareholders": dist, "tender_success_prob_20pct_premium": success_prob},
        "messages": [SystemMessage(content=f"CorporateSpy: Analysis done. Tender Success Prob @ +20% = {success_prob:.2%}")]
    }

def predator_oracle_node(state: AgentState) -> Dict:
    """Interroge le Neural Network."""
    # Ici on ferait l'inférence réelle si on avait les poids chargés.
    # Pour le test d'intégration, on utilise le générateur synthétique pour avoir un score cohérent
    # ou on mocke la forward pass si pas entrainé.
    # Disons qu'on retourne un score basé sur la dette (logique apprise par le Brain)
    md = state["market_data"]
    debt = md["net_debt_ebitda"]
    # Logique simple pour simuler le NN entrainé : High Debt = Low Score
    score = 1.0 / (1.0 + debt) # Simple proxy
    
    return {
        "brain_score": score,
        "messages": [SystemMessage(content=f"PredatorOracle: Neural Net Vulnerability Score = {score:.2f}")]
    }

def strategist_node(state: AgentState) -> Dict:
    """Le LLM décide."""
    llm = LLMClient()
    
    system_prompt = """You are GLASC Strategist, an elite M&A AI. 
    Analyze the reports. Decide: ATTACK, WAIT, or ABORT.
    Output JSON only: {"reasoning": "...", "decision": "COMMAND"}
    """
    
    user_prompt = f"""
    Target: {state['ticker']}
    Market: {state['market_data']}
    Risk (VaR): {state['risk_report']}
    Corporate Structure: {state['corp_structure']}
    AI Vulnerability Score: {state['brain_score']} (0=Safe, 1=Weak)
    
    Decision?
    """
    
    try:
        response = llm.generate(system_prompt, user_prompt)
        # Parse JSON crudely
        # LLMs often add markdown ```json ... ```
        clean_resp = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_resp)
        decision = data.get("decision", "WAIT").upper()
        reasoning = data.get("reasoning", "No reasoning provided.")
    except Exception as e:
        decision = "WAIT" # Fail safe
        reasoning = f"Error parsing LLM response: {e}"
        
    return {
        "decision": decision,
        "messages": [HumanMessage(content=f"Strategist: {decision}. Reason: {reasoning}")]
    }

# --- 3. The Graph ---

def create_orchestrator():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("DataScout", data_scout_node)
    workflow.add_node("QuantAnalyst", quant_analyst_node)
    workflow.add_node("CorporateSpy", corporate_spy_node)
    workflow.add_node("PredatorOracle", predator_oracle_node)
    workflow.add_node("Strategist", strategist_node)
    
    workflow.set_entry_point("DataScout")
    
    workflow.add_edge("DataScout", "QuantAnalyst")
    workflow.add_edge("QuantAnalyst", "CorporateSpy")
    workflow.add_edge("CorporateSpy", "PredatorOracle")
    workflow.add_edge("PredatorOracle", "Strategist")
    workflow.add_edge("Strategist", END) # Simple linear graph for V1
    
    return workflow.compile()

