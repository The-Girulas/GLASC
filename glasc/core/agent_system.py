from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import random

# --- 1. THE TOOLS (ACTIONS) ---
class AgentAction(BaseModel):
    agent_id: str
    action_type: str
    impact_price: float = 0.0
    impact_volatility: float = 0.0
    impact_probability: float = 0.0
    impact_attacker: float = 0.0
    impact_defender: float = 0.0
    description: str

class GameEngine:
    """
    Translates Agent Intent into Market Consequence.
    Acts as the 'Physics Engine' for Social Engineering.
    """
    
    @staticmethod
    def execute_action(action_name: str, agent_role: str) -> AgentAction:
        impacts = {
            # --- BANKER TOOLS ---
            "CUT_CREDIT": {"price": -0.08, "vol": 0.15, "att": 0.10, "def": -0.10, "desc": "Bank triggered Liquidity Crisis! Credit lines frozen."},
            "EXTEND_LOAN": {"price": 0.02, "vol": -0.05, "att": -0.05, "def": 0.05, "desc": "Bank stabilizes balance sheet. Loan extended."},
            
            # --- INSIDER TOOLS ---
            "LEAK_ALPHA": {"price": -0.05, "vol": 0.10, "att": 0.08, "def": -0.05, "desc": "Insider leaked confidential misses to press."},
            "BETRAY_CEO": {"price": -0.12, "vol": 0.20, "att": 0.15, "def": -0.15, "desc": "Key Insider flipped to Hostile! Board in shamble."},
            
            # --- DEFENDER TOOLS (AI CEO) ---
            "BUYBACK": {"price": 0.04, "vol": -0.02, "att": -0.05, "def": 0.10, "desc": "Target Corp launches emergency Share Buyback."},
            "POISON_PILL": {"price": -0.02, "vol": 0.05, "att": -0.20, "def": 0.10, "desc": "Poison Pill triggered! Acquisition cost skyrockets."},
            "WHITE_KNIGHT": {"price": 0.10, "vol": -0.10, "att": -0.30, "def": 0.20, "desc": "Friendly merge partner detected (White Knight)."}
        }

        if action_name not in impacts:
            return AgentAction(agent_id="SYSTEM", action_type="ERROR", description="Unknown Action")

        eff = impacts[action_name]
        return AgentAction(
            agent_id=agent_role,
            action_type=action_name,
            impact_price=eff["price"],
            impact_volatility=eff["vol"],
            impact_attacker=eff.get("att", 0.0),
            impact_defender=eff.get("def", 0.0),
            description=eff["desc"]
        )

# --- 2. THE PERSONAS (BRAINS) ---
class AgentPersona(BaseModel):
    role: str
    risk_tolerance: float # 0.0 (Chicken) to 1.0 (Cowboy)
    loyalty: float # 0.0 (Traitor) to 1.0 (Dog)
    
    def decide(self, market_state: dict, offer_value: float) -> str:
        """
        Simple Heuristic Brain (Can be upgraded to LLM later).
        Decides whether to accept a 'Deal' or 'Act' based on market stress.
        """
        # Fear Factor
        fear = market_state.get("volatility", 0.2) * (1 - self.risk_tolerance)
        
        # Greed Factor
        greed = offer_value 

        # Decision Logic
        if self.role == "Banker":
            # Bankers hate volatility. If fear is high, they cut credit.
            if fear > 0.6: return "CUT_CREDIT"
            if greed > 0.8: return "EXTEND_LOAN"

        if self.role == "Insider":
            # Insiders flip if Loyalty is low AND Fear is high (Rat fleeing ship)
            if self.loyalty < 0.3 and fear > 0.5: return "BETRAY_CEO"
            if greed > 0.0: return "LEAK_ALPHA" # Bribed

        return "HOLD"
