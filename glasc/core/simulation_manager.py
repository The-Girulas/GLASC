import asyncio
import json
import random
import time
import jax
import jax.numpy as jnp
from glasc.core.market_dynamics import MarketParameters, compute_success_probability
from glasc.core.agent_system import GameEngine
from glasc.core.negotiation_system import NegotiationEngine

class SimulationManager:
    def __init__(self):
        self.running = False
        self.ticker = "GLASC_CORP"
        self.current_price = 100.0
        self.target_price = 90.0 # If price drops below this, acquisition is easier
        self.params = MarketParameters(
            mu=0.05, 
            sigma=0.20, # Initial low vol
            lambda_j=0.1, 
            jump_mean=-0.1, 
            jump_std=0.1
        )
        self.prob = 0.50
        self.subscribers = []
        self.key = jax.random.PRNGKey(int(time.time()))
        self.power_balance = {"attacker": 0.1, "defender": 0.5}
        self.last_defense_time = 0
        
        # Society Engine
        self.negotiation_engine = NegotiationEngine()
        self.last_dialogue_time = 0

    async def broadcast(self, message: dict):
        disconnected = []
        for ws in self.subscribers:
            try:
                await ws.send_json(message)
            except:
                disconnected.append(ws)
        for ws in disconnected:
            self.subscribers.remove(ws)

    def update_params(self, volatility=None, sentiment=None):
        if volatility is not None:
            self.params = self.params.replace(sigma=volatility)
        # Sentiment could affect mu (drift)
        if sentiment == "bearish":
            self.params = self.params.replace(mu=-0.2)
        elif sentiment == "bullish":
            self.params = self.params.replace(mu=0.2)

    def apply_impact(self, price_change_pct: float, vol_change_pct: float, att_change: float = 0.0, def_change: float = 0.0):
        """ Apply instantaneous shock from Agent Actions """
        self.current_price *= (1.0 + price_change_pct)
        # Volatility is mean-reverting in long term but sticky in short term
        self.params = self.params.replace(sigma=max(0.1, min(1.5, self.params.sigma + vol_change_pct)))
        
        # Update Power Balance (Clamp between 0 and 1)
        self.power_balance["attacker"] = max(0.0, min(1.0, self.power_balance["attacker"] + att_change))
        self.power_balance["defender"] = max(0.0, min(1.0, self.power_balance["defender"] + def_change))
            
    async def run_loop(self):
        """Main Lifeblood Loop"""
        print("Starting Simulation Loop...")
        while self.running:
            # 1. Update Market Price (Micro-simulation / Random Walk)
            # This is the "Tick"
            # Increase volatility multiplier for more visual movement
            shock = random.gauss(0, self.params.sigma) 
            self.current_price *= (1.0 + shock * 0.05) # 5% of sigma per tick
            
            # Ensure price doesn't go below 1.0 (bankruptcy)
            self.current_price = max(1.0, self.current_price)
            
            # 2. Run JAX Engine -> Compute Probability of Takeover
            # We split the key every time
            self.key, subkey = jax.random.split(self.key)
            
            # Calculate Prob: P(Price < Target) in 1 year
            # High Volatility -> Wider distribution -> Higher prob of hitting low types if drift is neg?
            # Or just re-eval based on current price drop.
            
            prob_jax = compute_success_probability(
                subkey, self.params, self.current_price, 1.0, self.target_price
            )
            self.prob = float(prob_jax)

            # 3. Broadcast State
            msg = {
                "type": "TICK",
                "price": self.current_price,
                "probability": self.prob,
                "volatility": self.params.sigma,
                "power_balance": self.power_balance,
                "timestamp": time.time()
            }
            await self.broadcast(msg)
            
            # 4. Agent Thoughts (Adversarial AI)
            current_time = time.time()
            if current_time - self.last_defense_time > 8.0: # Cooldown
                chosen_action = None
                
                # Logic: If Probability > 70% or Defender Power < 30%
                if self.prob > 0.70:
                    chosen_action = "BUYBACK"
                if self.power_balance["defender"] < 0.30:
                    chosen_action = "WHITE_KNIGHT" if random.random() > 0.5 else "POISON_PILL"
                
                if chosen_action:
                    self.last_defense_time = current_time
                    
                    # Execute
                    print(f"DEFENDER AI TRIGGERING: {chosen_action}")
                    action_result = GameEngine.execute_action(chosen_action, "Defender_AI")
                    
                    # Apply
                    self.apply_impact(
                        action_result.impact_price, 
                        action_result.impact_volatility,
                        action_result.impact_attacker,
                        action_result.impact_defender
                    )
                    
                    # Broadcast Log
                    log_msg = {
                        "type": "AGENT_LOG",
                        "time": time.strftime("%H:%M:%S", time.localtime()),
                        "source": "DEFENDER_AI",
                        "action": chosen_action,
                        "message": action_result.description,
                        "impact_data": action_result.dict()
                    }
                    await self.broadcast(log_msg)

            # 5. Society Dialogue Loop (NegotiationEngine)
            if current_time - self.last_dialogue_time > 6.0:
                self.last_dialogue_time = current_time
                
                # Pick Random Pair
                initiator = random.choice(["Bank Syndicate", "Founder", "Hedge Fund A"])
                target = "GLASC_CORP" if initiator != "GLASC_CORP" else "Bank Syndicate"
                
                # Generate Dialogue
                context = {"probability": self.prob, "defender_power": self.power_balance["defender"]}
                msg_obj = self.negotiation_engine.generate_dialogue(initiator, target, context)
                
                # Evaluate Consequence
                consequence = self.negotiation_engine.evaluate_decision(msg_obj)
                
                # Broadcast Chat
                chat_msg = {
                    "type": "NEGOTIATION_EVENT",
                    "time": time.strftime("%H:%M:%S", time.localtime()),
                    "sender": msg_obj.sender,
                    "content": msg_obj.content,
                    "intent": msg_obj.intent
                }
                await self.broadcast(chat_msg)
                
                # Execute Consequence if any
                if consequence:
                    print(f"NEGOTIATION CONSEQUENCE: {consequence}")
                    # Reuse GameEngine for consequence
                    # ... (Implementation detail: map consequence to action)

            # 6. Check Game Over
            if self.power_balance["defender"] <= 0.05: # Allow small margin
                await self._trigger_game_over("VICTORY") # User wins (Attacker)
                break
            elif self.power_balance["defender"] >= 0.95:
                await self._trigger_game_over("DEFEAT") # User loses
                break

            await asyncio.sleep(0.1) # 10 Hz update for smoother UI

    async def _trigger_game_over(self, result: str):
        print(f"GAME OVER: {result}")
        msg = {
            "type": "GAME_OVER",
            "result": result,
            "stats": {
                "price": self.current_price,
                "duration": int(time.time()), # Placeholder duration
                "attackerInfluence": self.power_balance["attacker"],
                "defenderControl": self.power_balance["defender"]
            }
        }
        await self.broadcast(msg)
        self.running = False

    async def inject_user_message(self, content: str, target: str = "ALL"):
        """ Allows the user (God Mode) to inject a message into the stream """
        # Create Message
        from glasc.core.negotiation_system import DialogueMessage
        msg_obj = DialogueMessage(
            sender="USER_GOD_MODE",
            target=target,
            content=content,
            timestamp=time.time(),
            intent="INTERCERPTION"
        )

        # Broadcast
        chat_msg = {
            "type": "NEGOTIATION_EVENT",
            "time": time.strftime("%H:%M:%S", time.localtime()),
            "sender": msg_obj.sender,
            "content": msg_obj.content,
            "intent": msg_obj.intent
        }
        await self.broadcast(chat_msg)
        
        # Optional: Trigger consequence (Agents might panic if God speaks)
        if "SELL" in content.upper() or "CRASH" in content.upper():
            print("GOD MODE TRIGGERED PANIC")
            self.apply_impact(-0.05, 0.1, 0.1, -0.1) # Price drop, Vol up
