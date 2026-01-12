import random
import time
from typing import List, Dict, Optional
from pydantic import BaseModel

class DialogueMessage(BaseModel):
    sender: str
    target: str
    content: str
    timestamp: float
    intent: str # e.g., "THREAT", "OFFER", "IGNORE"

class NegotiationEngine:
    def __init__(self, use_local_llm: bool = False, llm_endpoint: str = "http://localhost:11434/api/generate"):
        self.use_local_llm = use_local_llm
        self.llm_endpoint = llm_endpoint
        self.history: List[DialogueMessage] = []
        
    def generate_dialogue(self, initiator: str, target: str, context: Dict) -> DialogueMessage:
        """
        Generates a message from Initiator to Target based on context.
        If use_local_llm is False, uses heuristics.
        """
        if self.use_local_llm:
            return self._call_llm(initiator, target, context)
        else:
            return self._heuristic_dialogue(initiator, target, context)

    def _heuristic_dialogue(self, sender: str, target: str, context: Dict) -> DialogueMessage:
        """ Fallback 'Mock' LLM for testing key mechanics """
        prob = context.get("probability", 0.5)
        power_def = context.get("defender_power", 0.5)
        
        # Simple Strategy Map
        templates = []
        intent = "NEUTRAL"
        
        if sender == "Bank Syndicate":
            if prob > 0.6:
                templates = [
                    "We are exposed here. The risk metrics are bleeding red.",
                    "If you don't stabilize the price, we're pulling the credit line.",
                    "This volatility is unacceptable. Fix it or we execute clauses."
                ]
                intent = "THREAT"
            else:
                templates = ["Monitoring the situation. Keep ratios steady.", "Liquidity looks fine for now."]
        
        elif sender == "Founder":
            if power_def < 0.4:
                templates = [
                    "I built this company! You can't let them take it!",
                    "Call the White Knight! Do something!",
                    "My legacy is at stake here."
                ]
                intent = "DESPERATE"
            else:
                templates = ["Hold the line. We have the votes.", "The employees are with us."]
        
        elif sender == "GLASC_CORP" (self): # The Defender AI
             templates = ["Initiating defense protocols.", "We are buying back shares to stabilize."]
             intent = "ACTION"

        # Default fallback
        if not templates:
            templates = [f"I am watching you, {target}.", "What is your move?"]

        content = random.choice(templates)
        
        return DialogueMessage(
            sender=sender,
            target=target,
            content=content,
            timestamp=time.time(),
            intent=intent
        )

    def _call_llm(self, sender: str, target: str, context: Dict) -> DialogueMessage:
        """ 
        Real Integration with Local/Remote LLM.
        Defaults to Ollama format (Llama3/Mistral).
        """
        import requests
        import json

        # prompt construction
        system_prompt = (
            f"You are {sender}, a stakeholder in a corporate takeover battle. "
            f"You are speaking to {target}. "
            f"Current Situation: Takeover Probability is {context.get('probability', 0.5):.2f}. "
            f"Defender Power is {context.get('defender_power', 0.5):.2f}. "
            "Your goal is to negotiate or threaten to protect your interests. "
            "Respond with a JSON object: { \"content\": \"...\", \"intent\": \"THREAT|OFFER|INFO\" }"
        )

        payload = {
            "model": "llama3", 
            "prompt": system_prompt, 
            "stream": False,
            "format": "json" 
        }

        try:
            # 5-second timeout to not block the game loop
            response = requests.post(self.llm_endpoint, json=payload, timeout=3.0)
            response.raise_for_status()
            
            data = response.json()
            # Ollama returns 'response' field, others choices[0].text
            raw_text = data.get("response", "") 
            
            # Parse inner JSON if model obeyed
            parsed = json.loads(raw_text)
            content = parsed.get("content", "I am uncertain.")
            intent = parsed.get("intent", "NEUTRAL")
            
            return DialogueMessage(
                sender=sender,
                target=target,
                content=content,
                timestamp=time.time(),
                intent=intent
            )
            
        except Exception as e:
            # Fallback only on explicit failure (connection refused/timeout)
            print(f"LLM CALL FAILED: {e}. Falling back to heuristic.")
            return self._heuristic_dialogue(sender, target, context)

    def evaluate_decision(self, message: DialogueMessage) -> Optional[str]:
        """
        Determines if a message triggers an IMMEDIATE Consequence/Action.
        E.g. "THREAT" might trigger a counter-action or submission.
        """
        if message.intent == "THREAT" and random.random() > 0.7:
             return "PANIC_SELL"
        if message.intent == "OFFER" and random.random() > 0.5:
             return "ACCEPT_DEAL"
             
        return None
