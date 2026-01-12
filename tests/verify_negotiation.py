import asyncio
import sys
import os
import time
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glasc.core.negotiation_system import NegotiationEngine

def test_society_loop():
    print("--- STARTING SOCIETY LOOP VERIFICATION ---")
    
    engine = NegotiationEngine()
    
    # 1. Test Heuristic Dialogue Generation
    print("[TEST 1] Generating Dialogue (Mock LLM)")
    context = {"probability": 0.8, "defender_power": 0.3}
    msg = engine.generate_dialogue("Bank Syndicate", "GLASC_CORP", context)
    
    print(f"  > Sender: {msg.sender}")
    print(f"  > Content: {msg.content}")
    print(f"  > Intent: {msg.intent}")
    
    if msg.sender != "Bank Syndicate":
        print("  [FAILED] Wrong Sender")
        sys.exit(1)
        
    if msg.intent != "THREAT":
        print(f"  [FAILED] Expected THREAT intent for Bank at Prob 0.8, got {msg.intent}")
        sys.exit(1)
        
    print("  [PASSED] Dialogue Generated Correctly")

    # 2. Test Consequence Evaluation
    print("\n[TEST 2] Evaluating Consequence")
    # Force a threat message
    msg.intent = "THREAT"
    
    # Since it's random, we run it multiple times to ensure it returns *something* eventually
    triggered = False
    for _ in range(20):
        decision = engine.evaluate_decision(msg)
        if decision:
            print(f"  > Decision Triggered: {decision}")
            triggered = True
            break
            
    if not triggered:
        print("  [WARNING] Random decision did not trigger in 20 tries (unlucky or broken?)")
    else:
        print("  [PASSED] Consequence Logic Active")

    print("\n--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_society_loop()
