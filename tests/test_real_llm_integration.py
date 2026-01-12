import asyncio
import sys
import os
import json
import requests
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glasc.core.negotiation_system import NegotiationEngine

def test_real_llm_call_structure():
    print("--- STARTING REAL LLM INTEGRATION TEST ---")
    
    # 1. Initialize Engine in "Real LLM Mode"
    engine = NegotiationEngine(use_local_llm=True, llm_endpoint="http://fake-ollama:11434/api/generate")
    
    context = {"probability": 0.9, "defender_power": 0.1}
    sender = "Bank Syndicate"
    target = "GLASC_CORP"
    
    # 2. Mock requests.post to intercept the call
    expected_response_json = {
        "response": json.dumps({
            "content": "This is a real generated threat.",
            "intent": "THREAT"
        })
    }
    
    mock_resp = MagicMock()
    mock_resp.json.return_value = expected_response_json
    mock_resp.status_code = 200
    
    with patch('requests.post', return_value=mock_resp) as mock_post:
        print("[TEST 1] Verifying Request Construction and Response Parsing")
        
        msg = engine.generate_dialogue(sender, target, context)
        
        # Verify call was made
        if not mock_post.called:
            print("  [FAILED] requests.post was not called!")
            sys.exit(1)
            
        # Verify Payload
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        
        print(f"  > Endpoint: {args[0]}")
        print(f"  > Model: {payload['model']}")
        print(f"  > Format: {payload['format']}")
        
        if payload['model'] != "llama3":
            print("  [FAILED] Incorrect model specified")
            sys.exit(1)
            
        if "Takeover Probability is 0.90" not in payload['prompt']:
            print(f"  [FAILED] Context missing from prompt. Got: {payload['prompt']}")
            sys.exit(1)

        # Verify Response Handled
        print(f"  > Parsed Content: {msg.content}")
        
        if msg.content != "This is a real generated threat.":
            print("  [FAILED] Engine did not parse LLM JSON correctly")
            sys.exit(1)
            
        print("  [PASSED] Request and Response handling correct.")

    # 3. Test Fallback
    print("\n[TEST 2] Verifying Broken Connection Fallback")
    with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Ollama Offline")):
         # Should not crash, but fall back to heuristic
         msg_fallback = engine.generate_dialogue(sender, target, context)
         
         if not msg_fallback.content:
             print("  [FAILED] No fallback content generated")
             sys.exit(1)
             
         print("  [PASSED] Gracefully fell back to heuristic on connection error.")

    print("\n--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_real_llm_call_structure()
