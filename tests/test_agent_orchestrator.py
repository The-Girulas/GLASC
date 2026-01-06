"""
Tests d'intégration pour Agent Orchestrator.
ATTENTION : Ce test charge un LLM réel (Qwen2.5-1.5B).
La première exécution peut prendre du temps (téléchargement).
"""

import unittest
import logging
from glasc.core.agent_orchestrator import create_orchestrator, AgentState
from glasc.utils.config_loader import load_config

# Configure logging to see LLM thoughts
logging.basicConfig(level=logging.INFO)

class AgentOrchestratorTest(unittest.TestCase):
    
    def setUp(self):
        self.app = create_orchestrator()
        self.config = load_config()

    def test_full_campaign_simulation(self):
        """
        Simule une campagne complète.
        Le Strategist (LLM) doit recevoir les rapports et prendre une décision.
        """
        initial_state = {
            "ticker": "EVIL_CORP",
            "messages": [],
            "market_data": {},    # Sera rempli par DataScout
            "risk_report": {},    # Sera rempli par QuantAnalyst
            "corp_structure": {}, # Sera rempli par CorporateSpy
            "brain_score": 0.0,   # Sera rempli par PredatorOracle
            "decision": "PENDING"
        }
        
        print("\n--- STARTING CAMPAIGN SIMULATION (May take time for LLM load) ---")
        
        # Run the graph
        final_state = self.app.invoke(initial_state)
        
        print("\n--- CAMPAIGN FINISHED ---")
        print(f"Final Decision: {final_state['decision']}")
        print("Messages History:")
        for m in final_state['messages']:
            print(f" - {m.type}: {m.content}")
            
        # Assertions
        self.assertIn(final_state['decision'], ["ATTACK", "WAIT", "ABORT"])
        self.assertTrue(len(final_state['messages']) >= 4) # At least 4 steps reported
        
        # Check if JAX worked
        r = final_state['risk_report']
        self.assertTrue("VaR_95" in r)
        self.assertGreater(r["VaR_95"], 0)

if __name__ == '__main__':
    unittest.main()
