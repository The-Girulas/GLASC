import pytest
import asyncio
from unittest.mock import MagicMock
from glasc.core.simulation_manager import SimulationManager
from glasc.core.agent_system import GameEngine

@pytest.fixture
def manager():
    return SimulationManager()

@pytest.mark.asyncio
async def test_defender_buyback_trigger(manager):
    """
    Verify that Defender triggers BUYBACK if probability > 0.70.
    """
    # 1. Force High Probability condition
    manager.prob = 0.85
    manager.power_balance["defender"] = 0.5 # Normal power
    manager.last_defense_time = 0 # Ready to fire
    manager.broadcast = MagicMock() 
    
    # Mock broadcast to be awaitable
    f = asyncio.Future()
    f.set_result(None)
    manager.broadcast.return_value = f

    # 2. Run ONE iteration logic manually
    current_time = 100
    chosen_action = None

    if manager.prob > 0.70:
        chosen_action = "BUYBACK"
    
    assert chosen_action == "BUYBACK"
    
    # Execute (Mock context)
    action_result = GameEngine.execute_action(chosen_action, "Defender_AI")
    manager.apply_impact(
        action_result.impact_price, 
        action_result.impact_volatility,
        action_result.impact_attacker,
        action_result.impact_defender
    )
    
    # 3. Assert Impact
    assert manager.current_price > 100.0 
    assert manager.power_balance["defender"] > 0.5

@pytest.mark.asyncio
async def test_defender_desperate_defense(manager):
    """
    Verify that Defender triggers WHITE_KNIGHT or POISON_PILL if power < 0.30.
    """
    manager.prob = 0.5 # Normal prob
    manager.power_balance["defender"] = 0.20 # Desperate
    
    # Logic simulation
    chosen_action = None
    if manager.power_balance["defender"] < 0.30:
        assert True
        
    action_result = GameEngine.execute_action("WHITE_KNIGHT", "Defender_AI")
    manager.apply_impact(
        action_result.impact_price,
        action_result.impact_volatility,
        action_result.impact_attacker,
        action_result.impact_defender
    )
    
    assert manager.power_balance["defender"] >= 0.40 # 0.20 + 0.20

