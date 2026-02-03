"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Risk Engine (Differentiable Metrics & Greeks)

This module provides tools to calculate risk metrics (VaR, ES)
and sensitivities (Greeks: Delta, Vega, etc.) using JAX's
automatic differentiation (AD) on Monte Carlo simulations.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Any
from jaxtyping import Float, Array, PRNGKeyArray

from glasc.core.market_dynamics import simulate_paths, MarketParameters

@partial(jax.jit, static_argnames=["confidence_level"])
def calculate_var_es(
    final_prices: Float[Array, "n_paths"],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculates Value at Risk (VaR) and Expected Shortfall (ES) on the final distribution.
    
    Args:
        final_prices: Tensor of terminal prices (or PnL).
        confidence_level: Confidence level (e.g., 0.95).
        
    Returns:
        (VaR, ES)
        Note: VaR is defined here as a positive loss level (e.g., VaR 95% = 5% Quantile).
    """
    # Relative or Absolute PnL?
    # Here, we assume final_prices represents the distribution of future portfolio values.
    # Often VaR is expressed as loss relative to mean or spot.
    # To simplify, we calculate the distribution quantiles.
    
    n_paths = final_prices.shape[0]
    
    # Sort to find empirical quantiles
    sorted_prices = jnp.sort(final_prices)
    
    # Index of alpha quantile (e.g., 5% for a 95% confidence level)
    alpha = 1.0 - confidence_level
    index = jnp.astype(alpha * n_paths, jnp.int32)
    
    # VaR = Price at alpha quantile.
    # If discussing loss: VaR = Mean - Quantile or Initial - Quantile.
    # Staying generic: we return the quantile.
    var_price_level = sorted_prices[index]
    
    # Expected Shortfall = Mean of prices below VaR (Tail of distribution)
    # ES = E[X | X <= VaR]
    # We take the first 'index' elements
    # Avoid division by zero if index=0
    safe_index = jnp.maximum(1, index)
    es_price_level = jnp.mean(sorted_prices[:safe_index])
    
    return var_price_level, es_price_level

def european_call_payoff(prices: Float[Array, "n_paths"], strike: float) -> Float[Array, "n_paths"]:
    """Simple payoff e.g., European Call."""
    return jnp.maximum(prices - strike, 0.0)

def pricing_function(
    params: MarketParameters,
    s0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    payoff_fn: Callable[[Array], Array],
    key: PRNGKeyArray
) -> float:
    """
    Pivot function for calculating prices and Greeks.
    Runs the 'end-to-end' simulation and returns the discounted mean price (Risk-Neutral).
    This scalar function is differentiable with respect to 'params' and 's0'.
    """
    # 1. Simulation
    paths = simulate_paths(key, params, s0, T, n_steps, n_paths)
    final_prices = paths[:, -1]
    
    # 2. Payoff
    payoffs = payoff_fn(final_prices)
    
    # 3. Discounting (Price = Discounted Expectation under Risk-Neutral measure)
    # Note: In Merton Jump Diffusion, drift 'mu' is not necessarily risk-free rate 'r'.
    # For pricing, we often replace mu with r - lambda*k.
    # HERE: We assume passed params are already 'Risk-Neutral' (Q-Measure).
    # So mu = r - compensator.
    # To simplify example, we discount by exp(-mu * T).
    discount_factor = jnp.exp(-params.mu * T)
    
    price = discount_factor * jnp.mean(payoffs)
    return price

@partial(jax.jit, static_argnames=["n_steps", "n_paths", "payoff_fn"])
def compute_greeks(
    params: MarketParameters,
    s0: float,
    T: float,
    key: PRNGKeyArray,
    payoff_fn: Callable = lambda x: european_call_payoff(x, 100.0), # Default dummy
    n_steps: int = 252,
    n_paths: int = 100_000
):
    """
    Calculates Price, Delta, and Vega in a single pass (or almost) via AD.
    Uses jax.value_and_grad.
    """
    
    # Wrapper to differentiate with respect to S0 (Delta)
    def price_given_s0(s):
        return pricing_function(params, s, T, n_steps, n_paths, payoff_fn, key)
    
    # Wrapper to differentiate with respect to Sigma (Vega)
    def price_given_sigma(sigma_val):
        # Recreate params with new vol (functional update)
        new_params = params.replace(sigma=sigma_val)
        return pricing_function(new_params, s0, T, n_steps, n_paths, payoff_fn, key)

    # Primal + Gradient calculation
    price, delta = jax.value_and_grad(price_given_s0)(s0)
    _, vega = jax.value_and_grad(price_given_sigma)(params.sigma)
    
    # Gamma (Second derivative)
    gamma = jax.grad(jax.grad(price_given_s0))(s0)
    
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega
    }
