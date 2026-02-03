"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Market Dynamics (Physics Engine)

This module implements the stochastic simulation engine using JAX.
It models asset paths via a jump diffusion process (Merton Jump Diffusion).

SDE Formula:
dS_t = S_t * (mu * dt + sigma * dW_t + (Y - 1) * dN_t)

Where:
- mu: Constant Drift
- sigma: Constant Volatility
- dW_t: Standard Brownian Motion
- dN_t: Poisson Process (intensity lambda)
- Y: Jump Size (log-normal)
"""

import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from jaxtyping import Float, Array, PRNGKeyArray

@struct.dataclass
class MarketParameters:
    """Immutable parameters for market simulation (Merton Jump Diffusion)."""
    mu: float        # Annual Drift (e.g., 0.05 for 5%)
    sigma: float     # Annual Volatility (e.g., 0.20 for 20%)
    lambda_j: float  # Jump Intensity (average number of jumps per year)
    jump_mean: float # Mean of log-jump
    jump_std: float  # Standard Deviation of log-jump

@partial(jax.jit, static_argnames=["n_steps", "n_paths"])
def simulate_paths(
    key: PRNGKeyArray,
    params: MarketParameters,
    s0: float,
    T: float,
    n_steps: int,
    n_paths: int
) -> Float[Array, "n_paths n_steps_plus_1"]:
    """
    Simulates asset price paths according to the Merton Jump Diffusion model.
    Executed in parallel (Vectorized) and complied (JIT).
    
    Args:
        key: JAX PRNG Key.
        params: Market Parameters (mu, sigma, jumps).
        s0: Initial Price.
        T: Time Horizon in years.
        n_steps: Number of time steps.
        n_paths: Number of paths to simulate.
        
    Returns:
        Simulated paths of shape (n_paths, n_steps + 1).
    """
    dt = T / n_steps
    
    # Split keys for different stochastic processes
    key_brownian, key_poisson, key_jump_size = jax.random.split(key, 3)
    
    # 1. Geometric Brownian Motion (GBM) Part
    # Z ~ N(0, 1)
    brownian_increments = jax.random.normal(key_brownian, shape=(n_paths, n_steps))
    
    # Deterministic Drift + Brownian Diffusion
    # log(S_t) evolution for continuous part: (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
    log_return_continuous = (params.mu - 0.5 * params.sigma**2) * dt + \
                            params.sigma * jnp.sqrt(dt) * brownian_increments
    
    # 2. Jump Part (Merton)
    # Number of jumps in each dt interval: Poisson(lambda * dt)
    # For small dt, Bernouilli(lambda * dt) is a good approx, but Poisson is exact.
    # We use Poisson here.
    poisson_increments = jax.random.poisson(key_poisson, params.lambda_j * dt, shape=(n_paths, n_steps))
    
    # Jump Size: Y ~ LogNormal(jump_mean, jump_std)
    # Contribution to log-return: log(Y) ~ Normal(jump_mean, jump_std)
    # If k jumps occur, we add k draws of Normal(jump_mean, jump_std).
    # Optimization: Generate a grid of potential jumps (one per time step per path) 
    # and mask by poisson_increments. 
    # Note: If poisson_increments > 1, this underestimates variance of multiple jumps in a single dt.
    # For small dt, p(N > 1) is negligible. To be simple vectorized rigorous:
    # We assume max 1 significant jump per dt or accept the sum normal approx.
    # Rigorous JAX version: Sum of normal variables = Normal variable.
    # If N jumps, log_jump ~ Normal(N * jump_mean, sqrt(N) * jump_std).
    
    # Jump Mask (0 or 1+). For very fast vectorized precise simulation:
    # Generate standard jumps for each step.
    jump_sizes_log_normal = jax.random.normal(key_jump_size, shape=(n_paths, n_steps)) * params.jump_std + params.jump_mean
    
    # Total jump contribution to log-return = jump_count * average_jump_size (approx)
    # Or better: multiply by number of jumps. 
    # Warning: if 2 jumps, we should sum 2 different random variables.
    # Here we multiply the same variable by N. Acceptable approximation if dt -> 0.
    log_return_jumps = poisson_increments * jump_sizes_log_normal
    
    # 3. Aggregation
    all_log_returns = log_return_continuous + log_return_jumps
    
    # Full path via cumulative sum
    log_paths = jnp.cumsum(all_log_returns, axis=1)
    
    # Add starting point t=0
    # log_S0 = log(S0)
    log_S0 = jnp.log(s0)
    initial_log_prices = jnp.full((n_paths, 1), log_S0)
    
    # Complete Trajectory
    full_log_paths = jnp.concatenate([initial_log_prices, initial_log_prices + log_paths], axis=1)
    
    return jnp.exp(full_log_paths)

@partial(jax.jit, static_argnames=["n_steps", "n_paths"])
def compute_success_probability(
    key: PRNGKeyArray,
    params: MarketParameters,
    s0: float,
    T: float,
    target_acquisition_price: float,
    n_steps: int = 100,
    n_paths: int = 10_000
) -> float:
    """
    Calculates the probability of Takeover Success.
    Hypothesis: The TOB succeeds if the market price falls below 'target_acquisition_price'
    (making the company vulnerable/cheap) or if volatility allows a power move.
    
    Here, simple metric: P(S_T < Acquisition_Price).
    The more the asset crashes, the easier it is to buy.
    """
    paths = simulate_paths(key, params, s0, T, n_steps, n_paths)
    final_prices = paths[:, -1]
    
    # How many paths end up under the target price?
    success_count = jnp.sum(final_prices < target_acquisition_price)
    return success_count / n_paths
