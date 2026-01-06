"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Risk Engine (Differentiable Metrics & Greeks)

Ce module fournit les outils pour calculer les mesures de risque (VaR, ES)
et les sensibilités (Grecques: Delta, Vega, etc.) en utilisant la différentiation
automatique (AD) de JAX sur les simulations Monte Carlo.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Any
from jaxtyping import Float, Array, PRNGKeyArray

from glasc.core.market_dynamics import simulate_paths, MarketParameters

def calculate_var_es(
    final_prices: Float[Array, "n_paths"],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calcule la Value at Risk (VaR) et l'Expected Shortfall (ES) sur la distribution finale.
    
    Args:
        final_prices: Tenseur des prix terminaux (ou PnL).
        confidence_level: Niveau de confiance (ex: 0.95).
        
    Returns:
        (VaR, ES)
        Note: VaR est ici définie comme une perte positive (ex: VaR 95% = Quantile 5%).
    """
    # Calcul du PnL relatif ou absolu ?
    # Ici, on suppose que final_prices représente la distribution des valeurs futures du portefeuille.
    # Souvent VaR est exprimée en perte par rapport à la moyenne ou au spot.
    # Pour simplifier, calculons les quantiles de la distribution.
    
    n_paths = final_prices.shape[0]
    
    # Tri pour trouver les quantiles empiriques
    sorted_prices = jnp.sort(final_prices)
    
    # Index du quantile alpha (ex: 5% pour un niveau de confiance de 95%)
    alpha = 1.0 - confidence_level
    index = jnp.astype(alpha * n_paths, jnp.int32)
    
    # VaR = Prix au quantile alpha.
    # Si on parle de perte: VaR = Mean - Quantile ou Initial - Quantile.
    # Restons génériques: on retourne le quantile.
    var_price_level = sorted_prices[index]
    
    # Expected Shortfall = Moyenne des prix inférieurs à la VaR (Queue de distribution)
    # ES = E[X | X <= VaR]
    # On prend les 'index' premiers éléments
    # Pour éviter la division par zéro si index=0
    safe_index = jnp.maximum(1, index)
    es_price_level = jnp.mean(sorted_prices[:safe_index])
    
    return var_price_level, es_price_level

def european_call_payoff(prices: Float[Array, "n_paths"], strike: float) -> Float[Array, "n_paths"]:
    """Payoff simple d'ex: Call Européen."""
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
    Fonction pivot pour le calcul des prix et des Grecques.
    Exécute la simulation 'end-to-end' et retourne le prix moyen actualisé (Risk-Neutral).
    Cette fonction scalaire est différentiable par rapport à 'params' et 's0'.
    """
    # 1. Simulation
    paths = simulate_paths(key, params, s0, T, n_steps, n_paths)
    final_prices = paths[:, -1]
    
    # 2. Payoff
    payoffs = payoff_fn(final_prices)
    
    # 3. Discounting (Prix = Espérance actualisée sous mesure risque-neutre)
    # Note: Dans Merton Jump Diffusion, le drift 'mu' n'est pas forcément le taux sans risque 'r'.
    # Pour le pricing, on remplace souvent mu par r - lambda*k.
    # ICI: On suppose que les params passés sont déjà les params 'Risk-Neutral' (Q-Measure).
    # Donc mu = r - compensateur.
    # Pour simplifier l'exemple, on actualise par exp(-mu * T).
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
    Calcule Prix, Delta et Vega en une seule passe (ou presque) via AD.
    Utilise jax.value_and_grad.
    """
    
    # Wrapper pour différentier par rapport à S0 (Delta)
    def price_given_s0(s):
        return pricing_function(params, s, T, n_steps, n_paths, payoff_fn, key)
    
    # Wrapper pour différentier par rapport à Sigma (Vega)
    def price_given_sigma(sigma_val):
        # On recrée params avec la nouvelle vol (functional update)
        new_params = params.replace(sigma=sigma_val)
        return pricing_function(new_params, s0, T, n_steps, n_paths, payoff_fn, key)

    # Calcul primal + gradient
    price, delta = jax.value_and_grad(price_given_s0)(s0)
    _, vega = jax.value_and_grad(price_given_sigma)(params.sigma)
    
    # Gamma (Dérivée seconde)
    gamma = jax.grad(jax.grad(price_given_s0))(s0)
    
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega
    }
