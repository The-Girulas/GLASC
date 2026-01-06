"""
Tests unitaires pour Risk Engine.
Validation croisée avec la formule analytique de Black-Scholes (BS) pour vérifier les Grecques JAX.
"""

import unittest
import jax
import jax.numpy as jnp
import chex
from scipy.stats import norm
import numpy as np

from glasc.core.market_dynamics import MarketParameters
from glasc.core.risk_engine import compute_greeks, european_call_payoff, calculate_var_es

# Formules analytiques Black-Scholes pour validation
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

class RiskEngineTest(chex.TestCase):

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)
        # Paramètres pour validation BS (Pas de sauts)
        self.params_bs = MarketParameters(
            mu=0.05,       # r pour BS
            sigma=0.20,
            lambda_j=0.0,
            jump_mean=0.0,
            jump_std=0.0
        )
        self.s0 = 100.0
        self.K = 100.0   # ATM
        self.T = 1.0
        self.n_steps = 100 # Pas besoin de trop fins pour GBM exact, mais utile pour approx
        self.n_paths = 200_000 # Beaucoup de paths pour converger vers la solution analytique
        
        self.payoff_fn = lambda x: european_call_payoff(x, self.K)

    def test_greeks_vs_black_scholes(self):
        """
        Le test ULTIME (Validation Gold Standard).
        Compare les Grecques calculées par JAX AD Monte Carlo vs Formule fermée BS.
        Nécessite une tolérance car MC est stochastique.
        """
        greeks = compute_greeks(
            self.params_bs, 
            self.s0, 
            self.T, 
            self.key, 
            self.payoff_fn, 
            self.n_steps, 
            self.n_paths
        )
        
        # Valeurs théoriques
        bs_p = bs_call_price(self.s0, self.K, self.T, self.params_bs.mu, self.params_bs.sigma)
        bs_d = bs_delta(self.s0, self.K, self.T, self.params_bs.mu, self.params_bs.sigma)
        bs_v = bs_vega(self.s0, self.K, self.T, self.params_bs.mu, self.params_bs.sigma)
        
        print(f"\nComparing JAX MC vs BS Analytical:")
        print(f"Price: JAX={greeks['price']:.4f}, BS={bs_p:.4f}")
        print(f"Delta: JAX={greeks['delta']:.4f}, BS={bs_d:.4f}")
        print(f"Vega : JAX={greeks['vega']:.4f}, BS={bs_v:.4f}")
        
        # Tolérances (MC converge en 1/sqrt(N))
        # Avec 200k paths, on attend ~0.5% d'erreur
        chex.assert_trees_all_close(greeks['price'], bs_p, rtol=0.01)
        chex.assert_trees_all_close(greeks['delta'], bs_d, rtol=0.02) # Delta est plus bruité
        chex.assert_trees_all_close(greeks['vega'], bs_v, rtol=0.05)  # Vega encore plus

    def test_var_calculation(self):
        """Test simple de cohérence VaR."""
        # Distribution normale déterministe artificielle pour tester le tri
        fake_paths = jnp.linspace(80, 120, 100) # Uniform 80-120
        var, es = calculate_var_es(fake_paths, confidence_level=0.95)
        
        # 5% quantile de 80..120 sur 100 points
        # Index 5 (approx) -> 80 + (40 * 0.05) = 82
        self.assertTrue(var < 90.0) # Doit être dans la queue basse
        self.assertTrue(es < var)   # ES est toujours plus extrême que VaR (pour pertes)
                                    # Ici on regarde les prix, donc ES < VaR (moyenne des pires)

if __name__ == '__main__':
    import unittest
    unittest.main()
