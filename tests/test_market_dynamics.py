"""
Tests unitaires pour le module Market Dynamics.
Utilise Chex pour la validation JAX et les tests de propriétés.
"""

import jax
import jax.numpy as jnp
import chex
from glasc.core.market_dynamics import simulate_paths, MarketParameters

class MarketDynamicsTest(chex.TestCase):
    
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(0)
        self.params = MarketParameters(
            mu=0.05,
            sigma=0.20,
            lambda_j=0.0,  # Pas de sauts pour le test GBM pur
            jump_mean=0.0,
            jump_std=0.0
        )
        self.s0 = 100.0
        self.T = 1.0
        self.n_steps = 252
        self.n_paths = 10_000 # Suffisant pour les stats, pas trop lourd pour le test

    def test_output_shape(self):
        """Vérifie que la forme du tenseur de sortie est correcte."""
        paths = simulate_paths(self.key, self.params, self.s0, self.T, self.n_steps, self.n_paths)
        
        # (n_paths, n_steps + 1) car inclut t=0
        expected_shape = (self.n_paths, self.n_steps + 1)
        self.assertEqual(paths.shape, expected_shape)
        
    def test_start_value(self):
        """Vérifie que toutes les trajectoires commencent à S0."""
        paths = simulate_paths(self.key, self.params, self.s0, self.T, self.n_steps, self.n_paths)
        chex.assert_trees_all_close(paths[:, 0], self.s0)

    def test_gbm_statistics(self):
        """
        Vérifie que la moyenne théorique du GBM correspond à la simulation.
        E[S_T] = S0 * exp(mu * T)
        """
        # On utilise beaucoup de paths pour la convergence
        n_paths_stat = 50_000
        paths = simulate_paths(self.key, self.params, self.s0, self.T, self.n_steps, n_paths_stat)
        
        final_prices = paths[:, -1]
        
        empirical_mean = jnp.mean(final_prices)
        theoretical_mean = self.s0 * jnp.exp(self.params.mu * self.T)
        
        # Tolérance de 1% (stochastique)
        chex.assert_trees_all_close(empirical_mean, theoretical_mean, rtol=0.01)

    @chex.all_variants # Teste JIT, pmap, etc. si applicable (ici surtout jit_cpu/jit_gpu)
    def test_jit_compatibility(self):
        """Vérifie que la fonction est JIT-compatible (variante Chex)."""
        # Chex va exécuter simulate_paths avec différentes transformations JAX
        # Pour ce test, on définit une fonction wrapper simple
        wrapper = lambda k, p: simulate_paths(k, self.params, self.s0, self.T, self.n_steps, self.n_paths)
        
        # Appel variant
        res = self.variant(wrapper)(self.key, self.params)
        self.assertEqual(res.shape, (self.n_paths, self.n_steps + 1))

    def test_determinism(self):
        """Vérifie que la simulation est déterministe avec la même clé."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)
        
        paths1 = simulate_paths(key1, self.params, self.s0, self.T, self.n_steps, 100)
        paths2 = simulate_paths(key2, self.params, self.s0, self.T, self.n_steps, 100)
        
        chex.assert_trees_all_equal(paths1, paths2)

if __name__ == '__main__':
    import unittest
    unittest.main()
