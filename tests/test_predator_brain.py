"""
Tests unitaires pour Predator Brain.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from glasc.core.predator_brain import PredatorModel, RealisticDataGenerator, PredatorTrainer

class PredatorBrainTest(unittest.TestCase):
    
    def setUp(self):
        self.generator = RealisticDataGenerator(seed=42)
        self.trainer = PredatorTrainer(learning_rate=0.01)
        self.key = jax.random.PRNGKey(0)

    def test_data_generator_sanity(self):
        """Vérifie que le générateur produit des données statistiques cohérentes."""
        features, labels = self.generator.generate_batch(100)
        
        # Check shapes
        self.assertEqual(features.shape, (100, 6))
        self.assertEqual(labels.shape, (100, 1))
        
        # Check values range (Debt/EBITDA normalized)
        # Debt feature is index 0. Un-normalize: x * 2 + 3
        debt_raw = features[:, 0] * 2.0 + 3.0
        # Check that we have valid debts (not all negative)
        self.assertTrue(jnp.mean(debt_raw) > 0)
        
    def test_brain_learning(self):
        """Vérifie que le réseau apprend (Loss qui descend sur un batch)."""
        # Create a fixed batch
        x, y = self.generator.generate_batch(32)
        
        # Init
        params, opt_state = self.trainer.init_state(self.key, x)
        
        # Train loop
        initial_loss = 999.0
        final_loss = 999.0
        
        # Overfit on this single batch
        for i in range(50):
            step_key = jax.random.fold_in(self.key, i)
            params, opt_state, loss = self.trainer.train_step(params, opt_state, x, y, step_key)
            if i == 0: initial_loss = loss
            final_loss = loss
            
        print(f"Training Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.5) # Should converge easily on fixed batch

    def test_vulnerability_logic(self):
        """Vérifie la logique du modèle (Inference Monotonicity)."""
        # On entraine rapidement le modèle pour qu'il ne soit pas aléatoire
        x_train, y_train = self.generator.generate_batch(200)
        params, opt_state = self.trainer.init_state(self.key, x_train)
        for i in range(20):
            params, opt_state, _ = self.trainer.train_step(params, opt_state, x_train, y_train, self.key)
            
        # Test Case:
        # Company A: Low Debt, High Insider (Fortress)
        # Company B: High Debt, Low Insider (Distressed)
        # B should have lower score (Failure risk due to debt?) OR higher score (Target?)
        # Oracle Logic says: Debt is BAD (-0.8 coeff), Insider is BAD (-2.5 coeff).
        # So both are bad?
        # Wait, Oracle Logic: 1.0 = Succès OPA.
        # High Debt -> Low Score (Fail because Covenant breach risk).
        # High Insider -> Low Score (Fail because Blocked).
        # So a "Good Target" has Low Debt AND Low Insider.
        
        # Let's create specific features
        # [Debt, Inst, Insider, Margin, Vol, MktCap]
        # Target A (Ideal): Debt=1x, Insider=0%
        target_A = jnp.array([[ (1.0 - 3.0)/2.0, 0.8, 0.0, 0.0, 0.0, 0.0 ]])
        
        # Target B (Toxic Debt): Debt=8x, Insider=0%
        target_B = jnp.array([[ (8.0 - 3.0)/2.0, 0.8, 0.0, 0.0, 0.0, 0.0 ]])
        
        # Target C (Locked): Debt=1x, Insider=80%
        target_C = jnp.array([[ (1.0 - 3.0)/2.0, 0.8, 0.8, 0.0, 0.0, 0.0 ]])
        
        score_A = self.trainer.model.apply({'params': params}, target_A)[0, 0]
        score_B = self.trainer.model.apply({'params': params}, target_B)[0, 0]
        score_C = self.trainer.model.apply({'params': params}, target_C)[0, 0]
        
        print(f"Scores: Ideal={score_A:.2f}, ToxicDebt={score_B:.2f}, Locked={score_C:.2f}")
        
        self.assertGreater(score_A, score_B) # Ideal est mieux que Toxic
        self.assertGreater(score_A, score_C) # Ideal est mieux que Locked

if __name__ == '__main__':
    unittest.main()
