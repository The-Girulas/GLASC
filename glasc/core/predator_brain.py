"""
GLASC: Global Leverage & Asset Strategy Controller
Module: Predator Brain (Vulnerability Scorer)

Ce module implémente l'intelligence artificielle (Neural Network) qui évalue la vulnérabilité des cibles.
Il inclut un Générateur de Données Synthétiques calibé sur des distributions financières réelles pour l'entraînement.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Tuple, Dict, Any
import numpy as np # Used for complex distributions generation logic in the generator
from functools import partial

# --- 1. The Neural Network (The Brain) ---

class PredatorModel(nn.Module):
    """
    Réseau de neurones simple (MLP) pour estimer la probabilité de succès d'une OPA.
    Input: Vecteur de features (Financières + Structurelles).
    Output: Score de vulnérabilité [0, 1].
    """
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Layer 1: Feature processing
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        
        # Layer 2: Reasoning
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        
        # Layer 3: Decision
        x = nn.Dense(features=1)(x)
        # Sigmoid pour obtenir une probabilité
        return nn.sigmoid(x)

# --- 2. Realistic Synthetic Data Generator ---

class RealisticDataGenerator:
    """
    Génère des données d'entraînement synthétiques mais réalistes pour le Predator Brain.
    Basé sur des recherches statistiques réelles (S&P 500 / High Yield stats).
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_batch(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Génère un batch de (Features, Labels).
        Features: [NetDebt/EBITDA, FreeFloat, InsiderOwn, Profitability, Volatility, MarketCapLog]
        Label: 1.0 (Succès OPA) ou 0.0 (Échec).
        """
        
        # 1. Net Debt / EBITDA
        # Dist: LogNormal pour avoir une queue à droite (entreprises très endettées).
        # Mode autour de 2.5x, mais queue jusqu'à 8x.
        debt_ebitda = self.rng.lognormal(mean=0.8, sigma=0.6, size=batch_size) 
        # Clip pour éviter les valeurs absurdes issues du LogNormal pur
        debt_ebitda = np.clip(debt_ebitda, -1.0, 10.0) 
        
        # 2. Institutional Ownership (Smart Money)
        # Dist: Beta distribution. Souvent bimodal ou concentré vers 60-80% pour les grosses capis.
        inst_own = self.rng.beta(a=2.0, b=2.0, size=batch_size) # Centré sur 0.5
        
        # 3. Insider Ownership (Bloquant)
        # Dist: Exponentielle. La plupart des boites ont peu d'insiders, certaines en ont beaucoup (Famille).
        insider_own = self.rng.exponential(scale=0.1, size=batch_size)
        insider_own = np.clip(insider_own, 0, 1.0)
        
        # 4. Profitability (EBITDA Margin)
        # Dist: Normal
        margin = self.rng.normal(loc=0.15, scale=0.10, size=batch_size)
        
        # 5. Volatility (30d annualized)
        vol = self.rng.lognormal(mean=-1.5, sigma=0.4, size=batch_size) # Median ~0.22
        
        # 6. Market Cap (Log scale)
        # Small caps are easier to eat ?
        log_mcap = self.rng.normal(loc=9.0, scale=2.0, size=batch_size) # $80M to $100B range

        # --- ORACLE LOGIC (Simulation de la Vérité Terrain) ---
        # On définit une heuristique complexe qui sert de "Vérité" que le NN doit apprendre.
        # Règle OPA hostile : 
        # - Trop de dette (>5x) = Covenants risqués = Échec.
        # - Trop d'Insiders (>30%) = Verrouillé = Échec.
        # - Volatilité haute = Cible en détresse = Succès potentiel.
        # - Profitabilité faible + Cash flow (Margin) = Repricage possible = Succès.
        
        score_logits = (
            -0.8 * debt_ebitda        # Dette toxique
            -2.5 * insider_own        # Forteresse familiale
            +0.5 * vol                # Opportunisme sur volatilité
            -0.3 * margin             # On cherche les boites mal gérées (Marge faible) pour les optimiser
            +0.1 * inst_own           # Les institutionnels vendent si le prix est bon
            + self.rng.normal(0, 1.0, size=batch_size) # Bruit de marché (Incertitude)
        )
        
        # Seuil de succès (Probabilité)
        probs = 1.0 / (1.0 + np.exp(-score_logits)) # Sigmoid manual
        labels = (probs > 0.5).astype(np.float32)
        
        # Stack features
        # Normalize inputs roughly [-1, 1] helps neural net training
        features = np.stack([
            (debt_ebitda - 3.0) / 2.0,
            inst_own,
            insider_own,
            (margin - 0.15) / 0.1,
            (vol - 0.2) / 0.1,
            (log_mcap - 9.0) / 2.0
        ], axis=1)
        
        return jnp.array(features), jnp.array(labels[:, None]) # (B, F), (B, 1)


# --- 3. Training Loop ---

class PredatorTrainer:
    def __init__(self, learning_rate: float = 1e-3):
        self.model = PredatorModel()
        self.tx = optax.adam(learning_rate)
        
    def init_state(self, key, sample_input):
        variables = self.model.init(key, sample_input)
        params = variables['params']
        opt_state = self.tx.init(params)
        return params, opt_state

    @partial(jax.jit, static_argnames=['self'])
    def train_step(self, params, opt_state, batch_x, batch_y, key):
        
        def loss_fn(p):
            # Forward pass with Dropout (requires rng)
            logits = self.model.apply({'params': p}, batch_x, training=True, rngs={'dropout': key})
            # Binary Cross Entropy
            # Note: logits output is Sigmoid-ed. optax.sigmoid_binary_cross_entropy takes logits BEFORE sigmoid.
            # ERROR in model definition above? No, user requested sigmoid output.
            # So let's compare Probs.
            loss = optax.l2_loss(logits, batch_y).mean() # Simple MSE for demonstration or BCE manual
            # Better: BCE
            epsilon = 1e-7
            loss_bce = -jnp.mean(batch_y * jnp.log(logits + epsilon) + (1 - batch_y) * jnp.log(1 - logits + epsilon))
            return loss_bce

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        updates, new_opt_state = self.tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss

from functools import partial
