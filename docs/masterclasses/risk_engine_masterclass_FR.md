# Masterclass : Risk Engine & Différentiation Automatique

## 1. L'Intention Mathématique : Pathwise Sensitivity

Dans la finance traditionnelle, pour calculer le **Delta** ($\frac{\partial V}{\partial S}$) d'un portefeuille complexe, on utilise souvent la méthode des "différences finies" (Bumping) :
$$ \Delta \approx \frac{V(S + \epsilon) - V(S)}{\epsilon} $$

Cela pose deux problèmes :
1.  **Lenteur** : Si vous avez 1000 paramètres (ex: volatilités locales), il faut relancer la simulation 1000 fois. Coût $O(N)$.
2.  **Bruit** : Simulation Monte Carlo est bruitée. La différence de deux simulations bruitées (même avec la même graine) peut être instable si $\epsilon$ est mal choisi.

**L'approche GLASC : Pathwise Derivative (via AD)**
Nous différentions directement le chemin de simulation.
$$ \frac{\partial}{\partial \theta} \mathbb{E}[f(X_T(\theta))] = \mathbb{E}[f'(X_T) \frac{\partial X_T}{\partial \theta}] $$

Grâce à `jax.grad`, nous obtenons le gradient exact de la fonction de simulation par rapport aux paramètres.

## 2. "Under the Hood" : AD vs Bumping

Dans `risk_engine.py`, nous définissons `pricing_function` qui prend en entrée `params` et `s0`.
```python
price, delta = jax.value_and_grad(price_given_s0)(s0)
```
En une seule passe (Forward + Backward), JAX calcule :
1.  Le Prix (Forward pass).
2.  La sensibilité de **chaque opération mathématique** par rapport à $S_0$ (Backward pass).

**Coût algorithmique** : $O(1)$ par rapport au nombre de paramètres.
Calculer 1 grecque ou 1000 grecques a quasiment le même coût computationnel en Backward Mode AD. C'est une révolution pour le Risk Management.

## 3. Focus Framework : `jax.jit` et `grad`

La combinaison `@jax.jit` + `jax.grad` est redoutable :
1.  `grad` génère le code Python du gradient.
2.  `jit` le compile en noyau CUDA optimisé.
3.  `vmap` vectorise le calcul du gradient sur 100 000 trajectoires.

Résultat : Nous calculons le Delta et le Vega de 100 000 options en quelques millisecondes.

## 4. Pro-Tip : Sensibilités aux paramètres de Monte Carlo

Attention, pour que `grad` fonctionne, la relation doit être continue.
- **Payoff discontinu** (ex: Digital Option $\mathbb{1}_{S>K}$) : La dérivée $\mathbb{1}'$ est un Dirac (0 partout, infini en K). `jax.grad` donnera 0 presque partout.
- **Solution** : Pour les payoffs discontinus, on utilise souvent le lissage (smearing) du payoff.
- **Note** : Pour un Call standard ($max(S-K, 0)$), la fonction est continue et dérivable presque partout (sauf en $S=K$), donc AD fonctionne très bien "out of the box".
