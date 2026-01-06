# Masterclass : Market Dynamics avec JAX

## 1. L'Intention Mathématique : Merton Jump Diffusion

Le modèle **Geometric Brownian Motion (GBM)** classique suppose que les prix évoluent de manière continue. Or, les marchés réels subissent des chocs soudains (Krachs, News). C'est pourquoi nous utilisons le **Merton Jump Diffusion Model**.

L'équation différentielle stochastique (SDE) est :

$$ \frac{dS_t}{S_t} = (\mu - \lambda k)dt + \sigma dW_t + (e^J - 1) dN_t $$

Où :
- $dW_t$ est la diffusion continue (le "bruit" normal du marché).
- $dN_t$ est un processus de Poisson comptant les sauts survenus.
- $J$ est la taille du saut (distribuée normalement en log).

Dans notre code, nous simulons directement l'évolution du log-prix $X_t = \ln(S_t)$ pour la stabilité numérique :

$$ X_{t+\Delta t} = X_t + (\mu - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z + \sum_{i=1}^{N_{\Delta t}} J_i $$

## 2. "Under the Hood" : Pourquoi JAX ?

### Le Problème Python
Une simulation Monte Carlo classique en Python pur utilise des boucles `for`. Pour 100 000 trajectoires de 252 pas, cela fait 25.2 millions d'opérations. En Python, l'overhead de l'interpréteur rendrait cela lent (plusieurs secondes, voire minutes).

### La Solution JAX (`@jax.jit` + `jax.vmap`)
Notre implémentation est **entièrement vectorisée**.

1.  **Vmap vs Boucle** : Au lieu de boucler sur les 100 000 trajectoires, nous demandons à JAX de traiter le tenseur `(100000, 252)` d'un coup. Sur GPU, cela lance des milliers de threads CUDA en parallèle.
2.  **Compilation XLA** : Le décorateur `@jax.jit` compile le graphe de calcul entier en code machine optimisé (fusion d'opérations).
    *   *Résultat :* Le temps d'exécution passe de ~10s (Python) à <10ms (JAX GPU) après compilation.

## 3. Focus Framework : Gestion du Hasard (PRNG)

En finance, la **reproductibilité** est critique. JAX gère l'aléatoire différemment de NumPy.
- **NumPy** : État global caché (`np.random.seed(0)`). Dangereux en parallèle.
- **JAX** : État explicite (`key`). On doit "splitter" la clé à chaque fois qu'on veut du nouveau hasard.

Dans `simulate_paths`, nous faisons :
```python
key_brownian, key_poisson, key_jump_size = jax.random.split(key, 3)
```
Cela garantit que l'aléatoire du mouvement Brownien est statistiquement indépendant des sauts, et que la simulation est **parfaitement déterministe** si on rejoue avec la même clé mère.

## 4. Pro-Tip : Vectorisation des Sauts

Implémenter des sauts aléatoires est piégeux en vectoriel. Une approche naïve serait :
> "Pour chaque pas de temps, si une pièce tombe pile, ajoute un saut."
Cela implique des `if/else`, très mauvais pour les GPUs (divergence de branche).

**Notre astuce** :
Nous générons une matrice de sauts potentiels (tailles) et une matrice d'occurrence (Poisson) séparément, puis nous multiplions.
```python
log_return_jumps = poisson_increments * jump_sizes_log_normal
```
Zéro branchement conditionnel. Le GPU exécute de l'algèbre linéaire pure (multiplications de matrices), ce pour quoi il est conçu. C'est ce qui nous permet de scaler à des millions de trajectoires sans ralentissement.
