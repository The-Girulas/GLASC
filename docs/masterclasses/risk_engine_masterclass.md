# Masterclass: Risk Engine & Automatic Differentiation

## 1. The Mathematical Intent: Pathwise Sensitivity

In traditional finance, to calculate the **Delta** ($\frac{\partial V}{\partial S}$) of a complex portfolio, the "finite difference" method (Bumping) is often used:
$$ \Delta \approx \frac{V(S + \epsilon) - V(S)}{\epsilon} $$

This poses two problems:
1.  **Slowness**: If you have 1000 parameters (e.g., local volatilities), you must rerun the simulation 1000 times. Risk cost $O(N)$.
2.  **Noise**: Monte Carlo simulation is noisy. The difference between two noisy simulations (even with the same seed) can be unstable if $\epsilon$ is poorly chosen.

**The GLASC Approach: Pathwise Derivative (via AD)**
We differentiate the simulation path directly.
$$ \frac{\partial}{\partial \theta} \mathbb{E}[f(X_T(\theta))] = \mathbb{E}[f'(X_T) \frac{\partial X_T}{\partial \theta}] $$

Thanks to `jax.grad`, we obtain the exact gradient of the simulation function with respect to the parameters.

## 2. "Under the Hood": AD vs Bumping

In `risk_engine.py`, we define `pricing_function` which takes `params` and `s0` as input.
```python
price, delta = jax.value_and_grad(price_given_s0)(s0)
```
In a single pass (Forward + Backward), JAX calculates:
1.  The Price (Forward pass).
2.  The sensitivity of **every mathematical operation** with respect to $S_0$ (Backward pass).

**Algorithmic Cost**: $O(1)$ relative to the number of parameters.
Calculating 1 Greek or 1000 Greeks has almost the same computational cost in Backward Mode AD. This is a revolution for Risk Management.

## 3. Framework Focus: `jax.jit` and `grad`

The combination `@jax.jit` + `jax.grad` is formidable:
1.  `grad` generates the Python code for the gradient.
2.  `jit` compiles it into an optimized CUDA kernel.
3.  `vmap` vectorizes the gradient calculation over 100,000 trajectories.

Result: We calculate the Delta and Vega of 100,000 options in a few milliseconds.

## 4. Pro-Tip: Monte Carlo Parameter Sensitivities

Warning: for `grad` to work, the relationship must be continuous.
- **Discontinuous Payoff** (e.g., Digital Option $\mathbb{1}_{S>K}$): The derivative $\mathbb{1}'$ is a Dirac (0 everywhere, infinite at K). `jax.grad` will give 0 almost everywhere.
- **Solution**: For discontinuous payoffs, payoff smoothing (smearing) is often used.
- **Note**: For a standard Call ($max(S-K, 0)$), the function is continuous and differentiable almost everywhere (except at $S=K$), so AD works very well "out of the box".
