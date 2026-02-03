# Masterclass: Market Dynamics with JAX

## 1. The Mathematical Intent: Merton Jump Diffusion

The classic **Geometric Brownian Motion (GBM)** model assumes prices evolve continuously. However, real markets suffer from sudden shocks (Crashes, News). That's why we use the **Merton Jump Diffusion Model**.

The Stochastic Differential Equation (SDE) is:

$$ \frac{dS_t}{S_t} = (\mu - \lambda k)dt + \sigma dW_t + (e^J - 1) dN_t $$

Where:
- $dW_t$ is the continuous diffusion (the market's standard "noise").
- $dN_t$ is a Poisson process counting the jumps that occur.
- $J$ is the jump size (log-normally distributed).

In our code, we directly simulate the evolution of the log-price $X_t = \ln(S_t)$ for numerical stability:

$$ X_{t+\Delta t} = X_t + (\mu - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z + \sum_{i=1}^{N_{\Delta t}} J_i $$

## 2. "Under the Hood": Why JAX?

### The Python Problem
A classic Monte Carlo simulation in pure Python uses `for` loops. For 100,000 paths of 252 steps, that's 25.2 million operations. In Python, the interpreter overhead would make this slow (several seconds or even minutes).

### The JAX Solution (`@jax.jit` + `jax.vmap`)
Our implementation is **fully vectorized**.

1.  **Vmap vs Loop**: Instead of looping over the 100,000 trajectories, we ask JAX to process the `(100000, 252)` tensor at once. On GPU, this launches thousands of CUDA threads in parallel.
2.  **XLA Compilation**: The `@jax.jit` decorator compiles the entire computation graph into optimized machine code (operation fusion).
    *   *Result:* Execution time drops from ~10s (Python) to <10ms (JAX GPU) after compilation.

## 3. Framework Focus: Randomness Management (PRNG)

In finance, **reproducibility** is critical. JAX handles randomness differently from NumPy.
- **NumPy**: Global hidden state (`np.random.seed(0)`). Dangerous in parallel contexts.
- **JAX**: Explicit state (`key`). We must "split" the key every time we want new randomness.

In `simulate_paths`, we do:
```python
key_brownian, key_poisson, key_jump_size = jax.random.split(key, 3)
```
This ensures that the Brownian motion randomness is statistically independent of the jumps, and that the simulation is **perfectly deterministic** if replayed with the same mother key.

## 4. Pro-Tip: Jump Vectorization

Implementing random jumps is tricky in vectorized code. A naive approach would be:
> "For each time step, if a coin flip is heads, add a jump."
This implies `if/else`, which is very bad for GPUs (branch divergence).

**Our Trick**:
We generate a matrix of potential jumps (sizes) and an occurrence matrix (Poisson) separately, then multiply them.
```python
log_return_jumps = poisson_increments * jump_sizes_log_normal
```
Zero conditional branching. The GPU executes pure linear algebra (matrix multiplications), which is what it's designed for. This is what allows us to scale to millions of trajectories without slowdown.
