"""
DPM-Solver adapted for Variance Exploding (VE) schedules.

Implements first and second-order DPM-Solver for the VE probability flow ODE,
bypassing diffrax for minimal overhead. The key advantage over generic ODE solvers
is that DPM-Solver exploits the semi-linear structure of the diffusion ODE,
achieving better quality at fewer neural network evaluations.

Reference: Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic
Model Sampling in Around 10 Steps" (NeurIPS 2022).
"""
import math
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial


class DPMSolverVE:
    """DPM-Solver for Variance Exploding schedules.

    Uses the log-SNR reparameterization to solve the probability flow ODE
    with fewer neural network evaluations than generic solvers.
    """

    def __init__(self, schedule, model, data_shape, order=1, dtype=None):
        """
        Args:
            schedule: VE noise schedule with σ(t), g2(t), σmax, get_timesteps().
            model: Denoising model f(x, σ) → denoised x.
            data_shape: Shape of data samples (e.g. (4, 96, 192)).
            order: Solver order (1 = DPM-Solver-1, 2 = DPM-Solver-2).
            dtype: Optional dtype for model inputs (e.g. jnp.bfloat16).
        """
        self.schedule = schedule
        self.data_shape = data_shape
        self.order = order
        self.dtype = dtype

        @eqx.filter_jit
        def model_fn(model, data_shape, dtype, x_flat, σ):
            """Call model on shaped input, return flat output."""
            x = jnp.reshape(x_flat, data_shape)
            if dtype is not None:
                x = x.astype(dtype)
                σ = σ.astype(dtype)
            out = model(x, σ)
            return jnp.reshape(out, x_flat.shape).astype(x_flat.dtype)

        self.model_fn = partial(model_fn, model, data_shape, dtype)

    def _denoise(self, x, σ):
        """Apply denoising: D(x/(1+σ), σ)."""
        scaling = 1 + σ
        return self.model_fn(x / scaling, σ)

    @eqx.filter_jit
    def _solve_single(self, timesteps, x0):
        """Solve the reverse ODE for a single sample using DPM-Solver."""
        x = x0
        n = len(timesteps)

        if self.order == 1:
            # DPM-Solver-1 (equivalent to DDIM-like update for VE)
            for i in range(n - 1):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                σ_cur = self.schedule.σ(t_cur)
                σ_next = self.schedule.σ(t_next)

                # Denoised prediction
                d = self._denoise(x, σ_cur)

                # VE update: exact solution of linear part + first-order correction
                # x_{t_next} = x_t + (σ_next² - σ_cur²)/(2σ_cur²) * (d - x_t)
                coeff = (σ_next**2 - σ_cur**2) / (2 * σ_cur**2)
                x = x + coeff * (d - x)
        else:
            # DPM-Solver-2: uses midpoint correction for second-order accuracy
            for i in range(n - 1):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                σ_cur = self.schedule.σ(t_cur)
                σ_next = self.schedule.σ(t_next)
                t_mid = (t_cur + t_next) / 2
                σ_mid = self.schedule.σ(t_mid)

                # First evaluation at current time
                d1 = self._denoise(x, σ_cur)
                coeff1 = (σ_mid**2 - σ_cur**2) / (2 * σ_cur**2)
                x_mid = x + coeff1 * (d1 - x)

                # Second evaluation at midpoint
                d2 = self._denoise(x_mid, σ_mid)
                coeff2 = (σ_next**2 - σ_cur**2) / (2 * σ_cur**2)
                x = x + coeff2 * (d2 - x)

        return x

    @eqx.filter_jit
    def sample(self, N, key=jr.PRNGKey(0), steps=10, warm_start=None):
        """Generate N samples.

        Args:
            N: Number of samples.
            key: PRNG key.
            steps: Number of solver steps (= number of NN evals for order=1,
                   2x for order=2).
            warm_start: Optional (mean_flat, t_start) tuple for truncated diffusion.
                       Starts from mean + σ(t_start)*noise instead of pure noise at t_max.
        """
        keys = jax.random.split(key, N)
        flat_dim = math.prod(self.data_shape)

        if warm_start is not None:
            mean_flat, t_start = warm_start
            σ_start = self.schedule.σ(t_start)
            x0 = mean_flat[None, :] + jr.normal(keys[0], (N, flat_dim)) * σ_start
            # Timesteps linearly spaced from t_start down to tmin
            tmin = self.schedule.get_timesteps(2)[0]
            reverse_timesteps = jnp.linspace(t_start, tmin, steps)
        else:
            x0 = jr.normal(keys[0], (N, flat_dim)) * self.schedule.σmax
            reverse_timesteps = self.schedule.get_timesteps(steps)[::-1]

        sampler = partial(self._solve_single, reverse_timesteps)

        samples = jax.vmap(sampler)(x0)
        samples = jnp.reshape(samples, (N, *self.data_shape))
        return samples
