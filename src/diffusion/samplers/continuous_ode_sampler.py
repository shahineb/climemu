"""
This sampler integrates the reverse-time ODE using diffrax to generate samples from the emulator.

Code adapted from : https://github.com/sandreza/JaxDiffusion/blob/main/jaxdiffusion/process/residual_sampler.py
"""
import math
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from functools import partial


class ContinuousHeunSampler:
    def __init__(self, schedule, model, data_shape):
        """
        Initializes the sampler with a noise schedule, denoising model, and data shape.

        Args:
            schedule: Noise schedule object with σ(t), g2(t), σmax, and get_timesteps().
            model: Denoising model, expects input of shape data_shape and noise scale σ.
            data_shape: Shape of the data to be sampled (tuple).
        """
        @eqx.filter_jit
        def denoiser_precursor(model, data_shape, x, σ):
            """
            Handles the fact that diffrax requires flat arrays.
            """
            xr = jnp.reshape(x, data_shape)
            denoised_x = model(xr, σ)
            return jnp.reshape(denoised_x, (math.prod(data_shape)))

        @eqx.filter_jit
        def drift_precursor(denoiser, schedule, x, t):
            """
            Computes the drift term for the reverse-time ODE.
            """
            g2 = schedule.g2(t)
            σ = schedule.σ(t)
            scaling = 1 + σ
            prefactor = g2 / (2 * σ**2)
            scaled_score = denoiser(x / scaling, σ) - x
            drift = - prefactor * scaled_score
            return drift

        denoiser = partial(denoiser_precursor, model, data_shape)
        drift = partial(drift_precursor, denoiser, schedule)
        self.schedule = schedule
        self.data_shape = data_shape
        self.drift = drift
    
    @eqx.filter_jit
    def precursor_solver(self, timesteps, x0):
        """
        Solves the reverse-time ODE using Heun's method over the given timesteps.

        Args:
            timesteps: Array of time points to integrate over (descending order).
            x0: Initial condition, flattened array.

        Returns:
            sol.ys: Solution trajectory (array of states at each timestep).
        """
        def drift_diffrax_signature(t, y, args=None):
            # Diffrax expects signature (t, y, args)
            return self.drift(y, t)
        f = dfx.ODETerm(drift_diffrax_signature)
        solversteps = dfx.StepTo(timesteps)
        sol = dfx.diffeqsolve(terms=f,
                              solver=dfx.Heun(),
                              t0=jnp.max(timesteps),
                              t1=jnp.min(timesteps),
                              stepsize_controller=solversteps,
                              dt0=None,
                              y0=x0)
        return sol.ys

    @eqx.filter_jit
    def sample(self, N, key=jr.PRNGKey(0), steps=300):
        """
        Generates N samples by integrating the reverse-time ODE.

        Args:
            N: Number of samples to generate.
            key: JAX PRNGKey for randomness.
            steps: Number of ODE integration steps.

        Returns:
            samples: Array of generated samples with shape (N, *data_shape).
        """
        # Draw initial Gausssian noise samples from N(0, σmax²)
        keys = jax.random.split(key, N)
        x0 = jr.normal(keys[0], (N, math.prod(self.data_shape))) * self.schedule.σmax

        # Define sample on timesteps defined by the noise schedule
        reverse_timesteps = self.schedule.get_timesteps(steps)[::-1]
        sampler = partial(self.precursor_solver, reverse_timesteps)

        # Vectorize sampler over batch dimension and draw samples
        samples = jax.vmap(sampler)(x0)
        samples = jnp.reshape(samples, (N, *self.data_shape))
        return samples
