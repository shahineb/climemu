"""
Continuous variance exploding (VE) noise schedule.

Provides methods for computing noise scales, time-to-noise mappings, drift coefficients,
sampling noise scales, and generating integration timesteps.

Code adapted from 
"""
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class ContinuousVESchedule(eqx.Module):
    σmin: float
    σmax: float
    logσmin: float
    logσmax: float
    σd: float
    timesteps: jnp.ndarray

    def __init__(self, σmin, σmax, timesteps=None):
        """
        Args:
            σmin: Minimum noise scale.
            σmax: Maximum noise scale.
            timesteps: Optional array of integration timesteps.
        """
        self.σmin = σmin
        self.σmax = σmax
        self.logσmin = jnp.log(σmin)
        self.logσmax = jnp.log(σmax)
        self.σd = σmax / σmin
        self.timesteps = timesteps

    @eqx.filter_jit
    def σ(self, t):
        """
        Computes the noise scale σ(t) = σmin * √(σd^(2t) - 1)

        Args:
            t ∈ [0, 1]

        Returns:
            σ(t)
        """
        return self.σmin * jnp.sqrt(self.σd ** (2 * t) - 1)
    
    @eqx.filter_jit
    def t(self, σ):
        """
        Computes the time t corresponding to a given noise scale σ.
        t(σ) = log(1 + (σ / σmin)^2) / (2 * log(σd))

        Args:
            σ ∈ [σmin, σmax]

        Returns:
            t(σ)
        """
        return jnp.log1p((σ / self.σmin) ** 2) / (2 * jnp.log(self.σd))

    @eqx.filter_jit
    def g2(self, t):
        """
        Computes squared diffusion coefficient g²(t) (used in the drift of the reverse-time ODE).

        g²(t) = d(σ(t)²)/dt = σmin² * (σd^(2t)) * (2 * log(σd))

        Args:
            t ∈ [0, 1]

        Returns:
            g²(t): Drift coefficient.
        """
        dσ2dt = (self.σmin ** 2) * ((self.σd) ** (2 * t)) * (2 * jnp.log(self.σd))
        return dσ2dt

    @eqx.filter_jit
    def sample_σ(self, key):
        """
        Samples a noise scale σ from the uniform distribution in log-space [logσmin, logσmax].

        Args:
            key: JAX PRNGKey.

        Returns:
            σ: Sampled noise scale.
        """
        logσ = jr.uniform(key, minval=self.logσmin, maxval=self.logσmax)
        return jnp.exp(logσ)

    def get_timesteps(self, steps):
        """
        Generates an array of integration timesteps for ODE solvers.

        Args:
            steps: Number of timesteps.

        Returns:
            timesteps: Array of time values in [tmin, tmax].
        """
        tmin = jnp.log(2) / (2 * jnp.log(self.σd))
        tmax = 1.0
        if self.timesteps is not None:
            timesteps = self.timesteps
            timesteps = timesteps.at[0].set(tmin)
            timesteps = timesteps.at[-1].set(tmax)
        else:
            timesteps = jnp.linspace(tmin, tmax, steps)
        return timesteps