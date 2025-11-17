import jax
import jax.numpy as jnp
import equinox as eqx


class GaussianFourierProjection(eqx.Module):
    """Time embedding using Gaussian Fourier features.
    
    Projects scalar time inputs into a higher-dimensional embedding using
    Gaussian Fourier features (sinusoidal basis with log-spaced frequencies).

    Attributes:
        B: jnp.ndarray of shape (d//2,), log-spaced frequency multipliers.
    """
    B: jax.Array

    def __init__(self, d: int):
        """
        Args:
            d: Output embedding dimension (must be even).

        Raises:
            ValueError: If `d` is not even.
        """
        if d % 2 != 0:
            raise ValueError(f"Output dimension d must be even, got {d}") 
        half_d = d // 2
        k = jnp.arange(half_d)
        self.B = jnp.exp(-jnp.log(10000) * k / (half_d - 1))   # Generate frequencies on logscale from 1 to 1/10000

    def __call__(self, t):
        Bt = self.B * t
        temb = jnp.concatenate((jnp.sin(Bt), jnp.cos(Bt)), axis=-1)
        return temb


class DoYFourierProjection(eqx.Module):
    """Day-of-Year embedding using Fourier features.

    Projects day-of-year inputs into a higher-dimensional embedding using
    Fourier features (sinusoidal basis with linearly spaced frequencies).

    Day-of-year is assumed to be in [1, 365] (no leap years).

    Attributes:
        k: jnp.ndarray of shape (d//2,), linearly spaced frequency multipliers.
    """
    k: jax.Array

    def __init__(self, d: int):
        """
        Args:
            d: Output embedding dimension (must be even).
        Raises:
            ValueError: If `d` is not even.
        """
        if d % 2 != 0:
            raise ValueError(f"Output dimension d must be even, got {d}") 
        half_d = d // 2
        self.k = 1 + jnp.arange(half_d)

    def __call__(self, doy):
        θ = 2 * jnp.pi * (((doy - 1.0) % 365) / 365)
        kθ = self.k * θ
        doy_emb = jnp.concatenate((jnp.sin(kθ), jnp.cos(kθ)), axis=-1)
        return doy_emb