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
