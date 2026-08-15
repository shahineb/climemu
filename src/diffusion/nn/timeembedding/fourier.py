import jax
import jax.numpy as jnp
import equinox as eqx


class FourierEmbedding(eqx.Module):
    """Scalar time embedding using Fourier features.

    Maps scalar time inputs into a higher-dimensional embedding using
    Fourier features (sinusoidal basis with different frequencies).

    Attributes:
        ω: jnp.ndarray of shape (d//2,), frequency multipliers.
        out_channels: int, output embedding dimension.
    """
    ω: jax.Array
    out_channels: int

    def __init__(self, ω: jnp.ndarray):
        """
        Args:
            ω: jnp.ndarray of shape (d//2,), frequency multipliers.
        """
        self.ω = ω
        self.out_channels = 2 * ω.shape[0]

    def __call__(self, t):
        ωt = self.ω * t
        temb = jnp.concatenate((jnp.sin(ωt), jnp.cos(ωt)), axis=-1)
        return temb


class LinearFourierEmbedding(FourierEmbedding):
    """Linear Fourier embedding using Fourier features.

    Maps scalar time inputs into a higher-dimensional embedding using
    Fourier features (sinusoidal basis with linearly spaced frequencies).

    Attributes:
        ω: jnp.ndarray of shape (d//2,), linearly-spaced frequency multipliers.
        out_channels: int, output embedding dimension.
    """
    def __init__(self, d: int):
        """
        Args:
            d: Output embedding dimension (must be even).
        """
        if d % 2 != 0:
            raise ValueError(f"Output dimension d must be even, got {d}")
        half_d = d // 2
        ω = 1 + jnp.arange(half_d)
        super().__init__(ω)


class LogFourierEmbedding(FourierEmbedding):
    """Logarithmic Fourier embedding using Fourier features.

    Maps scalar time inputs into a higher-dimensional embedding using
    Fourier features (sinusoidal basis with log-spaced frequencies).

    Attributes:
        ω: jnp.ndarray of shape (d//2,), log-spaced frequency multipliers.
        out_channels: int, output embedding dimension.
    """
    def __init__(self, d: int):
        """
        Args:
            d: Output embedding dimension (must be even).
        """
        if d % 2 != 0:
            raise ValueError(f"Output dimension d must be even, got {d}")
        half_d = d // 2
        ω = jnp.exp(-jnp.log(10000) * jnp.arange(half_d) / (half_d - 1))   # Generate frequencies on logscale from 1 to 1/10000
        super().__init__(ω)


class DoYFourierEmbedding(LinearFourierEmbedding):
    """Day-of-Year embedding using Fourier features.

    Maps day-of-year inputs into a higher-dimensional embedding using
    Fourier features (sinusoidal basis with linearly spaced frequencies).

    Day-of-year is assumed to be in [1, 365] (no leap years).

    Attributes:
        ω: jnp.ndarray of shape (d//2,), linearly spaced frequency multipliers.
        out_channels: int, output embedding dimension.
    """
    def __call__(self, doy):
        θ = 2 * jnp.pi * (((doy - 1.0) % 365) / 365)
        return super().__call__(θ)
