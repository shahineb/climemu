from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .conv import HealPIXFacetConvBlock, HealPIXFacetConvTransposeBlock, HealPIXConvBlock
from .attention import HealPIXAttention


class HealPIXResnetBlockDown(eqx.Module):
    """Downsampling residual block using facet-based convolutions.

    Attributes:
        down: Facet-based downsampling convolution
        conv: Standard convolution for feature processing
        proj: Skip connection projection using facet-based convolution
        linear: Linear layer for time embedding
        attention: Spatial attention mechanism (optional)
    """
    down: HealPIXFacetConvBlock
    proj: HealPIXFacetConvBlock
    conv: HealPIXConvBlock
    linear: eqx.nn.Linear
    attention: Optional[HealPIXAttention]
    in_channels: int
    out_channels: int

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_dim: int,
                 use_attention: bool = False,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
            use_attention: Whether to include spatial attention
            key: PRNG key for initialization
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        χ1, χ2, χ3, χ4, χ5 = jr.split(key, 5)
        self.down = HealPIXFacetConvBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          norm=True,
                                          activation='silu',
                                          key=χ1)
        self.conv = HealPIXConvBlock(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     padding=1,
                                     activation='silu',
                                     norm=True,
                                     key=χ2)
        self.proj = HealPIXFacetConvBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          key=χ3)
        self.linear = eqx.nn.Linear(in_features=temb_dim,
                                    out_features=out_channels,
                                    key=χ4)
        self.attention = HealPIXAttention(channels=out_channels,
                                          n_heads=4,
                                          key=χ5) if use_attention else None

    def __call__(self, x: jax.Array, temb: jax.Array, key: jax.random.PRNGKey = jr.PRNGKey(0)) -> jax.Array:
        """
        Forward pass through downsampling block.

        Parameters
        ----------
        x : jax.Array
            Input features with shape (channels, nodes).
        temb : jax.Array
            Time embedding vector (temb_dim,).
        key : jax.random.PRNGKey
            Key for randomness in attention and conv.

        Returns
        -------
        jax.Array
            Output features with downsampled nodes.
        """
        χ1, χ2 = jr.split(key, 2)
        # Downsample
        Fx = self.down(x, key=χ1)
        # Diffusion time embedding
        temb = self.linear(jax.nn.silu(temb))
        Fx = Fx + jnp.expand_dims(temb, axis=tuple(range(1, Fx.ndim)))
        # Convolution
        Fx = self.conv(Fx, key=χ2)
        # Residual connection
        x̃ = self.proj(x)
        y = Fx + x̃
        # Spatial attention
        if self.attention:
            y = self.attention(y)
        return y



class HealPIXResnetBlockUp(eqx.Module):
    """Upsampling residual block using facet-based convolutions.

    Attributes:
        up: Facet-based upsampling convolution
        conv: Standard convolution for feature processing
        proj: Skip connection projection using facet-based transposed convolution
        linear: Linear layer for time embedding
        attention: Spatial attention mechanism (optional)
    """
    up: HealPIXFacetConvTransposeBlock
    proj: HealPIXFacetConvTransposeBlock
    conv: HealPIXConvBlock
    linear: eqx.nn.Linear
    attention: Optional[HealPIXAttention]
    in_channels: int
    out_channels: int

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_dim: int,
                 use_attention: bool = False,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
            use_attention: Whether to include spatial attention
            key: PRNG key for initialization
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        χ1, χ2, χ3, χ4, χ5 = jr.split(key, 5)
        self.up = HealPIXFacetConvTransposeBlock(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 norm=True,
                                                 activation='silu',
                                                 key=χ1)
        self.conv = HealPIXConvBlock(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     padding=1,
                                     activation='silu',
                                     norm=True,
                                     key=χ2)
        self.proj = HealPIXFacetConvTransposeBlock(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   key=χ3)
        self.linear = eqx.nn.Linear(in_features=temb_dim,
                                    out_features=out_channels,
                                    key=χ4)
        self.attention = HealPIXAttention(channels=out_channels,
                                          n_heads=4,
                                          key=χ5) if use_attention else None

    def __call__(self, x: jax.Array, temb: jax.Array, key: jax.random.PRNGKey = jr.PRNGKey(0)) -> jax.Array:
        """Forward pass of the upsampling block.

        Args:
            x: Input tensor of shape (channels, height, width)
            temb: Time embedding tensor
            key: PRNG key for stochastic operations

        Returns:
            Output tensor of shape (out_channels, height*stride, width*stride)
        """
        χ1, χ2 = jr.split(key, 2)
        # Upsample
        Fx = self.up(x, key=χ1)
        # Diffusion time embedding
        temb = self.linear(jax.nn.silu(temb))
        Fx = Fx + jnp.expand_dims(temb, axis=tuple(range(1, Fx.ndim)))
        # Convolution
        Fx = self.conv(Fx, key=χ2)
        # Residual connection
        x̃ = self.proj(x)
        y = x̃ + Fx
        # Spatial attention
        if self.attention:
            y = self.attention(y)
        return y


class HealPIXResnetBlock(eqx.Module):
    """Residual block using HealPIX convolutions.

    Attributes:
        conv1: HealPIX convolution block
        conv2: HealPIX convolution block
        proj: Skip connection projection
        linear: Linear layer for diffusion time embedding
        attention: Spatial attention mechanism (optional)
    """
    conv1: HealPIXConvBlock
    conv2: HealPIXConvBlock
    proj: eqx.nn.Conv1d
    linear: eqx.nn.Linear
    attention: Optional[HealPIXAttention]
    in_channels: int
    out_channels: int

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_dim: int,
                 use_attention: bool = False,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
            use_attention: Whether to include spatial attention
            key: PRNG key for initialization
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        χ1, χ2, χ3, χ4, χ5 = jr.split(key, 5)
        self.conv1 = HealPIXConvBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      activation='silu',
                                      norm=True,
                                      key=χ1)
        self.conv2 = HealPIXConvBlock(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      activation='silu',
                                      norm=True,
                                      key=χ2)
        if in_channels == out_channels:
            self.proj = lambda x: x
        else:
            self.proj = eqx.nn.Conv1d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      key=χ3)
        self.linear = eqx.nn.Linear(in_features=temb_dim,
                                    out_features=out_channels,
                                    key=χ4)
        self.attention = HealPIXAttention(channels=out_channels,
                                          n_heads=4,
                                          key=χ5) if use_attention else None

    def __call__(self, x: jax.Array, temb: jax.Array, key: jax.random.PRNGKey = jr.PRNGKey(0)) -> jax.Array:
        """
        Forward pass through downsampling block.

        Parameters
        ----------
        x : jax.Array
            Input features with shape (channels, nodes).
        temb : jax.Array
            Time embedding vector (temb_dim,).
        key : jax.random.PRNGKey
            Key for randomness in attention and conv.

        Returns
        -------
        jax.Array
            Output features with downsampled nodes.
        """
        χ1, χ2 = jr.split(key, 2)
        # First conv
        Fx = self.conv1(x, key=χ1)
        # Time embedding
        temb = self.linear(jax.nn.silu(temb))
        Fx = Fx + jnp.expand_dims(temb, axis=tuple(range(1, Fx.ndim)))
        # Second conv
        Fx = self.conv2(Fx, key=χ2)
        # Residual connection
        x̃ = self.proj(x)
        y = Fx + x̃
        # Spatial attention
        if self.attention:
            y = self.attention(y)
        return y
