"""HEALPix-specific convolution modules.
"""
from typing import Optional, Any
from math import sqrt
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import einops
from .padding import pad


class HealPIXConv(eqx.Module):
    """2D convolution adapted for HEALPix data.

    This module implements convolution on HEALPix data by:
    1. Rearranging the HEALPix grid into a 2D representation
    2. Applying standard 2D convolution
    3. Rearranging back to HEALPix format

    This is adapted from https://github.com/NVlabs/cBottle

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        conv: 2D convolution layer
        padding: Padding size

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides (default: 1)
        use_bias: Whether to use bias in convolution (default: True)
        dilation: Spacing between kernel elements (default: 1)
        key: PRNG key for initialization
    """
    in_channels: int
    out_channels: int
    conv: eqx.nn.Conv2d
    padding: int

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, use_bias=True, dilation=1, key=jr.PRNGKey(0)):
        self.conv = eqx.nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  use_bias=use_bias,
                                  dilation=dilation,
                                  key=key)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

    def __call__(self, x):
        nside = int(sqrt((x.shape[-1] / 12)))
        x = einops.rearrange(x, "c (f x y) -> (c) f x y", y=nside, x=nside, f=12)
        x = pad(x, self.padding)
        x = einops.rearrange(x, "(c) f x y -> (f) c x y", f=12, c=self.in_channels)
        x = jax.vmap(self.conv)(x)
        x = einops.rearrange(x, "(f) c x y -> c (f x y)", f=12, c=self.out_channels)
        return x


class HealPIXConvTranspose(eqx.Module):
    """2D transposed convolution adapted for HEALPix data.
    
    This module implements transposed convolution on HEALPix data by:
    1. Rearranging the HEALPix grid into a 2D representation
    2. Applying standard 2D transposed convolution
    3. Rearranging back to HEALPix format

    This is adapted from https://github.com/NVlabs/cBottle

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        conv: 2D transposed convolution layer
        padding: Padding size

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides (default: 1)
        use_bias: Whether to use bias in convolution (default: True)
        dilation: Spacing between kernel elements (default: 1)
        key: PRNG key for initialization
    """
    in_channels: int
    out_channels: int
    conv: eqx.nn.ConvTranspose2d
    padding: int

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, use_bias=True, dilation=1, key=jr.PRNGKey(0)):
        self.conv = eqx.nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding + stride,
                                           use_bias=use_bias,
                                           dilation=dilation,
                                           key=key)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

    def __call__(self, x):
        nside = int(sqrt((x.shape[-1] / 12)))
        x = einops.rearrange(x, "c (f x y) -> (c) f x y", y=nside, x=nside, f=12)
        x = pad(x, 1)
        x = einops.rearrange(x, "(c) f x y -> (f) c x y", f=12, c=self.in_channels)
        x = jax.vmap(self.conv)(x)
        x = einops.rearrange(x, "(f) c x y -> c (f x y)", f=12, c=self.out_channels)
        return x


class HealPIXConvBlock(eqx.Module):
    """HEALPix convolution block with optional normalization and activation.
    
    This block implements a common pattern for HEALPix convolutions:
    1. Group Normalization (optional)
    2. Activation (optional)
    3. Dropout (optional)
    4. HEALPix Convolution

    Attributes:
        norm: Optional group normalization layer
        activation: Optional activation function
        dropout: Optional dropout layer
        conv: HEALPix convolution layer

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides (default: 1)
        use_bias: Whether to use bias in convolution (default: True)
        dilation: Spacing between kernel elements (default: 1)
        activation: Activation function to use ('relu', 'prelu', 'silu', or False)
        dropout: Dropout probability (default: 0.0)
        norm: Whether to use group normalization (default: False)
        key: PRNG key for initialization
    """
    norm: Optional[eqx.nn.GroupNorm]
    activation: Optional[Any]
    dropout: Optional[eqx.nn.Dropout]
    conv: HealPIXConv

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, use_bias=True,
                 dilation=1, activation=False, dropout=0., norm=False, key=jr.PRNGKey(0)):
        self.conv = HealPIXConv(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                use_bias=use_bias,
                                key=key)
        self.norm = eqx.nn.GroupNorm(groups=min(max(1, self.conv.conv.in_channels // 4), 32),
                                     channels=self.conv.conv.in_channels,
                                     channelwise_affine=True) if norm else None
        self.dropout = eqx.nn.Dropout(p=dropout) if dropout > 0 else None

        if activation:
            if activation == 'relu':
                self.activation = jax.nn.relu
            elif activation == 'prelu':
                self.activation = eqx.nn.PReLU()
            elif activation == 'silu':
                self.activation = jax.nn.silu
            else:
                raise ValueError("Unsupported argument specified for activation")
        else:
            self.activation = None

    def __call__(self, x, key=jr.PRNGKey(0)):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x, key=key)
        x = self.conv(x)
        return x

    @property
    def in_channels(self):
        return self.conv.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.conv.out_channels
    

class HealPIXConvTransposeBlock(eqx.Module):
    """HEALPix transposed convolution block with optional normalization and activation.
    
    This block implements a common pattern for HEALPix transposed convolutions:
    1. Group Normalization (optional)
    2. Activation (optional)
    3. Dropout (optional)
    4. HEALPix Transposed Convolution

    Attributes:
        norm: Optional group normalization layer
        activation: Optional activation function
        dropout: Optional dropout layer
        conv: HEALPix transposed convolution layer

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides (default: 1)
        use_bias: Whether to use bias in convolution (default: True)
        dilation: Spacing between kernel elements (default: 1)
        activation: Activation function to use ('relu', 'prelu', 'silu', or False)
        dropout: Dropout probability (default: 0.0)
        norm: Whether to use group normalization (default: False)
        key: PRNG key for initialization
    """
    norm: Optional[eqx.nn.GroupNorm]
    activation: Optional[Any]
    dropout: Optional[eqx.nn.Dropout]
    conv: HealPIXConvTranspose

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, use_bias=True,
                 dilation=1, activation=False, dropout=0., norm=False, key=jr.PRNGKey(0)):
        self.conv = HealPIXConvTranspose(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         use_bias=use_bias,
                                         key=key)
        self.norm = eqx.nn.GroupNorm(groups=min(max(1, self.conv.conv.in_channels // 4), 32),
                                     channels=self.conv.conv.in_channels,
                                     channelwise_affine=True) if norm else None
        self.dropout = eqx.nn.Dropout(p=dropout) if dropout > 0 else None

        if activation:
            if activation == 'relu':
                self.activation = jax.nn.relu
            elif activation == 'prelu':
                self.activation = eqx.nn.PReLU()
            elif activation == 'silu':
                self.activation = jax.nn.silu
            else:
                raise ValueError("Unsupported argument specified for activation")
        else:
            self.activation = None

    def __call__(self, x, key=jr.PRNGKey(0)):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x, key=key)
        x = self.conv(x)
        return x

    @property
    def in_channels(self):
        return self.conv.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.conv.out_channels
    

class HealPIXFacetConv(eqx.Module):
    """Facet-based convolution for HEALPix data.
    
    This module implements convolution on HEALPix data by processing each
    facet of the HEALPix grid separately. Downsamples the input.

    Adapted from https://github.com/deepsphere/deepsphere-cosmo-tf2

    Attributes:
        p: Input is downsampled by 4^p
        conv: 1D convolution layer applied to each facet

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        p: Input is downsampled by 4^p (default: 1)
        key: PRNG key for initialization
    """
    p: int
    conv: eqx.nn.Conv1d
    def __init__(self, in_channels: int, out_channels: int, p: int = 1, key: jr.PRNGKey = jr.PRNGKey(0)):
        if p < 1:
            raise ValueError("Reduction factor p must be >= 1")
        filter_size = 4**p
        self.p = p
        self.conv = eqx.nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=filter_size,
                                  stride=filter_size,
                                  padding=0,
                                  key=key)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (in_channels, n_nodes)
        returns: (out_channels, n_nodes//(4**p))
        """
        return self.conv(x)


class HealPIXFacetConvTranspose(eqx.Module):
    """Facet-based transposed convolution for HEALPix data.
    
    This module implements transposedconvolution on HEALPix data by processing each
    facet of the HEALPix grid separately. Upsamples the input.

    Adapted from https://github.com/deepsphere/deepsphere-cosmo-tf2

    Attributes:
        p: Input is upsampled by 4^p
        conv: 1D transposed convolution layer applied to each facet

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        p: Input is upsampled by 4^p (default: 1)
        key: PRNG key for initialization
    """
    p: int
    conv: eqx.nn.ConvTranspose1d
    def __init__(self, in_channels: int, out_channels: int, p: int = 1, key: jr.PRNGKey = jr.PRNGKey(0)):
        if p < 1:
            raise ValueError("Reduction factor p must be >= 1")
        filter_size = 4**p
        self.p = p
        self.conv = eqx.nn.ConvTranspose1d(in_channels,
                                           out_channels,
                                           kernel_size=filter_size,
                                           stride=filter_size,
                                           padding=0,
                                           key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (in_channels, n_nodes)
        returns: (out_channels, n_nodes//(4**p))
        """
        return self.conv(x)
    

class HealPIXFacetConvBlock(eqx.Module):
    """Facet-based convolution block with optional normalization and activation.
    
    This block implements a common pattern for facet-based HEALPix convolutions:
    1. Group Normalization (optional)
    2. Activation (optional)
    3. Dropout (optional)
    4. Facet-based HEALPix Convolution

    Attributes:
        norm: Optional group normalization layer
        activation: Optional activation function
        dropout: Optional dropout layer
        conv: Facet-based HEALPix convolution layer

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        p: Reduction factor for facet size (default: 1)
        activation: Activation function to use ('relu', 'prelu', 'silu', or False)
        dropout: Dropout probability (default: 0.0)
        norm: Whether to use group normalization (default: False)
        key: PRNG key for initialization
    """
    norm: Optional[eqx.nn.GroupNorm]
    activation: Optional[Any]
    dropout: Optional[eqx.nn.Dropout]
    conv: HealPIXFacetConv

    def __init__(self, in_channels, out_channels, p=1, activation=None, dropout=0., norm=False, key=jr.PRNGKey(0)):
        self.conv = HealPIXFacetConv(in_channels, out_channels, p=p, key=key)

        self.norm = eqx.nn.GroupNorm(groups=min(max(1, self.conv.conv.in_channels // 4), 32),
                                     channels=self.conv.conv.in_channels,
                                     channelwise_affine=True) if norm else None
        self.dropout = eqx.nn.Dropout(p=dropout) if dropout > 0 else None

        if activation:
            if activation == 'relu':
                self.activation = jax.nn.relu
            elif activation == 'prelu':
                self.activation = eqx.nn.PReLU()
            elif activation == 'silu':
                self.activation = jax.nn.silu
            else:
                raise ValueError("Unsupported argument specified for activation")
        else:
            self.activation = None

    def __call__(self, x, key=jr.PRNGKey(0)):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x, key=key)
        x = self.conv(x)
        return x

    @property
    def in_channels(self):
        return self.conv.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.conv.out_channels


class HealPIXFacetConvTransposeBlock(eqx.Module):
    """Facet-based transposed convolution block with optional normalization and activation.
    
    This block implements a common pattern for facet-based HEALPix transposed convolutions:
    1. Group Normalization (optional)
    2. Activation (optional)
    3. Dropout (optional)
    4. Facet-based HEALPix Transposed Convolution

    Attributes:
        norm: Optional group normalization layer
        activation: Optional activation function
        dropout: Optional dropout layer
        conv: Facet-based HEALPix transposed convolution layer

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        p: Reduction factor for facet size (default: 1)
        activation: Activation function to use ('relu', 'prelu', 'silu', or False)
        dropout: Dropout probability (default: 0.0)
        norm: Whether to use group normalization (default: False)
        key: PRNG key for initialization
    """
    norm: Optional[eqx.nn.GroupNorm]
    activation: Optional[Any]
    dropout: Optional[eqx.nn.Dropout]
    conv: HealPIXFacetConvTranspose

    def __init__(self, in_channels, out_channels, p=1, activation=None, dropout=0., norm=False, key=jr.PRNGKey(0)):
        self.conv = HealPIXFacetConvTranspose(in_channels, out_channels, p=p, key=key)

        self.norm = eqx.nn.GroupNorm(groups=min(max(1, self.conv.conv.in_channels // 4), 32),
                                     channels=self.conv.conv.in_channels,
                                     channelwise_affine=True) if norm else None
        self.dropout = eqx.nn.Dropout(p=dropout) if dropout > 0 else None

        if activation:
            if activation == 'relu':
                self.activation = jax.nn.relu
            elif activation == 'prelu':
                self.activation = eqx.nn.PReLU()
            elif activation == 'silu':
                self.activation = jax.nn.silu
            else:
                raise ValueError("Unsupported argument specified for activation")
        else:
            self.activation = None

    def __call__(self, x, key=jr.PRNGKey(0)):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x, key=key)
        x = self.conv(x)
        return x

    @property
    def in_channels(self):
        return self.conv.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.conv.out_channels