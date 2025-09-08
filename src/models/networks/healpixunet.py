from typing import List, Tuple, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .backbones import ConvNet
from .modules import HealPIXFacetConvBlock, HealPIXFacetConvTransposeBlock, HealPIXConvBlock, BipartiteRemap
from .timeencoder import GaussianFourierProjection


class ResnetBlockDown(eqx.Module):
    """Downsampling residual block using facet-based convolutions.
    
    This block implements a residual connection with downsampling, where:
    1. The main path uses facet-based convolutions for downsampling
    2. Time information is incorporated through embeddings
    3. A skip connection matches channels using facet-based convolution

    Attributes:
        down: Facet-based downsampling convolution
        conv: Standard convolution for feature processing
        proj: Skip connection projection using facet-based convolution
        linear: Linear layer for time embedding
    """
    down: HealPIXFacetConvBlock
    proj: HealPIXFacetConvBlock
    conv: HealPIXConvBlock
    linear: eqx.nn.Linear

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
            key: PRNG key for initialization
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding size
        """
        χ1, χ2, χ3, χ4 = jr.split(key, 4)
        self.down = HealPIXFacetConvBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          norm=True,
                                          activation='silu',
                                          key=χ1)
        self.conv = HealPIXConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation='silu',
            norm=True,
            key=χ2
        )
        self.proj = HealPIXFacetConvBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          key=χ3)
        self.linear = eqx.nn.Linear(
            in_features=temb_dim,
            out_features=out_channels,
            key=χ4
        )

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
        # Time embedding
        temb = self.linear(jax.nn.silu(temb))
        Fx = Fx + jnp.expand_dims(temb, axis=tuple(range(1, Fx.ndim)))
        # Convolution
        Fx = self.conv(Fx, key=χ2)
        # Residual connection
        x̃ = self.proj(x)
        y = Fx + x̃
        return y


class ResnetBlockUp(eqx.Module):
    """Upsampling residual block using facet-based convolutions.
    
    This block implements a residual connection with upsampling, where:
    1. The main path uses facet-based transposed convolutions for upsampling
    2. Time information is incorporated through embeddings
    3. A skip connection matches channels using facet-based transposed convolution

    Attributes:
        up: Facet-based upsampling convolution
        conv: Standard convolution for feature processing
        proj: Skip connection projection using facet-based transposed convolution
        linear: Linear layer for time embedding
    """
    up: HealPIXFacetConvTransposeBlock
    proj: HealPIXFacetConvTransposeBlock
    conv: HealPIXConvBlock
    linear: eqx.nn.Linear

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
            key: PRNG key for initialization
            kernel_size: Size of convolution kernel (default 4 for transposed conv)
            stride: Stride of convolution
            padding: Padding size
        """
        χ1, χ2, χ3, χ4 = jr.split(key, 4)
        self.up = HealPIXFacetConvTransposeBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm=True,
            activation='silu',
            key=χ1
        )
        self.conv = HealPIXConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation='silu',
            norm=True,
            key=χ2
        )
        self.proj = HealPIXFacetConvTransposeBlock(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   key=χ3)
        self.linear = eqx.nn.Linear(
            in_features=temb_dim,
            out_features=out_channels,
            key=χ4
        )

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
        # Time embedding
        temb = self.linear(jax.nn.silu(temb))
        Fx = Fx + jnp.expand_dims(temb, axis=tuple(range(1, Fx.ndim)))
        # Convolution
        Fx = self.conv(Fx, key=χ2)
        # Residual connection
        x̃ = self.proj(x)
        y = x̃ + Fx
        return y


class ResnetBlock(eqx.Module):
    """Downsampling residual block using facet-based convolutions.
    
    This block implements a residual connection with downsampling, where:
    1. The main path uses facet-based convolutions for downsampling
    2. Time information is incorporated through embeddings
    3. A skip connection matches channels using facet-based convolution

    Attributes:
        down: Facet-based downsampling convolution
        conv: Standard convolution for feature processing
        proj: Skip connection projection using facet-based convolution
        linear: Linear layer for time embedding
    """
    conv1: HealPIXConvBlock
    conv2: HealPIXConvBlock
    proj: Callable
    linear: eqx.nn.Linear

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
            key: PRNG key for initialization
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding size
        """
        χ1, χ2, χ3, χ4 = jr.split(key, 4)
        self.conv1 = HealPIXConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation='silu',
            norm=True,
            key=χ1
        )
        self.conv2 = HealPIXConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation='silu',
            norm=True,
            key=χ2
        )
        if in_channels == out_channels:
            self.proj = lambda x: x
        else:
            self.proj = eqx.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                key=χ3
            )
        self.linear = eqx.nn.Linear(
            in_features=temb_dim,
            out_features=out_channels,
            key=χ4
        )

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
        return y


class Encoder(ConvNet):
    """U-Net encoder with time-conditioned residual blocks.
    
    Attributes:
        encoding_layers: Sequential container of downsampling blocks
    """
    encoding_layers: eqx.nn.Sequential

    def __init__(self, 
                 input_size: Tuple[int, ...], 
                 n_filters: List[int], 
                 temb_dim: int, 
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the encoder.
        
        Args:
            input_size: Input shape (channels, height, width)
            n_filters: List of channel numbers for each layer
            temb_dim: Dimension of time embedding
        """
        super().__init__(input_size=input_size)
        keys = jr.split(key, len(n_filters))
        encoding_layers = [ResnetBlock(in_channels=input_size[0],
                                       out_channels=n_filters[0],
                                       temb_dim=temb_dim,
                                       key=keys[0])]
        encoding_layers += [ResnetBlockDown(in_channels=n_filters[i],
                                            out_channels=n_filters[i + 1],
                                            temb_dim=temb_dim,
                                            key=keys[i + 1]) for i in range(len(n_filters) - 1)]
        self.encoding_layers = eqx.nn.Sequential(encoding_layers)

    def _compute_output_size(self) -> Tuple[int, ...]:
        """Compute the output shape of the encoder.
        
        Returns:
            Shape of the encoder output (channels, height, width)
        """
        dummy_input = jnp.zeros(self._input_size)
        temb_dim = self.encoding_layers[0].linear.in_features
        dummy_temb = jnp.zeros((temb_dim,))
        output = self.__call__(dummy_input, dummy_temb)
        return output[-1].shape

    def __call__(self, x: jax.Array, temb: jax.Array, key: jax.random.PRNGKey = jr.PRNGKey(0)) -> List[jax.Array]:
        """Forward pass of the encoder.
        
        Args:
            x: Input tensor of shape (channels, height, width)
            temb: Time embedding tensor
            key: PRNG key for stochastic operations
        
        Returns:
            List of feature maps at different scales, ordered from 
            highest resolution to lowest.
        """
        features = []
        for layer in self.encoding_layers:
            key, χ = jr.split(key)
            x = layer(x, temb, key=χ)
            features += [x]
        return features


class Decoder(ConvNet):
    """U-Net decoder with time-conditioned residual blocks.

    Attributes:
        decoding_layers: Sequential container of upsampling blocks
    """
    decoding_layers: eqx.nn.Sequential

    def __init__(self,
                 input_size: Tuple[int, ...],
                 n_filters: List[int],
                 skip_filters: List[int],
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the decoder.
        
        Args:
            input_size: Input shape from encoder's deepest layer
            n_filters: List of channel numbers for each layer
            temb_dim: Dimension of time embedding
        """
        super().__init__(input_size=input_size)
        keys = jr.split(key, len(n_filters))
        decoding_layers = [ResnetBlockUp(in_channels=self.input_size[0],
                                         out_channels=n_filters[0],
                                         temb_dim=temb_dim,
                                         key=keys[0])]
        for i in range(len(n_filters) - 2):
            χ1, χ2, χ3, χ4 = jr.split(keys[i + 1], 4)
            decoding_layers.append(ResnetBlockUp(in_channels=skip_filters[i + 1] + n_filters[i],
                                                 out_channels=n_filters[i + 1],
                                                 temb_dim=temb_dim,
                                                 key=χ1))
            decoding_layers.append(ResnetBlock(in_channels=n_filters[i + 1],
                                               out_channels=n_filters[i + 1],
                                               temb_dim=temb_dim,
                                               key=χ2))
            decoding_layers.append(ResnetBlock(in_channels=n_filters[i + 1],
                                               out_channels=n_filters[i + 1],
                                               temb_dim=temb_dim,
                                               key=χ3))
            decoding_layers.append(ResnetBlock(in_channels=n_filters[i + 1],
                                               out_channels=n_filters[i + 1],
                                               temb_dim=temb_dim,
                                               key=χ4))
        χ1, χ2, χ3, χ4 = jr.split(keys[-1], 4)
        decoding_layers += [ResnetBlock(in_channels=skip_filters[-1] + n_filters[-2],
                                        out_channels=n_filters[-1],
                                        temb_dim=temb_dim,
                                        key=χ1)]
        decoding_layers += [ResnetBlock(in_channels=n_filters[-1],
                                        out_channels=n_filters[-1],
                                        temb_dim=temb_dim,
                                        key=χ2)]
        decoding_layers += [ResnetBlock(in_channels=n_filters[-1],
                                        out_channels=n_filters[-1],
                                        temb_dim=temb_dim,
                                        key=χ3)]
        decoding_layers += [ResnetBlock(in_channels=n_filters[-1],
                                        out_channels=n_filters[-1],
                                        temb_dim=temb_dim,
                                        key=χ4)]
        self.decoding_layers = eqx.nn.Sequential(decoding_layers)

    def __call__(self, features: List[jax.Array], temb: jax.Array, key: jax.random.PRNGKey = jr.PRNGKey(0)) -> jax.Array:
        """Forward pass of the decoder.
        
        Args:
            features: List of feature maps from encoder, ordered from 
                     highest resolution to lowest
            temb: Time embedding tensor
            key: PRNG key for stochastic operations
        
        Returns:
            Output tensor with upsampled spatial dimensions
        """
        x = features.pop()  # Start with bottleneck features
        for i, layer in enumerate(self.decoding_layers):
            key, χ = jr.split(key)
            x = layer(x, temb, key=χ)
            if i % 4 == 0 and len(features) > 0:
                x = jnp.concatenate([x, features.pop()], axis=0)
        return x


class HealPIXUNet(eqx.Module):
    """Time-conditioned Residual U-Net architecture for lat-lon to HEALPix processing.

    1. Remaps input from lat-lon grid to HEALPix grid using bipartite attention
    2. Processes the HEALPix data through a time-conditioned U-Net
    3. Remaps the output back to lat-lon grid


    Attributes:
        embedding: Time embedding module using Gaussian Fourier features
        to_healpix: Remapping layer from lat-lon to HEALPix grid
        to_latlon: Remapping layer from HEALPix to lat-lon grid
        encoder: Downsampling path with residual blocks
        decoder: Upsampling path with skip connections
        output_layer: Final convolution layer

    Args:
        input_size: Input shape (channels, nlat, nlon)
        nside: HEALPix nside parameter determining resolution (12 * nside^2 pixels)
        enc_filters: List of channel numbers for encoder layers
        dec_filters: List of channel numbers for decoder layers
        out_channels: Number of output channels
        temb_dim: Dimension of time embedding
        healpix_emb_dim: Dimension of intermediate HEALPix representation
        edges_to_healpix: Edge connectivity matrix for lat-lon to HEALPix remapping
        edges_to_latlon: Edge connectivity matrix for HEALPix to lat-lon remapping
        key: PRNG key for initialization
    """
    embedding: GaussianFourierProjection
    encoder: Encoder
    decoder: Decoder
    output_layer: eqx.nn.Conv1d
    to_healpix: BipartiteRemap
    to_latlon: BipartiteRemap

    def __init__(self,
                 input_size: Tuple[int, ...],
                 nside: int,
                 enc_filters: List[int],
                 dec_filters: List[int],
                 out_channels: int,
                 temb_dim: int,
                 healpix_emb_dim: int,
                 edges_to_healpix: jax.Array,
                 edges_to_latlon: jax.Array,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the U-Net architecture.

        Args:
            input_size: Input shape (channels, nlat, nlon)
            enc_filters: List of channel numbers for encoder layers
            dec_filters: List of channel numbers for decoder layers
            out_channels: Number of output channels
            temb_dim: Dimension of time embedding
        """
        in_channels = input_size[0]
        npix = 12 * nside**2
        self.embedding = GaussianFourierProjection(temb_dim)

        key, χ = jr.split(key)
        self.to_healpix = BipartiteRemap(in_channels=in_channels,
                                         out_channels=healpix_emb_dim,
                                         edges=edges_to_healpix,
                                         key=χ)

        key, χ = jr.split(key)
        self.to_latlon = BipartiteRemap(in_channels=out_channels,
                                        out_channels=out_channels,
                                        edges=edges_to_latlon,
                                        key=χ)

        key, χ = jr.split(key)
        self.encoder = Encoder(input_size=(healpix_emb_dim, npix),
                               n_filters=enc_filters,
                               temb_dim=temb_dim,
                               key=χ)

        key, χ = jr.split(key)
        bottleneck_size = npix // (4 ** len(enc_filters))
        self.decoder = Decoder(input_size=(enc_filters[-1], bottleneck_size),
                               n_filters=dec_filters,
                               skip_filters=enc_filters[::-1],
                               temb_dim=temb_dim,
                               key=χ)

        key, χ = jr.split(key)
        self.output_layer = eqx.nn.Conv1d(in_channels=dec_filters[-1],
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          key=χ)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Forward pass of the U-Net.

        Args:
            x: Input tensor of shape (channels, height, width)
            t: Time values

        Returns:
            Output tensor of shape (out_channels, height, width)
        """
        # Map to healpix
        c, nlat, nlon = x.shape
        x = self.to_healpix(x.reshape(c, -1))

        # Time embedding
        temb = self.embedding(t)

        # Encoder path with skip connections
        latent_features = self.encoder(x, temb)

        # Decoder path using skip connections
        output = self.decoder(latent_features, temb)

        # Final convolution
        output = self.output_layer(output)

        # Map back to latlon
        output = self.to_latlon(output).reshape(-1, nlat, nlon)
        return output
