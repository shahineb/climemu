# %%
from typing import List, Tuple, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# from .backbones import ConvNet
# from .modules import HealPIXFacetConvBlock, HealPIXFacetConvTransposeBlock, HealPIXConvBlock, BipartiteRemap
# from .timeencoder import GaussianFourierProjection, DoYFourierProjection

import os, sys
base_dir = os.path.join(os.getcwd(), '../../..')
if base_dir not in sys.path:
    sys.path.append(base_dir)
from src.diffusion.nn.backbones import ConvNet
from src.diffusion.nn.modules import HealPIXFacetConvBlock, HealPIXFacetConvTransposeBlock, HealPIXConvBlock, BipartiteRemap
from src.diffusion.nn.timeencoder.gaussianfourier import GaussianFourierProjection, DoYFourierProjection


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
    in_channels: int
    out_channels: int

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
        self.in_channels = in_channels
        self.out_channels = out_channels

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
    in_channels: int
    out_channels: int

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
        self.in_channels = in_channels
        self.out_channels = out_channels

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
    in_channels: int
    out_channels: int

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
        self.in_channels = in_channels
        self.out_channels = out_channels

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



# %%
class Encoder(ConvNet):
    encoding_layers: eqx.nn.Sequential
    skip_filters: List[List[int]]
    def __init__(self,
                 input_size: Tuple[int, ...],
                 n_filters: List[int],
                 n_blocks: List[int],
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        super().__init__(input_size=input_size)
        encoding_layers = []

        # Initial stage
        encoding_stage = []
        for j in range(n_blocks[0]):
            key, χ = jr.split(key)
            block = ResnetBlock(in_channels=input_size[0] if j == 0 else n_filters[0],
                                out_channels=n_filters[0],
                                temb_dim=temb_dim,
                                key=χ)
            encoding_stage.append(block)
        encoding_layers.append(eqx.nn.Sequential(encoding_stage))

        # Downsampling stages
        for i in range(len(n_filters) - 1):
            encoding_stage = []
            key, χ = jr.split(key)
            down_block = ResnetBlockDown(in_channels=n_filters[i],
                                        out_channels=n_filters[i],
                                        temb_dim=temb_dim,
                                        key=χ)
            encoding_stage.append(down_block)
            for j in range(n_blocks[i + 1]):
                key, χ = jr.split(key)
                block = ResnetBlock(in_channels=n_filters[i] if j == 0 else n_filters[i + 1],
                                    out_channels=n_filters[i + 1],
                                    temb_dim=temb_dim,
                                    key=χ)
                encoding_stage.append(block)
            encoding_layers.append(eqx.nn.Sequential(encoding_stage))
        self.encoding_layers = eqx.nn.Sequential(encoding_layers)

        # Record skip features sizes for decoder
        self.skip_filters = [[block.out_channels for block in layers] for layers in self.encoding_layers]


    def __call__(self, x, temb, key=jr.PRNGKey(0)):
        features = []
        for layer in self.encoding_layers:
            for block in layer:
                key, χ = jr.split(key)
                x = block(x, temb, key=χ)
                features += [x]
        return features


class Decoder(ConvNet):
    bottleneck_layers: eqx.nn.Sequential
    decoding_layers: eqx.nn.Sequential

    def __init__(self,
                 input_size: Tuple[int, ...],
                 skip_filters: List[int],
                 n_bottleneck_blocks: int,
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        super().__init__(input_size=input_size)
        # Bottleneck stage
        bottleneck_layers = []
        for _ in range(n_bottleneck_blocks):
            key, χ = jr.split(key)
            block = ResnetBlock(in_channels=input_size[0],
                                out_channels=input_size[0],
                                temb_dim=temb_dim,
                                key=χ)
            bottleneck_layers.append(block)
        self.bottleneck_layers = eqx.nn.Sequential(bottleneck_layers)

        decoding_layers = []
        n_blocks = list(map(len, skip_filters))
        n_filters = [x[0] for x in skip_filters]

        # Initial decoding stage
        decoding_stage = []
        for j in range(n_blocks[0]):
            key, χ = jr.split(key)
            block = ResnetBlock(in_channels=input_size[0] + skip_filters[0][j],
                                out_channels=input_size[0],
                                temb_dim=temb_dim,
                                key=χ)
            decoding_stage.append(block)
        decoding_layers.append(eqx.nn.Sequential(decoding_stage))

        # Upsampling stages
        for i in range(len(n_filters) - 1):
            decoding_stage = []
            key, χ = jr.split(key)
            up_block = ResnetBlockUp(in_channels=n_filters[i],
                                    out_channels=n_filters[i],
                                    temb_dim=temb_dim,
                                    key=χ)
            decoding_stage.append(up_block)
            for j in range(n_blocks[i + 1]):
                key, χ = jr.split(key)
                in_channels = n_filters[i] if j == 0 else n_filters[i + 1]
                block = ResnetBlock(in_channels=in_channels + skip_filters[i + 1][j],
                                    out_channels=n_filters[i + 1],
                                    temb_dim=temb_dim,
                                    key=χ)
                decoding_stage.append(block)
            decoding_layers.append(eqx.nn.Sequential(decoding_stage))
        self.decoding_layers = eqx.nn.Sequential(decoding_layers)

    def __call__(self, features, temb, key=jr.PRNGKey(0)):
        x = features[-1]
        for layer in self.bottleneck_layers:
            key, χ = jr.split(key)
            x = layer(x, temb, key=χ)
        for i, layer in enumerate(self.decoding_layers):
            for j, block in enumerate(layer):
                key, χ = jr.split(key)
                if i == 0 or j > 0:
                    x = jnp.concatenate([x, features.pop()], axis=0)
                x = block(x, temb, key=χ)
        return x
    

class Song2020HealPIXUNet(eqx.Module):
    embedding: GaussianFourierProjection
    doy_embedding: DoYFourierProjection
    pos_embedding: jax.Array
    conv_embedding: HealPIXConvBlock
    to_healpix: BipartiteRemap
    to_latlon: BipartiteRemap
    encoder: Encoder
    decoder: Decoder
    output_layer: eqx.nn.Conv1d

    def __init__(self,
                 input_size: Tuple[int, ...],
                 nside: int,
                 n_filters: List[int],
                 n_blocks: List[int],
                 n_bottleneck_blocks: int,
                 out_channels: int,
                 temb_dim: int,
                 healpix_emb_dim: int,
                 doyemb_dim: int,
                 posemb_dim: int,
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
        # Diffusion time embedding
        self.embedding = GaussianFourierProjection(temb_dim)

        # Day of year embedding
        self.doy_embedding = DoYFourierProjection(doyemb_dim)

        # Embedding fusing layer
        key, χ = jr.split(key)
        self.conv_embedding = HealPIXConvBlock(in_channels=healpix_emb_dim + doyemb_dim,
                                               out_channels=posemb_dim,
                                               kernel_size=3,
                                               key=χ)

        # Positional embedding tensor
        npix = 12 * nside**2
        key, χ = jr.split(key)
        self.pos_embedding = jr.normal(χ, (posemb_dim, npix)) / jnp.sqrt(posemb_dim)

        # LatLon <-> HealPIX remapping
        key, χ = jr.split(key)
        self.to_healpix = BipartiteRemap(in_channels=input_size[0],
                                         out_channels=healpix_emb_dim,
                                         edges=edges_to_healpix,
                                         key=χ)

        key, χ = jr.split(key)
        self.to_latlon = BipartiteRemap(in_channels=out_channels,
                                        out_channels=out_channels,
                                        edges=edges_to_latlon,
                                        key=χ)

        # Encoder
        key, χ = jr.split(key)
        self.encoder = Encoder(input_size=(posemb_dim, npix),
                               n_filters=n_filters,
                               n_blocks=n_blocks,
                               temb_dim=temb_dim,
                               key=χ)

        # Decoder
        key, χ = jr.split(key)
        bottleneck_size = npix // (4 ** (len(n_filters) - 1))
        skip_filters = [x[::-1] for x in self.encoder.skip_filters[::-1]]
        self.decoder = Decoder(input_size=(n_filters[-1], bottleneck_size),
                               skip_filters=skip_filters,
                               n_bottleneck_blocks=n_bottleneck_blocks,
                               temb_dim=temb_dim,
                               key=χ)

        # Output layer
        key, χ = jr.split(key)
        self.output_layer = eqx.nn.Conv1d(in_channels=n_filters[0],
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          key=χ)

    def __call__(self, x: jax.Array, doy: jax.Array, t: jax.Array) -> jax.Array:
        # Map to healpix
        c, nlat, nlon = x.shape
        x = self.to_healpix(x.reshape(c, -1))

        # DoY embedding
        doy_emb = self.doy_embedding(doy)
        doy_emb = jnp.broadcast_to(doy_emb[:, None], (doy_emb.shape[0], x.shape[1]))

        # Fuse and add positional embedding
        x = jnp.concatenate([x, doy_emb], axis=0)
        x = self.conv_embedding(x)
        x = x + self.pos_embedding

        # Diffusion time embedding
        temb = self.embedding(t)

        # Encoder path with skip connections
        features = self.encoder(x, temb)

        # Decoder path using skip connections
        output = self.decoder(features, temb)

        # Final convolution
        output = self.output_layer(output)

        # Map back to latlon
        output = self.to_latlon(output).reshape(-1, nlat, nlon)
        return output


# # %%
# input_size = (5, 96, 192)
# out_channels = 4
# n_filters = [64, 128, 128, 128]
# n_blocks = [2, 2, 2, 2]
# nside = 64
# temb_dim = 128
# doyemb_dim = 16
# n_bottleneck_blocks = 1
# key = jr.PRNGKey(0)
# healpix_emb_dim = 5
# posemb_dim = 64
# edges_data = jnp.load("/Users/shahine/Documents/Research/MIT/code/repos/climemu/sandbox/edges.npz")
# to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
# to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)

# def print_parameter_count(model):
#     leaves = eqx.filter(model, eqx.is_array)
#     n_parameters = sum(jnp.size(x) for x in jax.tree.leaves(leaves))
#     print(f"Number of parameters = {n_parameters/1e6:.2f}M \n")


# bottleneck_size = (12 * nside**2) // (4 ** (len(n_filters) - 1))
# print("Bottleneck size:", bottleneck_size)


# %%
# unet = HealPIXUNetDoY(input_size=input_size,
#                       nside=nside,
#                       n_filters=n_filters,
#                       n_blocks=n_blocks,
#                       n_bottleneck_blocks=n_bottleneck_blocks,
#                       out_channels=out_channels,
#                       temb_dim=temb_dim,
#                       healpix_emb_dim=healpix_emb_dim,
#                       doyemb_dim=doyemb_dim,
#                       posemb_dim=posemb_dim,
#                       edges_to_healpix=to_healpix,
#                       edges_to_latlon=to_latlon,
#                       key=jr.PRNGKey(0))
# print_parameter_count(unet)

# x = jnp.ones(input_size)
# doy = jnp.array([100])
# t = jnp.array([0.5])
# y = unet(x, doy, t)
