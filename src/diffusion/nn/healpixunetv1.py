from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .backbones import ConvNet
from .modules import HealPIXResnetBlockDown, HealPIXResnetBlockUp, HealPIXResnetBlock, BipartiteRemap
from .timeembedding import LogFourierEmbedding




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
            key: PRNG key used to initialize the blocks
        """
        super().__init__(input_size=input_size)
        keys = jr.split(key, len(n_filters))
        encoding_layers = [HealPIXResnetBlock(in_channels=input_size[0],
                                              out_channels=n_filters[0],
                                              temb_dim=temb_dim,
                                              key=keys[0])]
        encoding_layers += [HealPIXResnetBlockDown(in_channels=n_filters[i],
                                                   out_channels=n_filters[i + 1],
                                                   temb_dim=temb_dim,
                                                   key=keys[i + 1]) for i in range(len(n_filters) - 1)]
        self.encoding_layers = eqx.nn.Sequential(encoding_layers)

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
            skip_filters: Channel sizes coming from encoder skip connections
            temb_dim: Dimension of time embedding
            key: PRNG key used to seed the blocks
        """
        super().__init__(input_size=input_size)
        keys = jr.split(key, len(n_filters))
        decoding_layers = [HealPIXResnetBlockUp(in_channels=self.input_size[0],
                                                out_channels=n_filters[0],
                                                temb_dim=temb_dim,
                                                key=keys[0])]
        for i in range(len(n_filters) - 2):
            χ1, χ2, χ3, χ4 = jr.split(keys[i + 1], 4)
            decoding_layers.append(HealPIXResnetBlockUp(in_channels=skip_filters[i + 1] + n_filters[i],
                                                        out_channels=n_filters[i + 1],
                                                        temb_dim=temb_dim,
                                                        key=χ1))
            decoding_layers.append(HealPIXResnetBlock(in_channels=n_filters[i + 1],
                                                      out_channels=n_filters[i + 1],
                                                      temb_dim=temb_dim,
                                                      key=χ2))
            decoding_layers.append(HealPIXResnetBlock(in_channels=n_filters[i + 1],
                                                      out_channels=n_filters[i + 1],
                                                      temb_dim=temb_dim,
                                                      key=χ3))
            decoding_layers.append(HealPIXResnetBlock(in_channels=n_filters[i + 1],
                                                      out_channels=n_filters[i + 1],
                                                      temb_dim=temb_dim,
                                                      key=χ4))
        χ1, χ2, χ3, χ4 = jr.split(keys[-1], 4)
        decoding_layers += [HealPIXResnetBlock(in_channels=skip_filters[-1] + n_filters[-2],
                                               out_channels=n_filters[-1],
                                               temb_dim=temb_dim,
                                               key=χ1)]
        decoding_layers += [HealPIXResnetBlock(in_channels=n_filters[-1],
                                               out_channels=n_filters[-1],
                                               temb_dim=temb_dim,
                                               key=χ2)]
        decoding_layers += [HealPIXResnetBlock(in_channels=n_filters[-1],
                                               out_channels=n_filters[-1],
                                               temb_dim=temb_dim,
                                               key=χ3)]
        decoding_layers += [HealPIXResnetBlock(in_channels=n_filters[-1],
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


class HealPIXUNetv1(eqx.Module):
    """Time-conditioned Residual U-Net architecture for lat-lon to HEALPix processing.

    1. Remaps input from lat-lon grid to HEALPix grid using bipartite attention
    2. Processes the HEALPix data with U-Net
    3. Remaps the output back to lat-lon grid


    Attributes:
        embedding: Time embedding module using Fourier features
        to_healpix: Remapping layer from lat-lon to HEALPix grid
        to_latlon: Remapping layer from HEALPix to lat-lon grid
        encoder: Downsampling path with residual blocks
        decoder: Upsampling path with skip connections
        output_layer: Final convolution layer
    """
    embedding: LogFourierEmbedding
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
            nside: HEALPix resolution parameter controlling npix = 12 nside^2
            enc_filters: List of channel numbers for encoder layers
            dec_filters: List of channel numbers for decoder layers
            out_channels: Number of output channels
            temb_dim: Dimension of diffusion time embedding
            healpix_emb_dim: Channels after the lat-lon to HEALPix projection
            edges_to_healpix: Adjacency describing the lat-lon → HEALPix map
            edges_to_latlon: Adjacency describing the HEALPix → lat-lon map
            key: PRNG key for initializing all submodules
        """
        in_channels = input_size[0]
        npix = 12 * nside**2
        self.embedding = LogFourierEmbedding(temb_dim)

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
            t: Diffusion time

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
