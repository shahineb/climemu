from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .backbones import ConvNet
from .modules import HealPIXResnetBlockDown, HealPIXResnetBlockUp, HealPIXResnetBlock, BipartiteRemap, HealPIXConvBlock
from .timeembedding import LogFourierEmbedding, DoYFourierEmbedding



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
            key: PRNG key used to seed the residual blocks
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
        bottleneck_layers: Sequential container applied at the latent resolution
        decoding_layers: Sequential container of upsampling stages with skips
    """
    bottleneck_layers: eqx.nn.Sequential
    decoding_layers: eqx.nn.Sequential

    def __init__(self,
                 input_size: Tuple[int, ...],
                 n_filters: List[int],
                 skip_filters: List[int],
                 n_blocks: List[int],
                 n_bottleneck_blocks: int,
                 temb_dim: int,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the decoder.

        Args:
            input_size: Input shape from encoder's deepest layer
            n_filters: List of decoder channel widths per stage
            skip_filters: Channel sizes coming from encoder skip connections
            n_blocks: Number of residual refinement blocks per decoder stage
            n_bottleneck_blocks: Number of residual blocks at the bottleneck
            temb_dim: Dimension of time embedding
            key: PRNG key used to initialize submodules
        """
        super().__init__(input_size=input_size)
        # Bottleneck layers
        bottleneck_layers = []
        for i in range(n_bottleneck_blocks):
            key, χ = jr.split(key)
            bottleneck_layers.append(HealPIXResnetBlock(in_channels=self.input_size[0] if i == 0 else n_filters[0],
                                                        out_channels=n_filters[0],
                                                        temb_dim=temb_dim,
                                                        key=χ))
        self.bottleneck_layers = eqx.nn.Sequential(bottleneck_layers)

        # Upsampling layers
        decoding_layers = []
        n_filters.append(n_filters[-1])
        for i in range(len(n_filters) - 1):
            decoding_stage = []
            key, χ = jr.split(key)
            if i < len(n_filters) - 2:
                decoding_stage.append(HealPIXResnetBlockUp(in_channels=skip_filters[i] + n_filters[i],
                                                           out_channels=n_filters[i + 1],
                                                           temb_dim=temb_dim,
                                                           key=χ))
            else:
                decoding_stage.append(HealPIXResnetBlock(in_channels=skip_filters[i] + n_filters[i],
                                                         out_channels=n_filters[i + 1],
                                                         temb_dim=temb_dim,
                                                         key=χ))
            for _ in range(n_blocks[i]):
                key, χ = jr.split(key)
                decoding_stage.append(HealPIXResnetBlock(in_channels=n_filters[i + 1],
                                                         out_channels=n_filters[i + 1],
                                                         temb_dim=temb_dim,
                                                         key=χ))
            decoding_layers.append(eqx.nn.Sequential(decoding_stage))
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
        x = features[-1]
        for layer in self.bottleneck_layers:
            key, χ = jr.split(key)
            x = layer(x, temb, key=χ)
        for layer in self.decoding_layers:
            x = jnp.concatenate([x, features.pop()], axis=0)
            for block in layer:
                key, χ = jr.split(key)
                x = block(x, temb, key=χ)
        return x


class HealPIXUNetv2(eqx.Module):
    """Time-conditioned Residual U-Net architecture for lat-lon to HEALPix processing.
       Adds day-of-year conditioning and learned HEALPix positional embeddings to the previous version.

    1. Remaps input from lat-lon grid to HEALPix grid using bipartite attention
    2. Concatenates day-of-year embeddings and adds learned positional bias
    3. Processes the HEALPix data with U-Net
    4. Remaps the output back to lat-lon grid

    Attributes:
        embedding: LogFourierEmbedding for diffusion time
        doy_embedding: DoYFourierEmbedding for per-sample seasonal context
        pos_embedding: Learned positional bias added to HEALPix nodes
        conv_embedding: HealPIXConvBlock mixing physical+Doy channels to posemb_dim
        to_healpix: BipartiteRemap projecting lat-lon pixels to HEALPix nodes
        to_latlon: BipartiteRemap projecting HEALPix nodes back to lat-lon pixels
        encoder: Downsampling path operating on HEALPix features
        decoder: Upsampling path with skip connections and bottleneck stacks
        output_layer: 1x1 convolution that produces the decoded channels
    """
    embedding: LogFourierEmbedding
    encoder: Encoder
    decoder: Decoder
    output_layer: eqx.nn.Conv1d
    to_healpix: BipartiteRemap
    to_latlon: BipartiteRemap
    doy_embedding: DoYFourierEmbedding
    conv_embedding: HealPIXConvBlock
    pos_embedding: jax.Array

    def __init__(self,
                 input_size: Tuple[int, ...],
                 out_channels: int,
                 nside: int,
                 enc_filters: List[int],
                 dec_filters: List[int],
                 dec_blocks: List[int],
                 n_bottleneck_blocks: int,
                 temb_dim: int,
                 doyemb_dim: int,
                 healpix_emb_dim: int,
                 posemb_dim: int,
                 edges_to_healpix: jax.Array,
                 edges_to_latlon: jax.Array,
                 key: jax.random.PRNGKey = jr.PRNGKey(0)):
        """Initialize the DoY-augmented HEALPix U-Net.

        Args:
            input_size: Input tensor shape (channels, nlat, nlon)
            out_channels: Number of output channels
            nside: HEALPix resolution parameter controlling npix = 12 nside^2
            enc_filters: Channel counts for each encoder stage
            dec_filters: Channel counts for each decoder stage
            dec_blocks: Number of residual refinement blocks per decoder stage
            n_bottleneck_blocks: Number of residual blocks at the bottleneck
            temb_dim: Dimension of diffusion time embedding
            doyemb_dim: Dimension of the day-of-year embedding
            healpix_emb_dim: Channels after the lat-lon to HEALPix projection
            posemb_dim: Dimension of the learned positional embedding
            edges_to_healpix: Adjacency describing the lat-lon → HEALPix map
            edges_to_latlon: Adjacency describing the HEALPix → lat-lon map
            key: PRNG key for initializing all submodules
        """
        in_channels = input_size[0]
        npix = 12 * nside**2
        self.embedding = LogFourierEmbedding(temb_dim)
        self.doy_embedding = DoYFourierEmbedding(doyemb_dim)

        key, χ = jr.split(key)
        self.pos_embedding = jr.normal(χ, (posemb_dim, npix)) / jnp.sqrt(posemb_dim)

        key, χ = jr.split(key)
        self.conv_embedding = HealPIXConvBlock(in_channels=healpix_emb_dim + doyemb_dim,
                                               out_channels=posemb_dim,
                                               kernel_size=3,
                                               key=χ)

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
        self.encoder = Encoder(input_size=(posemb_dim, npix),
                               n_filters=enc_filters,
                               temb_dim=temb_dim,
                               key=χ)

        key, χ = jr.split(key)
        bottleneck_size = npix // (4 ** len(enc_filters))
        self.decoder = Decoder(input_size=(enc_filters[-1], bottleneck_size),
                               n_filters=dec_filters,
                               skip_filters=enc_filters[::-1],
                               n_blocks=dec_blocks,
                               n_bottleneck_blocks=n_bottleneck_blocks,
                               temb_dim=temb_dim,
                               key=χ)

        key, χ = jr.split(key)
        self.output_layer = eqx.nn.Conv1d(in_channels=dec_filters[-1],
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          key=χ)

    def __call__(self, x: jax.Array, t: jax.Array, doy: jax.Array) -> jax.Array:
        """Forward pass through the DoY-aware U-Net.

        Args:
            x: Input tensor (channels, nlat, nlon)
            doy: Day-of-year values in [0, 365) for seasonal conditioning
            t: Diffusion time

        Returns:
            Output tensor of shape (out_channels, nlat, nlon)
        """
        # Map to healpix
        c, nlat, nlon = x.shape
        x = self.to_healpix(x.reshape(c, -1))

        # DoY embedding
        doy_emb = self.doy_embedding(doy)
        doy_emb = jnp.broadcast_to(doy_emb[:, None], (doy_emb.shape[0], x.shape[1]))

        # Fuse with input
        x = jnp.concatenate([x, doy_emb], axis=0)
        x = self.conv_embedding(x)

        # Add positional embedding
        x = x + self.pos_embedding

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
