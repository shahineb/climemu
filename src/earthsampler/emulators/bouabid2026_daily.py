import os
import yaml
from functools import partial
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from huggingface_hub import hf_hub_download
from diffusion import HealPIXUNetv2, ContinuousVESchedule, ContinuousHeunSampler
from .abstractemulator import GriddedEmulator
from ..utils.datetime import parse_doy
from .. import EMULATORS


class Bouabid2026DailyEmulator(GriddedEmulator):
    """Daily climate emulator using score-based diffusion.

    Generates daily anomaly fields conditioned on GMST anomaly and
    day-of-year. Uses annual (not monthly) pattern scaling and
    HealPIXUNetv2 with day-of-year conditioning.

    Lifecycle: ``__init__`` → ``load()`` → ``compile(n_samples)`` → ``__call__(gmst, doy)``.
    """

    def __init__(self, esm_name: str, variables=None):
        """Args:
            esm_name: Earth System Model identifier (e.g. ``"MPI-ESM1-2-LR"``).
            variables: Optional subset of output variables (e.g. ``["tas", "pr"]``).
                When *None*, all available variables are returned.
        """
        self.esm = esm_name
        self.repo_id = "shahineb/climemu"
        self._vars = variables

    def load(self, which: str = "default"):
        """Download pretrained weights and data from HuggingFace Hub.

        Args:
            which: Weight variant — ``"default"`` or ``"paper"``.
        """
        self.files_dir = os.path.join(self.esm, "daily", which)

        # Load climatology data
        self.climatology = self._load_climatology()

        # Resolve variable selection
        self._resolve_variables()

        # Load pattern scaling coefficients (annual)
        self.β = self._load_pattern_scaling()

        # Load the generative model precursor
        self.precursor = self._load_precursor()

    def _resolve_variables(self):
        all_vars = list(self.climatology.data_vars)
        if self._vars is None:
            self._var_idx = list(range(len(all_vars)))
        else:
            invalid = set(self._vars) - set(all_vars)
            if invalid:
                raise ValueError(f"Unknown variables: {invalid}. Available: {all_vars}")
            self._var_idx = [all_vars.index(v) for v in self._vars]
            self.climatology = self.climatology[self._vars]

    def compile(self, n_samples, n_steps=30, batch_size=1):
        """JIT-compile the generative model for a fixed sample count.

        Args:
            n_samples: Number of ensemble members per call.
            n_steps: Diffusion sampler steps (higher = better quality, slower).
                Minimum ~30 recommended; ``n_steps=2`` produces NaN.
            batch_size: Number of days to generate in parallel. When >1,
                pass a list of doys to ``__call__``.
        """
        self.batch_size = batch_size
        self.generative_model = partial(self.precursor,
                                        n_samples=n_samples,
                                        n_steps=n_steps)

        # Dry run to compile JAX functions
        dummy_pattern = jnp.zeros((batch_size, self.nlat, self.nlon))
        dummy_doy = jnp.ones((batch_size,))
        _ = self.generative_model(pattern_batch=dummy_pattern, doy_batch=dummy_doy, key=jr.PRNGKey(0))

    def __call__(self, gmst, doy, seed=None, xarray=False):
        """Generate daily climate anomaly samples.

        Args:
            gmst: GMST anomaly relative to piControl (°C). Scalar, or list
                matching ``batch_size``.
            doy: Day of year — integer (1–365), or string ``"dd/mm"`` or
                ``"dd-mm"``. Scalar, or list matching ``batch_size``.
            seed: Random seed. If *None*, a random seed is drawn.
            xarray: If *True*, return an ``xr.Dataset`` instead of a JAX array.

        Returns:
            JAX array of shape ``(n_samples, n_vars, nlat, nlon)`` for scalar
            doy, or ``(batch, n_samples, n_vars, nlat, nlon)`` for batched.
            When ``xarray=True``, an ``xr.Dataset`` with coordinates
            ``member``, ``lat``, ``lon`` (and ``batch`` if batched).
        """
        key = jr.PRNGKey(seed) if seed else jr.PRNGKey(np.random.randint(0, 1000000))
        is_batch = isinstance(doy, list)

        # Parse and normalize doy inputs
        doys = np.array([parse_doy(d) for d in doy] if is_batch else [parse_doy(doy)], dtype=float)
        gmsts = np.broadcast_to(np.atleast_1d(np.asarray(gmst, dtype=float)), doys.shape)

        if len(doys) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} doys (batch_size), got {len(doys)}")

        # Apply annual pattern scaling: pattern = β₁ * ΔT + β₀
        patterns = self.β[:, 1] * gmsts[:, None] + self.β[:, 0]
        patterns = patterns.reshape((-1, self.nlat, self.nlon))

        # Generate samples using the diffusion model
        samples = self.generative_model(pattern_batch=jnp.array(patterns),
                                         doy_batch=jnp.array(doys), key=key)

        # Subset to requested variables
        samples = samples[:, :, self._var_idx]  # (B, n_samples, n_vars, nlat, nlon)

        # Squeeze batch dim for single-doy calls (backward compat)
        if not is_batch:
            samples = samples[0]  # (n_samples, n_vars, nlat, nlon)

        # Convert to xarray Dataset
        if xarray:
            if is_batch:
                samples = xr.Dataset(
                    {
                        var: (("batch", "member", "lat", "lon"), samples[:, :, i])
                        for i, var in enumerate(self.vars)
                    },
                    coords={
                        "doy": ("batch", doys.astype(int)),
                        "gmst_anomaly": ("batch", gmsts),
                        "member": jnp.arange(samples.shape[1]) + 1,
                        "lat": self.lat,
                        "lon": self.lon,
                    },
                )
            else:
                samples = xr.Dataset(
                    {
                        var: (("member", "lat", "lon"), samples[:, i, :, :])
                        for i, var in enumerate(self.vars)
                    },
                    coords={
                        "member": jnp.arange(len(samples)) + 1,
                        "lat": self.lat,
                        "lon": self.lon,
                    },
                )
        return samples

    def _load_precursor(self):
        config = self._load_config()
        nn = self._load_nn(config)
        schedule = self._load_schedule(config)
        μ, σ = self._load_normalization()
        output_size = (config['out_channels'], config['input_size'][1], config['input_size'][2])
        precursor = partial(draw_samples_daily_batch,
                            nn=nn,
                            schedule=schedule,
                            output_size=output_size,
                            μ=μ, σ=σ)
        return precursor

    def _load_config(self):
        config_path = hf_hub_download(self.repo_id, f"{self.files_dir}/config.yaml")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _load_normalization(self):
        norm_stats_path = hf_hub_download(self.repo_id, f"{self.files_dir}/μ_σ.npz")
        stats = jnp.load(norm_stats_path)
        μ, σ = stats['μ'], stats['σ']
        return μ, σ

    def _load_pattern_scaling(self):
        pattern_scaling_path = hf_hub_download(self.repo_id, f"{self.files_dir}/β.npy")
        β = jnp.load(pattern_scaling_path)
        return β

    def _load_climatology(self):
        climatology_path = hf_hub_download(self.repo_id, f"{self.esm}/daily/piControl_climatology.nc")
        climatology = xr.open_dataset(climatology_path)
        return climatology

    def _load_nn(self, config):
        edges_path = hf_hub_download(self.repo_id, f"{self.files_dir}/edges.npz")
        edges_data = jnp.load(edges_path)
        to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
        to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)

        nn = HealPIXUNetv2(input_size=config['input_size'],
                           nside=config['nside'],
                           enc_filters=config['enc_filters'],
                           dec_filters=config['dec_filters'],
                           dec_blocks=config['dec_blocks'],
                           n_bottleneck_blocks=config['n_bottleneck_blocks'],
                           out_channels=config['out_channels'],
                           temb_dim=config['temb_dim'],
                           doyemb_dim=config['doyemb_dim'],
                           healpix_emb_dim=config['healpix_emb_dim'],
                           posemb_dim=config['posemb_dim'],
                           edges_to_healpix=to_healpix,
                           edges_to_latlon=to_latlon)

        weights_path = hf_hub_download(self.repo_id, f"{self.files_dir}/weights.eqx")
        nn = eqx.tree_deserialise_leaves(weights_path, nn)
        return nn

    def _load_schedule(self, config):
        sigma_max_path = hf_hub_download(self.repo_id, f"{self.files_dir}/σmax.npy")
        σmax = jnp.load(sigma_max_path)
        schedule = ContinuousVESchedule(config['sigma_min'], σmax)
        return schedule

    @property
    def lat(self):
        return self.climatology['lat'].values

    @property
    def lon(self):
        return self.climatology['lon'].values

    @property
    def vars(self):
        if self._vars is None:
            return list(self.climatology.data_vars)
        return self._vars


@eqx.filter_jit
def normalize(x, μ, σ):
    return (x - μ) / σ

@eqx.filter_jit
def denormalize(x, μ, σ):
    return σ * x + μ


def create_sampler_daily(nn, schedule, pattern, doy, μ, σ, output_size):
    context = normalize(pattern, μ[-1], σ[-1])[None, ...]
    def nn_with_context(x, t):
        x = jnp.concatenate((x, context), axis=0)
        return nn(x, t, doy)
    return ContinuousHeunSampler(schedule, nn_with_context, output_size)

@eqx.filter_jit
def draw_samples_daily(nn, schedule, pattern, doy, n_samples, n_steps, μ, σ, output_size, key=jr.PRNGKey(0)):
    sampler = create_sampler_daily(nn, schedule, pattern, doy, μ, σ, output_size)
    samples = sampler.sample(n_samples, steps=n_steps, key=key)
    return denormalize(samples, μ[:-1], σ[:-1])


@eqx.filter_jit
def draw_samples_daily_batch(nn, schedule, pattern_batch, doy_batch, n_samples, n_steps, μ, σ, output_size, key=jr.PRNGKey(0)):
    """Draw samples for a batch of (pattern, doy) pairs."""
    keys = jr.split(key, pattern_batch.shape[0])
    Γ = partial(draw_samples_daily,
                nn=nn, schedule=schedule,
                n_samples=n_samples, n_steps=n_steps,
                μ=μ, σ=σ, output_size=output_size)
    return jax.vmap(Γ)(pattern=pattern_batch, doy=doy_batch, key=keys)


@EMULATORS.register(("MPI-ESM1-2-LR", "daily"))
class MPIDailyEmulator(Bouabid2026DailyEmulator):
    """Daily emulator for MPI-ESM1-2-LR."""
    def __init__(self, **kwargs):
        super().__init__(esm_name="MPI-ESM1-2-LR", **kwargs)
