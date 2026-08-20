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
from diffusion import HealPIXUNetv1, ContinuousVESchedule, ContinuousHeunSampler
from .abstractemulator import GriddedEmulator
from .. import EMULATORS


class Bouabid2026MonthlyEmulator(GriddedEmulator):
    def __init__(self, esm_name: str, variables=None):
        self.esm = esm_name
        self.repo_id = "shahineb/climemu"
        self._vars = variables

    def load(self, which: str = "default"):
        # Set files directory in hugging face repo
        self.files_dir = os.path.join(self.esm, "monthly", which)

        # Load climatology data
        self.climatology = self._load_climatology()

        # Resolve variable selection
        self._resolve_variables()

        # Load pattern scaling coefficients
        self.β = self._load_pattern_scaling()

        # Load the generative model precursor
        self.precursor = self._load_precursor()

    def _resolve_variables(self):
        # List all available variables
        all_vars = list(self.climatology.data_vars)

        # If no specific variables are requested, use all available variables
        if self._vars is None:
            self._var_idx = list(range(len(all_vars)))
        # Else, validate the requested variables, get their indices and subset climatology
        else:
            invalid = set(self._vars) - set(all_vars)
            if invalid:
                raise ValueError(f"Unknown variables: {invalid}. Available: {all_vars}")
            self._var_idx = [all_vars.index(v) for v in self._vars]
            self.climatology = self.climatology[self._vars]

    def compile(self, n_samples, n_steps=30, batch_size=1):
        # Fix number of samples, steps, and batch size for generation
        self.batch_size = batch_size
        self.generative_model = partial(self.precursor,
                                        n_samples=n_samples,
                                        n_steps=n_steps)

        # Perform a dry run to compile the JAX functions (important for performance)
        dummy_pattern = jnp.zeros((batch_size, self.nlat, self.nlon))
        _ = self.generative_model(pattern_batch=dummy_pattern, key=jr.PRNGKey(0))

    def __call__(self, gmst, month, seed=None, xarray=False):
        key = jr.PRNGKey(seed) if seed else jr.PRNGKey(np.random.randint(0, 1000000))
        is_batch = not np.isscalar(month)

        # Normalize inputs to arrays
        months = np.atleast_1d(np.asarray(month))
        gmsts = np.broadcast_to(np.atleast_1d(np.asarray(gmst)), months.shape)

        if len(months) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} months (batch_size), got {len(months)}")

        # Apply pattern scaling: pattern = β₁ * ΔT + β₀
        patterns = self.β[months - 1, :, 1] * gmsts[:, None] + self.β[months - 1, :, 0]
        patterns = patterns.reshape((-1, self.nlat, self.nlon))

        # Generate samples using the diffusion model
        samples = self.generative_model(pattern_batch=jnp.array(patterns), key=key)

        # Subset to requested variables
        samples = samples[:, :, self._var_idx]  # (B, n_samples, n_vars, nlat, nlon)

        # Squeeze batch dim for single-month calls (backward compat)
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
                        "month": ("batch", months),
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
        # Build the neural network and noise schedule
        config = self._load_config()
        nn = self._load_nn(config)
        schedule = self._load_schedule(config)

        # Load normalization statistics used during training (needed to denormalize the generated samples)
        μ, σ = self._load_normalization()

        # Define the output size for the generated samples
        output_size = (config['out_channels'], config['input_size'][1], config['input_size'][2])

        # Create a precursor for the generative model
        precursor = partial(draw_samples_batch,
                            nn=nn,
                            schedule=schedule,
                            output_size=output_size,
                            μ=μ, σ=σ)
        return precursor

    def _load_config(self):
        # Load configuration from YAML file
        config_path = hf_hub_download(self.repo_id, f"{self.files_dir}/config.yaml")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _load_normalization(self):
        # Load normalization statistics used during training
        norm_stats_path = hf_hub_download(self.repo_id, f"{self.files_dir}/μ_σ.npz")
        stats = jnp.load(norm_stats_path)
        μ, σ = stats['μ'], stats['σ']
        return μ, σ

    def _load_pattern_scaling(self):
        # Load pattern scaling coefficients
        pattern_scaling_path = hf_hub_download(self.repo_id, f"{self.files_dir}/β.npy")
        β = jnp.load(pattern_scaling_path)
        return β

    def _load_climatology(self):
        # Load climatology data
        climatology_path = hf_hub_download(self.repo_id, f"{self.esm}/monthly/piControl_climatology.nc")
        climatology = xr.open_dataset(climatology_path)
        return climatology

    def _load_nn(self, config):
        # Load graph edges for HEALPix to lat-lon connectivity
        edges_path = hf_hub_download(self.repo_id, f"{self.files_dir}/edges.npz")
        edges_data = jnp.load(edges_path)
        to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
        to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)

        # Initialize the neural network
        nn = HealPIXUNetv1(input_size=config['input_size'],
                           nside=config['nside'],
                           enc_filters=config['enc_filters'],
                           dec_filters=config['dec_filters'],
                           out_channels=config['out_channels'],
                           temb_dim=config['temb_dim'],
                           healpix_emb_dim=config['healpix_emb_dim'],
                           edges_to_healpix=to_healpix,
                           edges_to_latlon=to_latlon)

        # Load the pre-trained weights from the saved model file
        weights_path = hf_hub_download(self.repo_id, f"{self.files_dir}/weights.eqx")
        nn = eqx.tree_deserialise_leaves(weights_path, nn)
        return nn

    def _load_schedule(self, config):
        # Load the maximum noise level as used in training
        sigma_max_path = hf_hub_download(self.repo_id, f"{self.files_dir}/σmax.npy")
        σmax = jnp.load(sigma_max_path)

        # Create the variance exploding schedule
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
    """Normalize data using mean and standard deviation."""
    return (x - μ) / σ

@eqx.filter_jit
def denormalize(x, μ, σ):
    """Denormalize data using mean and standard deviation."""
    return σ * x + μ


def create_sampler(nn, schedule, pattern, μ, σ, output_size):
    """Create a sampler for a given pattern."""
    context = normalize(pattern, μ[-1], σ[-1])[None, ...]
    def nn_with_context(x, t):
        x = jnp.concatenate((x, context), axis=0)
        return nn(x, t)
    return ContinuousHeunSampler(schedule, nn_with_context, output_size)

@eqx.filter_jit
def draw_samples_single(nn, schedule, pattern, n_samples, n_steps, μ, σ, output_size, key=jr.PRNGKey(0)):
    """Draw samples for a given pattern."""
    sampler = create_sampler(nn, schedule, pattern, μ, σ, output_size)
    samples = sampler.sample(n_samples, steps=n_steps, key=key)
    return denormalize(samples, μ[:-1], σ[:-1])

@eqx.filter_jit
def draw_samples_batch(nn, schedule, pattern_batch, n_samples, n_steps, μ, σ, output_size, key=jr.PRNGKey(0)):
    """Draw samples for a batch of patterns."""
    keys = jr.split(key, pattern_batch.shape[0])
    Γ = partial(draw_samples_single,
                nn=nn, schedule=schedule,
                n_samples=n_samples, n_steps=n_steps,
                μ=μ, σ=σ, output_size=output_size)
    return jax.vmap(Γ)(pattern=pattern_batch, key=keys)


@EMULATORS.register(("MPI-ESM1-2-LR", "monthly"))
class MPIMonthlyEmulator(Bouabid2026MonthlyEmulator):
    def __init__(self, **kwargs):
        super().__init__(esm_name="MPI-ESM1-2-LR", **kwargs)


@EMULATORS.register(("MIROC6", "monthly"))
class MIROCMonthlyEmulator(Bouabid2026MonthlyEmulator):
    def __init__(self, **kwargs):
        super().__init__(esm_name="MIROC6", **kwargs)


@EMULATORS.register(("ACCESS-ESM1-5", "monthly"))
class ACCESSMonthlyEmulator(Bouabid2026MonthlyEmulator):
    def __init__(self, **kwargs):
        super().__init__(esm_name="ACCESS-ESM1-5", **kwargs)


@EMULATORS.register(("CanESM5", "monthly"))
class CanESMMonthlyEmulator(Bouabid2026MonthlyEmulator):
    def __init__(self, **kwargs):
        super().__init__(esm_name="CanESM5", **kwargs)


@EMULATORS.register(("IPSL-CM6A-LR", "monthly"))
class IPSLMonthlyEmulator(Bouabid2026MonthlyEmulator):
    def __init__(self, **kwargs):
        super().__init__(esm_name="IPSL-CM6A-LR", **kwargs)
