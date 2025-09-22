import os
import yaml
from functools import partial
import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from huggingface_hub import hf_hub_download
from diffusion import HealPIXUNet, ContinuousVESchedule, ContinuousHeunSampler
from .abstractemulator import GriddedEmulator
from .. import EMULATORS


class Bouabid2025Emulator(GriddedEmulator):
    def __init__(self, esm_name: str):
        self.esm = esm_name
        self.repo_id = "shahineb/climemu"

    def load(self, which: str = "default"):
        # Set files directory in hugging face repo
        self.files_dir = os.path.join(self.esm, which)

        # Load pattern scaling coefficients
        self.β = self._load_pattern_scaling()

        # Load climatology data
        self.climatology = self._load_climatology()
    
        # Load the generative model precursor
        self.precursor = self._load_precursor()

    def compile(self, n_samples, n_steps=30):
        # Fix number of samples and steps for generation
        self.generative_model = partial(self.precursor,
                                        n_samples=n_samples,
                                        n_steps=n_steps)

        # Perform a dry run to compile the JAX functions (important for performance)
        dummy_pattern = jnp.zeros((self.nlat, self.nlon))
        _ = self.generative_model(pattern=dummy_pattern, key=jr.PRNGKey(0))

    def __call__(self, gmst, month, seed=None, xarray=False):
        # Apply pattern scaling: pattern = β₀ + β₁ * ΔT
        pattern = self.β[month - 1, :, 1] * gmst + self.β[month - 1, :, 0]
        pattern = pattern.reshape((self.nlat, self.nlon))
        
        # Generate samples using the diffusion model
        key = jr.PRNGKey(seed) if seed else jr.PRNGKey(np.random.randint(0, 1000000))
        samples = self.generative_model(pattern=pattern, key=key)

        # Convert to xarray Dataset
        if xarray:
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

        # Create an precursor for the generative model
        precursor = partial(draw_samples_single,
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
        climatology_path = hf_hub_download(self.repo_id, f"{self.esm}/piControl_climatology.nc")
        climatology = xr.open_dataset(climatology_path)
        return climatology

    def _load_nn(self, config):
        # Load graph edges for HEALPix to lat-lon connectivity
        edges_path = hf_hub_download(self.repo_id, f"{self.files_dir}/edges.npz")
        edges_data = jnp.load(edges_path)
        to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
        to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)

        # Initialize the neural network
        nn = HealPIXUNet(input_size=config['input_size'],
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
        return list(self.climatology.data_vars)


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


@EMULATORS.register("MPI-ESM1-2-LR")
class MPIEmulator(Bouabid2025Emulator):
    def __init__(self, which="default"):
        super().__init__(esm_name="MPI-ESM1-2-LR")


@EMULATORS.register("MIROC6")
class MIROCEmulator(Bouabid2025Emulator):
    def __init__(self, which="default"):
        super().__init__(esm_name="MIROC6")


@EMULATORS.register("ACCESS-ESM1-5")
class ACCESSEmulator(Bouabid2025Emulator):
    def __init__(self, which="default"):
        super().__init__(esm_name="ACCESS-ESM1-5")