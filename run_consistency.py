import os
from src.diffusion import HealPIXUNet, ContinuousVESchedule
from paper.mpi.config import Config
CACHE_DIR = "paper/mpi/cache"
from paper.mpi import utils
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import numpy as np
import einops

config = Config()

# Load or compute Latlon-HEALPix edges (LOAD THIS FILE)
edges_to_healpix, edges_to_latlon = utils.load_or_compute_edges(
    nside=config.model.nside,
    lat=np.linspace(-90, 90, 96),
    lon=np.linspace(0, 360, 192),
    edges_path=config.model.edges_path
)
_edges = jnp.load(f"{CACHE_DIR}/edges.npz")
edges_to_healpix = _edges["to_healpix"]
edges_to_latlon  = _edges["to_latlon"]

# Initialize and load model (LOAD THIS FILE)
model = HealPIXUNet(
    input_size=config.model.input_size,
    nside=config.model.nside,
    enc_filters=list(config.model.enc_filters),
    dec_filters=list(config.model.dec_filters),
    out_channels=config.model.out_channels,
    temb_dim=config.model.temb_dim,
    healpix_emb_dim=config.model.healpix_emb_dim,
    edges_to_healpix=edges_to_healpix,
    edges_to_latlon=edges_to_latlon
)
from paper.mpi.main import Denoiser
denoiser = Denoiser(model, config.model.context_channels)
denoiser = eqx.tree_deserialise_leaves(f"{CACHE_DIR}/weights_consistency.eqx", denoiser)

# Load sigma max (LOAD THIS FILE)
σmax = 175
σmax = jnp.load(f"{CACHE_DIR}/σmax.npy")
schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)

# Stats (LOAD THIS FILE)
μ_train, σ_train = np.zeros((5, 96, 192)), np.ones((5, 96, 192))
_stats = jnp.load(f"{CACHE_DIR}/μ_σ.npz")
μ_train = _stats["μ"]
σ_train = _stats["σ"]

# Pattern scaling (LOAD THIS FILE)
β = jnp.ones((12, 96, 192, 2))
β = jnp.load(f"{CACHE_DIR}/β.npy")


# Initialize sampling function
χtest = jr.PRNGKey(config.sampling.random_seed)
output_size = (config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
generate_samples = partial(utils.draw_samples_batch_consistency,
                            denoiser=denoiser,
                            schedule=schedule,
                            n_samples=5, # config.sampling.n_samples,
                            n_steps=3,
                            μ=μ_train, σ=σ_train,
                            output_size=output_size)




# Generate samples
ΔT = jnp.array([2.0]*12)
months = jnp.array(range(12))
β = einops.rearrange(β, 'm (l1 l2) i -> m l1 l2 i', l1=96)
assert β.shape == (12, 96, 192, 2)
pattern_batch = β[months, :, :, 0] + β[months, :, :, 1] * ΔT.reshape(-1, 1, 1)
pred_samples = generate_samples(pattern_batch=pattern_batch, key=χtest) # (3 months, n_samples, 4 vars, 96 lat, 192 lon)
print(f"{pred_samples.shape=}")

# Plot pred_samples[0]: mean across samples for each of the 4 variables
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

var_names = ["tas", "pr", "hurs", "sfcWind"] # Deltas of surface temp, precipitation, relative humidity, surface wind speed
lat = np.linspace(-90, 90, 96)
lon = np.linspace(0, 360, 192)

for month in range(12):
    sample0 = np.array(pred_samples[month])  # (n_samples, 4 variables, 96, 192)
    mean0 = sample0.mean(axis=0)         # (4 variables, 96, 192)

    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    tas_vmax = np.abs(mean0[0]).max()
    for i, (ax, name) in enumerate(zip(axes.flat, var_names)):
        vmax = np.abs(mean0[i]).max()
        im = ax.pcolormesh(lon, lat, mean0[i], cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax)

    # Add pattern batch as the last subplot
    ax = axes.flat[-1]
    name = "pattern batch"
    im = ax.pcolormesh(lon, lat, pattern_batch[month], cmap="RdBu_r", shading="auto", vmin=-tas_vmax, vmax=tas_vmax)
    ax.set_title(name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax)

    fig.suptitle(f"mean pred_samples[month {month}] (consistency - 1/t weighting)", fontsize=14)
    plt.tight_layout()
    os.makedirs("outputs/consistency - 1/t weighting", exist_ok=True)
    plt.savefig(f"outputs/consistency - 1/t weighting/pred_samples_{month}.png", dpi=150)
    print(f"Saved outputs/consistency - 1/t weighting/pred_samples_{month}.png")

