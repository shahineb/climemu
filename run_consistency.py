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
denoiser = eqx.tree_deserialise_leaves(f"{CACHE_DIR}/weights_consistency_2.eqx", denoiser)

# Load sigma max (LOAD THIS FILE)
σmax = jnp.load(f"{CACHE_DIR}/σmax.npy")
schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)

# Stats (LOAD THIS FILE)
_stats = jnp.load(f"{CACHE_DIR}/μ_σ.npz")
μ_train = _stats["μ"]
σ_train = _stats["σ"]

# Pattern scaling (LOAD THIS FILE)
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
# pred_samples = generate_samples(pattern_batch=pattern_batch, key=χtest) # (3 months, n_samples, 4 vars, 96 lat, 192 lon)
# print(f"{pred_samples.shape=}")


import jax
from typing import Tuple, Any, List

key = jr.PRNGKey(10)
n_steps = 2


pattern = pattern_batch[5]
context = utils.normalize(pattern, μ_train[-1], σ_train[-1])[None, ...]
rho = 7
t = jnp.linspace(0, 1, n_steps + 1)
sigma_steps = (schedule.σmax**(1/rho) + t * (schedule.σmin**(1/rho) - schedule.σmax**(1/rho)))**rho
# sigma_steps = sigma_steps[:-1]
# sigma_steps = schedule.σ(schedule.get_timesteps(n_steps + 1))[1:]
# sigma_steps = sigma_steps[::-1]


init_key, *step_keys = jr.split(key, n_steps + 1)
x = jr.normal(init_key, output_size) * schedule.σmax

for i in range(n_steps):
    σi = sigma_steps[i]
    x = denoiser(jnp.concatenate([x / (1 + σi), context], axis=0), σi)
    if i < n_steps - 1:
        σip1 = sigma_steps[i + 1]
        x += jr.normal(step_keys[i], x.shape) * σip1
# x = denoiser(jnp.concatenate([x / (1 + schedule.σmin), context], axis=0), schedule.σmin)

x = utils.denormalize(x, μ_train[:-1], σ_train[:-1])


fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    vmax = jnp.abs(x[i]).max()
    im = ax.imshow(x[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("outputs/consistency_sample.jpg", dpi=300)
plt.close()


# n_plot = 50
# sigma_ve = schedule.σ(schedule.get_timesteps(n_plot))[::-1]
# rho_k = 7
# sigma_karras = (schedule.σmax ** (1 / rho_k) + jnp.linspace(0, 1, n_plot) * (schedule.σmin ** (1 / rho_k) - schedule.σmax ** (1 / rho_k))) ** rho_k
# t_vals = jnp.linspace(0, 1, n_plot)
# n_steps_dot = 3
# t_ve_dots = jnp.linspace(0, 1, n_steps_dot)
# sigma_ve_dots = schedule.σ(schedule.get_timesteps(n_steps_dot))[::-1]
# sigma_karras_dots = (schedule.σmax ** (1 / rho_k) + jnp.linspace(0, 1, n_steps_dot) * (schedule.σmin ** (1 / rho_k) - schedule.σmax ** (1 / rho_k))) ** rho_k
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.plot(t_vals, sigma_ve, label="Ours (Variance Exploding)")
# ax.plot(t_vals, sigma_karras, label="Karras et al. (ρ=7)")
# ax.scatter(t_ve_dots, sigma_ve_dots, color="C0", zorder=5, s=50)
# ax.scatter(t_ve_dots, sigma_karras_dots, color="C1", zorder=5, s=50)
# ax.set_xlabel("t")
# ax.set_ylabel("σ")
# ax.set_yscale('log')
# ax.legend()
# ax.set_title("VE vs Karras noise schedules")
# plt.tight_layout()
# plt.savefig("outputs/schedule_comparison.jpg", dpi=300)
# plt.close()



@eqx.filter_jit
def draw_samples_single_consistency(denoiser: eqx.Module, schedule: Any, pattern: jnp.ndarray,
                        n_samples: int, n_steps: int, μ: jnp.ndarray, σ: jnp.ndarray,
                        output_size: Tuple, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a given pattern using consistency model."""
    context = utils.normalize(pattern, μ[-1], σ[-1])[None, ...]
    sigma_steps = schedule.σ(schedule.get_timesteps(n_steps))

    def _sample_one(key):
        init_key, *step_keys = jr.split(key, 1 + n_steps)
        x = jr.normal(init_key, output_size) * sigma_steps[-1]
        for i in range(n_steps-1, -1, -1):
            σ_i = sigma_steps[i]
            x = denoiser(jnp.concatenate([x / (1+σ_i), context], axis=0), σ_i)
            if i > 0:
                x += jr.normal(step_keys[i-1], x.shape) * sigma_steps[i-1]
        return x

    keys = jr.split(key, n_samples)
    samples = jax.vmap(_sample_one)(keys)
    return utils.denormalize(samples, μ[:-1], σ[:-1])


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

