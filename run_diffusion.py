import os
import time
from src.diffusion import HealPIXUNet, ContinuousVESchedule
from paper.mpi.config import Config
from paper.mpi.data import load_dataset
from src.utils.collate import numpy_collate
CACHE_DIR = "paper/mpi/cache"
from paper.mpi import utils
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
import numpy as np
import einops
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

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
model = eqx.tree_deserialise_leaves(f"{CACHE_DIR}/weights.eqx", model)

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
β = jnp.load(f"{CACHE_DIR}/β.npy")
β_flat = np.array(β)  # (12, lat*lon, 2) — keep original shape for load_dataset

# Generate samples
χtest = jr.PRNGKey(config.sampling.random_seed)
output_size = (config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
ΔT = jnp.array([2.0])
months = jnp.array([5]) #range(12)
β = einops.rearrange(β, 'm (l1 l2) i -> m l1 l2 i', l1=96)
assert β.shape == (12, 96, 192, 2)
pattern_batch = β[months, :, :, 0] + β[months, :, :, 1] * ΔT.reshape(-1, 1, 1)

generate_samples = partial(utils.draw_samples_batch,
                            model=model,
                            schedule=schedule,
                            pattern_batch=pattern_batch,
                            n_samples=20,
                            n_steps=config.sampling.n_steps,
                            μ=μ_train, σ=σ_train,
                            output_size=output_size,
                            key=χtest)

# Generate samples with timing
t0 = time.perf_counter()
pred_samples = generate_samples()
jax.block_until_ready(pred_samples)
t1 = time.perf_counter()
n_total = pred_samples.shape[0] * pred_samples.shape[1]
print(f"Inference: {t1-t0:.2f}s total | {(t1-t0)/n_total:.3f}s per sample ({pred_samples.shape[0]} months x {pred_samples.shape[1]} draws)")

# FID (data-space Fréchet distance, PCA-reduced)
def _compute_fd(gen: np.ndarray, ref: np.ndarray, n_components: int = 50) -> float:
    n = min(gen.shape[0] - 1, ref.shape[0] - 1, n_components)
    if n < 2:
        return float('nan')
    pca = PCA(n_components=n)
    pca.fit(np.concatenate([gen, ref], axis=0))
    g, r = pca.transform(gen), pca.transform(ref)
    mu_g, mu_r = g.mean(0), r.mean(0)
    cov_g, cov_r = np.cov(g, rowvar=False), np.cov(r, rowvar=False)
    covmean = linalg.sqrtm(cov_g @ cov_r)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_g - mu_r
    return float(np.dot(diff, diff) + np.trace(cov_g + cov_r - 2 * covmean))

try:
    from collections import defaultdict
    print("Loading 1pctCO2 validation data for FID reference...")
    val_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=list(config.data.val_experiments),
        variables=config.data.variables,
        in_memory=False,
        external_β=β_flat
    )

    # Build per-month reference samples matching ΔT ≈ target
    ΔT_target = float(np.array(ΔT)[0])
    ΔT_tol = 0.5
    month_refs = defaultdict(list)
    n_search = min(len(val_dataset), 2000)
    rng = np.random.default_rng(0)
    print(f"Searching {n_search} val samples for ΔT≈{ΔT_target}±{ΔT_tol}...")
    for idx in rng.choice(len(val_dataset), n_search, replace=False):
        e_i, ω_i, t_i = val_dataset.cmip6data.indexmap[idx]
        exp = val_dataset.cmip6data.experiments[e_i]
        selected = val_dataset.cmip6data[exp].isel(time=t_i, member=ω_i)
        month = int(selected.time.dt.month.item()) - 1  # 0-indexed
        gmst = float(val_dataset.gmst[exp].isel(time=t_i).ds.tas.values.squeeze())
        if abs(gmst - ΔT_target) <= ΔT_tol:
            month_refs[month].append(selected.ds.to_array().values)  # (n_vars, lat, lon)

    _var_names = ["tas", "pr", "hurs", "sfcWind"]
    target_months_list = [int(m) for m in months]
    for m_idx, month in enumerate(target_months_list):
        refs = month_refs.get(month, [])
        gen_arr = np.array(pred_samples[m_idx])  # (n_samples, n_vars, lat, lon)
        if len(refs) < 5:
            print(f"Month {month+1}: only {len(refs)} refs at ΔT≈{ΔT_target}±{ΔT_tol} — FID skipped")
            continue
        ref_arr = np.stack(refs)
        if gen_arr.shape[0] < 5:
            print(f"Warning: only {gen_arr.shape[0]} generated samples for month {month+1} — FID estimate may be unreliable")
        gen_f = gen_arr.reshape(gen_arr.shape[0], -1)
        ref_f = ref_arr.reshape(ref_arr.shape[0], -1)
        print(f"Month {month+1} FID (data-space, {len(refs)} refs): {_compute_fd(gen_f, ref_f):.4f}")
        gen_bv = gen_arr.reshape(gen_arr.shape[0], len(_var_names), -1)
        ref_bv = ref_arr.reshape(ref_arr.shape[0], len(_var_names), -1)
        for i, vname in enumerate(_var_names):
            print(f"  FID [{vname}]: {_compute_fd(gen_bv[:, i, :], ref_bv[:, i, :]):.4f}")
except Exception as e:
    print(f"FID skipped: {e}")

matplotlib.use("Agg")

# Plot pred_samples: mean across samples for each of the 4 variables

var_names = ["tas", "pr", "hurs", "sfcWind"] # Deltas of surface temp, precipitation, relative humidity, surface wind speed
lat = np.linspace(-90, 90, 96)
lon = np.linspace(0, 360, 192)

for month in [5]:
    sample0 = np.array(pred_samples[month])  # (n_samples, 4 variables, 96, 192)
    mean0 = sample0.mean(axis=0)         # (4 variables, 96, 192)

    tas_vmax = np.abs(mean0[0]).max()
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
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

    fig.suptitle(f"pred_samples[month {month}] — mean across samples", fontsize=14)
    plt.tight_layout()
    os.makedirs("outputs/diffusion", exist_ok=True)
    plt.savefig(f"outputs/diffusion/pred_samples_{month}.png", dpi=150)
    print(f"Saved outputs/diffusion/pred_samples_{month}.png")

