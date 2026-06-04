import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils.collate import numpy_collate
from experiments.canesm.config import Config
from experiments.canesm.data import load_dataset, compute_normalization
from experiments.canesm import utils


# Load config and dataset
config = Config()
dataset = load_dataset(
    root=config.data.root_dir,
    model=config.data.model_name,
    experiments=config.data.train_experiments,
    variables=config.data.variables,
    in_memory=config.data.in_memory,
    pattern_scaling_path=config.data.pattern_scaling_path
)

# Load normalization stats
μ, σ = compute_normalization(
    dataset,
    config.training.batch_size,
    max_samples=config.data.norm_max_samples,
    seed=config.training.random_seed,
    norm_stats_path=config.data.norm_stats_path
)

# Create training loader (same as in trainer.py)
train_loader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    collate_fn=numpy_collate
)

# Output directory for figures
fig_dir = os.path.join("experiments/canesm/debug_figs")
os.makedirs(fig_dir, exist_ok=True)

# Iterate and visualize batches
variables = config.data.variables
n_batches_to_show = 5

for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= n_batches_to_show:
        break

    x = utils.process_batch(batch, μ, σ)
    patterns, samples = batch

    # Show raw data and normalized data side by side for first sample in batch
    n_vars = len(variables)
    fig, axes = plt.subplots(3, n_vars, figsize=(5 * n_vars, 12))

    for v, var in enumerate(variables):
        # Raw sample
        im0 = axes[0, v].imshow(samples[0, v, ::-1], aspect="auto")
        axes[0, v].set_title(f"Raw {var}")
        plt.colorbar(im0, ax=axes[0, v])

        # Normalized sample (x channels: 0..n_vars-1 are the variables)
        im1 = axes[1, v].imshow(x[0, v, ::-1], aspect="auto")
        axes[1, v].set_title(f"Normalized {var}")
        plt.colorbar(im1, ax=axes[1, v])

        # Histogram of normalized values
        axes[2, v].hist(x[:, v].ravel(), bins=100)
        axes[2, v].set_title(f"Normalized {var} (full batch)")
        axes[2, v].set_xlabel("value")

    # Also show the context channel (pattern)
    fig_ctx, ax_ctx = plt.subplots(1, 2, figsize=(10, 4))
    ax_ctx[0].imshow(patterns[0, ::-1], aspect="auto")
    ax_ctx[0].set_title("Raw pattern")
    im_ctx = ax_ctx[1].imshow(x[0, -1, ::-1], aspect="auto")
    ax_ctx[1].set_title("Normalized pattern")
    plt.colorbar(im_ctx, ax=ax_ctx[1])

    fig.suptitle(f"Batch {batch_idx}", fontsize=14)
    fig_ctx.suptitle(f"Batch {batch_idx} — context", fontsize=14)
    fig.savefig(os.path.join(fig_dir, f"batch_{batch_idx}.png"), dpi=150, bbox_inches="tight")
    fig_ctx.savefig(os.path.join(fig_dir, f"batch_{batch_idx}_context.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig_ctx)

    # Print summary stats
    print(f"\n--- Batch {batch_idx} ---")
    print(f"  Normalized x: min={x.min():.2f}, max={x.max():.2f}, "
          f"mean={x.mean():.2f}, std={x.std():.2f}")
    for v, var in enumerate(variables):
        vals = x[:, v]
        print(f"  {var}: min={vals.min():.2f}, max={vals.max():.2f}, "
              f"mean={vals.mean():.2f}, std={vals.std():.2f}")
    print(f"  Any NaN: {bool(jnp.isnan(x).any())}")
    print(f"  Any Inf: {bool(jnp.isinf(x).any())}")


# ── Full scan over training loader ──────────────────────────────────────────
print("\n\n========== Full training loader scan ==========\n")


# Create training loader (same as in trainer.py)
train_loader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    collate_fn=numpy_collate
)


n_channels = len(variables) + 1  # variables + context
channel_names = list(variables) + ["pattern"]
global_min = jnp.full(n_channels, jnp.inf)
global_max = jnp.full(n_channels, -jnp.inf)
nan_batches = []
inf_batches = []

for batch_idx, batch in enumerate(train_loader):
    x = utils.process_batch(batch, μ, σ)

    batch_min = x.min(axis=(0, 2, 3))  # min per channel
    batch_max = x.max(axis=(0, 2, 3))  # max per channel
    global_min = jnp.minimum(global_min, batch_min)
    global_max = jnp.maximum(global_max, batch_max)

    if jnp.isnan(x).any():
        nan_batches.append(batch_idx)
    if jnp.isinf(x).any():
        inf_batches.append(batch_idx)

    if (batch_idx + 1) % 100 == 0:
        print(f"  Scanned {batch_idx + 1} batches...")

# Report
print(f"\nScanned {batch_idx + 1} batches total\n")
print("Per-channel global min/max:")
for c, name in enumerate(channel_names):
    print(f"  {name:>10s}: min={global_min[c]:.4f}, max={global_max[c]:.4f}")

print(f"\nBatches with NaN: {nan_batches if nan_batches else 'None'}")
print(f"Batches with Inf: {inf_batches if inf_batches else 'None'}")
