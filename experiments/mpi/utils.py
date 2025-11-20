import os
from functools import partial
from typing import Tuple, Any, List
import numpy as np
from scipy.stats import kstest
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
import equinox as eqx
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.utils.collate import numpy_collate

from src.diffusion import ContinuousHeunSampler
from src.utils.graphs import compute_latlon_to_healpix_edges


################################################################################
#                               HEALPIX-LATLON EDGES                           #
################################################################################


def load_or_compute_edges(nside: int, edges_path: str, lat: jax.Array = None, lon: jax.Array = None) -> Tuple[jax.Array]:
    """
    Load precomputed edges if they exist, otherwise compute and save them.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)

    if os.path.exists(edges_path):
        print(f"Loading precomputed edges from {edges_path}")
        edges_data = jnp.load(edges_path)
        to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
        to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)
        return to_healpix, to_latlon
    else:
        print(f"Computing edges for nside={nside}, nlat={len(lat)}, nlon={len(lon)}")
        edges_list = compute_latlon_to_healpix_edges(lat, lon, nside, k=10)
        
        # Save edges
        print(f"Saving edges to {edges_path}")
        edges_dict = {'to_healpix': jnp.array(edges_list[0]).astype(jnp.int32),
                      'to_latlon': jnp.array(edges_list[1]).astype(jnp.int32)}
        jnp.savez(edges_path, **edges_dict)
        return edges_list



################################################################################
#                               BATCHES PROCESSING                             #
################################################################################


@eqx.filter_jit
def normalize(x: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Normalize data using mean and standard deviation."""
    return (x - μ) / σ

@eqx.filter_jit
def denormalize(x: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Denormalize data using mean and standard deviation."""
    return σ * x + μ

@eqx.filter_jit
def process_single(pattern: jnp.ndarray, sample: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Process a single sample for model input."""
    x = jnp.concatenate([sample, pattern[None, ...]], axis=0)
    return normalize(x, μ, σ)

@eqx.filter_jit
def process_batch(batch: Tuple, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Process a batch of samples."""
    doys, patterns, samples = batch
    x = jax.vmap(partial(process_single, μ=μ, σ=σ))(patterns, samples)
    return doys, x



################################################################################
#                              TEST POWER ESTIMATION                           #
################################################################################


def estimate_power(dataset, σmax, α, n_montecarlo, npool, μ, σ, ctx_size, key):
    # Initialize dataloader on subset of size n_iter
    dataset_size = len(dataset)
    indices = jr.permutation(key, dataset_size)[:n_montecarlo].tolist()
    rejections = 0
    dataset_subset = Subset(dataset, indices)
    dummy_loader = DataLoader(dataset_subset, batch_size=1, shuffle=True, collate_fn=numpy_collate)

    # Estimate power on this subset
    rejections = 0
    with tqdm(total=n_montecarlo) as pbar:
        pbar.set_description(f"Estimating power for σmax = {σmax:.1f}")
        for batch in dummy_loader:
            # Draw sample and flatten
            _, x = process_batch(batch, μ, σ)
            x = x[:, :-ctx_size]
            x0 = np.array(x.ravel())
            x0 = np.random.choice(x0, size=npool, replace=False)

            # Add noise and perform test
            xn = x0 + σmax * np.random.randn(len(x0))
            _, pvalue = kstest(xn, "norm", args=(0, σmax))
            rejections += (pvalue < α)
            _ = pbar.update(1)
    return rejections / n_montecarlo



################################################################################
#                               SAMPLING FUNCTIONS                             #
################################################################################


def create_sampler(model: eqx.Module, schedule: Any, doy: jnp.ndarray, pattern: jnp.ndarray,
                   μ: jnp.ndarray, σ: jnp.ndarray, output_size: Tuple) -> ContinuousHeunSampler:
    """Create a sampler for a given pattern."""
    context = normalize(pattern, μ[-1], σ[-1])[None, ...]
    def model_with_context(x, t):
        x = jnp.concatenate((x, context), axis=0)
        return model(x, doy, t)
    return ContinuousHeunSampler(schedule, model_with_context, output_size)

@eqx.filter_jit
def draw_samples_single(model: eqx.Module, schedule: Any, doy: jnp.ndarray, pattern: jnp.ndarray,
                        n_samples: int, n_steps: int, μ: jnp.ndarray, σ: jnp.ndarray,
                        output_size: Tuple, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a given pattern."""
    sampler = create_sampler(model, schedule, doy, pattern, μ, σ, output_size)
    samples = sampler.sample(n_samples, steps=n_steps, key=key)
    return denormalize(samples, μ[:-1], σ[:-1])

@eqx.filter_jit
def draw_samples_batch(model: eqx.Module, schedule: Any, pattern_batch: jnp.ndarray, doy_batch: jnp.ndarray,
                       n_samples: int, n_steps: int, μ: jnp.ndarray,
                       σ: jnp.ndarray, output_size: Tuple, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a batch of patterns."""
    keys = jr.split(key, pattern_batch.shape[0])
    Γ = partial(draw_samples_single,
                model=model,
                schedule=schedule,
                n_samples=n_samples,
                n_steps=n_steps, μ=μ, σ=σ,
                output_size=output_size)
    return jax.vmap(Γ)(doy=doy_batch, pattern=pattern_batch, key=keys)



################################################################################
#                               EMA FUNCTIONS                                  #
################################################################################


def is_float_array(x):
    return isinstance(x, jnp.ndarray) and np.issubdtype(x.dtype, np.floating)


@eqx.filter_jit
def update_ema(ema_model, model, decay):
    # Partition the model into numeric and static components.
    numeric_ema, static_ema = eqx.partition(ema_model, is_float_array)
    numeric_model, _ = eqx.partition(model, is_float_array)
    
    # Update only the numeric parameters.
    updated_numeric = jax.tree.map(lambda e, p: decay * e + (1 - decay) * p,
                                   numeric_ema,
                                   numeric_model)
    
    # Combine the updated numeric parameters with the static parts.
    return eqx.combine(updated_numeric, static_ema)




################################################################################
#                               METRICS COMPUTATION                            #
################################################################################


@eqx.filter_jit
def wasserstein_1d(pred_sorted: jnp.ndarray, target_sorted: jnp.ndarray, 
                   L: int = 50) -> jnp.ndarray:
    """Approximate the 1D Wasserstein distance using a quantile grid."""
    n_pred = pred_sorted.shape[0]
    n_target = target_sorted.shape[0]
    q_pred = (jnp.arange(n_pred) + 0.5) / n_pred
    q_target = (jnp.arange(n_target) + 0.5) / n_target
    q_common = jnp.linspace(0., 1., L)
    f_pred = jnp.interp(q_common, q_pred, pred_sorted)
    f_target = jnp.interp(q_common, q_target, target_sorted)
    return jnp.mean(jnp.abs(f_pred - f_target))


@eqx.filter_jit
def compute_emd_single(pred_samples: jnp.ndarray, target_data: jnp.ndarray, 
                       L: int = 50) -> jnp.ndarray:
    """Compute Earth Mover's Distance for a single sample."""
    pred_sorted = jnp.sort(pred_samples, axis=0)
    target_sorted = jnp.sort(target_data, axis=0)
    pred_flat = pred_sorted.reshape(pred_sorted.shape[0], -1)
    target_flat = target_sorted.reshape(target_sorted.shape[0], -1)
    emd_for_coordinate = partial(wasserstein_1d, L=L)
    emd_flat = jax.vmap(emd_for_coordinate)(pred_flat.T, target_flat.T)
    return emd_flat.reshape(pred_sorted.shape[1:])


@eqx.filter_jit
def compute_emd_batch(pred_samples: jnp.ndarray, target_data: jnp.ndarray,
                      L: int = 50) -> jnp.ndarray:
    """Compute Earth Mover's Distance for a batch of samples."""
    emd = jax.vmap(partial(compute_emd_single, L=L))(pred_samples, target_data)
    return jnp.mean(emd, axis=0)



################################################################################
#                   VISUALISATION/LOGGING FUNCTIONS                            #
################################################################################



def print_parameter_count(model):
    leaves = eqx.filter(model, eqx.is_array)
    n_parameters = sum(jnp.size(x) for x in jax.tree.leaves(leaves))
    print(f"Number of parameters = {n_parameters} \n")


def normalize_for_viz(x: jnp.ndarray) -> jnp.ndarray:
    """Map array to [0,1] range for visualization."""
    return (x - x.min()) / (x.max() - x.min())


def create_wandb_image(data: jnp.ndarray, flip: bool = True) -> wandb.Image:
    """Create a wandb image from data."""
    data = jax.device_get(data)
    if flip:
        data = data[::-1, :]
    return wandb.Image(normalize_for_viz(data))


def log_initial_context(context: jnp.ndarray) -> None:
    """Log initial context to wandb."""
    tas_context = create_wandb_image(context[0])
    wandb.log({"condition": tas_context}, step=0)


def log_samples(pred_samples: jnp.ndarray, target_data: jnp.ndarray, 
                variable_names: List[str], step: int) -> None:
    """Log generated samples and metrics to wandb.
    
    Args:
        pred_samples: Predicted samples
        target_data: Target data
        variable_names: List of variable names
        step: Current step
    """
    # Log samples
    sample_images = {
        f"{var}/sample": create_wandb_image(pred_samples[0, 0, i].squeeze())
        for i, var in enumerate(variable_names)
    }
    wandb.log(sample_images, step=step)
    
    # Compute and log metrics
    emd = compute_emd_batch(pred_samples, target_data[:, None]).mean(axis=(1, 2))
    emd_logs = {f"{var}/EMD": emd[i] for i, var in enumerate(variable_names)}
    metrics = {**emd_logs}
    wandb.log({k: v.item() for k, v in metrics.items()}, step=step)



def get_sample_batch(dataset, batch_size, key):
    """Get a sample batch for visualization.
    
    Args:
        dataset: PatternScalingCMIP6Dataset to sample from
        batch_size: Number of samples to get
        key: JAX random key
    Returns:
        Tuple of (pattern, target_data)
    """
    # Create a subset of the dataset for visualization
    indices = jr.randint(key, (batch_size,), 0, len(dataset))
    subset = Subset(dataset, indices.tolist())
    
    # Create a dataloader for the subset
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    
    # Get the first batch
    batch = next(iter(loader))
    return batch