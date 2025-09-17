import os
import copy
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
import jax.numpy as jnp
import jax.random as jr

from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.utils import arrays
from src.datasets import CMIP6Data, PatternToCMIP6Dataset
from src.utils.collate import numpy_collate
from . import utils


def load_dataset(
    root: str,
    model: str,
    experiments: List[str],
    variables: List[str],
    subset: Optional[Dict[str, Any]] = None,
    in_memory: bool = True,
    pattern_scaling_path: Optional[str] = None,
    external_β: Optional[jnp.ndarray] = None
) -> PatternToCMIP6Dataset:
    """Load and prepare a CMIP6 dataset with pattern scaling.

    Args:
        root: Path to CMIP6 data directory
        model: Climate model name (e.g., "MPI-ESM1-2-LR")
        experiments: List of experiment names (e.g., ["ssp585", "historical"])
                    Multiple experiments will be combined into a single dataset
        variables: Climate variables to load (e.g., ["tas", "pr"])
        subset: Optional dictionary to subset the data (e.g., time slice)
        in_memory: Whether to load the full dataset into memory
        pattern_scaling_path: Path to save/load pattern scaling coefficients
        external_β: Optional external pattern scaling coefficients

    Returns:
        A PatternToCMIP6Dataset ready for use with DataLoader
    """
    # Initialize CMIP6 data loader with specified parameters
    cmip6_kwargs = {
        "root": root,
        "model": model,
        "experiments": experiments,
        "variables": variables
    }
    if subset:
        cmip6_kwargs["subset"] = subset

    # Load CMIP6 data and compute global mean surface temperature
    cmip6data = CMIP6Data(**cmip6_kwargs)
    tas = cmip6data.dtree.map_over_datasets(arrays.filter_var('tas'))
    gmst = tas.map_over_datasets(arrays.global_mean).mean("member").compute()
    gmst = gmst.map_over_datasets(partial(arrays.moving_average, window=60))

    # Handle pattern scaling coefficients (β) in different scenarios
    if external_β is not None:
        # Use externally provided coefficients (e.g., from training dataset)
        dataset = PatternToCMIP6Dataset(gmst, cmip6data, external_β, in_memory=in_memory)
    elif pattern_scaling_path and not os.path.exists(pattern_scaling_path):
        # Compute and save new coefficients
        dataset = PatternToCMIP6Dataset(gmst, cmip6data, in_memory=in_memory)
        dataset.fit(["historical", "ssp585"])
        dataset.save_pattern_scaling(pattern_scaling_path)
    elif pattern_scaling_path:
        # Load existing coefficients
        dataset = PatternToCMIP6Dataset(gmst, cmip6data, in_memory=in_memory)
        dataset.load_pattern_scaling(pattern_scaling_path)
    else:
        # Compute coefficients but don't save them
        dataset = PatternToCMIP6Dataset(gmst, cmip6data, in_memory=in_memory)
        dataset.fit()

    return dataset


def compute_normalization(
    dataset: PatternToCMIP6Dataset,
    batch_size: int,
    max_samples: int = 1000,
    seed: int = 42,
    norm_stats_path: Optional[str] = None,
    force_recompute: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute mean and standard deviation for data normalization using a random subset.

    If norm_stats_path is provided and the file exists, it will load the statistics
    unless force_recompute is True. If the file doesn't exist, it will compute and
    save the statistics.

    Args:
        dataset: The PatternToCMIP6Dataset containing the data
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to use for normalization
        seed: Random seed for reproducibility
        norm_stats_path: Optional path to save/load normalization statistics
        force_recompute: If True, recompute statistics even if they exist

    Returns:
        Tuple of (mean, std) arrays for normalization
    """
    # Try to load existing statistics if path is provided
    if norm_stats_path and os.path.exists(norm_stats_path) and not force_recompute:
        print(f"Loading normalization statistics from {norm_stats_path}")
        stats = jnp.load(norm_stats_path)
        return stats['μ'], stats['σ']

    # Keep only piControl data
    indexmap = dataset.cmip6data.indexmap
    piControl_index = dataset.cmip6data.experiments.index("piControl")
    subindexmap = indexmap[indexmap[:, 0] == piControl_index]
    piControl_dataset = copy.deepcopy(dataset)
    piControl_dataset.cmip6data.indexmap = subindexmap

    # Create a random subset of the dataset
    dataset_size = len(piControl_dataset)
    # subset_size = min(max_samples, dataset_size // 6)  # ~6 lag month autocorrelation
    subset_size = min(max_samples, dataset_size)

    # Generate random indices
    key = jr.PRNGKey(seed)
    indices = jr.permutation(key, dataset_size)[:subset_size].tolist()
    dataset_subset = Subset(piControl_dataset, indices)
    
    # Create a dataloader for the subset
    dummy_loader = DataLoader(
        dataset_subset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=numpy_collate
    )
    
    # Process each batch and collect results
    x = []
    for batch in tqdm(dummy_loader, desc=f"Computing μ, σ  (using {subset_size} samples)"):
        sample_shape = batch[-1].shape
        sample_shape = (1 + sample_shape[1], sample_shape[2], sample_shape[3])
        x.append(utils.process_batch(batch, jnp.zeros(sample_shape), jnp.ones(sample_shape)))
    
    # Compute mean across all batches
    x = jnp.concatenate(x)
    μ = x.mean(axis=0)

    # Compute stddev with shrinkage estimator to prevent σ~0
    # σ = x.std(axis=0)
    N = subset_size
    λ = N / 10
    σ2 = x.var(axis=0, ddof=1)
    σ2glob = x.var(axis=(0, 2, 3), ddof=1)[:, None, None]
    σ2shrink = ((N - 1) * σ2 + λ * σ2glob) / (N - 1 + λ)
    σ = jnp.sqrt(σ2shrink)
    
    # Save statistics if path is provided
    if norm_stats_path:
        print(f"Saving normalization statistics to {norm_stats_path}")
        jnp.savez(norm_stats_path, μ=μ, σ=σ)
    return μ, σ 


def estimate_sigma_max(
    dataset: PatternToCMIP6Dataset,
    μ: jnp.ndarray,
    σ: jnp.ndarray,
    ctx_size: int,
    sigmas: Optional[np.ndarray],
    subset_size: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    sigma_max_path: str = None,
    force_recompute: bool = False
) -> float:
    """
    Estimate the maximum σ such that adding Gaussian noise with σ to the data
    produces a samples that don't reject a normality test.

    If sigma_max_path is provided and the file exists, it will load the value
    unless force_recompute is True. If the file doesn't exist, it will compute and
    save the value.

    Args:
        dataset: The PatternToCMIP6Dataset containing the data
        μ: Mean for normalization
        σ: Standard deviation for normalization
        sigmas: Grid of σ values to test for
        batch_size: Batch size for dataloading
        subset_size: Number of samples to use for estimation
        seed: Random seed for reproducibility
        alpha: Significance level for the normality test
        sigma_max_path: Optional path to save/load σmax
        force_recompute: If True, recompute σmax even if it exists

    Returns:
        Estimated σmax value
    """
    # Try to load existing σmax if path is provided
    if sigma_max_path and os.path.exists(sigma_max_path) and not force_recompute:
        sigma_max = float(np.load(sigma_max_path))
        print(f"Loading σmax = {sigma_max} from {sigma_max_path}")
        return sigma_max

    # Create loader over subset of training data
    dataset_size = len(dataset)
    subset_size = min(subset_size, dataset_size)
    key = jr.PRNGKey(seed)
    indices = jr.permutation(key, dataset_size)[:subset_size].tolist()
    dataset_subset = Subset(dataset, indices)
    dummy_loader = DataLoader(
        dataset_subset,
        batch_size=1,
        shuffle=True,
        collate_fn=numpy_collate
    )

    # Find smallest σmax that fails to reject normality test over all samples
    sigma_max = -np.inf
    N = len(sigmas)
    vmax = np.max(sigmas)
    reps = 30
    with tqdm(total=len(dummy_loader)) as pbar:
        for batch in dummy_loader:
            # Flatten sample
            x = utils.process_batch(batch, μ, σ)[:, :-ctx_size]
            x0 = x.ravel()

            # Iterate over noise levels
            for sigma in sigmas:
                # Compute frequency at which we fail to reject normality
                count = 0
                for _ in range(reps):
                    xn = x0 + sigma * np.random.randn(*x0.shape)
                    _, p = stats.normaltest(xn)
                    count += (p >= alpha)
                fail_to_reject_rate = count / reps

                # If >80% then update σmax and move to next sample
                if fail_to_reject_rate > 0.8:
                    sigma_max = max(sigma_max, sigma)
                    sigmas = np.linspace(sigma_max, vmax, N)
                    pbar.set_description(f"σmax {round(sigma_max, 2)}")
                    break
            _ = pbar.update(1)
            if np.isclose(sigma_max, vmax, atol=1):
                sigma_max = vmax
                break

    # Double it to prevent signal leak (just to be safe)
    sigma_max = 2 * sigma_max

    # Save and return
    if sigma_max_path:
        print(f"Saving σmax = {sigma_max} to {sigma_max_path}")
        np.save(sigma_max_path, sigma_max)
    return sigma_max
