import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import xarray as xr
import jax.numpy as jnp
import jax.random as jr
from dask.diagnostics import ProgressBar

from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset



# import sys
# base_dir = os.path.join(os.getcwd(), '../..')
# if base_dir not in sys.path:
#     sys.path.append(base_dir)
from src.utils import arrays, trees
from src.datasets import AmonCMIP6Data, DayCMIP6Data, PatternToDayCMIP6Data
from src.utils.collate import numpy_collate
from . import utils
# from experiments.mpi import utils



def compute_annual_gmst(
    root: str,
    model: str,
    experiments: List[str],
    gmst_path: str,
    force_recompute: bool = False
) -> xr.DataTree:
    # Try to load existing gmst data
    if gmst_path and os.path.exists(gmst_path):
        if force_recompute:
            os.remove(gmst_path)
        else:
            print(f"Loading gmst time series from {gmst_path}")
            gmst = xr.open_datatree(gmst_path)
            assert set(experiments) <= set(gmst.keys()), "GMST is missing some requested experiments, need to recompute."
            return gmst, None

    # Load monthly temperature data
    cmip6data = AmonCMIP6Data(root=root,
                              model=model,
                              experiments=experiments,
                              variables=["tas"])
    tas = cmip6data.dtree.map_over_datasets(arrays.filter_var('tas'))

    # Compute annual gmst time series + 5-year moving average smoothing
    ensemble_mean_tas = tas.map_over_datasets(arrays.annual_mean).mean("member")
    with ProgressBar():
        print("Computing GMST time series...")
        ensemble_mean_tas = ensemble_mean_tas.compute()
    gmst = ensemble_mean_tas.map_over_datasets(arrays.global_mean)
    smooth_gmst = gmst.map_over_datasets(partial(arrays.year_moving_average, window=5))

    # Save computed gmst data
    if gmst_path:
        smooth_gmst.to_netcdf(gmst_path)
        print(f"Saved GMST time series to {gmst_path}")
    return smooth_gmst, ensemble_mean_tas


def load_dataset(
    root: str,
    model: str,
    experiments: Dict[str, List[str]],
    variables: List[str],
    subset: Optional[Dict[str, Any]] = None,
    gmst_path: Optional[str] = None,
    pattern_scaling_path: Optional[str] = None,
    external_β: Optional[jnp.ndarray] = None
) -> PatternToDayCMIP6Data:
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

    # Create CMIP6 data instance
    cmip6data = DayCMIP6Data(**cmip6_kwargs)

    # Compute annual GMST time series
    gmst, ensemble_mean_tas = compute_annual_gmst(root=root,
                                                  model=model,
                                                  experiments=list(experiments.keys()),
                                                  gmst_path=gmst_path)

    # Handle pattern scaling coefficients (β) in different scenarios
    if external_β is not None:
        # Use externally provided coefficients (e.g., from training dataset)
        dataset = PatternToDayCMIP6Data(gmst, cmip6data, external_β)
    elif pattern_scaling_path and not os.path.exists(pattern_scaling_path):
        # Compute and save new coefficients
        dataset = PatternToDayCMIP6Data(gmst, cmip6data)
        dataset.fit(["historical", "ssp585"], ensemble_mean_tas)
        dataset.save_pattern_scaling(pattern_scaling_path)
        print(f"Saved pattern scaling coefficients to {pattern_scaling_path}")
    elif pattern_scaling_path:
        # Load existing coefficients
        dataset = PatternToDayCMIP6Data(gmst, cmip6data)
        dataset.load_pattern_scaling(pattern_scaling_path)
        print(f"Loaded pattern scaling coefficients from {pattern_scaling_path}")
    else:
        # Compute coefficients but don't save them
        dataset = PatternToDayCMIP6Data(gmst, cmip6data)
        dataset.fit(["historical", "ssp585"], ensemble_mean_tas)
    return dataset


def compute_normalization(
    dataset: PatternToDayCMIP6Data,
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
    piControldata = DayCMIP6Data(root=dataset.cmip6data.root,
                                model=dataset.cmip6data.model,
                                experiments={"piControl": ["r1i1p1f1"]},
                                variables=dataset.cmip6data.variables)

    piControl_dataset = PatternToDayCMIP6Data(cmip6data=piControldata,
                                              gmst=trees.filter_datatree(dataset.gmst, ["piControl"]),
                                              β=dataset.β)

    # Create a random subset of the dataset
    dataset_size = len(piControl_dataset)
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
        μ0 = jnp.zeros(sample_shape)
        σ0 = jnp.ones(sample_shape)
        _, x0 = utils.process_batch(batch, μ0, σ0)
        x.append(x0)
    
    # Compute mean and stddev across all batches
    x = jnp.concatenate(x)
    μ = x.mean(axis=0)
    σ = x.std(axis=0)

    # Save statistics if path is provided
    if norm_stats_path:
        print(f"Saving normalization statistics to {norm_stats_path}")
        jnp.savez(norm_stats_path, μ=μ, σ=σ)
    return μ, σ


def estimate_sigma_max(
    dataset: PatternToDayCMIP6Data,
    μ: jnp.ndarray,
    σ: jnp.ndarray,
    ctx_size: int,
    search_interval: List[float],
    seed: int = 42,
    alpha: float = 0.05,
    sigma_max_path: str = None,
    force_recompute: bool = False
) -> float:
    """
    Binary search to find the smallest σmax such that a KS test
    cannot reject that the data with added Gaussian noise comes from N(0,σmax).

    If sigma_max_path is provided and the file exists, it will load the value
    unless force_recompute is True. If the file doesn't exist, it will compute and
    save the value.

    Args:
        dataset: The PatternToCMIP6Dataset containing the data
        μ: Mean for normalization
        σ: Standard deviation for normalization
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
        σmax = float(np.load(sigma_max_path))
        print(f"Loading σmax = {σmax} from {sigma_max_path}")
        return σmax
    
    # Define search parameters
    σmax_low, σmax_high = search_interval
    max_split = 20
    n_montecarlo = 100
    max_montecarlo = 10000
    npool = 50000
    tgt_pow = 0.1
    tol = 0.001 + 1.96 * np.sqrt(tgt_pow * (1 - tgt_pow)  / max_montecarlo)
    key = jr.PRNGKey(seed)

    # Select σmax such that test power < 0.1
    with tqdm(total=max_split) as pbar:
        for _ in range(max_split):
            # Set σmax in the middle of search interval
            σmax = 0.5 * (σmax_low + σmax_high)

            # Estimate CI on test power
            key, χ = jr.split(key)
            power = utils.estimate_power(dataset=dataset,
                                         σmax=σmax,
                                         α=alpha,
                                         n_montecarlo=n_montecarlo,
                                         npool=npool,
                                         μ=μ,
                                         σ=σ,
                                         ctx_size=ctx_size,
                                         key=χ)
            spread = 1.96 * np.sqrt(power * (1 - power) / n_montecarlo)
            lb, ub = power - spread, power + spread
            pbar.set_description(f"σmax = {σmax} -> Power ∈ ({lb:.3f}, {ub:.3f})")
            _ = pbar.update(1)

            # Case 1: CI is fully below 0.1
            if ub < tgt_pow:
                # Early stop if CI is tight and close to 0.1 from below
                if (spread < tol) and (ub + tol / 2 > tgt_pow):
                    break
                # Else look for smaller values
                σmax_high = σmax
            # Case 2: CI is fully above 0.1 OR CI is tight and straddles 0.1
            elif (lb > tgt_pow) or ((spread < tol) and (lb < tgt_pow) and (ub > tgt_pow)):
                # Look for larger values
                σmax_low = σmax
            # Case 3: Ambiguous overlap with 0.1 and CI not tight enough
            else:
                print("Uncertain, increasing nb of monte carlo samples \n")
                n_montecarlo = min(2 * n_montecarlo, max_montecarlo)
            if np.allclose(σmax_low, σmax_high, atol=1):
                break

    # Save and return
    if sigma_max_path:
        print(f"Saving σmax = {σmax} to {sigma_max_path}")
        np.save(sigma_max_path, σmax)
    return σmax


# # %%
# dataset = load_dataset(
#     root="/home/shahineb/fs06/data/products/cmip6/processed",
#     model="MPI-ESM1-2-LR",
#     experiments={"ssp126": ["r1i1p1f1", "r2i1p1f1"]},
#     variables=["tas"],
#     gmst_path="./gmst.nc",
#     pattern_scaling_path="./β.npy",
# )


# # %%
# μ, σ = compute_normalization(
#     dataset=dataset,
#     batch_size=16,
#     max_samples=1000,
#     norm_stats_path="./norm_stats.npz",
# )


# # %%
# σmax = estimate_sigma_max(
#     dataset=dataset,
#     μ=μ,
#     σ=σ,
#     ctx_size=1,
#     search_interval=[1, 200],
#     sigma_max_path="./sigma_max.npy",
# )
