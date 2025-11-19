# %%
from typing import Dict, List, Optional, Tuple, Any
import os
from functools import partial
import xarray as xr
import numpy as np
import einops
from sklearn.linear_model import LinearRegression
from dask.diagnostics import ProgressBar
from src.datasets import AmonCMIP6Data
from src.utils import arrays


# %%
def compute_gmst(
    root: str,
    model: str,
    experiments: List[str],
    gmst_path: str,
    force_recompute: bool = False
) -> xr.DataTree:
    # Try to load existing gmst data
    if gmst_path and os.path.exists(gmst_path) and not force_recompute:
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
    smooth_gmst.to_netcdf(gmst_path)
    print(f"Saved GMST time series to {gmst_path}")
    return smooth_gmst, ensemble_mean_tas


def fit_pattern_scaling(
    gmst: xr.DataTree,
    ensemble_mean_tas: xr.DataTree,
    experiments: List[str],
    pattern_scaling_path: str,
    force_recompute: bool = False
) -> np.ndarray:
    # Try to load existing pattern scaling parameters
    if pattern_scaling_path and os.path.exists(pattern_scaling_path) and not force_recompute:
        print(f"Loading pattern scaling parameters from {pattern_scaling_path}")
        β = np.load(pattern_scaling_path)
        return β

    # Extract arrays from datatree
    print("Fitting pattern scaling parameters...")
    X = []
    Y = []
    for e in experiments:
        gmst = gmst[e].tas.values
        tas = ensemble_mean_tas[e].tas.values
        X.append(gmst)
        Y.append(tas)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # Reshape for sklearn
    X = X.reshape(-1, 1)
    Y = einops.rearrange(Y, 't lat lon -> t (lat lon)')

    # Fit linear regression model
    lm = LinearRegression()
    lm.fit(X, Y)

    # Extract coefficients
    β1 = lm.coef_.squeeze()
    β0 = lm.intercept_.squeeze()
    β = np.stack([β0, β1], axis=-1)

    # Save computed pattern scaling parameters
    if pattern_scaling_path:
        np.save(pattern_scaling_path, β)
        print(f"Saved pattern scaling parameters to {pattern_scaling_path}")
    return β



# # %%
# gmst, ensemble_mean_tas = compute_gmst(
#     root="/home/shahineb/fs06/data/products/cmip6/processed",
#     model="MPI-ESM1-2-LR",
#     experiments=["ssp126"],
#     gmst_path="gmst.nc",
#     force_recompute=True
# )


# # %%
# β = fit_pattern_scaling(
#     ensemble_mean_tas=ensemble_mean_tas,
#     gmst=gmst,
#     experiments=["ssp126"],
#     pattern_scaling_path="β.npy",
#     force_recompute=True
# )