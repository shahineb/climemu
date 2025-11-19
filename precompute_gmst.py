# %%
from typing import Dict, List, Optional, Tuple, Any
import os
from functools import partial
import xarray as xr
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
) -> arrays.DataTree:
    # Try to load existing gmst data
    if gmst_path and os.path.exists(gmst_path) and not force_recompute:
        print(f"Loading gmst time series from {gmst_path}")
        gmst = xr.open_datatree(gmst_path)
        assert set(experiments) <= set(gmst.keys()), "GMST is missing some requested experiments, need to recompute."
        return gmst

    # Load monthly temperature data
    cmip6data = AmonCMIP6Data(root=root,
                              model=model,
                              experiments=experiments,
                              variables=["tas"])
    tas = cmip6data.dtree.map_over_datasets(arrays.filter_var('tas'))

    # Compute annual gmst time series + 5-year moving average smoothing
    gmst = tas.map_over_datasets(arrays.global_mean).mean("member")
    gmst = gmst.map_over_datasets(arrays.annual_mean)
    with ProgressBar():
        print("Computing GMST time series...")
        gmst = gmst.compute()
    smooth_gmst = gmst.map_over_datasets(partial(arrays.year_moving_average, window=5))

    # Save computed gmst data
    smooth_gmst.to_netcdf(gmst_path)