# %%
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import shutil
from functools import partial
import xarray as xr
import numpy as np
import einops
from sklearn.linear_model import LinearRegression
from dask.diagnostics import ProgressBar
from src.datasets import AmonCMIP6Data, DayCMIP6Data, PatternToDayCMIP6Data
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
    smooth_gmst.to_netcdf(gmst_path)
    print(f"Saved GMST time series to {gmst_path}")
    return smooth_gmst, ensemble_mean_tas


# %%
# gmst, ensemble_mean_tas = compute_gmst(
#     root="/home/shahineb/fs06/data/products/cmip6/processed",
#     model="MPI-ESM1-2-LR",
#     experiments=["ssp126"],
#     gmst_path="gmst.nc",
#     force_recompute=True
# )


# # %%
# cmip6data  = DayCMIP6Data(root="/home/shahineb/fs06/data/products/cmip6/processed",
#                        model="MPI-ESM1-2-LR",
#                        variables=["tas"],
#                        experiments={
#                            "ssp126": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
#                        })


# # %%
# dataset = PatternToDayCMIP6Data(gmst=gmst, cmip6data=cmip6data)
# dataset.fit(experiments=["ssp126"], ensemble_mean_tas=ensemble_mean_tas)

# # %%
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# def numpy_collate(batch):
#     return [np.stack(b, axis=0) for b in zip(*batch)]

# train_loader = DataLoader(
#         dataset,
#         batch_size=8, 
#         shuffle=True,
#         collate_fn=numpy_collate
#     )

# # %%
# %%time
# i = 0
# for batch in tqdm(train_loader):
#     doy_batch, pattern_batch, cmip6_array_batch = batch
#     i += 1
#     if i >= 100:
#         break

# %%
# # %%
# doy, pattern, cmip6_array = dataset[366]

# # %%
# cmip6_slice = cmip6data[366]
# cmip6_array = cmip6_slice.to_array().values

# # %%
# dataset.β

# # %%

# # %%
# idx = 366

# selected_data = cmip6data[idx]
# year = selected_data.time.dt.year.item()
# doy = selected_data.time.dt.dayofyear.item()
# selected_data_array = selected_data.to_array().values
# leaf_idx, _ = cmip6data.indexmap(idx)
# experiment = cmip6data.dtree.leaves[leaf_idx].parent.name
# gmst = selfgmst[experiment].sel(year=year).tas.item()
# pattern = β[:, 0] + β[:, 1] * gmst



# # selected_data = cmip6data[idx]
# # year = selected_data.time.dt.year.item()

# # %%
# leaf_idx, _ = cmip6data.indexmap(idx)
# experiment = cmip6data.dtree.leaves[leaf_idx].parent.name

# # %%
# gmst[experiment].sel(year=year).tas.item()


# # %%
# β = fit_pattern_scaling(
#     ensemble_mean_tas=ensemble_mean_tas,
#     gmst=selfgmst,
#     experiments=["ssp126"],
#     pattern_scaling_path="β.npy",
#     force_recompute=False
# )