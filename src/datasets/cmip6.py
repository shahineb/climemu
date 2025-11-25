# %%
import os
from typing import List, Optional, Dict, Union
import functools
from tqdm import tqdm
import numpy as np
import xarray as xr
import zarr
from torch.utils.data import Dataset
from dask.diagnostics import ProgressBar
from .constants import SECONDS_PER_DAY


class AmonCMIP6Data(Dataset):
    """Dataset for loading and managing CMIP6 climate model data.

    This class provides a unified interface for accessing CMIP6 climate model data across:
    - Multiple experiments (e.g., 'ssp245', 'ssp585')
    - Different climate variables (e.g., 'tas', 'pr')
    - Various ensemble members

    The data is organized in a hierarchical structure using xarray.DataTree for efficient
    access and manipulation of multi-dimensional climate data.
    """
    def __init__(self, root: str, model: str, experiments: List[str], variables: List[str], subset: Optional[Dict] = None):
        """Initialize the CMIP6 dataset.

        Args:
            root: Root directory containing CMIP6 data.
            model: Climate model name (e.g., 'MPI-ESM1-2-LR').
            experiments: List of experiment names (e.g., ['ssp245', 'ssp585']).
            variables: List of climate variables (e.g., ['tas', 'pr']).
            frequency: Data frequency ('monthly', 'daily').
            subset: Optional dictionary of dimensions to subset the data.

        Raises:
            ValueError: If experiments or variables lists are empty.
        """
        self.root = root
        self.model = model
        self.variables = variables
        self._init_dtree(experiments, subset)
        self._init_indexmap()

    @staticmethod
    def get_file_path(root: str, model: str, experiment: str, variable: str, variant: str) -> str:
        """Construct the file path for a given model configuration.

        Args:
            root: Root directory.
            model: Model name.
            experiment: Experiment name.
            variable: Variable name.
            variant: Variant identifier.
            frequency: Data frequency.
            table_id: Table identifier.

        Returns:
            str: Constructed file path.
        """
        filename = f"{variable}_Amon_{model}_{experiment}_{variant}_monthly_anomaly.nc"
        path = os.path.join(root, model, experiment, variant, f"{variable}_anomaly", "Amon", filename)
        return path

    def _load_xarray(self, experiment: str, variable: str) -> xr.Dataset:
        """Load a dataset for a given experiment and variable by combining available variants.

        Args:
            experiment: Experiment name.
            variable: Variable name.

        Returns:
            xr.Dataset: Combined dataset across all variants.

        Raises:
            FileNotFoundError: If no files are found for the given experiment and variable.
        """
        experiment_dir = os.path.join(self.root, self.model, experiment)
        if not os.path.isdir(experiment_dir):
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        variants = os.listdir(experiment_dir)
        file_paths = [
            self.get_file_path(self.root, self.model, experiment, variable, variant)
            for variant in variants
        ]
        existing_paths = list(filter(os.path.isfile, file_paths))
        
        if not existing_paths:
            raise FileNotFoundError(f"No files found for {variable} in {experiment}")
        
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        dataset = xr.open_mfdataset(existing_paths, combine="nested", concat_dim="member", join="outer", decode_times=time_coder)
        if variable == "pr":
            dataset = dataset * SECONDS_PER_DAY
        return dataset

    def _init_dtree(self, experiments: List[str], subset: Optional[Dict]) -> None:
        """Initialize the xarray.Dataset by merging variables and experiments.

        This method:
        1. Loads data for each experiment and variable.
        2. Aligns ensemble members across variables.
        3. Merges variables for each experiment.
        4. Concatenates experiments into a single datatree.

        Args:
            experiments: List of experiment names.
            subset: Dictionary of dimensions to subset the data (optional).
        """
        merged_experiments = dict()
        with tqdm(total=len(experiments), desc="Progress") as pbar:
            for e in experiments:
                pbar.set_description(f"Building datatree for {e}")
                ds_list = [self._load_xarray(e, var) for var in self.variables]
                common_members = functools.reduce(np.intersect1d, [ds.member.values for ds in ds_list])
                ds_list = [ds.sel(member=common_members) for ds in ds_list]
                merged_experiments[e] = xr.merge(ds_list)
                _ = pbar.update(1)

            self.dtree = xr.DataTree.from_dict(merged_experiments, name="AmonCMIP6Data")
            if subset is not None:
                self.dtree = self.dtree.sel(subset)
            pbar.set_description("Datatree initialized")

    def load(self) -> None:
        """Load the xarray dataset into memory.
        """
        print("Loading xarray into memory...")
        with ProgressBar():
            self.dtree.load()

    def _init_indexmap(self) -> None:
        """Initialize the indexmap for accessing data points.

        Creates a mapping of valid (non-null) data points across all dimensions.
        """
        indexmap = []
        with tqdm(total=len(self.experiments)) as pbar:
            for i, e in enumerate(self.experiments):
                pbar.set_description(f"Building indices for {e}")
                da = self.dtree[e].ds.to_array()
                valid = ~da.isnull().any(dim=["lat", "lon", "variable"])
                argvalid = np.argwhere(valid.values)
                argvalid = np.column_stack((np.full(len(argvalid), i), argvalid))
                indexmap.append(argvalid)
                _ = pbar.update(1)
            self.indexmap = np.concatenate(indexmap)
            pbar.set_description("Indexmap initialized")

    def isel(self, *args, **kwargs) -> xr.Dataset:
        """Perform index-based selection on the dataset.

        Args:
            *args: Positional arguments passed to xarray.Dataset.isel().
            **kwargs: Keyword arguments passed to xarray.Dataset.isel().

        Returns:
            xr.Dataset: Selected subset of the dataset.
        """
        return self.dtree.isel(*args, **kwargs)

    def sel(self, *args, **kwargs) -> xr.Dataset:
        """Perform label-based selection on the dataset.

        Args:
            *args: Positional arguments passed to xarray.Dataset.sel().
            **kwargs: Keyword arguments passed to xarray.Dataset.sel().

        Returns:
            xr.Dataset: Selected subset of the dataset.
        """
        return self.dtree.sel(*args, **kwargs)

    def __len__(self) -> int:
        """Get the number of valid data points in the dataset.

        Returns:
            int: Number of valid data points.
        """
        return len(self.indexmap)

    def __repr__(self):
        """String representation of the dataset.

        Returns:
            str: String representation of the xarray.DataTree.
        """
        return self.dtree.__repr__()

    def __getitem__(self, idx: Union[str, int]) -> xr.Dataset:
        """Get item by string or integer indexing.

        Args:
            idx: If str: variable name to select.
                 If int: position in the indexmap to select.

        Returns:
            xr.Dataset: Selected data.

        Raises:
            ValueError: If idx is not a string or integer.
        """
        if isinstance(idx, str):
            return self.dtree[idx]
        elif isinstance(idx, int):
            ei, ωi, ti = self.indexmap[idx]
            e = self.experiments[ei]
            return self[e].ds.isel(member=ωi, time=ti)
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")
        
    @functools.cached_property
    def experiments(self) -> List[str]:
        """Get list of experiment names.

        Returns:
            List[str]: List of experiment names.
        """
        return list(self.dtree.keys())
    
    @functools.cached_property
    def lat(self) -> np.ndarray:
        """Get latitude coordinates.

        Returns:
            np.ndarray: Array of latitude values.
        """
        e = self.experiments[0]
        return self.dtree[e].lat.values
    
    @functools.cached_property
    def lon(self) -> np.ndarray:
        """Get longitude coordinates.

        Returns:
            np.ndarray: Array of longitude values.
        """
        e = self.experiments[0]
        return self.dtree[e].lon.values
    
    @functools.cached_property
    def nlat(self) -> int:
        """Get number of latitude points.

        Returns:
            int: Number of latitude points.
        """
        return len(self.lat)
    
    @functools.cached_property
    def nlon(self) -> int:
        """Get number of longitude points.

        Returns:
            int: Number of longitude points.
        """
        return len(self.lon)
    

class DayCMIP6Data(Dataset):
    def __init__(self, root: str, model: str, variables: List, experiments: Dict):
        self.root = root
        self.model = model
        self.variables = list(variables)
        self.experiments = experiments
        self._init_dtree(experiments)
        self._init_indexmap()

    @staticmethod
    def get_store_path(root: str, model: str, experiment: str, variable: str, variant: str) -> str:
        filename = f"{variable}_day_{model}_{experiment}_{variant}_daily_anomaly.zarr"
        path = os.path.join(root, model, experiment, variant, f"{variable}_anomaly", "day", filename)
        return path

    def _load_zarr(self, experiment: str, variable: str, variant: str) -> xr.Dataset:
        store_dir = self.get_store_path(self.root, self.model, experiment, variable, variant)
        if not os.path.isdir(store_dir):
            raise FileNotFoundError(f"No files found for {variable} in {experiment}, {variant}")
        dataset = zarr.open_consolidated(store_dir, mode='r')[variable]
        return dataset
    
    def _init_dtree(self, experiments: Dict):
        n_dataset = sum(map(len, experiments.values()))
        self.dtree = dict()
        with tqdm(total=n_dataset, desc="Progress") as pbar:
            for e, variants in experiments.items():
                for ω in variants:
                    pbar.set_description(f"Building datatree for {e}/{ω}")
                    path = f"{e}/{ω}"
                    self.dtree[path] = {v: self._load_zarr(e, v, ω) for v in self.variables}
                    _ = pbar.update(1)
            pbar.set_description("Datatree initialized")

    def _init_indexmap(self):
        """Initialize the indexmap for accessing data points.
        """
        leaves_lengths = [leaf["tas"].shape[0] for leaf in self.leaves]
        self._cumulative_leaves_lengths = np.cumsum([0] + leaves_lengths)
        def indexmap(idx):
            leaf_idx = np.searchsorted(self._cumulative_leaves_lengths, idx, "right") - 1
            time_idx = idx - self._cumulative_leaves_lengths[leaf_idx]
            return leaf_idx, time_idx
        self.indexmap = indexmap

    def __len__(self) -> int:
        """Get the number of data points in the dataset.

        Returns:
            int: Number of data points.
        """
        return self._cumulative_leaves_lengths[-1]

    def __repr__(self):
        """String representation of the dataset.

        Returns:
            str: String representation of the xarray.DataTree.
        """
        return self.dtree.__repr__()

    def __getitem__(self, idx: Union[str, int]) -> xr.Dataset:
        """Get item by string or integer indexing.

        Args:
            idx: If str: variable name to select.
                 If int: position in the indexmap to select.

        Returns:
            xr.Dataset: Selected data.

        Raises:
            ValueError: If idx is not a string or integer.
        """
        if isinstance(idx, str):
            return self.dtree[idx]
        elif isinstance(idx, int):
            leaf_idx, time_idx = self.indexmap(idx)
            leaf = self.leaves[leaf_idx]
            return {var: leaf[var][time_idx, :, :] for var in self.variables}
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    @functools.cached_property
    def paths(self):
        """Get list of experiment names.

        Returns:
            List[str]: List of leaves paths.
        """
        return list(self.dtree.keys())
    
    @functools.cached_property
    def leaves(self):
        return list(self.dtree.values())

    @functools.cached_property
    def lat(self):
        e = list(self.experiments.keys())[0]
        ω = self.experiments[e][0]
        v = self.variables[0]
        store_dir = self.get_store_path(self.root, self.model, e, v, ω)
        lat = zarr.open_consolidated(store_dir, mode='r')["lat"][:]
        return lat
    
    @functools.cached_property
    def lon(self):
        e = list(self.experiments.keys())[0]
        ω = self.experiments[e][0]
        v = self.variables[0]
        store_dir = self.get_store_path(self.root, self.model, e, v, ω)
        lat = zarr.open_consolidated(store_dir, mode='r')["lon"][:]
        return lat
    
    @functools.cached_property
    def nlat(self):
        return len(self.lat)

    @functools.cached_property
    def nlon(self):
        return len(self.lon)
    


# store_dir = DayCMIP6Data.get_store_path(root, model, "piControl", "tas", "r1i1p1f1")
# tas = zarr.open_consolidated(store_dir, mode='r')

# cmip6 = DayCMIP6Data(root, model, variables, experiments)

# %%
# import time
# start = time.perf_counter()
# for i in range(100):
#     _ = cmip6[i + 1]
# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.4f} seconds")



# # %%
# class DayCMIP6Data(Dataset):
#     def __init__(self, root: str, model: str, variables: List, experiments: Dict, subset: Optional[Dict] = None):
#         self.root = root
#         self.model = model
#         self.variables = list(variables)
#         self.experiments = experiments
#         self.time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
#         self._init_dtree(experiments, subset)
#         self._init_indexmap()

#     @staticmethod
#     def get_store_path(root: str, model: str, experiment: str, variable: str, variant: str) -> str:
#         filename = f"{variable}_day_{model}_{experiment}_{variant}_daily_anomaly.zarr"
#         path = os.path.join(root, model, experiment, variant, f"{variable}_anomaly", "day", filename)
#         return path
    
#     def _load_zarr(self, experiment: str, variable: str, variant: str) -> xr.Dataset:
#         store_dir = self.get_store_path(self.root, self.model, experiment, variable, variant)
#         if not os.path.isdir(store_dir):
#             raise FileNotFoundError(f"No files found for {variable} in {experiment}, {variant}")
        
#         dataset = xr.open_zarr(store_dir,
#                                decode_times=False,
#                                consolidated=True).unify_chunks()
#         if variable == "pr":
#             dataset = dataset * SECONDS_PER_DAY
#             dataset["pr"].attrs["units"] = "mm/day"
#         return dataset
    
#     def _init_dtree(self, experiments: Dict, subset: Optional[Dict]):
#         n_dataset = sum(map(len, experiments.values()))
#         merged_experiments = dict()
#         with tqdm(total=n_dataset, desc="Progress") as pbar:
#             for e, variants in experiments.items():
#                 merged_variants = dict()
#                 for ω in variants:
#                     pbar.set_description(f"Building datatree for {e}/{ω}")
#                     ds_list = [self._load_zarr(e, v, ω) for v in self.variables]
#                     merged_variants[ω] = xr.merge(ds_list, compat="override")
#                     _ = pbar.update(1)
#                 merged_experiments[e] = merged_variants

#             self.dtree = xr.DataTree.from_dict(merged_experiments, nested=True, name="DayCMIP6Data")
#             if subset is not None:
#                 self.dtree = self.dtree.sel(subset)
#             pbar.set_description("Datatree initialized")

#     def _init_indexmap(self):
#         """Initialize the indexmap for accessing data points.
#         """
#         leaves_lengths = [leaf.ds.sizes["time"] for leaf in self.dtree.leaves]
#         self._cumulative_leaves_lengths = np.cumsum([0] + leaves_lengths)
#         def indexmap(idx):
#             leaf_idx = np.searchsorted(self._cumulative_leaves_lengths, idx, "right") - 1
#             time_idx = idx - self._cumulative_leaves_lengths[leaf_idx]
#             return leaf_idx, time_idx
#         self.indexmap = indexmap

#     def isel(self, *args, **kwargs) -> xr.Dataset:
#         """Perform index-based selection on the dataset.

#         Args:
#             *args: Positional arguments passed to xarray.Dataset.isel().
#             **kwargs: Keyword arguments passed to xarray.Dataset.isel().

#         Returns:
#             xr.Dataset: Selected subset of the dataset.
#         """
#         return self.dtree.isel(*args, **kwargs)

#     def sel(self, *args, **kwargs) -> xr.Dataset:
#         """Perform label-based selection on the dataset.

#         Args:
#             *args: Positional arguments passed to xarray.Dataset.sel().
#             **kwargs: Keyword arguments passed to xarray.Dataset.sel().

#         Returns:
#             xr.Dataset: Selected subset of the dataset.
#         """
#         return self.dtree.sel(*args, **kwargs)

#     def __len__(self) -> int:
#         """Get the number of data points in the dataset.

#         Returns:
#             int: Number of data points.
#         """
#         return self._cumulative_leaves_lengths[-1]

#     def __repr__(self):
#         """String representation of the dataset.

#         Returns:
#             str: String representation of the xarray.DataTree.
#         """
#         return self.dtree.__repr__()

#     def __getitem__(self, idx: Union[str, int]) -> xr.Dataset:
#         """Get item by string or integer indexing.

#         Args:
#             idx: If str: variable name to select.
#                  If int: position in the indexmap to select.

#         Returns:
#             xr.Dataset: Selected data.

#         Raises:
#             ValueError: If idx is not a string or integer.
#         """
#         if isinstance(idx, str):
#             return self.dtree[idx]
#         elif isinstance(idx, int):
#             leaf_idx, time_idx = self.indexmap(idx)
#             return self.dtree.leaves[leaf_idx].ds.isel(time=time_idx)
#         else:
#             raise ValueError(f"Invalid index type: {type(idx)}")

#     @functools.cached_property
#     def paths(self) -> List[str]:
#         """Get list of experiment names.

#         Returns:
#             List[str]: List of leaves paths.
#         """
#         return list([leaf.path for leaf in self.dtree.leaves])
    
#     @functools.cached_property
#     def lat(self) -> np.ndarray:
#         """Get latitude coordinates.

#         Returns:
#             np.ndarray: Array of latitude values.
#         """
#         return self.dtree.leaves[0].lat.values
    
#     @functools.cached_property
#     def lon(self) -> np.ndarray:
#         """Get longitude coordinates.

#         Returns:
#             np.ndarray: Array of longitude values.
#         """
#         return self.dtree.leaves[0].lon.values
    
#     @functools.cached_property
#     def nlat(self) -> int:
#         """Get number of latitude points.

#         Returns:
#             int: Number of latitude points.
#         """
#         return len(self.lat)
    
#     @functools.cached_property
#     def nlon(self) -> int:
#         """Get number of longitude points.

#         Returns:
#             int: Number of longitude points.
#         """
#         return len(self.lon)



# %%
# SECONDS_PER_DAY = 86400
# cmip6  = DayCMIP6Data(
#     root="/home/shahineb/fs06/data/products/cmip6/processed",
#                        model="MPI-ESM1-2-LR",
#                        variables=["tas", "pr", "hurs", "sfcWind"],
#                       experiments={
#                           "piControl": ["r1i1p1f1"],
#                           "ssp126": ["r1i1p1f1", "r2i1p1f1"]
#                       })


# %%
# import time
# start = time.perf_counter()
# for i in range(10):
#     foo = cmip6[i + 1]
#     _ = np.stack([foo[v] for v in cmip6.variables])
# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.4f} seconds")



# %%

# cmip6  = DayCMIP6Data(root="/orcd/data/raffaele/001/shahineb/products/cmip6/processed",
#                       model="MPI-ESM1-2-LR",
#                       variables=["tas", "pr", "hurs", "sfcWind"],
#                       experiments={
#                           "piControl": ["r1i1p1f1"],
#                           "ssp126": ["r1i1p1f1", "r2i1p1f1"]
#                       })


# import time
# start = time.perf_counter()
# for i in range(100):
#     _ = cmip6[i + 1].to_array().values
# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.4f} seconds")


# import xarray as xr
# import zarr


# filepath = "/orcd/home/002/shahineb/data/products/cmip6/processed/MPI-ESM1-2-LR/piControl/r1i1p1f1/tas_anomaly/day/tas_day_MPI-ESM1-2-LR_piControl_r1i1p1f1_daily_anomaly.zarr"
# ds_xr = xr.open_zarr(filepath, consolidated=True, decode_times=False)
# ds_zarr = zarr.open_consolidated(filepath, mode='r')


# start = time.perf_counter()
# for i in range(100):
#     _ = ds_xr["tas"].isel(time=i).values
# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.4f} seconds")  # Elapsed time: 12.3002 seconds



# start = time.perf_counter()
# for i in range(100):
#     _ = ds_zarr["tas"][i, :, :]
# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.4f} seconds")  # Elapsed time: 0.0478 seconds
# %%
