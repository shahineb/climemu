import os
from typing import List, Optional, Dict, Union
import functools
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from dask.diagnostics import ProgressBar
from .constants import SECONDS_PER_DAY, TABLE_ID


class CMIP6Data(Dataset):
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
    def get_file_path(root: str, model: str, experiment: str, variable: str, 
                      variant: str, table_id: str = TABLE_ID) -> str:
        """Construct the file path for a given model configuration.

        Args:
            root: Root directory.
            model: Model name.
            experiment: Experiment name.
            variable: Variable name.
            variant: Variant identifier.
            table_id: Table identifier (default: TABLE_ID).

        Returns:
            str: Constructed file path.
        """
        filename = f"{variable}_{table_id}_{model}_{experiment}_{variant}_monthly_anomaly.nc"
        path = os.path.join(root, model, experiment, variant, f"{variable}_anomaly", table_id, filename)
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
        for exp in experiments:
            ds_list = [self._load_xarray(exp, var) for var in self.variables]
            common_members = functools.reduce(np.intersect1d, [ds.member.values for ds in ds_list])
            ds_list = [ds.sel(member=common_members) for ds in ds_list]
            merged_experiments[exp] = xr.merge(ds_list)

        self.dtree = xr.DataTree.from_dict(merged_experiments, name="CMIP6Data")
        if subset is not None:
            self.dtree = self.dtree.sel(subset)

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
        for i, e in enumerate(self.experiments):
            da = self.dtree[e].ds.to_array()
            valid = ~da.isnull().any(dim=["lat", "lon", "variable"])
            argvalid = np.argwhere(valid.values)
            argvalid = np.column_stack((np.full(len(argvalid), i), argvalid))
            indexmap.append(argvalid)
        self.indexmap = np.concatenate(indexmap)

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