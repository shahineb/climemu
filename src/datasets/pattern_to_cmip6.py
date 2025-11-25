from typing import Optional, Union, Tuple, List
import numpy as np
import xarray as xr
import einops
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from src.utils import arrays
from src.datasets import AmonCMIP6Data, DayCMIP6Data
# from .cmip6 import AmonCMIP6Data, DayCMIP6Data



class PatternToAmonCMIP6Data(Dataset):
    """Dataset for CMIP6 pattern scaling using linear regression.

    The pattern scaling equation is:
        local_temperature_change = β₀ + β₁ * global_temperature

    where:
    - β₀ is the intercept (base climate)
    - β₁ is the scaling factor (rate of local temperature change per degree of global warming)

    Features:
    - Monthly pattern scaling models for surface temperature
    - Ensemble mean computation
    - Save/load functionality for scaling parameters
    - Prediction of spatial temperature patterns
    """

    def __init__(self, gmst: xr.DataTree, cmip6data: AmonCMIP6Data, 
                 β: Optional[np.ndarray] = None, in_memory: bool = False):
        """Initialize the pattern scaling dataset.

        Args:
            gmst: Global Mean Surface Temperature data tree.
            cmip6data: CMIP6 dataset containing climate variables.
            β: Pre-computed pattern scaling parameters (shape: months, spatial_points, 2).
            in_memory: Whether to load data into memory.
        """        
        self.cmip6data = cmip6data
        self.gmst = gmst
        self.β = β
        self.in_memory = in_memory
        if self.in_memory:
            self.cmip6data.load()

    def fit(self, experiments: Optional[List[str]] = None) -> None:
        """Fit pattern scaling models for each month.

        Args:
            experiments: Optional list of experiments to fit over. If None, uses all experiments.
        """
        # Compute surface temperature ensemble mean
        dt_tas = self.cmip6data.dtree.map_over_datasets(arrays.filter_var('tas')).mean("member")
        if not self.in_memory:
            with ProgressBar():
                dt_tas = dt_tas.compute()

        # Group GMST and surface temperature by month
        dt_gmst_by_month = self.gmst.map_over_datasets(arrays.groupby_month_and_year)
        dt_ensemble_mean_by_month = dt_tas.map_over_datasets(arrays.groupby_month_and_year)

        # Setup list of experiments to fit over
        experiments = self.cmip6data.experiments if experiments is None else experiments

        # Fit linear regression models for each month
        β1: List[np.ndarray] = []
        β0: List[np.ndarray] = []

        for month in tqdm(range(1, 13), desc="Fitting monthly patterns"):
            gmst_month = dt_gmst_by_month.sel(month=month)
            ensemble_mean_month = dt_ensemble_mean_by_month.sel(month=month)

            # Extract arrays from datatree
            X = []
            Y = []
            for e in experiments:
                gmst = gmst_month[e].tas.values
                tas = ensemble_mean_month[e].tas.values
                X.append(gmst)
                Y.append(tas)
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=-1)

            # Reshape for sklearn
            X = X.reshape(-1, 1)
            Y = einops.rearrange(Y, 'lat lon t -> t (lat lon)')

            # Fit linear regression model
            lm = LinearRegression()
            lm.fit(X, Y)

            # Extract coefficients
            β1_month = lm.coef_.squeeze()
            β0_month = lm.intercept_.squeeze()
            β1.append(β1_month)
            β0.append(β0_month)

        # Stack coefficients into final β array
        β1 = np.stack(β1, axis=0)
        β0 = np.stack(β0, axis=0)
        self.β = np.stack([β0, β1], axis=-1)

    def save_pattern_scaling(self, path: str) -> None:
        """Save the pattern scaling model to a file.

        Args:
            path: Path to save the model parameters.
        """
        np.save(path, self.β)

    def load_pattern_scaling(self, path: str) -> None:
        """Load the pattern scaling model from a file.

        Args:
            path: Path to load the model parameters from.
        """
        self.β = np.load(path)

    def predict_single_pattern(self, gmst: float, month: int) -> np.ndarray:
        """Predict spatial temperature pattern for a given month and GMST.

        Args:
            gmst: Global Mean Surface Temperature value.
            month: Month number (1-12).

        Returns:
            np.ndarray: Predicted spatial temperature pattern with shape (lat, lon).
        """
        β = self.β[month - 1]
        pattern = β[:, 0] + β[:, 1] * gmst
        return pattern.reshape(self.cmip6data.nlat, self.cmip6data.nlon)

    def isel(self, time: int, experiment: int, member: int, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Select data point and return pattern scaling components.

        Args:
            time: Time index.
            experiment: Experiment index.
            member: Ensemble member index.
            *args: Additional positional arguments passed to cmip6data.isel().
            **kwargs: Additional keyword arguments passed to cmip6data.isel().

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted temperature pattern and actual values for the selected point.
        """
        e = self.cmip6data.experiments[experiment]
        selected_data = self.cmip6data[e].isel(time=time, member=member, *args, **kwargs)
        month = int(selected_data.time.dt.month.item())
        actual_values = selected_data.ds.to_array().values
        gmst_values = self.gmst[e].isel(time=time).ds.tas.values.squeeze()
        pattern = self.predict_single_pattern(gmst_values, month)
        return pattern, actual_values

    def __len__(self) -> int:
        """Get the number of valid data points in the dataset.

        Returns:
            int: Number of time steps.
        """
        return len(self.cmip6data)

    def __getitem__(self, idx: Union[str, int]) -> Union[Dataset, Tuple[np.ndarray, np.ndarray]]:
        """Get item by string or integer indexing.

        Args:
            idx: If str: variable name to select.
                 If int: position in the indexmap to select.

        Returns:
            Union[Dataset, Tuple[np.ndarray, np.ndarray]]: If idx is str: returns the CMIP6 dataset for that variable.
            If idx is int: returns a tuple of (predicted_pattern, actual_values).

        Raises:
            ValueError: If idx is not a string or integer.
        """
        if isinstance(idx, str):
            return self.cmip6data[idx]
        elif isinstance(idx, int):
            e, ω, t = self.cmip6data.indexmap[idx]
            return self.isel(experiment=e, member=ω, time=t)
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def __repr__(self) -> str:
        """Get string representation of the dataset.

        Returns:
            str: String representation.
        """
        return f"PatternScalingtoAmonCMIP6Data(\n gmst={self.gmst}, \n cmip6data={self.cmip6data}\n)"
    



class PatternToDayCMIP6Data(Dataset):
    def __init__(self, gmst: xr.DataTree, cmip6data: DayCMIP6Data, β: Optional[np.ndarray] = None):
        self.cmip6data = cmip6data
        self.gmst = gmst
        self.β = β

    def fit(self, experiments, ensemble_mean_tas):
        # Extract arrays from datatree
        print("Fitting pattern scaling parameters...")
        X = []
        Y = []
        for e in experiments:
            gmst = self.gmst[e].tas.values
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
        self.β = np.stack([β0, β1], axis=-1)

    def save_pattern_scaling(self, path: str) -> None:
        np.save(path, self.β)

    def load_pattern_scaling(self, path: str) -> None:
        self.β = np.load(path)

    def predict_single_pattern(self, gmst: float) -> np.ndarray:
        pattern = self.β[:, 0] + self.β[:, 1] * gmst
        return pattern.reshape(self.cmip6data.nlat, self.cmip6data.nlon)
    
    def __len__(self) -> int:
        return len(self.cmip6data)

    def __getitem__(self, idx: Union[str, int]) -> Union[Dataset, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(idx, str):
            return self.cmip6data[idx]
        elif isinstance(idx, int):
            # Get the selected data slice
            cmip6_slice = self.cmip6data[idx]
            year = cmip6_slice.time.dt.year.item()
            doy = cmip6_slice.time.dt.dayofyear.item()

            # Get corresponding gmst
            leaf_idx, _ = self.cmip6data.indexmap(idx)
            experiment = self.cmip6data.dtree.leaves[leaf_idx].parent.name
            gmst = self.gmst[experiment].sel(year=year).tas.item()

            # Return pattern scaling and data array
            pattern = self.predict_single_pattern(gmst)
            cmip6_array = cmip6_slice[self.cmip6data.variables].to_array().values
            return doy, pattern, cmip6_array
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")
        
    def __repr__(self) -> str:
        """Get string representation of the dataset.

        Returns:
            str: String representation.
        """
        return f"PatternScalingtoDayCMIP6Data(\n gmst={self.gmst}, \n cmip6data={self.cmip6data}\n)"
    


###
import numpy as np
gmst = xr.open_datatree("experiments/mpi/cache/gmsttrain.nc")
cmip6  = DayCMIP6Data(root="/orcd/data/raffaele/001/shahineb/products/cmip6/processed",
                      model="MPI-ESM1-2-LR",
                      variables=["tas", "pr", "hurs", "sfcWind"],
                      experiments={
                          "piControl": ["r1i1p1f1"],
                          "ssp126": ["r1i1p1f1", "r2i1p1f1"]
                      })
β = np.load("experiments/mpi/cache/β.npy")

dataset = PatternToDayCMIP6Data(gmst, cmip6, β)


import time
start = time.perf_counter()
for i in range(100):
    # Get the selected data slice
    cmip6_slice = dataset.cmip6data[i]
    # cmip6_array = cmip6_slice.to_array().values
    cmip6_array = cmip6_slice['tas'].values
    cmip6_array = cmip6_slice['pr'].values
    cmip6_array = cmip6_slice['hurs'].values
    cmip6_array = cmip6_slice['sfcWind'].values
    # _ = dataset[i + 1]
end = time.perf_counter()
print(f"Elapsed time: {end - start:.4f} seconds")