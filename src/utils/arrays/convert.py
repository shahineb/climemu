import numpy as np
import xarray as xr


def xarray_like(numpy_array, example_xr_dataarray):
    """
    Create a new DataArray from a numpy array with the same dimensions and coordinates as an existing DataArray,
    """
    # Ensure the input is a NumPy array
    numpy_array = np.asarray(numpy_array)

    # Check if the shape matches
    if numpy_array.shape != example_xr_dataarray.shape:
        raise ValueError(f"Shape mismatch: array shape {numpy_array.shape} does not match "
                         f"xr_dataarray shape {example_xr_dataarray.shape}.")

    # Create the new DataArray
    field = xr.DataArray(
        data=numpy_array,
        dims=example_xr_dataarray.dims,
        coords=example_xr_dataarray.coords,
    )
    return field