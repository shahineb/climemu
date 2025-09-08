from functools import singledispatch
import numpy as np
import xarray as xr


@singledispatch
def global_mean(data, *args, **kwargs):
    raise TypeError("Unsupported data type. Only xarray.DataArray and numpy.ndarray are supported.")


@global_mean.register
def _global_mean_datarray(data: xr.DataArray, lat_dim='lat', lon_dim='lon'):
    """
    Computes the global mean of a gridded DataArray
    """
    try:
        wlat = np.cos(np.deg2rad(data[lat_dim]))
        return data.weighted(wlat).mean((lat_dim, lon_dim))
    except (AttributeError, KeyError):
        return None
    

@global_mean.register
def _global_mean_dataset(data: xr.Dataset, lat_dim='lat', lon_dim='lon'):
    """
    Computes the global mean of a gridded DataArray
    """
    try:
        wlat = np.cos(np.deg2rad(data[lat_dim]))
        return data.weighted(wlat).mean((lat_dim, lon_dim))
    except (AttributeError, KeyError):
        return None


@global_mean.register
def _global_mean_numpy(data: np.ndarray, lat, lat_dim=0, lon_dim=1):
    """
    Computes the global mean of a gridded numpy array with latitude as the first dimension
    """
    # Compute latitude weights
    wlat = np.cos(np.deg2rad(lat))
    wlat /= wlat.sum()

    # Expand weights to match the data shape (broadcast)
    shape = [1] * data.ndim
    shape[lat_dim] = len(lat) 
    wlat = wlat.reshape(shape)

    # Apply weights and compute global mean
    weighted_sum = np.sum(data * wlat, axis=(lat_dim, lon_dim))
    global_mean = weighted_sum / data.shape[lon_dim]
    return global_mean