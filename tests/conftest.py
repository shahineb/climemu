"""Shared test fixtures and configuration for climemu tests."""

import pytest
import numpy as np
import jax.numpy as jnp
import xarray as xr
from unittest.mock import Mock, patch


@pytest.fixture
def sample_climate_data():
    """Create sample climate data for testing."""
    # Create a small synthetic climate dataset
    lat = np.linspace(-90, 90, 10)  # 10 latitude points
    lon = np.linspace(0, 360, 20)   # 20 longitude points
    time = np.arange(12)  # 12 months
    
    # Create synthetic data
    tas = np.random.randn(12, 10, 20) + 15  # Temperature in °C
    pr = np.random.randn(12, 10, 20) + 2    # Precipitation in mm/day
    hurs = np.random.randn(12, 10, 20) + 70 # Relative humidity in %
    sfcWind = np.random.randn(12, 10, 20) + 5 # Wind speed in m/s
    
    ds = xr.Dataset({
        'tas': (('time', 'lat', 'lon'), tas),
        'pr': (('time', 'lat', 'lon'), pr),
        'hurs': (('time', 'lat', 'lon'), hurs),
        'sfcWind': (('time', 'lat', 'lon'), sfcWind),
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon,
    })
    
    return ds


@pytest.fixture
def sample_pattern_scaling_coeffs():
    """Create sample pattern scaling coefficients."""
    # β has shape (12, nlat*nlon, 2) where 2 is for β₀ and β₁
    nlat, nlon = 10, 20
    β = np.random.randn(12, nlat * nlon, 2)
    return jnp.array(β)


@pytest.fixture
def mock_hf_download():
    """Mock HuggingFace Hub download function."""
    with patch('huggingface_hub.hf_hub_download') as mock:
        # Create temporary files for testing
        mock.return_value = "/tmp/test_file.npy"
        yield mock


@pytest.fixture
def mock_xarray_open():
    """Mock xarray.open_dataset for testing."""
    with patch('xarray.open_dataset') as mock:
        # Return a mock dataset
        mock_ds = Mock()
        mock_ds.lat.values = np.linspace(-90, 90, 10)
        mock_ds.lon.values = np.linspace(0, 360, 20)
        mock_ds.data_vars = ['tas', 'pr', 'hurs', 'sfcWind']
        mock.return_value = mock_ds
        yield mock


@pytest.fixture
def mock_jax_load():
    """Mock jax.numpy.load for testing."""
    with patch('jax.numpy.load') as mock:
        # Return mock arrays
        mock_array = jnp.array([1.0, 2.0, 3.0])
        mock.return_value = mock_array
        yield mock
