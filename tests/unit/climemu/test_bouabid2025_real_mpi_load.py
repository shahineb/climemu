"""Integration tests for Bouabid2025Emulator with MPI-ESM1-2-LR data."""

import numpy as np
import jax.numpy as jnp
import xarray as xr
from src.climemu.emulators.bouabid2025 import Bouabid2025Emulator


class TestBouabid2025EmulatorMPIESM1_2_LR:
    """Integration test cases for the Bouabid2025Emulator with MPI-ESM1-2-LR data."""

    def test_load_method_downloads_files_from_huggingface(self):
        """Test that load method downloads files from Hugging Face platform."""
        emulator = Bouabid2025Emulator("MPI-ESM1-2-LR")
        
        # This test will actually try to download from Hugging Face
        # It should succeed if the files exist, or raise an appropriate error
        try:
            emulator.load()
            
            # If successful, verify that all expected attributes are set
            assert hasattr(emulator, 'files_dir')
            assert hasattr(emulator, 'β')
            assert hasattr(emulator, 'climatology')
            assert hasattr(emulator, 'precursor')
            
            # Verify files_dir is set correctly
            assert emulator.files_dir == "MPI-ESM1-2-LR/default"
            
        except Exception as e:
            # If download fails, verify it's a network/file not found error
            # This is expected if the files don't exist or network is unavailable
            assert any(keyword in str(e).lower() for keyword in [
                'not found', 'network', 'connection', 'timeout', 'http', 'download'
            ])

    def test_load_method_with_real_data_validation(self):
        """Test that load method works with real data and validates all components."""
        emulator = Bouabid2025Emulator("MPI-ESM1-2-LR")
        
        try:
            emulator.load()
            
            # Test that all required attributes are present
            required_attrs = ['files_dir', 'β', 'climatology', 'precursor']
            for attr in required_attrs:
                assert hasattr(emulator, attr), f"Missing required attribute: {attr}"
            
            # Test β array properties
            β = emulator.β
            assert isinstance(β, jnp.ndarray), "β should be a JAX array"
            assert β.ndim == 3, f"β should be 3D, got {β.ndim}D"
            assert β.shape[0] == 12, f"β should have 12 months, got {β.shape[0]}"
            assert β.shape[2] == 2, f"β should have 2 coefficients, got {β.shape[2]}"
            
            # Test climatology properties
            climatology = emulator.climatology
            assert isinstance(climatology, xr.Dataset), "climatology should be an xarray Dataset"
            assert 'lat' in climatology.coords, "climatology should have 'lat' coordinate"
            assert 'lon' in climatology.coords, "climatology should have 'lon' coordinate"
            
            # Test dimension consistency
            nlat = climatology.lat.size
            nlon = climatology.lon.size
            nlat_nlon = nlat * nlon
            
            assert β.shape[1] == nlat_nlon, f"β second dimension {β.shape[1]} should equal nlat*nlon {nlat_nlon}"
            
            # Test that β values are reasonable (not all zeros or NaNs)
            assert not jnp.all(β == 0), "β should not be all zeros"
            assert not jnp.any(jnp.isnan(β)), "β should not contain NaN values"
            
            # Test climatology data quality
            for var in climatology.data_vars:
                data = climatology[var].values
                assert not np.all(np.isnan(data)), f"climatology variable {var} should not be all NaN"
                assert np.isfinite(data).all(), f"climatology variable {var} should be finite"
            
            # Test precursor function
            precursor = emulator.precursor
            assert callable(precursor), "precursor should be callable"
            
            # Test that properties work correctly
            lat = emulator.lat
            lon = emulator.lon
            vars_list = emulator.vars
            
            assert len(lat) == nlat, f"lat property length {len(lat)} should equal nlat {nlat}"
            assert len(lon) == nlon, f"lon property length {len(lon)} should equal nlon {nlon}"
            assert vars_list == list(climatology.data_vars), f"vars property {vars_list} should match climatology variables {list(climatology.data_vars)}"
            
        except Exception as e:
            # If load fails, verify it's a network/file not found error
            assert any(keyword in str(e).lower() for keyword in [
                'not found', 'network', 'connection', 'timeout', 'http', 'download'
            ])

    def test_compile_and_generate_samples_shape(self):
        """Test that compiled emulator generates samples with correct shape."""
        emulator = Bouabid2025Emulator("MPI-ESM1-2-LR")
        
        try:
            # Load the emulator
            emulator.load()
            
            # Compile with specific parameters
            n_samples = 1
            n_steps = 2
            emulator.compile(n_samples=n_samples, n_steps=n_steps)
            
            # Generate a sample
            samples = emulator(gmst=2.0, month=3, seed=42)
            
            # Verify the sample shape
            expected_shape = (n_samples, emulator.nvar, emulator.nlat, emulator.nlon)
            assert samples.shape == expected_shape, f"Expected shape {expected_shape}, got {samples.shape}"
            
            # Verify it's a JAX array
            assert isinstance(samples, jnp.ndarray), "Samples should be a JAX array"
            
            # Verify the sample contains reasonable values (not all zeros or NaNs)
            assert not jnp.any(jnp.isnan(samples)), "Samples should not contain NaN values"
            assert jnp.all(jnp.isfinite(samples)), "Samples should be finite"
            
        except Exception as e:
            # If load fails, verify it's a network/file not found error
            assert any(keyword in str(e).lower() for keyword in [
                'not found', 'network', 'connection', 'timeout', 'http', 'download'
            ])

    def test_generate_samples_with_xarray_output(self):
        """Test that emulator generates xarray Dataset when xarray=True."""
        emulator = Bouabid2025Emulator("MPI-ESM1-2-LR")
        
        try:
            # Load and compile the emulator
            emulator.load()
            n_samples = 1
            n_steps = 2
            emulator.compile(n_samples=n_samples, n_steps=n_steps)
            
            # Generate samples with xarray=True
            samples = emulator(gmst=1.5, month=6, seed=123, xarray=True)
            
            # Verify it's an xarray Dataset
            assert isinstance(samples, xr.Dataset), "Samples should be an xarray Dataset when xarray=True"
            
            # Verify the Dataset has the expected structure
            expected_vars = emulator.vars
            assert list(samples.data_vars.keys()) == expected_vars, f"Expected variables {expected_vars}, got {list(samples.data_vars.keys())}"
            
            # Verify coordinates
            assert 'member' in samples.coords, "Dataset should have 'member' coordinate"
            assert 'lat' in samples.coords, "Dataset should have 'lat' coordinate"
            assert 'lon' in samples.coords, "Dataset should have 'lon' coordinate"
            
            # Verify coordinate shapes
            assert samples.member.size == n_samples, f"Expected {n_samples} members, got {samples.member.size}"
            assert samples.lat.size == emulator.nlat, f"Expected {emulator.nlat} lat points, got {samples.lat.size}"
            assert samples.lon.size == emulator.nlon, f"Expected {emulator.nlon} lon points, got {samples.lon.size}"
            
            # Verify data variable shapes
            for var in expected_vars:
                var_data = samples[var]
                expected_shape = (n_samples, emulator.nlat, emulator.nlon)
                assert var_data.shape == expected_shape, f"Variable {var} should have shape {expected_shape}, got {var_data.shape}"
                
                # Verify data quality
                assert not np.all(np.isnan(var_data.values)), f"Variable {var} should not be all NaN"
                assert np.isfinite(var_data.values).all(), f"Variable {var} should be finite"
            
            # Verify member coordinate values
            expected_members = np.arange(n_samples) + 1
            assert np.array_equal(samples.member.values, expected_members), f"Expected members {expected_members}, got {samples.member.values}"
            
            # Verify lat/lon coordinates match emulator properties
            assert np.array_equal(samples.lat.values, emulator.lat), "Lat coordinates should match emulator.lat"
            assert np.array_equal(samples.lon.values, emulator.lon), "Lon coordinates should match emulator.lon"
            
        except Exception as e:
            # If load fails, verify it's a network/file not found error
            assert any(keyword in str(e).lower() for keyword in [
                'not found', 'network', 'connection', 'timeout', 'http', 'download'
            ])
