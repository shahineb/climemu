"""Tests for Bouabid2025Emulator core functionality."""

import pytest
import numpy as np
import jax.numpy as jnp
import xarray as xr
from unittest.mock import Mock, patch
from climemu.emulators.bouabid2025 import Bouabid2025Emulator


class TestBouabid2025Emulator:
    """Test cases for the Bouabid2025Emulator class."""

    def test_initialization(self):
        """Test Bouabid2025Emulator initialization."""
        emulator = Bouabid2025Emulator("test_esm")
        
        assert emulator.esm == "test_esm"
        assert emulator.repo_id == "shahineb/climemu"

    def test_load_method_raises_error_without_files(self):
        """Test that load method raises error when files don't exist."""
        emulator = Bouabid2025Emulator("test_esm")
        
        # This should raise an error since the files don't exist
        with pytest.raises(Exception):  # Could be FileNotFoundError, HTTPError, etc.
            emulator.load()

    def test_call_method_without_load_raises_error(self):
        """Test that __call__ fails if load hasn't been called first."""
        emulator = Bouabid2025Emulator("test_esm")
        
        with pytest.raises(AttributeError):
            emulator(gmst=2.0, month=3)

    def test_call_method_without_compile_raises_error(self):
        """Test that __call__ fails if compile hasn't been called first."""
        emulator = Bouabid2025Emulator("test_esm")
        emulator.β = jnp.ones((12, 9, 2))
        
        # Mock climatology properly
        mock_climatology = Mock()
        mock_climatology.__getitem__ = Mock(return_value=Mock(values=np.array([1, 2, 3])))
        emulator.climatology = mock_climatology
        
        with pytest.raises(AttributeError):
            emulator(gmst=2.0, month=3)

    def test_properties_after_load(self):
        """Test properties after loading."""
        emulator = Bouabid2025Emulator("test_esm")
        
        # Mock climatology properly
        mock_climatology = Mock()
        mock_climatology.__getitem__ = Mock(side_effect=lambda key: Mock(values=np.linspace(-90, 90, 10) if key == 'lat' else np.linspace(0, 360, 20)))
        mock_climatology.data_vars = ['tas', 'pr', 'hurs', 'sfcWind']
        emulator.climatology = mock_climatology
        
        # Test properties
        assert np.array_equal(emulator.lat, np.linspace(-90, 90, 10))
        assert np.array_equal(emulator.lon, np.linspace(0, 360, 20))
        assert emulator.vars == ['tas', 'pr', 'hurs', 'sfcWind']

    @patch('climemu.emulators.bouabid2025.Bouabid2025Emulator._load_precursor')
    @patch('climemu.emulators.bouabid2025.Bouabid2025Emulator._load_climatology')
    @patch('climemu.emulators.bouabid2025.Bouabid2025Emulator._load_pattern_scaling')
    def test_load_method_sets_files_dir_correctly(self, mock_pattern_scaling, mock_climatology, mock_precursor):
        """Test that load method sets files_dir correctly using mocked data."""
        emulator = Bouabid2025Emulator("test_esm")
        
        # Mock the internal methods
        mock_pattern_scaling.return_value = Mock()
        mock_climatology.return_value = Mock()
        mock_precursor.return_value = Mock()
        
        # Call load with default which
        emulator.load()
        assert emulator.files_dir == "test_esm/default"
        
        # Call load with custom which
        emulator.load(which="paper")
        assert emulator.files_dir == "test_esm/paper"
