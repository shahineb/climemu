"""Tests for abstract emulator classes."""

import pytest
import numpy as np
from climemu.emulators.abstractemulator import AbstractEmulator, GriddedEmulator


class TestAbstractEmulator:
    """Test cases for the AbstractEmulator class."""

    def test_abstract_emulator_is_abstract(self):
        """Test that AbstractEmulator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractEmulator()

    def test_abstract_emulator_build_classmethod(self):
        """Test the build class method."""
        class ConcreteEmulator(AbstractEmulator):
            def __init__(self, value="test"):
                self.value = value
            
            def __call__(self, *args, **kwargs):
                return f"called with {self.value}"
        
        # Test build method
        emulator = ConcreteEmulator.build(value="custom")
        assert isinstance(emulator, ConcreteEmulator)
        assert emulator.value == "custom"
        assert emulator() == "called with custom"


class TestGriddedEmulator:
    """Test cases for the GriddedEmulator class."""

    def test_gridded_emulator_is_abstract(self):
        """Test that GriddedEmulator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GriddedEmulator()

    def test_gridded_emulator_concrete_properties(self):
        """Test the concrete properties that depend on abstract ones."""
        class ConcreteGriddedEmulator(GriddedEmulator):
            def __init__(self, lat_coords, lon_coords, var_list):
                self._lat = lat_coords
                self._lon = lon_coords
                self._variables = var_list
            
            @property
            def lat(self):
                return self._lat
            
            @property
            def lon(self):
                return self._lon
            
            @property
            def vars(self):
                return self._variables
            
            def __call__(self, *args, **kwargs):
                return "concrete_emulator"
        
        # Test with sample data
        lat_coords = np.linspace(-90, 90, 10)
        lon_coords = np.linspace(0, 360, 20)
        var_list = ['tas', 'pr', 'hurs', 'sfcWind']
        
        emulator = ConcreteGriddedEmulator(lat_coords, lon_coords, var_list)
        
        # Test abstract properties
        assert np.array_equal(emulator.lat, lat_coords)
        assert np.array_equal(emulator.lon, lon_coords)
        assert emulator.vars == var_list
        
        # Test concrete properties
        assert emulator.nlat == len(lat_coords)
        assert emulator.nlon == len(lon_coords)
        assert emulator.nvar == len(var_list)
