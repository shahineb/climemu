"""Unit tests for emulator lifecycle — load/compile ordering and path construction."""

import pytest
import jax.numpy as jnp
from unittest.mock import Mock, patch
from earthsampler.emulators.bouabid2026_monthly import Bouabid2026MonthlyEmulator


class TestEmulatorLifecycle:
    def test_call_without_load_raises(self):
        emulator = Bouabid2026MonthlyEmulator("test_esm")
        with pytest.raises(AttributeError):
            emulator(gmst=2.0, month=3)

    def test_call_without_compile_raises(self):
        emulator = Bouabid2026MonthlyEmulator("test_esm")
        emulator.β = jnp.ones((12, 9, 2))
        mock_clim = Mock()
        mock_clim.__getitem__ = Mock(return_value=Mock(values=jnp.array([1, 2, 3])))
        emulator.climatology = mock_clim
        with pytest.raises(AttributeError):
            emulator(gmst=2.0, month=3)

    @patch('earthsampler.emulators.bouabid2026_monthly.Bouabid2026MonthlyEmulator._load_precursor')
    @patch('earthsampler.emulators.bouabid2026_monthly.Bouabid2026MonthlyEmulator._load_climatology')
    @patch('earthsampler.emulators.bouabid2026_monthly.Bouabid2026MonthlyEmulator._load_pattern_scaling')
    def test_load_sets_files_dir(self, mock_ps, mock_clim, mock_precursor):
        mock_ps.return_value = Mock()
        mock_ds = Mock()
        mock_ds.data_vars = ['tas', 'pr', 'hurs', 'sfcWind']
        mock_clim.return_value = mock_ds
        mock_precursor.return_value = Mock()

        emulator = Bouabid2026MonthlyEmulator("test_esm")

        emulator.load()
        assert emulator.files_dir == "test_esm/monthly/default"

        emulator.load(which="paper")
        assert emulator.files_dir == "test_esm/monthly/paper"
