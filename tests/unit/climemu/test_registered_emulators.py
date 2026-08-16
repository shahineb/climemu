"""Tests for emulator registration, factory function, and user-facing input validation."""

import pytest
from unittest.mock import Mock
from climemu import build_emulator, EMULATORS
from climemu.emulators.bouabid2026_monthly import Bouabid2026MonthlyEmulator
from climemu.emulators.bouabid2026_daily import Bouabid2026DailyEmulator


# All ESM/frequency pairs that should be registered
MONTHLY_ESMS = ["MPI-ESM1-2-LR", "MIROC6", "ACCESS-ESM1-5", "CanESM5", "IPSL-CM6A-LR"]
DAILY_ESMS = ["MPI-ESM1-2-LR"]


class TestRegistration:
    @pytest.mark.parametrize("esm", MONTHLY_ESMS)
    def test_monthly_esm_registered(self, esm):
        assert (esm, "monthly") in EMULATORS

    @pytest.mark.parametrize("esm", DAILY_ESMS)
    def test_daily_esm_registered(self, esm):
        assert (esm, "daily") in EMULATORS

    @pytest.mark.parametrize("esm", MONTHLY_ESMS)
    def test_build_monthly_returns_correct_type(self, esm):
        emulator = build_emulator(esm)
        assert isinstance(emulator, Bouabid2026MonthlyEmulator)
        assert emulator.esm == esm

    @pytest.mark.parametrize("esm", DAILY_ESMS)
    def test_build_daily_returns_correct_type(self, esm):
        emulator = build_emulator(esm, frequency="daily")
        assert isinstance(emulator, Bouabid2026DailyEmulator)
        assert emulator.esm == esm


class TestBuildEmulatorErrors:
    def test_unknown_esm_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_emulator("NonexistentESM")

    def test_wrong_frequency_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_emulator("MPI-ESM1-2-LR", frequency="hourly")

    def test_daily_for_monthly_only_esm_raises_keyerror(self):
        """MIROC6 only has monthly — requesting daily should fail."""
        with pytest.raises(KeyError):
            build_emulator("MIROC6", frequency="daily")


class TestVariableSubsetting:
    """Test the variables= kwarg that users pass at build time."""

    def _make_loaded_emulator(self, variables=None):
        """Create an emulator with mocked load, to test variable subsetting."""
        emulator = Bouabid2026MonthlyEmulator("test_esm", variables=variables)
        # Mock a climatology with known variables
        mock_clim = Mock()
        mock_clim.data_vars = ["tas", "pr", "hurs", "sfcWind"]
        mock_clim.__getitem__ = Mock(return_value=mock_clim)
        emulator.climatology = mock_clim
        emulator._resolve_variables()
        return emulator

    def test_all_variables_by_default(self):
        emulator = self._make_loaded_emulator()
        assert emulator.vars == ["tas", "pr", "hurs", "sfcWind"]
        assert emulator._var_idx == [0, 1, 2, 3]

    def test_subset_variables(self):
        emulator = self._make_loaded_emulator(variables=["tas", "pr"])
        assert emulator.vars == ["tas", "pr"]
        assert emulator._var_idx == [0, 1]

    def test_single_variable(self):
        emulator = self._make_loaded_emulator(variables=["pr"])
        assert emulator.vars == ["pr"]
        assert emulator._var_idx == [1]

    def test_invalid_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variables"):
            self._make_loaded_emulator(variables=["tas", "fake_var"])

    def test_variables_kwarg_forwarded_by_build(self):
        emulator = build_emulator("MPI-ESM1-2-LR", variables=["tas"])
        assert emulator._vars == ["tas"]
