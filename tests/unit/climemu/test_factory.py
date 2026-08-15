"""Tests for the factory function in climemu.__init__."""

import pytest
from unittest.mock import patch, Mock
from climemu import build_emulator


class TestBuildEmulator:
    """Test cases for the build_emulator factory function."""

    def test_build_emulator_with_valid_name(self):
        """Test build_emulator with a valid registered emulator name."""
        mock_emulator_class = Mock()
        mock_emulator_instance = Mock()
        mock_emulator_class.return_value = mock_emulator_instance

        with patch('climemu.EMULATORS', {('MPI-ESM1-2-LR', 'monthly'): mock_emulator_class}):
            result = build_emulator('MPI-ESM1-2-LR')

            mock_emulator_class.assert_called_once()
            assert result == mock_emulator_instance

    def test_build_emulator_with_frequency(self):
        """Test build_emulator with explicit frequency parameter."""
        mock_monthly = Mock(return_value=Mock(name="monthly"))
        mock_daily = Mock(return_value=Mock(name="daily"))

        registry = {
            ('MPI-ESM1-2-LR', 'monthly'): mock_monthly,
            ('MPI-ESM1-2-LR', 'daily'): mock_daily,
        }
        with patch('climemu.EMULATORS', registry):
            monthly = build_emulator('MPI-ESM1-2-LR', frequency='monthly')
            mock_monthly.assert_called_once()

            daily = build_emulator('MPI-ESM1-2-LR', frequency='daily')
            mock_daily.assert_called_once()

    def test_build_emulator_with_keyerror(self):
        """Test build_emulator with an unregistered emulator name."""
        with patch('climemu.EMULATORS', {}):
            with pytest.raises(KeyError):
                build_emulator('nonexistent_emulator')

    def test_build_emulator_preserves_emulator_type(self):
        """Test that build_emulator returns the correct emulator type."""
        class TestEmulator:
            def __init__(self):
                self.name = "test_emulator"

        with patch('climemu.EMULATORS', {('test', 'monthly'): TestEmulator}):
            result = build_emulator('test')
            assert isinstance(result, TestEmulator)
            assert result.name == "test_emulator"
