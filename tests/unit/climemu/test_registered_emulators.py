"""Tests for registered emulator classes."""

from climemu.emulators.bouabid2026_monthly import (
    Bouabid2026MonthlyEmulator,
    MPIMonthlyEmulator,
    MIROCMonthlyEmulator,
    ACCESSMonthlyEmulator
)
from climemu.emulators.bouabid2026_daily import (
    Bouabid2026DailyEmulator,
    MPIDailyEmulator
)


class TestBouabid2026MonthlyEmulator:
    """Test cases for the Bouabid2026MonthlyEmulator class."""

    def test_bouabid2026_emulator_initialization(self):
        """Test Bouabid2026MonthlyEmulator initialization."""
        emulator = Bouabid2026MonthlyEmulator("test_esm")
        
        assert emulator.esm == "test_esm"
        assert emulator.repo_id == "shahineb/climemu"

    def test_bouabid2026_emulator_initialization_default_which(self):
        """Test Bouabid2026MonthlyEmulator initialization with default which parameter."""
        emulator = Bouabid2026MonthlyEmulator("test_esm")
        
        assert emulator.esm == "test_esm"
        assert emulator.repo_id == "shahineb/climemu"


class TestMPIMonthlyEmulator:
    """Test cases for the MPIMonthlyEmulator class."""

    def test_mpi_emulator_initialization(self):
        """Test MPIMonthlyEmulator initialization."""
        emulator = MPIMonthlyEmulator()
        
        assert emulator.esm == "MPI-ESM1-2-LR"
        assert emulator.repo_id == "shahineb/climemu"

    def test_mpi_emulator_inheritance(self):
        """Test that MPIMonthlyEmulator inherits from Bouabid2026MonthlyEmulator."""
        emulator = MPIMonthlyEmulator()
        assert isinstance(emulator, Bouabid2026MonthlyEmulator)


class TestMIROCMonthlyEmulator:
    """Test cases for the MIROCMonthlyEmulator class."""

    def test_miroc_emulator_initialization(self):
        """Test MIROCMonthlyEmulator initialization."""
        emulator = MIROCMonthlyEmulator()
        
        assert emulator.esm == "MIROC6"
        assert emulator.repo_id == "shahineb/climemu"

    def test_miroc_emulator_inheritance(self):
        """Test that MIROCMonthlyEmulator inherits from Bouabid2026MonthlyEmulator."""
        emulator = MIROCMonthlyEmulator()
        assert isinstance(emulator, Bouabid2026MonthlyEmulator)


class TestACCESSMonthlyEmulator:
    """Test cases for the ACCESSMonthlyEmulator class."""

    def test_access_emulator_initialization(self):
        """Test ACCESSMonthlyEmulator initialization."""
        emulator = ACCESSMonthlyEmulator()
        
        assert emulator.esm == "ACCESS-ESM1-5"
        assert emulator.repo_id == "shahineb/climemu"

    def test_access_emulator_inheritance(self):
        """Test that ACCESSMonthlyEmulator inherits from Bouabid2026MonthlyEmulator."""
        emulator = ACCESSMonthlyEmulator()
        assert isinstance(emulator, Bouabid2026MonthlyEmulator)


class TestBouabid2026DailyEmulator:
    """Test cases for the Bouabid2026DailyEmulator class."""

    def test_daily_emulator_initialization(self):
        """Test Bouabid2026DailyEmulator initialization."""
        emulator = Bouabid2026DailyEmulator("test_esm")

        assert emulator.esm == "test_esm"
        assert emulator.repo_id == "shahineb/climemu"


class TestMPIDailyEmulator:
    """Test cases for the MPIDailyEmulator class."""

    def test_mpi_daily_emulator_initialization(self):
        """Test MPIDailyEmulator initialization."""
        emulator = MPIDailyEmulator()

        assert emulator.esm == "MPI-ESM1-2-LR"
        assert emulator.repo_id == "shahineb/climemu"

    def test_mpi_daily_emulator_inheritance(self):
        """Test that MPIDailyEmulator inherits from Bouabid2026DailyEmulator."""
        emulator = MPIDailyEmulator()
        assert isinstance(emulator, Bouabid2026DailyEmulator)
