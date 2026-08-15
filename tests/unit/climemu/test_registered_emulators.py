"""Tests for registered emulator classes."""

from climemu.emulators.bouabid2026 import (
    Bouabid2026Emulator, 
    MPIEmulator, 
    MIROCEmulator, 
    ACCESSEmulator
)


class TestBouabid2026Emulator:
    """Test cases for the Bouabid2026Emulator class."""

    def test_bouabid2026_emulator_initialization(self):
        """Test Bouabid2026Emulator initialization."""
        emulator = Bouabid2026Emulator("test_esm")
        
        assert emulator.esm == "test_esm"
        assert emulator.repo_id == "shahineb/climemu"

    def test_bouabid2026_emulator_initialization_default_which(self):
        """Test Bouabid2026Emulator initialization with default which parameter."""
        emulator = Bouabid2026Emulator("test_esm")
        
        assert emulator.esm == "test_esm"
        assert emulator.repo_id == "shahineb/climemu"


class TestMPIEmulator:
    """Test cases for the MPIEmulator class."""

    def test_mpi_emulator_initialization(self):
        """Test MPIEmulator initialization."""
        emulator = MPIEmulator()
        
        assert emulator.esm == "MPI-ESM1-2-LR"
        assert emulator.repo_id == "shahineb/climemu"

    def test_mpi_emulator_inheritance(self):
        """Test that MPIEmulator inherits from Bouabid2026Emulator."""
        emulator = MPIEmulator()
        assert isinstance(emulator, Bouabid2026Emulator)


class TestMIROCEmulator:
    """Test cases for the MIROCEmulator class."""

    def test_miroc_emulator_initialization(self):
        """Test MIROCEmulator initialization."""
        emulator = MIROCEmulator()
        
        assert emulator.esm == "MIROC6"
        assert emulator.repo_id == "shahineb/climemu"

    def test_miroc_emulator_inheritance(self):
        """Test that MIROCEmulator inherits from Bouabid2026Emulator."""
        emulator = MIROCEmulator()
        assert isinstance(emulator, Bouabid2026Emulator)


class TestACCESSEmulator:
    """Test cases for the ACCESSEmulator class."""

    def test_access_emulator_initialization(self):
        """Test ACCESSEmulator initialization."""
        emulator = ACCESSEmulator()
        
        assert emulator.esm == "ACCESS-ESM1-5"
        assert emulator.repo_id == "shahineb/climemu"

    def test_access_emulator_inheritance(self):
        """Test that ACCESSEmulator inherits from Bouabid2026Emulator."""
        emulator = ACCESSEmulator()
        assert isinstance(emulator, Bouabid2026Emulator)
