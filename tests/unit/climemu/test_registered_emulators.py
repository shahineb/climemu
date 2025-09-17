"""Tests for registered emulator classes."""

from src.climemu.emulators.bouabid2025 import (
    Bouabid2025Emulator, 
    MPIEmulator, 
    MIROCEmulator, 
    ACCESSEmulator
)


class TestBouabid2025Emulator:
    """Test cases for the Bouabid2025Emulator class."""

    def test_bouabid2025_emulator_initialization(self):
        """Test Bouabid2025Emulator initialization."""
        emulator = Bouabid2025Emulator("test_esm", "default")
        
        assert emulator.esm == "test_esm"
        assert emulator.files_dir == "test_esm/default"
        assert emulator.repo_id == "shahineb/jax-esm-emulation"

    def test_bouabid2025_emulator_initialization_default_which(self):
        """Test Bouabid2025Emulator initialization with default which parameter."""
        emulator = Bouabid2025Emulator("test_esm")
        
        assert emulator.esm == "test_esm"
        assert emulator.files_dir == "test_esm/default"
        assert emulator.repo_id == "shahineb/jax-esm-emulation"


class TestMPIEmulator:
    """Test cases for the MPIEmulator class."""

    def test_mpi_emulator_initialization(self):
        """Test MPIEmulator initialization."""
        emulator = MPIEmulator()
        
        assert emulator.esm == "MPI-ESM1-2-LR"
        assert emulator.files_dir == "MPI-ESM1-2-LR/default"
        assert emulator.repo_id == "shahineb/jax-esm-emulation"

    def test_mpi_emulator_inheritance(self):
        """Test that MPIEmulator inherits from Bouabid2025Emulator."""
        emulator = MPIEmulator()
        assert isinstance(emulator, Bouabid2025Emulator)


class TestMIROCEmulator:
    """Test cases for the MIROCEmulator class."""

    def test_miroc_emulator_initialization(self):
        """Test MIROCEmulator initialization."""
        emulator = MIROCEmulator()
        
        assert emulator.esm == "MIROC6"
        assert emulator.files_dir == "MIROC6/default"
        assert emulator.repo_id == "shahineb/jax-esm-emulation"

    def test_miroc_emulator_inheritance(self):
        """Test that MIROCEmulator inherits from Bouabid2025Emulator."""
        emulator = MIROCEmulator()
        assert isinstance(emulator, Bouabid2025Emulator)


class TestACCESSEmulator:
    """Test cases for the ACCESSEmulator class."""

    def test_access_emulator_initialization(self):
        """Test ACCESSEmulator initialization."""
        emulator = ACCESSEmulator()
        
        assert emulator.esm == "ACCESS-ESM1-5"
        assert emulator.files_dir == "ACCESS-ESM1-5/default"
        assert emulator.repo_id == "shahineb/jax-esm-emulation"

    def test_access_emulator_inheritance(self):
        """Test that ACCESSEmulator inherits from Bouabid2025Emulator."""
        emulator = ACCESSEmulator()
        assert isinstance(emulator, Bouabid2025Emulator)
