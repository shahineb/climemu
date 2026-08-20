"""Integration tests — full load -> compile -> generate flow using HuggingFace weights."""

import pytest
import numpy as np
import jax.numpy as jnp
import xarray as xr
from climemu import build_emulator


class TestMonthlyMPI:
    """Monthly MPI-ESM1-2-LR end-to-end tests."""

    def test_load_and_validate_data(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()

        # β shape: (12 months, nlat*nlon, 2 coefficients)
        assert emulator.β.shape[0] == 12
        assert emulator.β.shape[2] == 2
        assert emulator.β.shape[1] == emulator.nlat * emulator.nlon

        # No NaN in weights or climatology
        assert not jnp.any(jnp.isnan(emulator.β))
        for var in emulator.climatology.data_vars:
            assert np.isfinite(emulator.climatology[var].values).all()

    def test_generate_samples(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2)

        samples = emulator(gmst=1.5, month=6, seed=42)
        assert samples.shape == (1, emulator.nvar, emulator.nlat, emulator.nlon)
        assert isinstance(samples, jnp.ndarray)

    def test_xarray_output(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=2, n_steps=2)

        ds = emulator(gmst=1.5, month=6, seed=42, xarray=True)
        assert isinstance(ds, xr.Dataset)
        assert list(ds.data_vars) == emulator.vars
        assert set(ds.coords) == {"member", "lat", "lon"}
        assert ds.member.size == 2
        assert np.array_equal(ds.lat.values, emulator.lat)
        assert np.array_equal(ds.lon.values, emulator.lon)


class TestMonthlyMPIBatch:
    """Monthly MPI-ESM1-2-LR batched sampling tests."""

    def test_batch_generate(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2, batch_size=3)

        samples = emulator(gmst=1.5, month=[1, 6, 12])
        assert samples.shape == (3, 1, emulator.nvar, emulator.nlat, emulator.nlon)
        assert isinstance(samples, jnp.ndarray)

    def test_batch_with_paired_gmst(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2, batch_size=3)

        samples = emulator(gmst=[1.0, 1.5, 2.0], month=[1, 6, 12])
        assert samples.shape == (3, 1, emulator.nvar, emulator.nlat, emulator.nlon)

    def test_batch_scalar_gmst_broadcast(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=2, n_steps=2, batch_size=2)

        samples = emulator(gmst=1.5, month=[3, 9])
        assert samples.shape == (2, 2, emulator.nvar, emulator.nlat, emulator.nlon)

    def test_batch_wrong_size_raises(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2, batch_size=3)

        with pytest.raises(ValueError, match="Expected 3 months"):
            emulator(gmst=1.5, month=[1, 6])

    def test_batch_xarray(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR")
        emulator.load()
        emulator.compile(n_samples=2, n_steps=2, batch_size=3)

        ds = emulator(gmst=[1.0, 1.5, 2.0], month=[2, 2, 2], seed=42, xarray=True)
        assert isinstance(ds, xr.Dataset)
        assert list(ds.data_vars) == emulator.vars
        assert set(ds.dims) == {"batch", "member", "lat", "lon"}
        assert ds.sizes["batch"] == 3
        assert ds.sizes["member"] == 2
        assert "month" in ds.coords
        assert "gmst_anomaly" in ds.coords
        assert np.array_equal(ds.month.values, [2, 2, 2])
        assert np.array_equal(ds.gmst_anomaly.values, [1.0, 1.5, 2.0])


class TestDailyMPI:
    """Daily MPI-ESM1-2-LR end-to-end tests."""

    def test_load_and_validate_data(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()

        # Daily β shape: (nlat*nlon, 2) — no month dimension
        assert emulator.β.ndim == 2
        assert emulator.β.shape[1] == 2
        assert emulator.β.shape[0] == emulator.nlat * emulator.nlon
        assert not jnp.any(jnp.isnan(emulator.β))

    def test_generate_with_int_doy(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2)

        samples = emulator(gmst=1.5, doy=172, seed=42)
        assert samples.shape == (1, emulator.nvar, emulator.nlat, emulator.nlon)

    def test_generate_with_str_doy(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2)

        samples = emulator(gmst=1.5, doy="21/06", seed=42)
        assert samples.shape == (1, emulator.nvar, emulator.nlat, emulator.nlon)

    def test_xarray_output(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=2, n_steps=2)

        ds = emulator(gmst=1.5, doy=172, seed=42, xarray=True)
        assert isinstance(ds, xr.Dataset)
        assert list(ds.data_vars) == emulator.vars
        assert ds.member.size == 2


class TestDailyMPIBatch:
    """Daily MPI-ESM1-2-LR batched sampling tests."""

    def test_batch_generate(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2, batch_size=3)

        samples = emulator(gmst=1.5, doy=[1, 172, 365])
        assert samples.shape == (3, 1, emulator.nvar, emulator.nlat, emulator.nlon)
        assert isinstance(samples, jnp.ndarray)

    def test_batch_with_paired_gmst(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2, batch_size=3)

        samples = emulator(gmst=[1.0, 1.5, 2.0], doy=[1, 172, 365])
        assert samples.shape == (3, 1, emulator.nvar, emulator.nlat, emulator.nlon)

    def test_batch_scalar_gmst_broadcast(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=2, n_steps=2, batch_size=2)

        samples = emulator(gmst=1.5, doy=[90, 270])
        assert samples.shape == (2, 2, emulator.nvar, emulator.nlat, emulator.nlon)

    def test_batch_wrong_size_raises(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=1, n_steps=2, batch_size=3)

        with pytest.raises(ValueError, match="Expected 3 doys"):
            emulator(gmst=1.5, doy=[1, 172])

    def test_batch_xarray(self, requires_hf):
        emulator = build_emulator("MPI-ESM1-2-LR", frequency="daily")
        emulator.load()
        emulator.compile(n_samples=2, n_steps=2, batch_size=3)

        ds = emulator(gmst=[1.0, 1.5, 2.0], doy=[172, 172, 172], seed=42, xarray=True)
        assert isinstance(ds, xr.Dataset)
        assert list(ds.data_vars) == emulator.vars
        assert set(ds.dims) == {"batch", "member", "lat", "lon"}
        assert ds.sizes["batch"] == 3
        assert ds.sizes["member"] == 2
        assert "doy" in ds.coords
        assert "gmst_anomaly" in ds.coords
        assert np.array_equal(ds.doy.values, [172, 172, 172])
        assert np.array_equal(ds.gmst_anomaly.values, [1.0, 1.5, 2.0])
