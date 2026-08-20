# Quickstart

## Monthly emulation

```python
import earthsampler

# Create emulator for MPI-ESM1-2-LR
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR")

# Download pretrained weights (~1 min on first call, cached afterwards)
emulator.load()

# JIT-compile the sampler for 5 ensemble members
emulator.compile(n_samples=5)

# Generate samples for a 2°C GMST anomaly in March
samples = emulator(gmst=2.0, month=3, seed=42, xarray=True)
```

The returned `xr.Dataset` has variables `tas`, `pr`, `hurs`, `sfcWind` with
dimensions `(member, lat, lon)`.

## Daily emulation

```python
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR", frequency="daily")
emulator.load()
emulator.compile(n_samples=5)

# Day-of-year as integer (1–365) or string ("dd/mm")
samples = emulator(gmst=2.0, doy="21/06", seed=42, xarray=True)
```

## Variable subsetting

Only generate the variables you need:

```python
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR", variables=["tas", "pr"])
emulator.load()
emulator.compile(n_samples=5)
samples = emulator(gmst=2.0, month=6, xarray=True)
# samples has only 'tas' and 'pr'
```

## Batch generation

Generate multiple months (or days) in a single call:

```python
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR")
emulator.load()
emulator.compile(n_samples=3, batch_size=4)

# Generate for 4 months at once
samples = emulator(gmst=1.5, month=[3, 6, 9, 12], xarray=True)
# samples has an extra 'batch' dimension
```

## Supported ESMs

| ESM | Monthly | Daily |
|-----|---------|-------|
| MPI-ESM1-2-LR | Yes | Yes |
| MIROC6 | Yes | — |
| ACCESS-ESM1-5 | Yes | — |
| CanESM5 | Yes | — |
| IPSL-CM6A-LR | Yes | — |
