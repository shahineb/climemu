# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --all-extras

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single test by name
.venv/bin/python -m pytest tests/ -v -k test_name

# Build package
uv build

# CI mirrors: uv run pytest -v --import-mode=importlib
```

Python 3.11, 3.12, 3.13 supported. Python 3.12 is the default (.python-version).

## Architecture

Climate emulator using score-based diffusion models on JAX/Equinox. Two-stage pipeline: **pattern scaling** (linear regression on global mean surface temperature) → **conditional diffusion** (HealPIXUNet on spherical grids).

### Class hierarchy

```
AbstractEmulator (ABC)
  └── GriddedEmulator (ABC)
        └── Bouabid2025Emulator
              ├── MPIEmulator      ["MPI-ESM1-2-LR"]
              ├── MIROCEmulator    ["MIROC6"]
              ├── ACCESSEmulator   ["ACCESS-ESM1-5"]
              ├── CanESMEmulator   ["CanESM5"]
              └── IPSLEmulator     ["IPSL-CM6A-LR"]
```

ESM-specific subclasses are registered via `Registry` (a custom dict in `src/climemu/utils/registry.py`) and instantiated through `build_emulator(name, **kwargs)` in `src/climemu/__init__.py`.

### User API flow

```python
emulator = climemu.build_emulator("MPI-ESM1-2-LR", variables=["tas", "pr"])
emulator.load(which="default")       # downloads weights from HuggingFace
emulator.compile(n_samples=5)        # JIT compilation dry run
samples = emulator(gmst=2.0, month=3, seed=0, xarray=True)
```

Methods must be called in order: `load()` → `compile()` → `__call__()`. Grid properties (`lat`, `lon`) require climatology to be loaded first.

### Data pipeline (inside `__call__`)

`gmst, month` → pattern scaling (β coefficients) → normalize → HealPIXUNet diffusion sampler (reverse-time ODE via diffrax) → denormalize → output `[n_samples, n_vars, nlat, nlon]`

### Key source locations

- `src/climemu/emulators/bouabid2025.py` — main emulator implementation, registered ESM subclasses, normalize/denormalize, sampler creation
- `src/climemu/emulators/abstractemulator.py` — base ABCs
- `src/diffusion/nn/healpixunet.py` — U-Net on HealPIX sphere (facet-based convolutions, time embeddings)
- `src/diffusion/samplers/continuous_ode_sampler.py` — Heun's method ODE sampler
- `src/diffusion/schedules/variance_exploding.py` — noise schedule σ(t)

### Variable subsetting

Passed at build time via `variables=["tas", "pr"]`. Resolved in `_resolve_variables()` after climatology loads — indexes into the full variable set.

### Pretrained weights

Hosted on HuggingFace (`shahineb/climemu`). Each ESM has: `config.yaml`, `weights.eqx`, `μ_σ.npz`, `β.npy`, `edges.npz`, `σmax.npy`, and a `piControl_climatology.nc`.

## Code Conventions

- JAX/Equinox idioms: `@eqx.filter_jit`, `jr.PRNGKey`, `eqx.tree_deserialise_leaves`
- Greek letters used in math code: β, σ, μ, χ
- Concise naming (`_var_idx` not `_var_indices`)
- Comments explain *why*, not *what*
