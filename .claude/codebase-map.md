# Codebase Map

**Last Updated:** 2026-06-15
**Update Trigger:** Initial creation

## Directory Structure
```
climemu/
├── src/
│   ├── climemu/                     # Published package
│   │   ├── __init__.py              # Registry + build_emulator() factory
│   │   ├── emulators/
│   │   │   ├── abstractemulator.py  # AbstractEmulator, GriddedEmulator ABCs
│   │   │   └── bouabid2025.py       # Main implementation + 5 ESM subclasses
│   │   └── utils/
│   │       └── registry.py          # Registry dict pattern
│   ├── diffusion/                   # Diffusion model internals
│   │   ├── losses/                  # denoising_score_matching.py
│   │   ├── nn/
│   │   │   ├── healpixunet.py       # U-Net on HealPIX sphere
│   │   │   ├── backbones/convnet.py # ConvNet backbone
│   │   │   ├── modules/             # HealPIX conv, padding, remap
│   │   │   └── timeencoder/         # GaussianFourierProjection
│   │   ├── samplers/                # ContinuousODESampler (Heun's method)
│   │   └── schedules/               # ContinuousVESchedule (variance exploding)
│   ├── datasets/                    # CMIP6 data handling
│   └── utils/                       # Array ops, graph utilities
├── tests/unit/climemu/              # pytest unit tests (31 tests)
├── paper/                           # Training & inference scripts per ESM
│   ├── mpi/, miroc/, access/        # ESM-specific experiments
│   └── misc/                        # Baselines, misc experiments
├── examples/collab-demo.ipynb       # Google Colab demo
├── .github/workflows/ci.yml         # CI: Python 3.11/3.12/3.13
└── pyproject.toml                   # uv-managed build config
```

## Key Files
| File | Purpose | Notes |
|------|---------|-------|
| `src/climemu/__init__.py` | Public API, registry, `build_emulator()` | Entry point |
| `src/climemu/emulators/bouabid2025.py` | Core emulator: load, compile, __call__ | Most important file |
| `src/climemu/emulators/abstractemulator.py` | Base ABCs | Defines interface |
| `src/climemu/utils/registry.py` | Registry pattern | Maps ESM names → classes |
| `src/diffusion/nn/healpixunet.py` | HealPIXUNet architecture | Facet-based convolutions |
| `src/diffusion/samplers/continuous_ode_sampler.py` | ODE sampler | Heun's method via diffrax |
| `src/diffusion/schedules/variance_exploding.py` | Noise schedule σ(t) | VE schedule |
| `tests/conftest.py` | Shared pytest fixtures | Mock data, HF downloads |

## Patterns & Conventions
- Registry pattern: ESM subclasses registered via `@emulators.register("name")`
- JAX/Equinox idioms: `@eqx.filter_jit`, `jr.PRNGKey`, `eqx.tree_deserialise_leaves`
- Greek letters in math: β, σ, μ, χ
- Concise naming: `_var_idx` not `_var_indices`

## Dependencies
| Package | Purpose | Version |
|---------|---------|---------|
| jax | Numerical computing | 0.5–0.8 |
| equinox | Neural networks for JAX | 0.13–0.14 |
| diffrax | ODE/SDE solvers | 0.7–0.8 |
| xarray | Gridded data output | 2024.1+ |
| huggingface_hub | Weight downloads | — |

## Gotchas
- Methods must be called in order: `load()` → `compile()` → `__call__()`
- `lat`/`lon` properties require climatology loaded first
- Variable subsetting passed at build time, resolved after `load()`
- Model operates on HealPIX grid internally, remapped to lat/lon on output
