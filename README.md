[![image](https://github.com/shahineb/climemu/actions/workflows/ci.yml/badge.svg)](https://github.com/shahineb/climemu/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/climemu)](https://pypi.org/project/climemu/) 

# ESM Emulation with diffusion models in JAX

This repository contains the implementation for the paper **"Score-based generative emulation of impact-relevant earth system model outputs"**. Codebase allows to run emulators of monthly averaged near-surface temperature, precipitation, relative humidity, wind speed for MPI-ESM1-2-LR, MIROC6, ACCESS-ESM1-5.


## Installation

Code tested on Python ≥3.11. GPU support is required for practical usage. Install from PyPI:

|-----------------|----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install climemu`                                                                                          |
| NVIDIA GPU      | `pip install climemu[cuda12]`                                                                                  |
| Google TPU      | `pip install climemu[tpu]`                                                                                     |


## Usage

```python
import climemu

# Instantiate emulator
emulator = climemu.build_emulator("MPI-ESM1-2-LR")

# Download pretrained weights and compile (~1min)
emulator.load()
emulator.compile(n_samples=5)   # Nb of samples generated at each function call

# Generate 5 samples for a given gmst and month
samples = emulator(gmst=2,       # GMST anomaly wrt piControl (°C)
                   month=3,      # Month index (1-12)
                   seed=0,       # Random seed
                   xarray=True)  # Return xr.Dataset
```

:warning: _Default model files for usage are trained on the full set of Tier I SSP simulations. To reproduce the paper results follow [instructions](paper/README.md)_.


## Citing
_add_