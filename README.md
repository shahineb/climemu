[![image](https://github.com/shahineb/climemu/actions/workflows/ci.yml/badge.svg)](https://github.com/shahineb/climemu/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/climemu)](https://pypi.org/project/climemu/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.04358-b31b1b.svg)](https://arxiv.org/abs/2510.04358)

# ESM Emulation with diffusion models in JAX

Codebase allows to run emulators of monthly averaged near-surface temperature, precipitation, relative humidity, wind speed for MPI-ESM1-2-LR, MIROC6, ACCESS-ESM1-5.

## Installation

Code tested on Python ≥3.11. GPU support is required for practical usage. Install from PyPI:

<table>
  <tr><td>CPU</td><td><code>pip install climemu</code></td></tr>
  <tr><td>NVIDIA GPU</td><td><code>pip install climemu[cuda12]</code></td></tr>
  <tr><td>Google TPU</td><td><code>pip install climemu[tpu]</code></td></tr>
</table>


## Usage
[![Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahineb/climemu/blob/main/examples/collab-demo.ipynb)

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
```bibtex
@article{bouabid2025score,
  title={Score-based generative emulation of impact-relevant Earth system model outputs},
  author={Bouabid, Shahine and Souza, Andre N and Ferrari, Raffaele},
  journal={arXiv preprint arXiv:2510.04358},
  year={2025}
}
```