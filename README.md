[![image](https://github.com/shahineb/mit-earthsampler/actions/workflows/ci.yml/badge.svg)](https://github.com/shahineb/mit-earthsampler/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/earthsampler)](https://pypi.org/project/earthsampler/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.04358-b31b1b.svg)](https://arxiv.org/abs/2510.04358)

# MIT EarthSampler

Pretrained generative emulators on CMIP6 Earth system model for sampling spatially coherent projections of impact-relevant climate variables.

## Installation

Code tested on Python ≥3.11. GPU support is required for practical usage. Install from PyPI:

<table>
  <tr><td>CPU</td><td><code>pip install earthsampler</code></td></tr>
  <tr><td>NVIDIA GPU</td><td><code>pip install earthsampler[cuda12]</code></td></tr>
  <tr><td>Google TPU</td><td><code>pip install earthsampler[tpu]</code></td></tr>
</table>


## Usage

```python
import earthsampler

# Instantiate emulator
emulator = earthsampler.build_emulator("MPI-ESM1-2-LR")

# Download pretrained weights and compile (~1min)
emulator.load()
emulator.compile(n_samples=5)   # Nb of samples generated at each function call

# Generate 5 samples for a given gmst and month
samples = emulator(gmst=2,       # GMST anomaly wrt piControl (°C)
                   month=3,      # Month index (1-12)
                   seed=0,       # Random seed
                   xarray=True)  # Return xr.Dataset
```


## Reference
```bibtex
@article{bouabid2026score,
  title={Score-based generative emulation of impact-relevant Earth system model outputs},
  author={Bouabid, Shahine and Souza, Andre Nogueira and Ferrari, Raffaele},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={18},
  number={3},
  pages={e2025MS005558},
  year={2026},
  publisher={Wiley Online Library}
}
```
:warning: _Code and instructions to reproduce the paper results have been moved to a [legacy branch](https://github.com/shahineb/mit-earthsampler/tree/150826-paper/paper)_
