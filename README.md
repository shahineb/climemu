# ESM Emulation with diffusion models in JAX

This repository contains the implementation for the paper **"Score-based generative emulation of impact-relevant earth system model outputs"**. Codebase allows to train and run emulators of monthly averaged near-surface temperature, precipitation, relative humidity, windspeed for MPI-ESM1-2-LR, MIROC6, ACCESS-ESM1-5.


## Project Structure
```
├── src/                   # Core source code
│   ├── datasets/          # Dataset handling (CMIP6, pattern scaling)
│   ├── diffusion/         # Diffusion model
│   └── utils/             # Utility functions
│
└── experiments/           # Experiment-specific code
    ├── access/            # ACCESS-ESM1-5 emulator
    ├── miroc/             # MIROC6 emulator
    └── mpi/               # MPI-ESM1-2-LR emulator
        ├── main.py        # Training script
        ├── trainer.py     # Training loop
        ├── data.py        # Data loading and preprocessing
        ├── utils.py       # Experiment utilities
        └── config.py      # Configuration
```



## Usage

### Training

Key training configurations can be modified in `config.py`. To train a model for a specific ESM (e.g., MPI-ESM1-2-LR):
```bash
python -m experiments.mpi.main
```
This will generate files in `experiments/mpi/cache/`.


### Inference
Generate emulated climate projections:
```bash
# Pre-industrial control
python -m experiments.mpi.inference.piControl

# Future scenarios
python inference/ssp245.py
```



## Installation

### Requirements

- Python 3.8+
- JAX with GPU support
- Key dependencies: `equinox`, `xarray`, `numpy`, `wandb`

### Setup

```bash
# Clone the repository
git clone https://github.com/shahineb/jax-esm-emulation.git
cd jax-esm-emulation

# Install dependencies
pip install -r requirements.txt  # Create this file with your dependencies
```
