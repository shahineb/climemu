# ESM Emulation with diffusion models in JAX

This repository contains the implementation for the paper **"Score-based generative emulation of impact-relevant earth system model outputs"**. Codebase allows to train and run emulators of monthly averaged near-surface temperature, precipitation, relative humidity, windspeed for MPI-ESM1-2-LR, MIROC6, ACCESS-ESM1-5.



## Example

Add example directory


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

### Instalation

Code tested on Python 3.10+. Training and inference require GPU support.
```bash
# Clone the repository and install dependencies
git clone https://github.com/shahineb/jax-esm-emulation.git
cd jax-esm-emulation
uv sync
```


### Training

Key training configurations can be modified in `config.py`. To train a model for a specific ESM (e.g., MPI-ESM1-2-LR):
```bash
uv run -m experiments.mpi.main
```
This will generate files in `experiments/mpi/cache/`.


### Inference
Generate emulated climate projections:
```bash
# Pre-industrial control
uv run -m experiments.mpi.inference.piControl

# Future scenarios
uv run -m experiments.mpi.inference.ssp245
```
