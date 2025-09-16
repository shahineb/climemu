# ESM Emulation with diffusion models in JAX

This repository contains the implementation for the paper **"Score-based generative emulation of impact-relevant earth system model outputs"**. Codebase allows to train and run emulators of monthly averaged near-surface temperature, precipitation, relative humidity, windspeed for MPI-ESM1-2-LR, MIROC6, ACCESS-ESM1-5. See [Installation](#usage) instructions.



## Example

_Add instructions on how to download model files and draw samples from the emulator with the example scripts. (:warning: Model files are meant for usage and trained on the full set of SSPs, to reproduce paper results follow instructions)_


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
Instructions use [uv](https://docs.astral.sh/uv/) for Python package and project management. Code tested on Python 3.12. Training and inference require GPU support.

### Installation
```bash
# Clone the repository and install dependencies
git clone https://github.com/shahineb/jax-esm-emulation.git
cd jax-esm-emulation
uv sync
```

### Training
To train an emulator for a specific ESM (e.g. MPI-ESM1-2-LR):
```bash
uv run -m experiments.mpi.main
```
This will generate model files in `experiments/mpi/cache`. Training configurations can be modified in `experiments/mpi/config.py`. The code expect CMIP6 archives precomputed as anomalies and organized as follows
```
root_dir/
└── model_name/
    └── experiment/
        └── variant/
            └── variable_anomaly/
                └── table_id/
                    └── variable_table-id_model_experiment_variant_monthly_anomaly.nc
```

### Inference
Generate emulated climate projections:
```bash
# Pre-industrial control
uv run -m experiments.mpi.inference.piControl

# Future scenarios
uv run -m experiments.mpi.inference.ssp245
```
