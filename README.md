# ESM Emulation with diffusion models in JAX

This repository contains the implementation for the paper **"Score-based generative emulation of impact-relevant earth system model outputs"**. Codebase allows to run emulators of monthly averaged near-surface temperature, precipitation, relative humidity, windspeed for MPI-ESM1-2-LR, MIROC6, ACCESS-ESM1-5. See [Installation](#instructions) instructions.



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

:warning: _Default model files for usage are trained on the full set of SSPs. To reproduce paper results use `climemu.build_emulator("MPI-ESM1-2-LR", which="paper")` or follow [instructions](#instructions)_.


## Project Structure
```
├── src/                   # Core source code
│   ├── datasets/          # Dataset handling (CMIP6, pattern scaling)
│   ├── diffusion/         # Diffusion model
│   └── utils/             # Utility functions
│
└── experiments/           # Experiment-specific code
    ├── access/            # ACCESS-ESM1-5 emulation
    ├── miroc/             # MIROC6 emulation
    └── mpi/               # MPI-ESM1-2-LR emulation
        ├── main.py        # Training script
        ├── trainer.py     # Training loop
        ├── data.py        # Data loading and preprocessing
        ├── utils.py       # Experiment utilities
        └── config.py      # Configuration
```




## Instructions
Instructions use [uv](https://docs.astral.sh/uv/) for Python package and project management. Code tested on Python 3.12. Training and inference require GPU support.

### Installation
```bash
# Clone the repository and install dependencies
git clone https://github.com/shahineb/climemu.git
cd climemu
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
