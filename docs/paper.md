The results in the paper are for an emulator trained on a subset of the experiments (piControl, historical, ssp126, ssp585) to leave some experiments out for validation. The package allows you to load this emulator following:
```python
import climemu

# Instantiate emulator
emulator = climemu.build_emulator("MPI-ESM1-2-LR")

# Download pretrained weights on piControl, historical, ssp126, ssp585 only
emulator.load(which="paper")
```


Alternatively, we provide instructions below to retrain the emulator and reproduce the results from the paper.



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
