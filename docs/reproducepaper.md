## Reproducing Paper Results

The paper results are based on an emulator trained with a subset of CMIP6 experiments (piControl, historical, ssp126, ssp585). Other experiments were left out for validation.

You can load this pretrained emulator directly:
```python
import climemu

# Instantiate emulator
emulator = climemu.build_emulator("MPI-ESM1-2-LR")

# Download weights trained on piControl, historical, ssp126, ssp585
emulator.load(which="paper")
```
Alternatively, you can also retrain the emulator and reproduce the results by following the steps below.




## Project Structure
```
├── src/                   # Core source code
│   ├── climemu/           # Package
│   ├── datasets/          # Dataset handling (CMIP6, pattern scaling)
│   ├── diffusion/         # Diffusion model
│   └── utils/             # Utility functions
│
└── experiments/           # Paper experiment-specific code
    ├── intermodel/        # Scripts for inter-model diagnosis
    ├── access/            # ACCESS-ESM1-5 emulation
    ├── miroc/             # MIROC6 emulation
    └── mpi/               # MPI-ESM1-2-LR emulation
        ├── main.py        # Training script
        ├── trainer.py     # Training loop
        ├── data.py        # Data loading and preprocessing
        ├── utils.py       # Experiment utilities
        ├── config.py      # Configuration
        ├── inference/     # Scripts to generate large emulated ensembles
        └── plots/         # Scripts for evaluation
```




## Instructions
Instructions use [uv](https://docs.astral.sh/uv/) for Python package and project management. Code tested on Python 3.12. Training and inference require GPU support.

### Installation
```bash
# Clone the repository and install all dependencies
git clone https://github.com/shahineb/climemu.git
cd climemu
uv sync --all-extras
```

### Training
To train an emulator for a specific ESM, update the paths to your CMIP6 data directory in `experiments/<esm>/config.py`. The code expects CMIP6 data **already converted to anomalies**, organized as:
```
root_dir/
└── model_name/
    └── experiment/
        └── variant/
            └── variable_anomaly/
                └── table_id/
                    └── variable_table-id_model_experiment_variant_monthly_anomaly.nc
```
This can be modified in [CMIP6Data.get_file_path](https://github.com/shahineb/climemu/blob/main/src/datasets/cmip6.py#L42).


Run training (example for MPI-ESM1-2-LR):
```bash
uv run -m experiments.mpi.main
```
This will generate model files in `experiments/mpi/cache`.


### Inference
Adapt the output paths for the emulated ensembles in `experiments/<esm>/config.py`.

Generate emulated climate projections:
```bash
# Pre-industrial control
uv run -m experiments.mpi.inference.piControl

# Future scenarios
uv run -m experiments.mpi.inference.ssp245
uv run -m experiments.mpi.inference.ssp270
```


### Diagnostics

The `plots/` and `intermodel/` directories contains all the scripts used to recreate the diagnostics shown in the paper from the generated large ensembles.
There are many of them, covering a wide range of evaluations, so we do not list them individually here.