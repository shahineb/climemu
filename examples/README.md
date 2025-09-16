# Examples

This directory contains example scripts for using the pre-trained emulator.

## draw_sample.py

A simple script to generate climate emulator samples.

### Usage

```bash
python draw_sample.py --help
```

### Basic Examples

```bash
# Generate samples for MPI model with default settings
python draw_sample.py --model MPI

# Generate samples for MIROC model with custom GMST anomaly
python draw_sample.py --model MIROC --gmst 3.0 --month 6

# Generate multiple samples for ACCESS model
python draw_sample.py --model ACCESS --n-samples 5 --output access_samples.nc

# Generate samples with custom diffusion steps
python draw_sample.py --model MPI --n-steps 50 --seed 42
```

### Arguments

- `--model`: Climate model to use (MPI, MIROC, or ACCESS) [default: MPI]
- `--gmst`: Global Mean Surface Temperature (°C) anomaly wrt piControl [default: 2.0]
- `--month`: Month index (1-12) [default: 1]
- `--seed`: Random seed for reproducibility [default: 0]
- `--n-samples`: Number of samples to draw [default: 1]
- `--n-steps`: Number of steps for diffusion model [default: 30]
- `--output`: Output NetCDF path [default: ./emulator_samples.nc]

### Output

The script generates a NetCDF file containing the emulator samples with the following structure:

- **Variables**: Climate variables (e.g., temperature, precipitation, etc.)
- **Dimensions**: `member`, `lat`, `lon`
- **Attributes**: Model metadata including model name, GMST anomaly, month, seed, etc.