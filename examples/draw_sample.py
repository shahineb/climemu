"""
This script demonstrates how to use the the emulator to generate samples from the ESM output distribution.
It supports three climate models: MPI-ESM1-2-LR, MIROC6, and ACCESS-ESM1-5.

The emulator takes a global mean surface temperature (GMST) anomaly and month
as input, and generates climate field samples that follow the learned
distribution of the ESM outputs.

Usage:
    python draw_sample.py --model MPI --gmst 2.0 --month 1
    python draw_sample.py --help  # for full options
"""

import os
import sys
from functools import partial
import argparse
import xarray as xr
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

base_dir = os.path.join(os.getcwd())
sys.path.append(base_dir)

from src.diffusion import HealPIXUNet, ContinuousVESchedule
import examples.utils as utils



def load_model_config(model_name):
    """
    Load configuration for the specified climate model (Data shapes and paths are model-specific).
    """
    model_name = model_name.upper()
    # Dynamically import the appropriate configuration class
    if model_name == "MPI":
        from experiments.mpi.config import Config
        return Config()
    elif model_name == "MIROC":
        from experiments.miroc.config import Config
        return Config()
    elif model_name == "ACCESS":
        from experiments.access.config import Config
        return Config()
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: MPI, MIROC, ACCESS")



def build_nn(config):
    """
    Build and load the trained neural network
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        HealPIXUNet: Loaded and initialized neural network with trained weights
    """
    # Load graph edges for HEALPix to lat-lon connectivity
    edges_to_healpix, edges_to_latlon = utils.load_edges(config.model.edges_path)

    # Initialize the neural network
    nn = HealPIXUNet(input_size=config.model.input_size,
                     nside=config.model.nside,
                     enc_filters=list(config.model.enc_filters),
                     dec_filters=list(config.model.dec_filters),
                     out_channels=config.model.out_channels,
                     temb_dim=config.model.temb_dim,
                     healpix_emb_dim=config.model.healpix_emb_dim,
                     edges_to_healpix=edges_to_healpix,
                     edges_to_latlon=edges_to_latlon)
    
    # Load the pre-trained weights from the saved model file
    nn = eqx.tree_deserialise_leaves(config.training.model_filename, nn)
    
    # Print the total number of parameters for reference
    utils.print_parameter_count(nn)
    return nn

def build_schedule(config):
    """
    Build the noise schedule for the diffusion process. This controls the denoising sequence.

    Args:
        config: Configuration object containing schedule parameters
        
    Returns:
        ContinuousVESchedule: Variance Exploding noise schedule
    """
    # Load the maximum noise level as used in training
    σmax = utils.load_σmax(config.data.sigma_max_path)
    
    # Create the variance exploding schedule
    schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
    return schedule
    

def build_generative_model(config, n_samples, n_steps):
    """
    Build the core generative model that can draw samples by evolving the reverse diffusion process.
    
    Args:
        config: Configuration object containing model parameters
        n_samples (int): Number of parallel samples to generate per function call
        n_steps (int): Number of denoising steps in the reverse diffusion process

    Returns:
        callable: Function that generates ESM output samples given a pattern scaling input and random key
    """
    # Build the neural network and noise schedule
    nn = build_nn(config)
    schedule = build_schedule(config)
    
    # Load normalization statistics used during training (needed to denormalize the generated samples)
    μ, σ = utils.load_normalization(config.data.norm_stats_path)
    
    # Define the output size for the generated samples
    output_size = (config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
    
    # Create a function that takes (pattern, key) and returns samples
    generative_model = partial(utils.draw_samples_single,
                               model=nn,
                               schedule=schedule,
                               n_samples=n_samples,
                               n_steps=n_steps,
                               output_size=output_size,
                               μ=μ, σ=σ)
    return generative_model

def build_emulator(config, n_samples=1, n_steps=30):
    """
    Build the emulator that maps GMST anomaly to ESM output anomalies.
    
    The emulator combines pattern scaling (β coefficients) with the generative
    model to create a simple interface: given a GMST anomaly and month, it
    generates samples from the learned ESM output distribution.
    
    Pattern scaling = β₀ + β₁ * ΔT
    where β₀ and β₁ are learned coefficients that map GMST to spatial patterns
    
    Args:
        config: Configuration object containing model parameters
        n_samples (int): Number of parallel samples to generate per function call
        n_steps (int): Number of denoising steps in the reverse diffusion process
        
    Returns:
        callable: Emulator function that takes (ΔT, month, seed) and returns samples
    """
    # Load the generative model and pattern scaling coefficients
    generative_model = build_generative_model(config, n_samples, n_steps)
    β = utils.load_β(config.data.pattern_scaling_path)  # Shape: (12, n_vars, 2)

    # Perform a dry run to compile the JAX functions (important for performance)
    nlat, nlon = config.model.input_size[1], config.model.input_size[2]
    _ = generative_model(pattern=jnp.zeros((nlat, nlon)), key=jr.PRNGKey(0))

    def emulator(ΔT, month, seed):
        """
        Generate climate samples for a given GMST anomaly and month.
        
        Args:
            ΔT (float): Global Mean Surface Temperature anomaly wrt piControl in °C
            month (int): Month index (1-12)
            seed (int): Random seed for reproducibility
            
        Returns:
            jnp.ndarray: Generated climate samples of shape (n_samples, n_vars, nlat, nlon)
        """
        # Apply pattern scaling: pattern = β₀ + β₁ * ΔT
        pattern = β[month - 1, :, 1] * ΔT + β[month - 1, :, 0]
        pattern = pattern.reshape((nlat, nlon))
        
        # Generate samples using the diffusion model
        key = jr.PRNGKey(seed)
        samples = generative_model(pattern=pattern, key=key)
        return samples
    
    return emulator


def parse_args():
    parser = argparse.ArgumentParser(description="Uses emulator to generate a sample from the ESM output distribution.")
    parser.add_argument("--model", type=str, default="MPI", choices=["MPI", "MIROC", "ACCESS"], 
                       help="ESM to emulate (MPI, MIROC, or ACCESS)")
    parser.add_argument("--gmst", type=float, default=2.0, help="Global Mean Surface Temperature (°C) anomaly wrt piControl")
    parser.add_argument("--month", type=int, default=1, help="Month index (1-12)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--n-samples", type=int, default=1, dest="n_samples", help="Number of samples to draw")
    parser.add_argument("--n-steps", type=int, default=30, dest="n_steps", help="Number of steps for diffusion model")
    parser.add_argument("--output", type=str, default="./emulator_samples.nc", help="Output NetCDF path")
    return parser.parse_args()


def main():
    """
    1. Parses command-line arguments
    2. Loads model configuration
    3. Builds the emulator
    4. Generates climate samples
    5. Saves results to a NetCDF file
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Validate month input
    if not 1 <= args.month <= 12:
        raise ValueError(f"Month must be between 1 and 12, got {args.month}")
    
    # Load configuration for the specified model
    config = load_model_config(args.model)
    print(f"Using {args.model} model configuration")
    
    # Build emulator with user-specified parameters
    print("Building emulator... (takes ~1min)")
    emulator = build_emulator(config, args.n_samples, args.n_steps)

    # Generate climate samples
    print(f"Generating {args.n_samples} sample(s) for GMST anomaly = {args.gmst}°C, month = {args.month}")
    ΔT = args.gmst
    month = args.month
    seed = args.seed
    samples = emulator(ΔT=ΔT, month=month, seed=seed)

    # Create xarray Dataset for easy data handling
    print("Creating NetCDF dataset...")
    nlat, nlon = config.model.input_size[1], config.model.input_size[2]
    ds = xr.Dataset(
        {
            # Each variable gets its own data array
            var: (("member", "lat", "lon"), samples[:, i, :, :])
            for i, var in enumerate(config.data.variables)
        },
        coords={
            "member": jnp.arange(len(samples)) + 1,  # Sample indices
            "lat": jnp.arange(nlat),                 # Latitude indices
            "lon": jnp.arange(nlon),                 # Longitude indices
        },
    )
    
    # Add metadata to the dataset for reproducibility and documentation
    ds.attrs["model"] = args.model
    ds.attrs["gmst_anomaly"] = ΔT
    ds.attrs["month"] = month
    ds.attrs["seed"] = seed
    ds.attrs["n_samples"] = args.n_samples
    ds.attrs["n_steps"] = args.n_steps
    ds.attrs["description"] = "Climate emulator samples generated by JAX-ESM-emulation"
    
    # Save to NetCDF file
    ds.to_netcdf(args.output)
    print(f"{ds}")
    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()