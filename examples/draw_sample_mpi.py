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
from experiments.mpi.config import Config
import examples.utils as utils



def build_nn(config):
    # Load edges
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
    nn = eqx.tree_deserialise_leaves(config.training.model_filename, nn)
    utils.print_parameter_count(nn)
    return nn

def build_schedule(config):
    σmax = utils.load_σmax(config.data.sigma_max_path)
    schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
    return schedule
    

def build_generative_model(config, n_samples, n_steps):
    nn = build_nn(config)
    schedule = build_schedule(config)
    μ, σ = utils.load_normalization(config.data.norm_stats_path)
    output_size = (config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
    generative_model = partial(utils.draw_samples_single,
                               model=nn,
                               schedule=schedule,
                               n_samples=n_samples,
                               n_steps=n_steps,
                               output_size=output_size,
                               μ=μ, σ=σ)
    return generative_model

def build_emulator(config, n_samples=1, n_steps=30):
    # Load generative model and pattern scaling coefficients
    generative_model = build_generative_model(config, n_samples, n_steps)
    β = utils.load_β(config.data.pattern_scaling_path)

    # dry run to compile
    nlat, nlon = config.model.input_size[1], config.model.input_size[2]
    _ = generative_model(pattern=jnp.zeros((nlat, nlon)), key=jr.PRNGKey(0))

    def emulator(ΔT, month, seed):
        pattern = β[month - 1, :, 1] * ΔT + β[month - 1, :, 0]
        pattern = pattern.reshape((nlat, nlon))
        key = jr.PRNGKey(seed)
        samples = generative_model(pattern=pattern, key=key)
        return samples
    return emulator


def parse_args():
    parser = argparse.ArgumentParser(description="Draw emulator samples for MPI experiment.")
    parser.add_argument("--gmst", type=float, default=2.0, help="Global Mean Surface Temperature (°C) anomaly wrt piControl")
    parser.add_argument("--month", type=int, default=1, help="Month index (1-12)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--n-samples", type=int, default=1, dest="n_samples", help="Number of samples to draw")
    parser.add_argument("--n-steps", type=int, default=30, dest="n_steps", help="Number of steps for diffusion model")
    parser.add_argument("--output", type=str, default="./emulator_samples.nc", help="Output NetCDF path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    emulator = build_emulator(config, 1, 30)

    ΔT = args.gmst
    month = args.month
    seed = args.seed
    samples = emulator(ΔT=ΔT, month=month, seed=seed)

    # Save samples to a NetCDF file
    nlat, nlon = config.model.input_size[1], config.model.input_size[2]
    ds = xr.Dataset(
        {
            var: (("member", "lat", "lon"), samples[:, i, :, :])
            for i, var in enumerate(config.data.variables)
        },
        coords={
            "member": jnp.arange(len(samples)) + 1,
            "lat": jnp.arange(nlat),
            "lon": jnp.arange(nlon),
        },
    )
    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()