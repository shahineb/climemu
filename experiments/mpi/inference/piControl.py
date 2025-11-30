import os
from typing import Tuple, List, Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import xarray as xr
from tqdm import tqdm
from functools import partial

from src.diffusion import HealPIXUNetDoY, ContinuousVESchedule

from ..config import Config
from ..data import load_dataset
from .. import utils
# from experiments.mpi.config import Config
# from experiments.mpi.data import load_dataset
# import experiments.mpi.utils as utils



def make_year_dataloader(batch_size: int, β: jnp.ndarray, σpiControl: float, key: jr.PRNGKey, nlat: int, nlon: int) -> Iterator[jnp.ndarray]:
    doys = jnp.array_split(jnp.arange(1, 366), int(jnp.ceil(365 / batch_size)))
    for doy_batch in doys:
        key, _ = jr.split(key)
        ΔT = σpiControl * jr.normal(key, len(doy_batch))
        pattern_batch = β[:, 0] + β[:, 1] * ΔT.reshape(-1, 1)
        pattern_batch = pattern_batch.reshape(-1, nlat, nlon)
        yield doy_batch, pattern_batch



def load_model_and_data(config: Config):
    """Load the trained model, normalization stats, and σmax. Returns (model, β, lat, lon, μ, σ, σmax)."""
    # Load pattern scaling coefficients and normalization statistics
    β = jnp.load(config.data.pattern_scaling_path)  # shape: (n_lat * n_lon, 2)
    stats = jnp.load(config.data.norm_stats_path)
    μ_train, σ_train = jnp.array(stats['μ']), jnp.array(stats['σ'])

    # Load piControl data
    piControl_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments={"piControl": ["r1i1p1f1"]},
        gmst_path=config.data.train_gmst_path,
        variables=["tas"],
        external_β=β
    )

    # Estimate σPI
    piControl_gmst = piControl_dataset.gmst['piControl'].ds['tas']
    σpiControl = piControl_gmst.std().item()

    # Extract coordinates
    lat = piControl_dataset.cmip6data.lat
    lon = piControl_dataset.cmip6data.lon

    # Load sigma max
    σmax = float(np.load(config.data.sigma_max_path))

    # Load or compute Latlon-HEALPix edges
    edges_to_healpix, edges_to_latlon = utils.load_or_compute_edges(
        nside=config.model.nside,
        edges_path=config.model.edges_path
    )

    # Initialize and load model
    model = HealPIXUNetDoY(
        input_size=config.model.input_size,
        nside=config.model.nside,
        enc_filters=list(config.model.enc_filters),
        dec_filters=list(config.model.dec_filters),
        out_channels=config.model.out_channels,
        temb_dim=config.model.temb_dim,
        doyemb_dim=config.model.doyemb_dim,
        posemb_dim=config.model.posemb_dim,
        healpix_emb_dim=config.model.healpix_emb_dim,
        edges_to_healpix=edges_to_healpix,
        edges_to_latlon=edges_to_latlon
    )
    model = eqx.tree_deserialise_leaves(config.training.model_filename, model)
    return model, β, lat, lon, μ_train, σ_train, σmax, σpiControl



def save_predictions(
    pred_samples: List[np.ndarray],
    lat: jnp.array,
    lon: jnp.array,
    output_dir: str,
    variables: List[str]
) -> None:
    """Save predictions to a NetCDF file."""
    os.makedirs(output_dir, exist_ok=True)
    numpy_array = np.concatenate(pred_samples, axis=1)
    n_sample = numpy_array.shape[1]
    da = xr.DataArray(
        numpy_array,
        dims=["dayofyear", "sample", "variable", "lat", "lon"],
        coords={
            "dayofyear": np.arange(1, 366),
            "sample": np.arange(1, n_sample + 1),
            "variable": list(variables),
            "lat": lat,
            "lon": lon,
        }
    )
    ds = da.to_dataset(dim="variable")

    # Save to NetCDF
    output_path = os.path.join(output_dir, "emulated_piControl.zarr")
    ds.to_zarr(output_path, mode='w', zarr_format=2, consolidated=True)
    print(f"Predictions saved to {output_path}")



def main():
    """Main function to run inference."""
    config = Config()
    model, β, lat, lon, μ_train, σ_train, σmax, σpiControl = load_model_and_data(config)

    # Prepare data loader
    n_samples = 30
    batch_size = 32
    key = jr.PRNGKey(0)
    nlat, nlon = len(lat), len(lon)
    dataloader = make_year_dataloader(n_samples, β, σpiControl, key, nlat, nlon)

    # Initialize noise schedule and random key
    schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
    χtest = jr.PRNGKey(config.sampling.random_seed)

    # Initialize sampling function
    output_size = (config.model.out_channels, nlat, nlon)
    generate_samples = partial(utils.draw_samples_batch,
                               model=model,
                               schedule=schedule,
                               n_samples=1,
                               n_steps=config.sampling.n_steps,
                               μ=μ_train, σ=σ_train,
                               output_size=output_size)

    # Generate predictions
    pred_samples = []
    n_total = n_samples * int(np.ceil(365 / batch_size))
    with tqdm(total=n_total) as pbar:
        for _ in range(n_samples):
            pred_samples_year = []
            dataloader = make_year_dataloader(batch_size, β, σpiControl, key, nlat, nlon)
            for doy, pattern in dataloader:
                pred_sample = generate_samples(doy_batch=doy, pattern_batch=pattern, key=χtest)
                pred_samples_year.append(jax.device_get(pred_sample))
                χtest, _ = jr.split(χtest)
                _ = pbar.update(1)
            pred_samples_year = jnp.concatenate(pred_samples_year)
            pred_samples.append(pred_samples_year)
            save_predictions(
                    pred_samples,
                    lat, lon,
                    config.sampling.output_dir,
                    config.data.variables
                )

    # Save final version
    save_predictions(
        pred_samples,
        lat, lon,
        config.sampling.output_dir,
        config.data.variables
    )


if __name__ == "__main__":
    main()


# import xarray as xr
# import matplotlib.pyplot as plt

# path = "/orcd/data/raffaele/001/shahineb/emulated/climemu/experiments/daily/mpi/outputs/emulated_piControl.zarr"
# ds = xr.open_zarr(path, consolidated=True)

# tas_climatology = xr.open_dataset("/orcd/home/002/shahineb/data/products/cmip6/processed/MPI-ESM1-2-LR/piControl/r1i1p1f1/tas_climatology/day/tas_day_MPI-ESM1-2-LR_r1i1p1f1_climatology.nc")
# pr_climatology = xr.open_dataset("/orcd/home/002/shahineb/data/products/cmip6/processed/MPI-ESM1-2-LR/piControl/r1i1p1f1/pr_climatology/day/pr_day_MPI-ESM1-2-LR_r1i1p1f1_climatology.nc")
# hurs_climatology = xr.open_dataset("/orcd/home/002/shahineb/data/products/cmip6/processed/MPI-ESM1-2-LR/piControl/r1i1p1f1/hurs_climatology/day/hurs_day_MPI-ESM1-2-LR_r1i1p1f1_climatology.nc")
# sfcWind_climatology = xr.open_dataset("/orcd/home/002/shahineb/data/products/cmip6/processed/MPI-ESM1-2-LR/piControl/r1i1p1f1/sfcWind_climatology/day/sfcWind_day_MPI-ESM1-2-LR_r1i1p1f1_climatology.nc")


# da = ds["hurs"] + hurs_climatology["hurs"]

# mean = da.mean(["dayofyear", "sample"]).compute()
# stddev = da.std(["dayofyear", "sample"]).compute()

# mean.plot(vmin=0)
# plt.savefig('mean.jpg', dpi=300)
# plt.close()

# stddev.plot()
# plt.savefig('stddev.jpg', dpi=300)
# plt.close()

# tas = ds
# stddev = ds['sfcWind'].mean(["doy", "sample"]).compute()
# stddev.plot()
# plt.savefig('stddev.jpg', dpi=300)
# plt.close()


# config = Config()
# model, β, lat, lon, μ_train, σ_train, σmax, σpiControl = load_model_and_data(config)



# n_samples = 2
# batch_size = 32
# key = jr.PRNGKey(1)
# nlat, nlon = len(lat), len(lon)


# # Initialize noise schedule and random key
# schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
# χtest = jr.PRNGKey(config.sampling.random_seed)

# # Initialize sampling function
# output_size = (config.model.out_channels, nlat, nlon)
# generate_samples = partial(utils.draw_samples_batch,
#                            model=model,
#                            schedule=schedule,
#                            n_samples=1,
#                            n_steps=config.sampling.n_steps,
#                            μ=μ_train, σ=σ_train,
#                            output_size=output_size)

# pred_samples = []
# n_total = n_samples * int(np.ceil(365 / batch_size))
# with tqdm(total=n_samples * 365) as pbar:
#     for _ in range(n_samples):
#         pred_samples_year = []
#         dataloader = make_year_dataloader(batch_size, β, σpiControl, key, nlat, nlon)
#         for doy, pattern in dataloader:
#             pred_sample = generate_samples(doy_batch=doy, pattern_batch=pattern, key=χtest)
#             pred_samples_year.append(jax.device_get(pred_sample))
#             χtest, _ = jr.split(χtest)
#             _ = pbar.update(1)
#         pred_samples_year = jnp.concatenate(pred_samples_year)
#         pred_samples.append(pred_samples_year)



# numpy_array = np.concatenate(pred_samples, axis=1)
# n_sample = numpy_array.shape[1]
# variables = ["tas", "pr", "hurs", "sfcWind"]
# da = xr.DataArray(
#     numpy_array,
#     dims=["doy", "sample", "variable", "lat", "lon"],
#     coords={
#         "doy": np.arange(1, 366),
#         "sample": np.arange(1, n_sample + 1),
#         "variable": list(variables),
#         "lat": lat,
#         "lon": lon,
#     }
# )
# ds = da.to_dataset(dim="variable")

# # Save to NetCDF
# output_path = os.path.join(output_dir, "emulated_piControl.nc")
# ds = ds.chunk({"doy": 1, "sample": 1, "lat": -1, "lon": -1})
# ds.to_zarr("emulated_piControl.zarr", mode='w', zarr_format=2, consolidated=True)
# print(f"Predictions saved to {output_path}")


# def healpix_to_latlon(hp_map, n_lat=96, n_lon=192, nest=False):
#     # Lat/lon centers (in degrees)
#     lats = np.linspace(-89.5, 89.5, n_lat)
#     lons = np.linspace(-179.5, 179.5, n_lon)
#     lon2d, lat2d = np.meshgrid(lons, lats)
#     # Convert to theta, phi (Healpy convention)
#     theta = np.deg2rad(90 - lat2d)   # colatitude
#     phi   = np.deg2rad(lon2d)        # longitude
#     # Interpolate the healpix map onto the grid
#     grid = hp.get_interp_val(hp_map, theta, phi, nest=nest)
#     return lat2d, lon2d, grid
