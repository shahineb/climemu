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

from src.diffusion import HealPIXUNet, ContinuousVESchedule
from src.datasets import PatternToCMIP6Dataset

from ..config import Config
from ..data import load_dataset
from .. import utils


def make_dataloader(test_dataset: PatternToCMIP6Dataset, batch_size: int) -> Iterator[jnp.ndarray]:
    gmst = test_dataset.gmst["historical"].ds['tas']
    ΔT = jnp.array(gmst.values).flatten()
    months = jnp.asarray(gmst.time.dt.month.values)

    def get_pattern_batch(gmst: jnp.ndarray, month: jnp.array) -> jnp.ndarray:
        x = jax.vmap(test_dataset.predict_single_pattern)(gmst, month)
        return x

    idx = jnp.arange(0, len(ΔT))
    idx = jnp.array_split(idx, len(ΔT) // batch_size)
    for batch_idx in idx:
        gmst = ΔT[batch_idx]
        m = months[batch_idx]
        context = get_pattern_batch(gmst, m)
        yield context


def load_model_and_data(config: Config) -> Tuple[HealPIXUNet, PatternToCMIP6Dataset, jnp.ndarray, jnp.ndarray]:
    """Load the trained model, test dataset, and normalization statistics."""
    # Load pattern scaling coefficients and normalization statistics
    β = jnp.load(config.data.pattern_scaling_path)  # shape: (12, n_lat * n_lon, 2)
    stats = jnp.load(config.data.norm_stats_path)
    μ_train, σ_train = jnp.array(stats['μ']), jnp.array(stats['σ'])

    # Load test dataset
    test_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=["historical"],
        variables=config.data.variables,
        in_memory=config.data.in_memory,
        external_β=β
    )

    # Load sigma max
    σmax = float(np.load(config.data.sigma_max_path))

    # Load or compute Latlon-HEALPix edges
    edges_to_healpix, edges_to_latlon = utils.load_or_compute_edges(
        nside=config.model.nside,
        lat=test_dataset.cmip6data.lat,
        lon=test_dataset.cmip6data.lon,
        edges_path=config.model.edges_path
    )

    # Initialize and load model
    model = HealPIXUNet(
        input_size=config.model.input_size,
        nside=config.model.nside,
        enc_filters=list(config.model.enc_filters),
        dec_filters=list(config.model.dec_filters),
        out_channels=config.model.out_channels,
        temb_dim=config.model.temb_dim,
        healpix_emb_dim=config.model.healpix_emb_dim,
        edges_to_healpix=edges_to_healpix,
        edges_to_latlon=edges_to_latlon
    )
    model = eqx.tree_deserialise_leaves(config.training.model_filename, model)
    return model, test_dataset, μ_train, σ_train, σmax



def save_predictions(
    pred_samples: List[np.ndarray],
    test_dataset: PatternToCMIP6Dataset,
    output_dir: str,
    variables: List[str]
) -> None:
    """Save predictions to a NetCDF file."""
    os.makedirs(output_dir, exist_ok=True)
    numpy_array = np.concatenate(pred_samples, axis=0)  # shape: (total_time, member, channel, lat, lon)
    coords = dict(test_dataset.cmip6data.dtree.leaves[0].ds.coords)
    coords['time'] = coords['time'].isel(time=slice(0, numpy_array.shape[0]))

    data_vars = {
        var_name: (['time', 'member', 'lat', 'lon'], numpy_array[:, :, i, :, :])
        for i, var_name in enumerate(variables)
    }

    # Save to NetCDF
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    output_path = os.path.join(output_dir, "emulated_historical.nc")
    ds.to_netcdf(output_path, mode='w')
    print(f"Predictions saved to {output_path}")



def main():
    """Main function to run inference."""
    config = Config()
    model, test_dataset, μ_train, σ_train, σmax = load_model_and_data(config)

    # Prepare data loader
    test_loader = make_dataloader(test_dataset, config.sampling.batch_size)

    # Initialize noise schedule and random key
    schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
    χtest = jr.PRNGKey(config.sampling.random_seed)

    # Initialize sampling function
    output_size = (config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
    generate_samples = partial(utils.draw_samples_batch,
                               model=model,
                               schedule=schedule,
                               n_samples=config.sampling.n_samples,
                               n_steps=config.sampling.n_steps,
                               μ=μ_train, σ=σ_train,
                               output_size=output_size)

    # Generate predictions
    pred_samples = []
    n_batch = len(test_dataset.gmst.leaves[0].time) // config.sampling.batch_size
    with tqdm(total=n_batch) as pbar:
        for batch_idx, pattern_batch in enumerate(test_loader):
            batch_pred_samples = generate_samples(pattern_batch=pattern_batch, key=χtest)
            pred_samples.append(jax.device_get(batch_pred_samples))
            χtest, _ = jr.split(χtest)

            # Save ensemble periodically
            if (batch_idx + 1) % 10 == 0:
                save_predictions(
                    pred_samples,
                    test_dataset,
                    config.sampling.output_dir,
                    config.data.variables
                )
            _ = pbar.update(1)

    # Save final version
    save_predictions(
        pred_samples,
        test_dataset,
        config.sampling.output_dir,
        config.data.variables
    )


if __name__ == "__main__":
    main()