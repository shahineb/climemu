"""Shared utilities for plotting scripts."""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from dask.diagnostics import ProgressBar

# Module-level path configuration
CLIMATOLOGY_ROOT = "/home/shahineb/data/products/cmip6/processed"
CLIMATOLOGY_MODEL = 'MPI-ESM1-2-LR'
CLIMATOLOGY_MEMBER = 'r1i1p1f1'
RAW_CMIP6_ROOT = "/orcd/home/002/shahineb/data/products/cmip6/raw"

from experiments.mpi.config import Config
config = Config()


def casttofloat64(ds):
    ds = ds.astype({var: 'float64' for var in ds.data_vars})
    return ds


def load_data(config, in_memory=False):
    base_path = os.path.join(CLIMATOLOGY_ROOT, CLIMATOLOGY_MODEL, 'piControl', CLIMATOLOGY_MEMBER)

    # Load climatology
    climatology_tas = xr.open_dataset(os.path.join(base_path, f"tas_climatology/day/tas_day_{CLIMATOLOGY_MODEL}_{CLIMATOLOGY_MEMBER}_climatology.nc"))
    climatology_pr = xr.open_dataset(os.path.join(base_path, f"pr_climatology/day/pr_day_{CLIMATOLOGY_MODEL}_{CLIMATOLOGY_MEMBER}_climatology.nc")) * 86400
    climatology_hurs = xr.open_dataset(os.path.join(base_path, f"hurs_climatology/day/hurs_day_{CLIMATOLOGY_MODEL}_{CLIMATOLOGY_MEMBER}_climatology.nc"))
    climatology_sfcWind = xr.open_dataset(os.path.join(base_path, f"sfcWind_climatology/day/sfcWind_day_{CLIMATOLOGY_MODEL}_{CLIMATOLOGY_MEMBER}_climatology.nc"))
    climatology = xr.merge([climatology_tas, climatology_pr, climatology_hurs, climatology_sfcWind], compat="override")

    # Load emulated data
    pi_path = os.path.join(config.sampling.output_dir, "emulated_piControl.zarr")
    piControl_diffusion = xr.open_zarr(pi_path, consolidated=True)

    # Load cmip6 data
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    piControl_cmip6_tas = xr.open_zarr(os.path.join(base_path, "tas_anomaly/day/tas_day_MPI-ESM1-2-LR_piControl_r1i1p1f1_daily_anomaly.zarr"), consolidated=True, decode_times=time_coder)
    piControl_cmip6_pr = xr.open_zarr(os.path.join(base_path, "pr_anomaly/day/pr_day_MPI-ESM1-2-LR_piControl_r1i1p1f1_daily_anomaly.zarr"), consolidated=True, decode_times=time_coder) * 86400
    piControl_cmip6_hurs = xr.open_zarr(os.path.join(base_path, "hurs_anomaly/day/hurs_day_MPI-ESM1-2-LR_piControl_r1i1p1f1_daily_anomaly.zarr"), consolidated=True, decode_times=time_coder)
    piControl_cmip6_sfcWind = xr.open_zarr(os.path.join(base_path, "sfcWind_anomaly/day/sfcWind_day_MPI-ESM1-2-LR_piControl_r1i1p1f1_daily_anomaly.zarr"), consolidated=True, decode_times=time_coder)
    piControl_cmip6 = xr.merge([
            piControl_cmip6_tas[['tas']],
            piControl_cmip6_pr[['pr']],
            piControl_cmip6_hurs[['hurs']],
            piControl_cmip6_sfcWind[['sfcWind']]
        ], compat="override")
    if in_memory:
        with ProgressBar():
            climatology = casttofloat64(climatology).load()
            piControl_diffusion = casttofloat64(piControl_diffusion).load()
            piControl_cmip6 = casttofloat64(piControl_cmip6).load()

    return climatology, piControl_diffusion, piControl_cmip6




# Variable metadata
VARIABLES = {
    'tas': {
        'channel': 0,
        'name': 'Temperature',
        'unit': 'K',
        'cmap': 'coolwarm',
        'color': 'cornflowerblue'
    },
    'pr': {
        'channel': 1,
        'name': 'Precipitation',
        'unit': 'mm/day',
        'cmap': 'BrBG',
        'color': 'cornflowerblue'
    },
    'hurs': {
        'channel': 2,
        'name': 'Relative Humidity',
        'unit': '%',
        'cmap': 'RdBu',
        'color': 'cornflowerblue'
    },
    'sfcWind': {
        'channel': 3,
        'name': 'Windspeed',
        'unit': 'm/s',
        'cmap': 'PRGn',
        'color': 'cornflowerblue'
    }
}

# =============================================================================
# COMMON PLOTTING FUNCTIONS
# =============================================================================

def setup_figure(width_ratios, height_ratios, width_multiplier=1.0, height_multiplier=1.0, wspace=0.01, hspace=0.01):
    """Standard figure setup with configurable width and height multipliers."""
    nrow, ncol = len(height_ratios), len(width_ratios)
    fig = plt.figure(figsize=(width_multiplier * sum(width_ratios), 
                             height_multiplier * sum(height_ratios)))
    gs = GridSpec(nrow, ncol, figure=fig, width_ratios=width_ratios, 
                  height_ratios=height_ratios, hspace=hspace, wspace=wspace)
    return fig, gs

def save_plot(fig, output_dir, filename, dpi=300, show=False):
    """Standard plot saving with optional display."""
    if show:
        plt.show()
    else:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
        plt.close()

def wrap_lon(ds):
    """Convert longitude from 0-360 to -180-180 range."""
    lon360 = ds.lon.values
    lon180 = ((lon360 + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon180).sortby("lon")
    return ds

def assign_month_and_season_from_doy(ds, dim="time"):
    doy = ds["dayofyear"]
    month = xr.where(doy <= 31, 1,
            xr.where(doy <= 59, 2,
            xr.where(doy <= 90, 3,
            xr.where(doy <= 120, 4,
            xr.where(doy <= 151, 5,
            xr.where(doy <= 181, 6,
            xr.where(doy <= 212, 7,
            xr.where(doy <= 243, 8,
            xr.where(doy <= 273, 9,
            xr.where(doy <= 304, 10,
            xr.where(doy <= 334, 11, 12)))))))))))
    season = xr.where(month.isin([12, 1, 2]), "DJF",
             xr.where(month.isin([3, 4, 5]),  "MAM",
             xr.where(month.isin([6, 7, 8]),  "JJA", "SON")))
    return ds.assign_coords(month=(dim, month.values), season=(dim, season.values))


def emphasize_mid_cmap(cmap="RdPu", strength=4.0, N=256):
    base = plt.get_cmap(cmap, N)
    if strength <= 0:
        return base
    x = np.linspace(0., 1., N)
    x_warp = 0.5 + 0.5 * np.tanh(strength * (x - 0.5)) / np.tanh(strength / 2)
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap}_mid", base(x_warp))

myRdPu = emphasize_mid_cmap(cmap="RdPu", strength=4.0)