"""Shared utilities for plotting scripts."""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dask.diagnostics import ProgressBar

# Module-level path configuration
CLIMATOLOGY_ROOT = "/home/shahineb/data/cmip6/processed"
CLIMATOLOGY_MODEL = 'ACCESS-ESM1-5'
CLIMATOLOGY_MEMBER = 'r1i1p1f1'
RAW_CMIP6_ROOT = "/orcd/home/002/shahineb/data/cmip6/raw"


def groupby_month_and_year(ds):
    ds = ds.assign_coords(year=ds.time.dt.year, month=ds.time.dt.month)
    return ds.set_index(time=("year", "month")).unstack("time")


def casttofloat64(ds):
    ds = ds.astype({var: 'float64' for var in ds.data_vars})
    return ds


def load_data(config, in_memory=False):
    months = np.arange(1, 13).astype('int64')

    base_path = os.path.join(CLIMATOLOGY_ROOT, CLIMATOLOGY_MODEL, 'piControl', CLIMATOLOGY_MEMBER)

    climatology_tas = xr.open_dataset(os.path.join(base_path, f"tas_climatology/Amon/tas_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc"))
    climatology_tas = climatology_tas.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time')

    climatology_pr = xr.open_dataset(os.path.join(base_path, f"pr_climatology/Amon/pr_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc"))
    climatology_pr = climatology_pr.assign_coords(month=('time', months)).swap_dims({'time', 'month'}).drop_vars('time') * 86400

    climatology_hurs = xr.open_dataset(os.path.join(base_path, f"hurs_climatology/Amon/hurs_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc"))
    climatology_hurs = climatology_hurs.assign_coords(month=('time', months)).swap_dims({'time', 'month'}).drop_vars('time')

    climatology_sfcWind = xr.open_dataset(os.path.join(base_path, f"sfcWind_climatology/Amon/sfcWind_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc"))
    climatology_sfcWind = climatology_sfcWind.assign_coords(month=('time', months)).swap_dims({'time', 'month'}).drop_vars('time')

    climatology = xr.merge([climatology_tas, climatology_pr, climatology_hurs, climatology_sfcWind])

    pi_path = os.path.join(config.sampling.output_dir, "emulated_piControl.nc")
    piControl_diffusion = xr.open_dataset(pi_path)

    raw_base = os.path.join(RAW_CMIP6_ROOT, CLIMATOLOGY_MODEL, 'piControl', CLIMATOLOGY_MEMBER)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    piControl_cmip6_tas = xr.open_mfdataset(os.path.join(raw_base, "tas/Amon/*"), decode_times=time_coder).drop_vars('height')
    piControl_cmip6_pr = xr.open_mfdataset(os.path.join(raw_base, "pr/Amon/*"), decode_times=time_coder).astype("float64") * 86400
    piControl_cmip6_hurs = xr.open_mfdataset(os.path.join(raw_base, "hurs/Amon/*"), decode_times=time_coder).drop_vars('height')
    piControl_cmip6_sfcWind = xr.open_mfdataset(os.path.join(raw_base, "sfcWind/Amon/*"), decode_times=time_coder).drop_vars('height')
    piControl_cmip6 = xr.merge([
        piControl_cmip6_tas[['tas']],
        piControl_cmip6_pr[['pr']],
        piControl_cmip6_hurs[['hurs']],
        piControl_cmip6_sfcWind[['sfcWind']]
    ])
    if in_memory:
        with ProgressBar():
            climatology = casttofloat64(climatology).load()
            piControl_diffusion = casttofloat64(piControl_diffusion).load()
            piControl_cmip6 = casttofloat64(piControl_cmip6).load()

    piControl_cmip6 = groupby_month_and_year(piControl_cmip6)
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

def add_seasonal_coords(data):
    """Add seasonal coordinate mapping to data."""
    month_to_season = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3:  "MAM", 4:  "MAM", 5:  "MAM",
        6:  "JJA", 7:  "JJA", 8:  "JJA",
        9:  "SON", 10: "SON", 11: "SON"
    }
    seasons = np.array([month_to_season[m] for m in data['month'].values])
    return data.assign_coords(season=("month", seasons))