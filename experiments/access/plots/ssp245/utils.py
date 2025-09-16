"""Shared utilities for plotting scripts."""
import os
import xarray as xr
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dask.diagnostics import ProgressBar
from ...data import load_dataset

# Module-level path configuration
CLIMATOLOGY_ROOT = '/home/shahineb/data/cmip6/processed'
CLIMATOLOGY_MODEL = 'ACCESS-ESM1-5'
CLIMATOLOGY_MEMBER = 'r1i1p1f1'


def casttofloat64(ds):
    ds = ds.astype({var: 'float64' for var in ds.data_vars})
    return ds


def load_data(config, in_memory=True):
    """Load data and necessary statistics.
    
    Args:
        config: Configuration object containing model and data parameters.
        
    Returns:
        tuple: (test_dataset, pred_samples, μ_train, σ_train)
    """
    β = jnp.load(config.data.pattern_scaling_path)
    stats = jnp.load(config.data.norm_stats_path)
    μ_train, σ_train = jnp.array(stats['μ']), jnp.array(stats['σ'])
    test_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=["ssp245"],
        variables=config.data.variables,
        in_memory=in_memory,
        external_β=β)
    
    pred_path = os.path.join(config.sampling.output_dir, "emulated_ssp245.nc")
    pred_samples = xr.open_dataset(pred_path, chunks={})
    if in_memory:
        pred_samples = casttofloat64(pred_samples).load()
    return test_dataset, pred_samples, μ_train, σ_train


def groupby_month_and_year(ds):
    ds = ds.assign_coords(year=ds.time.dt.year, month=ds.time.dt.month)
    return ds.set_index(time=("year", "month")).unstack("time")


def load_climatology(in_memory=False):
    months = np.arange(1, 13).astype('int64')

    base_path = os.path.join(CLIMATOLOGY_ROOT, CLIMATOLOGY_MODEL, 'piControl', CLIMATOLOGY_MEMBER)

    tas_path = os.path.join(base_path, 'tas_climatology/Amon', f'tas_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc')
    pr_path = os.path.join(base_path, 'pr_climatology/Amon', f'pr_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc')
    hurs_path = os.path.join(base_path, 'hurs_climatology/Amon', f'hurs_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc')
    sfcwind_path = os.path.join(base_path, 'sfcWind_climatology/Amon', f'sfcWind_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc')

    climatology_tas = xr.open_dataset(tas_path)
    climatology_tas = climatology_tas.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time')

    climatology_pr = xr.open_dataset(pr_path)
    climatology_pr = climatology_pr.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time') * 86400

    climatology_hurs = xr.open_dataset(hurs_path)
    climatology_hurs = climatology_hurs.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time')

    climatology_sfcWind = xr.open_dataset(sfcwind_path)
    climatology_sfcWind = climatology_sfcWind.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time')

    climatology = xr.merge([climatology_tas, climatology_pr, climatology_hurs, climatology_sfcWind])

    if in_memory:
        with ProgressBar():
            climatology = casttofloat64(climatology).load()

    return climatology


def generate_patterns(test_dataset, time_mask=None):
    gmst = test_dataset.gmst["ssp245"].tas
    if time_mask is not None:
        gmst = gmst.sel(time=time_mask)
    months = gmst.time.dt.month.values
    ΔT = jnp.asarray(gmst.values).reshape(-1, 1)
    β_ = test_dataset.β[months - 1]
    patterns = β_[..., 0] + β_[..., 1] * ΔT
    nlat = len(test_dataset.cmip6data.lat)
    nlon = len(test_dataset.cmip6data.lon)
    patterns = patterns.reshape(-1, nlat, nlon)
    return patterns


def setup_figure(width_ratios, height_ratios, width_multiplier=1.0, height_multiplier=1.0, wspace=0.05, hspace=0.05):
    """Setup figure with GridSpec and return figure and gridspec objects.
    
    Args:
        width_ratios: List of width ratios for columns
        height_ratios: List of height ratios for rows
        width_multiplier: Multiplier for figure width
        height_multiplier: Multiplier for figure height
        wspace: Width spacing between subplots
        hspace: Height spacing between subplots
        
    Returns:
        tuple: (fig, gs) matplotlib figure and gridspec objects
    """
    nrow = len(height_ratios)
    ncol = len(width_ratios)
    nroweff = sum(height_ratios)
    ncoleff = sum(width_ratios)
    
    fig = plt.figure(figsize=(width_multiplier * ncoleff, height_multiplier * nroweff))
    gs = gridspec.GridSpec(
        nrows=nrow,
        ncols=ncol,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace
    )
    return fig, gs


def save_plot(fig, output_dir, filename, dpi=300):
    """Save plot to file.
    
    Args:
        fig: Matplotlib figure object
        output_dir: Directory to save the plot
        filename: Name of the file to save
        dpi: Resolution for saving
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def wrap_lon(ds):
    """Convert longitude from 0-360 to -180-180 range.
    
    Args:
        ds: xarray Dataset with longitude coordinates
        
    Returns:
        xarray Dataset with wrapped longitude coordinates
    """
    # assumes ds.lon runs 0…360
    lon360 = ds.lon.values
    lon180 = ((lon360 + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon180).sortby("lon")
    return ds


def add_seasonal_coords(data):
    """Add seasonal coordinate mapping to data.
    
    Args:
        data: xarray Dataset with month coordinate
        
    Returns:
        xarray Dataset with added season coordinate
    """
    month_to_season = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3:  "MAM", 4:  "MAM", 5:  "MAM",
        6:  "JJA", 7:  "JJA", 8:  "JJA",
        9:  "SON", 10: "SON", 11: "SON"
    }
    seasons = np.array([month_to_season[m] for m in data['month'].values])
    return data.assign_coords(season=("month", seasons))


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
        'cmap': 'BrBG',
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