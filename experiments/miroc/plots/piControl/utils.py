"""Shared utilities for plotting scripts."""
import os
import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

# Module-level path configuration
CLIMATOLOGY_ROOT = "/home/shahineb/data/cmip6/processed"
CLIMATOLOGY_MODEL = 'MIROC6'
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