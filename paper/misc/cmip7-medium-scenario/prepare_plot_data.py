import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import regionmask

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.datasets import CMIP6Data

# Constants
base_dir = os.path.dirname(__file__)
CLIMATOLOGY_ROOT = "/home/shahineb/data/products/cmip6/processed"
CLIMATOLOGY_MODEL = 'MPI-ESM1-2-LR'
CLIMATOLOGY_MEMBER = 'r1i1p1f1'
CMIP6_ROOT = "/orcd/data/raffaele/001/shahineb/products/cmip6/processed"
CMIP7_DATA_PATH = "/home/shahineb/data/emulated/climemu-private/mpi/cmip7_medium/2100.nc"
REGION_IDX = 10
SUBSET_SIZE = 10000
RANDOM_SEED = 5

def load_climatology():
    """Load climatology data."""
    months = np.arange(1, 13).astype('int64')
    base_path = os.path.join(CLIMATOLOGY_ROOT, CLIMATOLOGY_MODEL, 'piControl', CLIMATOLOGY_MEMBER)
    
    tas_path = os.path.join(base_path, f"tas_climatology/Amon/tas_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc")
    pr_path = os.path.join(base_path, f"pr_climatology/Amon/pr_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc")
    
    climatology_tas = xr.open_dataset(tas_path)
    climatology_tas = climatology_tas.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time') - 273.15
    
    climatology_pr = xr.open_dataset(pr_path)
    climatology_pr = climatology_pr.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time') * 86400
    
    return xr.merge([climatology_tas, climatology_pr])


def load_cmip_data(climatology):
    """Load CMIP6 and CMIP7 data."""
    cmip6data = CMIP6Data(CMIP6_ROOT, CLIMATOLOGY_MODEL, ["ssp126", "ssp585"],
                          ["tas", "pr"], {"time": slice("2100-01", "2100-12")})
    cmip6data.load()
    
    ssp126 = cmip6data["ssp126"].ds + climatology.sel(month=cmip6data["ssp126"].time.dt.month)
    ssp585 = cmip6data["ssp585"].ds + climatology.sel(month=cmip6data["ssp585"].time.dt.month)
    
    cmip7 = xr.open_dataset(CMIP7_DATA_PATH)[["tas", "pr"]]
    cmip7.load()
    cmip7 = cmip7 + climatology
    
    return ssp126, ssp585, cmip7


def extract_region_data(ds, region_mask):
    """Extract and flatten region data, removing NaNs."""
    region_ds = ds.where(region_mask)
    tas = region_ds['tas'].values.ravel()
    pr = region_ds['pr'].values.ravel()
    
    nan_mask = ~np.isnan(tas)
    tas = tas[nan_mask]
    pr = pr[nan_mask]
    
    return tas, pr


def prepare_data(ssp126, ssp585, cmip7):
    """Prepare temperature and precipitation data for plotting."""
    ar6 = regionmask.defined_regions.ar6.all
    region = ar6[[REGION_IDX]]
    print(region)
    
    region_mask = ~np.isnan(region.mask(cmip7.lon, cmip7.lat))
    
    ssp126_tas, ssp126_pr = extract_region_data(ssp126, region_mask)
    ssp585_tas, ssp585_pr = extract_region_data(ssp585, region_mask)
    cmip7_tas, cmip7_pr = extract_region_data(cmip7, region_mask)
    cmip7_pr = cmip7_pr.clip(min=0)
    
    # Convert to log space
    ssp126_logpr = np.log1p(ssp126_pr)
    ssp585_logpr = np.log1p(ssp585_pr)
    cmip7_logpr = np.log1p(cmip7_pr)
    
    # Subsample for plotting
    np.random.seed(RANDOM_SEED)
    subset_idx = np.random.choice(len(ssp126_tas), SUBSET_SIZE, replace=False)
    ssp126_tas, ssp126_logpr = ssp126_tas[subset_idx], ssp126_logpr[subset_idx]
    ssp585_tas, ssp585_logpr = ssp585_tas[subset_idx], ssp585_logpr[subset_idx]
    
    subset_idx = np.random.choice(len(cmip7_tas), SUBSET_SIZE, replace=False)
    cmip7_tas, cmip7_logpr = cmip7_tas[subset_idx], cmip7_logpr[subset_idx]
    
    # Save into dataframe for plotting
    df = pd.DataFrame(data=np.stack([ssp126_tas, ssp126_logpr,
                                     ssp585_tas, ssp585_logpr,
                                     cmip7_tas, cmip7_logpr], axis=-1),
                      columns=["ssp126_tas", "ssp126_logpr",
                               "ssp585_tas", "ssp585_logpr",
                               "cmip7_tas",  "cmip7_logpr"])
    output_path = os.path.join(base_dir, "outputs", "tas_pr_2100_region.csv")
    df.to_csv(output_path)


def main():
    """Main plotting function."""
    # Load data
    climatology = load_climatology()
    ssp126, ssp585, cmip7 = load_cmip_data(climatology)
    prepare_data(ssp126, ssp585, cmip7)


if __name__ == "__main__":
    main()
