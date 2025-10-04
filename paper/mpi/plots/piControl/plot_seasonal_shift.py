import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.piControl.utils import load_data, setup_figure, save_plot, wrap_lon, add_seasonal_coords, myRdPu

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'paper/mpi/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 6.0
HEIGHT_MULTIPLIER = 3.5
WSPACE = 0.05
HSPACE = 0.05

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================


def compute_emd(foo, bar):
    """Compute Earth Mover's Distance between two datasets."""
    emd = xr.apply_ufunc(
        wasserstein_distance,
        foo,
        bar,
        input_core_dims=[['flat'], ['flat']],
        exclude_dims={'flat'},
        vectorize=True,
        output_dtypes=[float],
        dask="parallelized")
    return emd

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

def get_plot_data(piControl_cmip6, piControl_diffusion):
    """Compute EMD-to-noise ratios for precipitation across seasons."""
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        print(f"Computing EMD for pr in {season}")
        emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
        esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
        σesm = esm_data.std('flat')
        σesm = σesm.where(σesm > 0.1, 0.1)
        emd[season] = compute_emd(emulator_data, esm_data) / σesm
    return emd

def compute_precipitation_data():
    """Compute precipitation data for different regions."""
    lat_range = {"Central Africa": slice(5, 18),
                 "India": slice(15, 30),
                 "Northern Australia": slice(-20, -5)}
    
    lon_range = {"Central Africa": slice(-20, 40),
                 "India": slice(65, 80),
                 "Northern Australia": slice(115, 145)}
    
    pr_data = {}
    for region in lat_range:
        piControl_ds = wrap_lon(piControl_cmip6["pr"]).sel(lat=lat_range[region], lon=lon_range[region])
        pr_data[region] = []
        for month in range(1, 13):
            pr_data[region].append(piControl_ds.sel(month=month).values.ravel())
        pr_data[region] = np.asarray(pr_data[region])
    
    return pr_data

def compute_precipitation_ratio():
    """Compute precipitation ratio for seasonal analysis."""
    piControl_data = piControl_cmip6["pr"]
    q95_ds = piControl_data.quantile(0.95, dim=("year")).compute()
    
    maxq95 = q95_ds.max(dim="month")
    minq95 = q95_ds.min(dim="month")
    minq95 = minq95.where(minq95 > 0.1, 0.1)
    precipratio = maxq95 / minq95
    
    return precipratio

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = add_seasonal_coords(piControl_cmip6)
piControl_diffusion = add_seasonal_coords(piControl_diffusion)

# Compute EMD data
emd = get_plot_data(piControl_cmip6['pr'] - climatology["pr"], piControl_diffusion['pr'])
max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")

# Compute precipitation data for regions
pr_data = compute_precipitation_data()

# Compute precipitation ratio
precipratio = compute_precipitation_ratio()

# =============================================================================
# PLOTTING
# =============================================================================

def create_seasonal_shift_plot():
    """Create the seasonal shift analysis plot."""
    width_ratios = [0.7, 0.1, 1, 0.05, 0.08, 1, 0.05]
    height_ratios = [1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Plot 1: 95th percentile of monthly precipitations
    ax = fig.add_subplot(gs[0, 0])
    for region in pr_data:
        q95 = np.quantile(pr_data[region], 0.95, axis=1)
        m = list(range(12))
        ax.plot(m, q95, marker="o", ms=3, lw=2, label=region)
    ax.margins(0.03)
    ax.set_xticks(m)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("[mm/day]")
    ax.legend()
    ax.set_title("95th percentile of \n monthly precipitations", weight="bold")
    
    # Plot 2: 95th percentile max-min ratio
    ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
    mesh = precipratio.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Spectral_r", add_colorbar=False)
    ax.coastlines()
    ax.set_title("95th percentile max-min ratio", weight="bold")
    mesh.set_clim(0, 100)
    
    # Colorbar for plot 2
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend='max')
    cbar.ax.set_yticks([0, 50, 100])
    cbar.set_label("[1]", labelpad=-1)
    pos = cax.get_position()
    new_width = pos.width * 0.4    # thinner
    new_height = pos.height * 0.7   # shorter
    new_x0 = pos.x0 - 0.00      # move left
    new_y0 = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
    cax.set_position([new_x0, new_y0, new_width, new_height])
    
    # Plot 3: Max EMD-to-noise across seasons
    ax = fig.add_subplot(gs[0, 5], projection=ccrs.Robinson())
    mesh = max_emd.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=myRdPu, add_colorbar=False)
    ax.coastlines()
    ax.set_title("Max. EMD-to-noise across seasons", weight="bold")
    mesh.set_clim(0, 1)
    
    # Colorbar for plot 3
    cax = fig.add_subplot(gs[0, 6])
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend='max')
    cbar.ax.set_yticks([0, 1])
    cbar.set_label("[1]", labelpad=-1)
    pos = cax.get_position()
    new_width = pos.width * 0.4    # thinner
    new_height = pos.height * 0.7   # shorter
    new_x0 = pos.x0 - 0.00      # move left
    new_y0 = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
    cax.set_position([new_x0, new_y0, new_width, new_height])
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate seasonal shift plot."""
    fig = create_seasonal_shift_plot()
    save_plot(fig, OUTPUT_DIR, 'precip_discrepancy.jpg', dpi=DPI)

if __name__ == "__main__":
    main()