import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn as sns
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.miroc.config import Config
from paper.miroc.plots.piControl.utils import load_data, setup_figure, save_plot, wrap_lon, add_seasonal_coords, myRdPu


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/miroc/plots/piControl/files'
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


def get_plot_data(piControl_cmip6, piControl_diffusion):
    """Compute max EMD to noise for sfcWind across seasons."""
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        print(f"Computing EMD for sfcWind in {season}")
        emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
        esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
        σesm = esm_data.std('flat')
        σesm = σesm.where(σesm > 0.1, 0.1)
        emd[season] = compute_emd(emulator_data, esm_data) / σesm
    return emd


def get_south_american_data(piControl_cmip6, piControl_diffusion):
    """Compute sfcWind data over south american monsoon region."""
    lon_range = slice(-54, -38)
    lat_range = slice(-13, -5)
    piControl_ds = wrap_lon(piControl_cmip6["sfcWind"]).sel(lat=lat_range, lon=lon_range)
    emulator_ds = wrap_lon(piControl_diffusion["sfcWind"]).sel(lat=lat_range, lon=lon_range)
    
    sfcWind_data = piControl_ds.values.ravel()
    sfcWind_emulator = emulator_ds.values.ravel()
    
    return sfcWind_data, sfcWind_emulator


def get_quantile_data(piControl_cmip6):
    """Find regions that have a strong concentration of zero windspeed signal."""
    piControl_ds = piControl_cmip6["sfcWind"]
    q10_ds = piControl_ds.quantile(0.1, dim=("year", "month")).compute()
    return q10_ds


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)

# Add seasonal coordinates
piControl_diffusion = add_seasonal_coords(piControl_diffusion)
piControl_cmip6 = add_seasonal_coords(piControl_cmip6)

# Compute emd to noise data
emd = get_plot_data(piControl_cmip6['sfcWind'] - climatology['sfcWind'], piControl_diffusion['sfcWind'])
max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")

# Get data for plotting
piControl_diffusion = piControl_diffusion + climatology
sfcWind_data, sfcWind_emulator = get_south_american_data(piControl_cmip6, piControl_diffusion)
q10_ds = get_quantile_data(piControl_cmip6)



# =============================================================================
# PLOTTING
# =============================================================================

def create_artefact_plot():
    """Create the windspeed artefact plot."""
    width_ratios = [0.7, 0.1, 1, 0.05, 0.08, 1, 0.05]
    height_ratios = [1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Windspeed distribution histogram
    ax = fig.add_subplot(gs[0, 0])
    nbins = 2 * np.ceil(2 * len(sfcWind_data) ** (1 / 3)).astype(int)
    sns.histplot(sfcWind_data, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", alpha=0.6, edgecolor=None, label="MIROC6")
    nbins = 2 * np.ceil(2 * len(sfcWind_emulator) ** (1 / 3)).astype(int)
    sns.histplot(sfcWind_emulator, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", alpha=0.6, edgecolor=None, label="Emulator")
    ax.set_yticks([])
    ax.set_yscale('log')
    ax.set_xlabel("[m/s]")
    ax.legend()
    ax.set_title("Windspeed distribution over \n South American Monsoon region", weight="bold")
    
    # 10th percentile map
    ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
    mesh = q10_ds.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Greens", add_colorbar=False)
    ax.coastlines()
    ax.set_title("10th percentile of piControl windspeed", weight="bold")
    mesh.set_clim(0, 2)
    
    # Colorbar for 10th percentile
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend='max')
    cbar.set_label("[m/s]", labelpad=-1)
    pos = cax.get_position()
    new_width = pos.width * 0.4    # thinner
    new_height = pos.height * 0.7   # shorter
    new_x0 = pos.x0 - 0.00      # move left
    new_y0 = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
    cax.set_position([new_x0, new_y0, new_width, new_height])
    
    # Max EMD map
    ax = fig.add_subplot(gs[0, 5], projection=ccrs.Robinson())
    mesh = max_emd.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=myRdPu, add_colorbar=False)
    ax.coastlines()
    ax.set_title("Max. EMD-to-noise across seasons", weight="bold")
    mesh.set_clim(0, 1)
    
    # Colorbar for EMD
    cax = fig.add_subplot(gs[0, 6])
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend="max")
    cbar.ax.set_yticks([0, 1])
    cbar.set_label("[1]", labelpad=-1)
    pos = cax.get_position()
    new_width = pos.width * 0.4    # thinner
    new_height = pos.height * 0.7   # shorter
    new_x0 = pos.x0 - 0.00      # move left
    new_y0 = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
    cax.set_position([new_x0, new_y0, new_width, new_height])
    
    return fig

def main():
    """Main function to generate artefact plot."""
    fig = create_artefact_plot()
    save_plot(fig, OUTPUT_DIR, 'windspeed_discrepancy.jpg', dpi=DPI)


if __name__ == "__main__":
    main()