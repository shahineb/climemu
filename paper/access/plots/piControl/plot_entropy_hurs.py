import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import seaborn as sns
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.access.config import Config
from paper.access.plots.piControl.utils import load_data, setup_figure, save_plot, wrap_lon, add_seasonal_coords, myRdPu


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/access/plots/piControl/files'
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
    """Compute max EMD to noise for hurs across seasons."""
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        print(f"Computing EMD for hurs in {season}")
        emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
        esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
        σesm = esm_data.std('flat')
        σesm = σesm.where(σesm > 0.1, 0.1)
        emd[season] = compute_emd(emulator_data, esm_data) / σesm
    return emd


def compute_entropy(da):
    """Compute Shannon entropy for humidity data."""
    nbins = np.ceil(2 * (3 * len(da.year)) ** (1 / 3)).astype(int).item()
    vmin = da.quantile(0.001).values.item()
    vmax = da.quantile(0.999).values.item()

    def entropy(samples):
        hist, _ = np.histogram(samples, bins=nbins, range=(vmin, vmax), density=True)
        hist = hist[hist > 0]
        H = -np.sum(hist * np.log2(hist))
        return H

    H = xr.apply_ufunc(
        entropy,
        da.groupby('season'),
        input_core_dims=[["year", "month"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return H.min('season')


def get_region_data(piControl_cmip6, piControl_diffusion, climatology, region_name, lat_range, lon_range, season="SON"):
    """Extract data for a specific region and season."""
    piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
    emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
    
    piControl_data = piControl_ds.where(piControl_ds.season == season, drop=True).values.ravel()
    emulator_data = emulator_ds.where(emulator_ds.season == season, drop=True).values.ravel()
    
    return piControl_data, emulator_data


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)

# Add seasonal coordinates
piControl_diffusion = add_seasonal_coords(piControl_diffusion)
piControl_cmip6 = add_seasonal_coords(piControl_cmip6)

# Compute EMD data
emd = get_plot_data(piControl_cmip6['hurs'] - climatology["hurs"], piControl_diffusion['hurs'])
max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")

# Compute entropy data
da = piControl_cmip6["hurs"] - climatology["hurs"]
Hmin = compute_entropy(da)

# Extract region data
regions = {
    "Colombian coast": (slice(-6, 9), slice(-81, -73)),
    "Equatorial Atlantic Ocean": (slice(2, 7), slice(-34, -10)),
    "Southern Brasil": (slice(-22, -8), slice(-56, -39))
}

hurs_data = {}
hurs_emulator = {}
for region_name, (lat_range, lon_range) in regions.items():
    piControl_data, emulator_data = get_region_data(piControl_cmip6, piControl_diffusion, climatology, region_name, lat_range, lon_range)
    hurs_data[region_name] = piControl_data
    hurs_emulator[region_name] = emulator_data


# =============================================================================
# PLOTTING
# =============================================================================

def create_entropy_hurs_plot():
    """Create the entropy and humidity discrepancy plot."""
    width_ratios = [0.1, 0.7, 0.1, 1, 0.05, 0.08, 1, 0.05]
    height_ratios = [0.5, 0.1, 0.5]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Colombian coast histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "Colombian coast", va="center", ha="center", rotation="vertical", weight="bold")
    
    ax = fig.add_subplot(gs[0, 1])
    data = hurs_data["Colombian coast"]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    logbins = np.concatenate([-np.logspace(-3, np.log(10), nbins // 2)[::-1], np.zeros(1), np.logspace(-3, np.log(10), nbins // 2)])
    sns.histplot(data, ax=ax, kde=False, stat="density", bins=logbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data = hurs_emulator["Colombian coast"]
    sns.histplot(data, ax=ax, kde=False, stat="density", bins=logbins, color="tomato", edgecolor=None, alpha=0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("")
    ax.set_xlabel("[%]")
    ax.set_frame_on(False)
    ax.set_title("SON relative humidity anomaly", weight="bold")
    
    # Southern Brasil histogram
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "South Brasil", va="center", ha="center", rotation="vertical", weight="bold")
    
    ax = fig.add_subplot(gs[2, 1])
    data = hurs_data["Southern Brasil"]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    sns.histplot(data, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", label="ACCESS-ESM1-5", edgecolor=None, alpha=0.5)
    data = hurs_emulator["Southern Brasil"]
    sns.histplot(data, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", label="Emulator", edgecolor=None, alpha=0.5)
    ax.set_yticks([])
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("")
    ax.set_xlabel("[%]")
    ax.legend(frameon=False)
    ax.set_frame_on(False)
    
    # Shannon Entropy map
    ax = fig.add_subplot(gs[:, 3], projection=ccrs.Robinson())
    mesh = Hmin.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Spectral", add_colorbar=False)
    ax.coastlines()
    ax.set_title("Shannon Entropy", weight="bold")
    mesh.set_clim(1., 2.)
    
    # Colorbar for entropy
    cax = fig.add_subplot(gs[:, 4])
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend="both")
    cbar.ax.set_yticks([1, 2])
    cbar.set_label("[bits]", labelpad=1)
    pos = cax.get_position()
    new_width = pos.width * 0.4    # thinner
    new_height = pos.height * 0.7   # shorter
    new_x0 = pos.x0 - 0.00      # move left
    new_y0 = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
    cax.set_position([new_x0, new_y0, new_width, new_height])
    
    # Max EMD map
    ax = fig.add_subplot(gs[:, 6], projection=ccrs.Robinson())
    mesh = max_emd.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=myRdPu, add_colorbar=False)
    ax.coastlines()
    ax.set_title("Max. EMD-to-noise across seasons", weight="bold")
    mesh.set_clim(0, 1)
    
    # Colorbar for EMD
    cax = fig.add_subplot(gs[:, 7])
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
    """Main function to generate entropy humidity plot."""
    fig = create_entropy_hurs_plot()
    save_plot(fig, OUTPUT_DIR, 'hurs_discrepancy.jpg', dpi=DPI)


if __name__ == "__main__":
    main()