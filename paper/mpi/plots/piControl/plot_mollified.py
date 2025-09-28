import os
import sys
import numpy as np
import cartopy.crs as ccrs
import seaborn as sns
from shapely.geometry import box

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import load_data, setup_figure, save_plot, wrap_lon, add_seasonal_coords

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'experiments/mpi/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 4.0
HEIGHT_MULTIPLIER = 2.4
WSPACE = 0.05
HSPACE = 0.05

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================


def compute_djf_hurs_caf(piControl_cmip6, piControl_diffusion):
    """DJF hurs central Africa data."""
    lat_range = slice(8, 13)
    lon_range = slice(14, 33)
    piControl_ds = wrap_lon(piControl_cmip6["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
    emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
    esm_data = piControl_ds.where(piControl_ds.season == "DJF", drop=True).values.ravel()
    emulator_data = emulator_ds.where(emulator_ds.season == "DJF", drop=True).values.ravel()
    print(f"DJF hurs CAF - ESM: {len(esm_data)}, Emulator: {len(emulator_data)}")
    return esm_data, emulator_data

def compute_jja_tas_arctic(piControl_cmip6, piControl_diffusion):
    """JJA tas arctic data."""
    lat_range = slice(70, 80)
    lon_range = slice(160, 220)
    piControl_ds = wrap_lon(piControl_cmip6["tas"]).sel(lat=lat_range, lon=lon_range).compute()
    emulator_ds = wrap_lon(piControl_diffusion["tas"]).sel(lat=lat_range, lon=lon_range).compute()
    esm_data = piControl_ds.where(piControl_ds.season == "JJA", drop=True).values.ravel()
    emulator_data = emulator_ds.where(emulator_ds.season == "JJA", drop=True).values.ravel()
    print(f"JJA tas arctic - ESM: {len(esm_data)}, Emulator: {len(emulator_data)}")
    return esm_data, emulator_data

def compute_mam_pr_india(piControl_cmip6, piControl_diffusion):
    """MAM pr india data."""
    lat_range = slice(16.768440, 31.441990)
    lon_range = slice(67.456055, 79.672852)
    piControl_ds = wrap_lon(piControl_cmip6["pr"]).sel(lat=lat_range, lon=lon_range).compute()
    emulator_ds = wrap_lon(piControl_diffusion["pr"]).sel(lat=lat_range, lon=lon_range).compute()
    esm_data = piControl_ds.where(piControl_ds.season == "MAM", drop=True).values.ravel()
    emulator_data = emulator_ds.where(emulator_ds.season == "MAM", drop=True).values.ravel()
    print(f"MAM pr india - ESM: {len(esm_data)}, Emulator: {len(emulator_data)}")
    return esm_data, emulator_data

def compute_all_region_data(piControl_cmip6, piControl_diffusion):
    """Compute data for all overmollified regions."""
    esm_data = {}
    emulator_data = {}
    
    esm_data["DJF hurs CAF"], emulator_data["DJF hurs CAF"] = compute_djf_hurs_caf(piControl_cmip6, piControl_diffusion)
    esm_data["JJA tas arctic"], emulator_data["JJA tas arctic"] = compute_jja_tas_arctic(piControl_cmip6, piControl_diffusion)
    esm_data["MAM pr india"], emulator_data["MAM pr india"] = compute_mam_pr_india(piControl_cmip6, piControl_diffusion)
    
    return esm_data, emulator_data

def create_log_bins(data, nbins):
    """Create logarithmic bins for precipitation data."""
    vmin, vmax = np.quantile(data, [0.01, 0.99])
    return np.concatenate([-np.logspace(-6, np.log(-vmin), nbins // 2)[::-1], 
                          np.zeros(1), 
                          np.logspace(-6, np.log(vmax), nbins // 2)])

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
piControl_cmip6 = add_seasonal_coords(piControl_cmip6) - climatology
piControl_diffusion = add_seasonal_coords(piControl_diffusion)

# Compute region data
esm_data, emulator_data = compute_all_region_data(piControl_cmip6, piControl_diffusion)

# =============================================================================
# PLOTTING
# =============================================================================

def create_mollified_plot():
    """Create the mollified analysis plot."""
    width_ratios = [1, 1, 1, 0.1, 0.66]
    height_ratios = [1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Plot 1: DJF hurs central Africa
    ax = fig.add_subplot(gs[0, 0])
    data1 = esm_data["DJF hurs CAF"]
    nbins = 3 * np.ceil(2 * len(data1) ** (1 / 3)).astype(int)
    sns.histplot(data1, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data2 = emulator_data["DJF hurs CAF"]
    nbins = 3 * np.ceil(2 * len(data2) ** (1 / 3)).astype(int)
    sns.histplot(data2, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[%]")
    data12 = np.concatenate([data1, data2])
    vmin, vmax = np.quantile(data12, [0.001, 0.999])
    ax.set_xlim(vmin, vmax)
    ax.set_title("(a)", weight="bold")
    
    # Plot 2: JJA tas arctic
    ax = fig.add_subplot(gs[0, 1])
    data1 = esm_data["JJA tas arctic"]
    nbins = 3 * np.ceil(2 * len(data1) ** (1 / 3)).astype(int)
    sns.histplot(data1, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data2 = emulator_data["JJA tas arctic"]
    nbins = 3 * np.ceil(2 * len(data2) ** (1 / 3)).astype(int)
    sns.histplot(data2, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[°C]")
    data12 = np.concatenate([data1, data2])
    vmin, vmax = np.quantile(data12, [0.001, 0.999])
    ax.set_xlim(vmin, vmax)
    ax.set_title("(b)", weight="bold")
    
    # Plot 3: MAM pr india
    ax = fig.add_subplot(gs[0, 2])
    data1 = esm_data["MAM pr india"]
    nbins = 3 * np.ceil(2 * len(data1) ** (1 / 3)).astype(int)
    logbins = create_log_bins(data1, nbins)
    sns.histplot(data1, ax=ax, kde=False, stat="density", bins=logbins, color="dodgerblue", label="MPI-ESM1-2-LR", edgecolor=None, alpha=0.5)
    data2 = emulator_data["MAM pr india"]
    nbins = 3 * np.ceil(2 * len(data2) ** (1 / 3)).astype(int)
    logbins = create_log_bins(data2, nbins)
    sns.histplot(data2, ax=ax, kde=False, stat="density", bins=logbins, color="tomato", label="Emulator", edgecolor=None, alpha=0.5)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[mm/day]")
    data12 = np.concatenate([data1, data2])
    vmin, vmax = np.quantile(data12, [0.01, 0.99])
    ax.set_xlim(vmin, vmax)
    ax.set_title("(c)", weight="bold")
    ax.legend(frameon=False)
    
    # Map showing regions
    ax_map = fig.add_subplot(gs[:, -1], projection=ccrs.Robinson())
    ax_map.coastlines(linewidth=0.5, color="black")
    
    regions = {
        "DJF hurs CAF": dict(lat=[8, 13], lon=[14, 33], color="red"),
        "JJA tas arctic": dict(lat=[70, 80], lon=[160, 220], color="red"),
        "MAM pr india": dict(lat=[16.768440, 31.441990], lon=[67.456055, 79.672852], color="red"),
    }
    for r in regions.values():
        lon0, lon1 = r["lon"]
        lat0, lat1 = r["lat"]
        geom = box(lon0, lat0, lon1, lat1)  # makes a rectangle polygon
        ax_map.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor=r["color"],
            linewidth=1,
        )
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate mollified plot."""
    fig = create_mollified_plot()
    save_plot(fig, OUTPUT_DIR, 'mollified.jpg', dpi=DPI)

if __name__ == "__main__":
    main()