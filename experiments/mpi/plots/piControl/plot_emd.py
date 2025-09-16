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

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import load_data, VARIABLES, setup_figure, save_plot

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'experiments/mpi/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 3.0
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

def get_plot_data():
    """Compute EMD-to-noise ratios for all variables and seasons."""
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    
    for var in VARIABLES.keys():
        emd_var = dict()
        piControl_diffusion_flat_var = piControl_diffusion_flat[var]
        piControl_cmip6_flat_var = piControl_cmip6_flat[var]
        
        for season in ["DJF", "MAM", "JJA", "SON"]:
            print(f"Computing EMD for {var} in {season}")
            emulator_data = piControl_diffusion_flat_var.where(piControl_diffusion_flat_var.season == season, drop=True)
            esm_data = piControl_cmip6_flat_var.where(piControl_cmip6_flat_var.season == season, drop=True)
            σesm = esm_data.std('flat')
            σesm = σesm.where(σesm > 0.1, 0.1)
            emd_var[season] = compute_emd(emulator_data, esm_data) / σesm
        emd[var] = emd_var
    return emd

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology

# Add seasonal coordinates
piControl_diffusion = add_seasonal_coords(piControl_diffusion)
piControl_cmip6 = add_seasonal_coords(piControl_cmip6)

# Compute EMD data
emd = get_plot_data()



# =============================================================================
# PLOTTING
# =============================================================================

def plot_variable(fig, gs, var, i):
    """Plot EMD maps for a single variable across all seasons."""
    var_info = VARIABLES[var]
    var_name = var_info['name']

    # Variable label
    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, var_name, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")

    # Plot each season
    meshes = []
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        ax = fig.add_subplot(gs[i, j + 1], projection=ccrs.Robinson())
        mesh = emd[var][season].plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap='RdPu', add_colorbar=False
        )
        ax.coastlines()
        meshes.append(mesh)
        if i == 0:
            ax.set_title(f"{season}", fontsize=16, weight="bold")
    
    # Set consistent color limits
    for mesh in meshes:
        mesh.set_clim(0, 1)
    
    # Add colorbar for first variable
    if i == 0:
        cax = fig.add_subplot(gs[1:-1, 5])
        cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend='max')
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticks([0, 0.5, 1])
        cbar.set_label("EMD-to-noise ratio", labelpad=4, fontsize=16, weight="bold")


def create_emd_plot():
    """Create the EMD comparison plot."""
    width_ratios = [0.05, 1, 1, 1, 1, 0.05]
    height_ratios = [1, 1, 1, 1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Plot each variable
    plot_variable(fig, gs, 'tas', 0)
    plot_variable(fig, gs, 'pr', 1)
    plot_variable(fig, gs, 'hurs', 2)
    plot_variable(fig, gs, 'sfcWind', 3)
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate EMD plot."""
    fig = create_emd_plot()
    save_plot(fig, OUTPUT_DIR, 'emd.jpg', dpi=DPI)

if __name__ == "__main__":
    main()