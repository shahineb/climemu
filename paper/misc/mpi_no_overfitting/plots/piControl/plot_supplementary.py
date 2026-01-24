import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import cartopy.crs as ccrs
import seaborn as sns

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.misc.mpi_no_overfitting.config import Config
from paper.misc.mpi_no_overfitting.plots.piControl.utils import VARIABLES, load_data, setup_figure, save_plot, wrap_lon, add_seasonal_coords
from paper.misc.mpi_no_overfitting.data import load_dataset
from paper.mpi.config import Config as Configmpi
from paper.mpi.plots.piControl.utils import load_data as load_data_mpi

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'paper/misc/mpi_no_overfitting/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.05
HSPACE = 0.05

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def load_historical_data(config):
    """Load dataset for historical comparison."""
    β = jnp.load(config.data.pattern_scaling_path)
    test_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=["historical"],
        variables=config.data.variables,
        in_memory=False,
        external_β=β)
    return test_dataset, β

def compute_regional_data(emulated_no_overfit_ds, emulated_ds, test_dataset, config):
    """Compute regional data for sfcWind and hurs variables."""
    lon_range = slice(-130, -55)
    lat_range = slice(20, 65)
    time_range = slice("1850-01", "1900-01")
    
    emulated_no_overfit = {}
    emulated = {}
    historical = {}
    
    for var in ["sfcWind", "hurs"]:
        emulated_no_overfit[var] = wrap_lon(emulated_no_overfit_ds[var]).sel(lat=lat_range, lon=lon_range).mean(['year', 'month']).compute()
        emulated[var] = wrap_lon(emulated_ds[var]).sel(lat=lat_range, lon=lon_range).mean(['year', 'month']).compute()
        historical[var] = wrap_lon(test_dataset['historical'][var]).sel(time=time_range, lat=lat_range, lon=lon_range).mean(['time', 'member']).compute()
    
    return emulated_no_overfit, emulated, historical


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
_, emulated_no_overfit_ds, _ = load_data(config, in_memory=False)

configmpi = Configmpi()
_, emulated_ds, _ = load_data_mpi(configmpi, in_memory=False)

# Load test data
test_dataset, β = load_historical_data(config)

# Compute regional data
emulated_no_overfit, emulated, historical = compute_regional_data(emulated_no_overfit_ds, emulated_ds, test_dataset, config)


# =============================================================================
# PLOTTING
# =============================================================================

def create_overfitting_plot():
    """Create the overfitting analysis plot."""
    width_ratios = [0.1, 1, 1, 0.05]
    height_ratios = [1, 1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    ax = fig.add_subplot(gs[0, -4])
    ax.axis("off")
    ax.text(0.5, 0.5, "Wind speed", va="center", ha="center", rotation="vertical", fontsize=14, weight="bold")


    ax = fig.add_subplot(gs[1, -4])
    ax.axis("off")
    ax.text(0.5, 0.5, "Relative humidity", va="center", ha="center", rotation="vertical", fontsize=14, weight="bold")

    # Plot variable maps
    for i, var in enumerate(["sfcWind", "hurs"]):
        var_info = VARIABLES[var]
        unit = var_info['unit']
        cmap = var_info['cmap']
        flatvalues = []
        meshes = []

        # Emulator map
        ax = fig.add_subplot(gs[i, -3], projection=ccrs.PlateCarree())
        mesh = emulated[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.coastlines()
        flatvalues.append(emulated[var].values.ravel())
        meshes.append(mesh)
        if i == 0:
            ax.set_title("(a) Train data includes \n historical r1-50i1p1f1", fontsize=16, weight="bold")

        # Historical map
        ax = fig.add_subplot(gs[i, -2], projection=ccrs.PlateCarree())
        mesh = emulated_no_overfit[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.coastlines()
        flatvalues.append(emulated_no_overfit[var].values.ravel())
        meshes.append(mesh)
        if i == 0:
            ax.set_title(f"(b) Train data includes \n historical r1i1p1f1 only", fontsize=16, weight="bold")

        # Set consistent color limits
        vmax = np.quantile(np.concatenate(flatvalues), 0.999)
        vmin = np.quantile(np.concatenate(flatvalues), 0.001)
        vmax = max(np.abs(vmax), np.abs(vmin))
        for mesh in meshes:
            mesh.set_clim(-vmax, vmax)
        
        # Add colorbar
        cax = fig.add_subplot(gs[i, -1])
        cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cbar.set_label(f"[{unit}]", labelpad=4)
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate overfitting plot."""
    fig = create_overfitting_plot()
    save_plot(fig, OUTPUT_DIR, 'overfitting_supplementary.jpg', dpi=DPI)

if __name__ == "__main__":
    main()