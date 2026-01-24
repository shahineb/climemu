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

from paper.mpi.config import Config
from paper.mpi.plots.piControl.utils import VARIABLES, load_data, setup_figure, save_plot, wrap_lon
from paper.mpi.data import load_dataset
from paper.misc.mpi_no_overfitting.config import Config as Confignooverfit
from paper.misc.mpi_no_overfitting.plots.piControl.utils import load_data as load_data_no_overfit

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
    """Load test dataset for historical comparison."""
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

def compute_gmst_data(test_dataset, config, β):
    """Compute GMST data for different experiments."""
    time_range = slice("1850-01", "1900-01")
    gmst_historical = test_dataset.gmst['historical']['tas'].sel(time=time_range).values.ravel()
    
    piControl_dataset = load_dataset(root=config.data.root_dir,
                                model=config.data.model_name,
                                experiments=["piControl"],
                                variables=["tas"],
                                in_memory=False,
                                external_β=β)
    gmst_piControl = piControl_dataset.gmst['piControl']['tas'].values.ravel()
    σpiControl = gmst_piControl.std()
    
    np.random.seed(5)
    gmst_emulator = σpiControl * np.random.randn(gmst_piControl.shape[0])
    
    df = pd.DataFrame({
        "GMST": np.concatenate([gmst_piControl, gmst_historical, gmst_emulator]),
        "Dataset": (["piControl"] * len(gmst_piControl)
                  + ["historical \n (1850-1900)"] * len(gmst_historical)
                  + ["N(0,σ²PI)"] * len(gmst_emulator))
    })
    
    return df

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
_, emulated_ds, _ = load_data(config, in_memory=False)


config_no_overfit = Confignooverfit()
_, emulated_no_overfit_ds, _ = load_data_no_overfit(config_no_overfit, in_memory=False)

# Load test data
test_dataset, β = load_historical_data(config)

# Compute regional data
emulated_no_overfit, emulated, historical = compute_regional_data(emulated_no_overfit_ds, emulated_ds, test_dataset, config)

# Compute GMST data
df = compute_gmst_data(test_dataset, config, β)

# =============================================================================
# PLOTTING
# =============================================================================

def create_overfitting_plot():
    """Create the overfitting analysis plot."""
    width_ratios = [0.6, 0.02, 0.05, 1, 1, 1, 0.05]
    height_ratios = [1, 1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Plot GMST boxplot
    ax = fig.add_subplot(gs[:, 0])
    sns.boxplot(data=df, x="Dataset", y="GMST", ax=ax, fill=False, showfliers=True, color=".2", fliersize=1, legend=False)
    sns.despine()
    ax.set_xlabel("")
    ax.set_ylabel("GMST anomaly [°C]", weight="bold", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    

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

        # Historical map
        ax = fig.add_subplot(gs[i, -4], projection=ccrs.PlateCarree())
        mesh = historical[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.coastlines()
        flatvalues.append(historical[var].values.ravel())
        meshes.append(mesh)
        if i == 0:
            ax.set_title(f"(a) MPI-ESM1-2-LR \n historical (1850-1900)", fontsize=18, weight="bold", pad=5)

        # Emulator map
        ax = fig.add_subplot(gs[i, -3], projection=ccrs.PlateCarree())
        mesh = emulated[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.coastlines()
        flatvalues.append(emulated[var].values.ravel())
        meshes.append(mesh)
        if i == 0:
            ax.set_title("(b) Emulator \n under ΔTₜ∼N(0,σ²PI)", fontsize=18, weight="bold", pad=20)
            ax.text(0.5, 1.02, "Train data includes historical r1-50i1p1f1", transform=ax.transAxes, ha="center", va="bottom", fontstyle="italic", fontsize=12)


        # Emulator no overfit map
        ax = fig.add_subplot(gs[i, -2], projection=ccrs.PlateCarree())
        mesh = emulated_no_overfit[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        ax.coastlines()
        flatvalues.append(emulated_no_overfit[var].values.ravel())
        meshes.append(mesh)
        if i == 0:
            ax.set_title(f"(c) Emulator \n under ΔTₜ∼N(0,σ²PI)", fontsize=18, pad=20, weight="bold")
            ax.text(0.5, 1.02, "Train data includes historical r1i1p1f1 only", transform=ax.transAxes, ha="center", va="bottom", fontstyle="italic", fontsize=12)

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
    save_plot(fig, OUTPUT_DIR, 'overfitting.jpg', dpi=DPI)

if __name__ == "__main__":
    main()