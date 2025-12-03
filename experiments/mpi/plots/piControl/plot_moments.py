import os
import sys
import numpy as np
from scipy.stats import skew
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar


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
WIDTH_MULTIPLIER = 6.0
HEIGHT_MULTIPLIER = 2.9
WSPACE = 0.05
HSPACE = 0.2

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def subsample_years(ds, n_years):
    unique_years = np.unique(ds.time.dt.year.values)
    idx = np.linspace(0, len(unique_years) - 1, n_years).round().astype(int)
    years_to_keep = unique_years[idx]
    return ds.sel(time=ds.time.dt.year.isin(years_to_keep))

def compute_moments(data, axis):
    """Compute statistical moments (mean, std, skewness) for data."""
    stack = data.stack(flat=axis)
    μ = stack.mean(['flat'])
    σ = stack.std(['flat'])
    γ1 = xr.apply_ufunc(skew, stack, input_core_dims=[['flat']], 
                       kwargs={'axis': -1, 'nan_policy': 'omit'}, 
                       output_dtypes=[float])
    return μ, σ, γ1

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
piControl_cmip6 = subsample_years(piControl_cmip6, piControl_diffusion.sizes['sample'])

# pr = piControl_diffusion["pr"] + climatology["pr"]
# pr = pr.where(piControl_diffusion["wet"] >= 0.5, 0)
# piControl_diffusion["pr"] = pr - climatology["pr"]
# piControl_diffusion = piControl_diffusion.drop_vars("wet")

# piControl_cmip6 = piControl_cmip6 + climatology.sel(dayofyear=piControl_cmip6["time"].dt.dayofyear)

with ProgressBar():
    piControl_diffusion = piControl_diffusion.compute()
    piControl_cmip6 = piControl_cmip6.compute()

# piControl_diffusion['pr'] = piControl_diffusion['pr'].clip(min=0)
# piControl_diffusion['hurs'] = piControl_diffusion['hurs'].clip(min=0, max=100)

# Compute statistical moments for both datasets
μ_diffusion, σ_diffusion, γ1_diffusion = compute_moments(piControl_diffusion, axis=["dayofyear", "sample"])
μ_cmip6, σ_cmip6, γ1_cmip6 = compute_moments(piControl_cmip6, axis=["time"])


# =============================================================================
# PLOTTING
# =============================================================================

def plot_variable(fig, gs, var, i):
    var_info = VARIABLES[var]
    var_name = var_info['name']
    unit = var_info['unit']
    cmap = plt.get_cmap(var_info['cmap'])

    ax = fig.add_subplot(gs[i:i + 3, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, var_name, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, config.data.model_name, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")
    ax = fig.add_subplot(gs[i + 2, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, "Emulator", va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")


    # Plot mean
    flatvalues = []
    flatvalues.append(μ_cmip6[var].values.ravel())
    ax = fig.add_subplot(gs[i, 2], projection=ccrs.Robinson())
    μ_mesh1 = μ_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap, add_colorbar=False)
    ax.coastlines()
    if i == 0:
        ax.set_title("Mean", fontsize=16, weight="bold")


    flatvalues.append(μ_diffusion[var].values.ravel())
    ax = fig.add_subplot(gs[i + 2, 2], projection=ccrs.Robinson())
    μ_mesh2 = μ_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap, add_colorbar=False)
    ax.coastlines()


    # Plot stddev
    cmap_upper = mcolors.LinearSegmentedColormap.from_list("upper_half", cmap(np.linspace(0.5, 1.0, 256)))
    flatvalues.append(σ_cmip6[var].values.ravel())
    ax = fig.add_subplot(gs[i, 3], projection=ccrs.Robinson())
    σ_mesh1 = σ_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap_upper, add_colorbar=False)
    ax.coastlines()
    if i == 0:
        ax.set_title("Standard deviation", fontsize=16, weight="bold")


    flatvalues.append(σ_diffusion[var].values.ravel())
    ax = fig.add_subplot(gs[i + 2, 3], projection=ccrs.Robinson())
    σ_mesh2 = σ_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap_upper, add_colorbar=False)
    ax.coastlines()


    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    vmin = np.quantile(np.concatenate(flatvalues), 0.001)
    vmax = max(np.abs(vmax), np.abs(vmin))
    μ_mesh1.set_clim(-vmax, vmax)
    μ_mesh2.set_clim(-vmax, vmax)
    σ_mesh1.set_clim(0, vmax)
    σ_mesh2.set_clim(0, vmax)

    # Colorbar for mean
    cax = fig.add_subplot(gs[i + 1, 2])
    cbar = fig.colorbar(μ_mesh1,
                        cax=cax,
                        orientation='horizontal',
                        extend='both')
    cbar.locator = ticker.MaxNLocator(nbins=3, integer=True)
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
    pos = cax.get_position()
    new_width = pos.width * 0.4
    new_height = pos.height * 0.6
    new_x0  = pos.x0 + (pos.width - new_width)/2
    cax.set_position([new_x0, pos.y0, new_width, new_height])

    # Colorbar for stddev
    cax = fig.add_subplot(gs[i + 1, 3])
    cbar = fig.colorbar(σ_mesh1,
                        cax=cax,
                        orientation='horizontal',
                        extend='max')
    cbar.locator = ticker.MaxNLocator(nbins=3, integer=True)
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
    pos = cax.get_position()
    new_width = pos.width * 0.4
    new_height = pos.height * 0.6
    new_x0  = pos.x0 + (pos.width - new_width)/2
    cax.set_position([new_x0, pos.y0, new_width, new_height])


    # Plot skew
    flatvalues = []
    flatvalues.append(γ1_cmip6[var].values.ravel())
    ax = fig.add_subplot(gs[i, 4], projection=ccrs.Robinson())
    mesh1 = γ1_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='PiYG', add_colorbar=False)
    ax.coastlines()
    if i == 0:
        ax.set_title("Skewness", fontsize=16, weight="bold")


    flatvalues.append(γ1_diffusion[var].values.ravel())
    ax = fig.add_subplot(gs[i + 2, 4], projection=ccrs.Robinson())
    mesh2 = γ1_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='PiYG', add_colorbar=False)
    ax.coastlines()


    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    vmin = np.quantile(np.concatenate(flatvalues), 0.001)
    vmax = max(np.abs(vmax), np.abs(vmin))
    mesh1.set_clim(-vmax, vmax)
    mesh2.set_clim(-vmax, vmax)

    # Colorbar for skew
    cax = fig.add_subplot(gs[i + 1, 4])
    cbar = fig.colorbar(mesh1,
                        cax=cax,
                        orientation='horizontal',
                        extend='both')
    cbar.locator = ticker.MaxNLocator(nbins=3, integer=True)
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("[1]", fontsize=8, labelpad=4)
    pos = cax.get_position()
    new_width = pos.width * 0.4
    new_height = pos.height * 0.6
    new_x0    = pos.x0 + (pos.width - new_width)/2
    cax.set_position([new_x0, pos.y0, new_width, new_height])



def create_moments_plot():
    """Create the statistical moments comparison plot."""
    width_ratios = [0.06, 0.05, 1, 1, 1]
    height_ratios = [1, 0.05, 1, 0.1] * 4
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Plot each variable
    plot_variable(fig, gs, 'tas', 0)
    plot_variable(fig, gs, 'pr', 4)
    plot_variable(fig, gs, 'hurs', 8)
    plot_variable(fig, gs, 'sfcWind', 12)
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate moments plot."""
    fig = create_moments_plot()
    save_plot(fig, OUTPUT_DIR, 'moments.jpg', dpi=DPI)

if __name__ == "__main__":
    main()