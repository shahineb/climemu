import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import itertools
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.piControl.utils import load_data, setup_figure, save_plot

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'paper/mpi/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.01
HSPACE = 0.01

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================


def pixelwise_correlation_maps(ds):
    """Compute pixelwise correlation maps between all variable pairs."""
    ds_stacked = ds.stack(sample=('year', 'month'))
    varnames = list(ds_stacked.data_vars.keys())
    pairs = list(itertools.combinations(varnames, 2))

    def corr(a, b):
        return np.corrcoef(a, b)[0, 1]

    # Compute for each pair
    corr_dict = {}
    for v1, v2 in pairs:
        name = f'corr_{v1}_{v2}'
        corr_map = xr.apply_ufunc(
            corr,
            ds_stacked[v1], ds_stacked[v2],
            input_core_dims=[['sample'], ['sample']],
            vectorize=True,
        )
        corr_dict[name] = corr_map

    return xr.Dataset(corr_dict)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6_deseasonalized = piControl_cmip6 - climatology

# Compute cross-correlation maps for both datasets
corr_ds_cmip6 = pixelwise_correlation_maps(piControl_cmip6_deseasonalized)
corr_ds_diffusion = pixelwise_correlation_maps(piControl_diffusion)


# =============================================================================
# PLOTTING
# =============================================================================

def add_triangle_background(fig, gs):
    """Add triangular background pattern to distinguish emulator vs CMIP6."""
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    pos = ax.get_position()
    x0, y0 = pos.x0, pos.y0
    w, h = pos.width, pos.height
    x0 = x0 - w
    y0 = y0 + 3 * h

    ax = fig.add_subplot(gs[-1, -3])
    ax.axis("off")
    pos = ax.get_position()
    x1, y1 = pos.x1, pos.y1
    w, h = pos.width, pos.height
    y1 = y1 - h

    triangleup = [(x0, y0), (x1, y1), (x1, y0)]
    triangledown = [(x0, y0), (x1, y1), (x0, y1)]

    tri = mpatches.Polygon(triangleup, transform=fig.transFigure, 
                          facecolor='tomato', zorder=0, alpha=0.1)
    fig.add_artist(tri)
    tri = mpatches.Polygon(triangledown, transform=fig.transFigure, 
                          facecolor='dodgerblue', zorder=0, alpha=0.1)
    fig.add_artist(tri)

def add_legend(fig, gs, config):
    """Add legend to distinguish emulator vs CMIP6."""
    legend_handles = [
        Patch(color='tomato', alpha=0.5, label='Emulator'),
        Patch(color='dodgerblue', alpha=0.5, label=config.data.model_name),
    ]
    ax_leg = fig.add_subplot(gs[-1, -3])
    ax_leg.axis('off')
    ax_leg.legend(handles=legend_handles, loc='center', frameon=False,
                 facecolor='white', fontsize=20)

def add_variable_labels(fig, gs):
    """Add variable labels around the correlation matrix."""
    variables = ['Temperature', 'Precipitation', 'Relative humidity', 'Windspeed']
    
    # Vertical labels (left side)
    for i, var in enumerate(variables):
        ax = fig.add_subplot(gs[i+1, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, var, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")
    
    # Horizontal labels (top side)
    for i, var in enumerate(variables):
        ax = fig.add_subplot(gs[0, i+1])
        ax.axis("off")
        ax.text(0.5, 0.5, var, va="center", ha="center",
                rotation="horizontal", fontsize=16, weight="bold")

def plot_correlation_pair(fig, gs, var_name, row, col, is_emulator=True):
    """Plot a single correlation map."""
    ax = fig.add_subplot(gs[row, col], projection=ccrs.Robinson())
    dataset = corr_ds_diffusion if is_emulator else corr_ds_cmip6
    mesh = dataset[var_name].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='Spectral_r', add_colorbar=False)
    mesh.set_clim(0, 1)
    ax.coastlines()
    return mesh

def create_correlations_plot():
    """Create the correlation matrix plot."""
    width_ratios = [0.05, 1, 1, 1, 1, 0.01, 0.05]
    height_ratios = [0.05, 1, 1, 1, 1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Add background and labels
    add_triangle_background(fig, gs)
    add_legend(fig, gs, config)
    add_variable_labels(fig, gs)
    
    # Define correlation pairs and their positions
    correlation_pairs = [
        ('corr_tas_pr', 1, 2, 2, 1),      # tas-pr: emulator(1,2), cmip6(2,1)
        ('corr_tas_hurs', 1, 3, 3, 1),    # tas-hurs: emulator(1,3), cmip6(3,1)
        ('corr_tas_sfcWind', 1, 4, 4, 1), # tas-sfcWind: emulator(1,4), cmip6(4,1)
        ('corr_pr_hurs', 2, 3, 3, 2),     # pr-hurs: emulator(2,3), cmip6(3,2)
        ('corr_pr_sfcWind', 2, 4, 4, 2),  # pr-sfcWind: emulator(2,4), cmip6(4,2)
        ('corr_hurs_sfcWind', 3, 4, 4, 3) # hurs-sfcWind: emulator(3,4), cmip6(4,3)
    ]
    
    # Plot all correlation pairs
    for var_name, emu_row, emu_col, cmip6_row, cmip6_col in correlation_pairs:
        plot_correlation_pair(fig, gs, var_name, emu_row, emu_col, is_emulator=True)
        mesh = plot_correlation_pair(fig, gs, var_name, cmip6_row, cmip6_col, is_emulator=False)
    
    # Add colorbar
    cax = fig.add_subplot(gs[2:-1, -1])
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_yticks([0, 1])
    cbar.set_label("Correlation", labelpad=0, fontsize=16, weight="bold")
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate correlations plot."""
    fig = create_correlations_plot()
    save_plot(fig, OUTPUT_DIR, 'crosscorrelations.jpg', dpi=DPI)

if __name__ == "__main__":
    main()