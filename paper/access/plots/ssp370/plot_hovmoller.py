import os
import sys
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.access.config import Config
from experiments.access.plots.ssp370.utils import load_data, setup_figure, save_plot
from experiments.access.plots.historical.utils import load_data as load_historical_data


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'experiments/access/plots/ssp370/files'
DPI = 300
WIDTH_MULTIPLIER = 4.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.1
HSPACE = 0.01


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def process_hovmoller_data(da_cmip6, da_diffusion):
    """Process data for Hovmöller diagram."""
    # Select tropical region and compute statistics
    σ_cmip6 = da_cmip6.sel(lat=slice(-30, 30)).mean('lat').std('member').groupby('time.year').mean().compute()
    
    da_cmip6 = da_cmip6.sel(lat=slice(-30, 30)).mean(['member', 'lat']).groupby('time.year').mean().compute()
    da_diffusion = da_diffusion.sel(lat=slice(-30, 30)).mean(['member', 'lat']).groupby('time.year').mean().compute()
    
    # Convert longitude to -180 to 180 range
    lons = da_cmip6['lon'].values
    lons = ((lons + 180) % 360) - 180
    sort_idx = np.argsort(lons)
    lons = lons[sort_idx]
    years = da_cmip6['year'].values
    
    # Sort data according to longitude
    cmip6_vals = da_cmip6.values[:, sort_idx]
    diff_vals = da_diffusion.values[:, sort_idx]
    σ_cmip6_vals = σ_cmip6.values[:, sort_idx]
    bnr = np.abs(diff_vals - cmip6_vals) / σ_cmip6_vals
    
    # Create meshgrid for plotting
    Lon, Year = np.meshgrid(lons, years)
    
    return cmip6_vals, diff_vals, bnr, Lon, Year, lons, years


def compute_contour_levels(cmip6_vals, diff_vals, bnr):
    """Compute contour levels for plotting."""
    # Compute levels for precipitation data
    flat_values = np.concatenate([cmip6_vals, diff_vals])
    vmax = np.quantile(flat_values, 0.99)
    vmin = np.quantile(flat_values, 0.01)
    vmax = max(np.abs(vmax), np.abs(vmin))
    locator = ticker.MaxNLocator(nbins=14, prune=None)
    levels = locator.tick_values(-vmax, vmax)
    
    # Compute levels for error-to-noise ratio
    bnrmax = max(1, np.quantile(bnr, 0.99))
    locator = ticker.MaxNLocator(nbins=14, prune=None)
    bnrlevels = locator.tick_values(0, bnrmax)
    
    return levels, bnrlevels


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp370'].ds

test_dataset_hist, pred_samples_hist, _, __ = load_historical_data(config, in_memory=False)
target_data_hist = test_dataset_hist['historical'].ds

da_cmip6 = xr.concat([target_data_hist['pr'], target_data['pr']], dim='time')
da_diffusion = xr.concat([pred_samples_hist['pr'], pred_samples['pr']], dim='time')

# Process data for Hovmöller diagram
cmip6_vals, diff_vals, bnr, Lon, Year, lons, years = process_hovmoller_data(da_cmip6, da_diffusion)
levels, bnrlevels = compute_contour_levels(cmip6_vals, diff_vals, bnr)


# =============================================================================
# PLOTTING
# =============================================================================

def create_hovmoller_plot():
    """Create the Hovmöller diagram plot."""
    # Setup figure
    width_ratios = [1, 1, 0.05, 0.3, 1, 0.05]
    height_ratios = [1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Longitude tick settings
    xticks = [-120, -60, 0, 60, 120]
    xtick_labels = ["120W", "60W", "0", "60E", "120E"]
    
    # CMIP6 plot
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(Lon, Year, cmip6_vals, levels=levels, extend='both', cmap='BrBG')
    ax1.set_title(config.data.model_name, weight="bold")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_yticks([1900, 2000, 2100])
    
    # Emulator plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.contourf(Lon, Year, diff_vals, levels=levels, extend='both', cmap='BrBG')
    ax2.set_title("Emulator", weight="bold")
    ax2.set_yticklabels([]) 
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels)
    ax2.yaxis.set_visible(False)
    
    # Precipitation colorbar
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(c1, cax=cax)
    cb.set_ticks([-0.25, 0, 0.25])
    cb.set_label("Precipitation anomaly [mm/day]")
    
    # Error-to-noise ratio plot
    ax3 = fig.add_subplot(gs[0, 4])
    c3 = ax3.contourf(Lon, Year, bnr, levels=bnrlevels, extend='max', cmap='RdPu')
    ax3.set_title("Error-to-noise ratio", weight="bold")
    ax3.set_yticklabels([])
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xtick_labels)
    ax3.yaxis.set_visible(False)
    
    # Error-to-noise ratio colorbar
    cax2 = fig.add_subplot(gs[0, 5])
    cb = fig.colorbar(c3, cax=cax2)
    cb.ax.set_yticks([0, 1])
    cb.set_label("[1]")
    
    return fig


def main():
    """Main function to generate Hovmöller diagram."""
    fig = create_hovmoller_plot()
    save_plot(fig, OUTPUT_DIR, 'hovmoller.jpg', dpi=DPI)


if __name__ == "__main__":
    main()