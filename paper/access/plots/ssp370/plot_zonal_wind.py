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
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 2.5
WSPACE = 0.01
HSPACE = 0.01


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def process_zonal_wind_data(da_cmip6, da_diffusion):
    """Process data for zonal wind plot."""
    # Compute zonal mean and statistics
    σ_cmip6 = da_cmip6.mean('lon').std('member').groupby('time.year').mean().compute()
    
    da_cmip6 = da_cmip6.mean(['member', 'lon']).groupby('time.year').mean().compute()
    da_diffusion = da_diffusion.mean(['member', 'lon']).groupby('time.year').mean().compute()
    
    # Transpose for plotting (lat x time)
    cmip6_vals = da_cmip6.values.T
    diff_vals = da_diffusion.values.T
    σ_cmip6_vals = σ_cmip6.where(σ_cmip6 > 0.1, 0.1).values.T
    bnr = np.abs(diff_vals - cmip6_vals) / σ_cmip6_vals
    
    # Get coordinates
    lats = da_cmip6['lat'].values
    years = da_cmip6['year'].values
    
    # Create meshgrid for plotting
    Year, Lat = np.meshgrid(years, lats)
    
    return cmip6_vals, diff_vals, bnr, Year, Lat, lats, years


def compute_contour_levels(cmip6_vals, diff_vals, bnr):
    """Compute contour levels for plotting."""
    # Compute levels for windspeed data
    flat_values = np.concatenate([cmip6_vals, diff_vals])
    vmax = np.quantile(flat_values, 0.99)
    vmin = np.quantile(flat_values, 0.01)
    vmax = max(np.abs(vmax), np.abs(vmin))
    locator = ticker.MaxNLocator(nbins=29, prune=None)
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

da_cmip6 = xr.concat([target_data_hist['sfcWind'], target_data['sfcWind']], dim='time')
da_diffusion = xr.concat([pred_samples_hist['sfcWind'], pred_samples['sfcWind']], dim='time')

# Process data for zonal wind plot
cmip6_vals, diff_vals, bnr, Year, Lat, lats, years = process_zonal_wind_data(da_cmip6, da_diffusion)
levels, bnrlevels = compute_contour_levels(cmip6_vals, diff_vals, bnr)


# =============================================================================
# PLOTTING
# =============================================================================

def create_zonal_wind_plot():
    """Create the zonal wind plot."""
    # Setup figure
    width_ratios = [1, 1, 0.02, 0.05, 0.2, 1, 0.02, 0.05]
    height_ratios = [1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Latitude tick settings
    yticks = [-60, -30, 0, 30, 60]
    ytick_labels = ["60S", "30S", "0", "30N", "60N"]
    
    # CMIP6 plot
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(Year, Lat, cmip6_vals, levels=levels, extend='both', cmap='PRGn')
    ax1.set_title(config.data.model_name, weight="bold")
    ax1.set_ylabel("Latitude")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels)
    ax1.set_xticks([1900, 2000, 2100])
    ax1.set_xticklabels([1900, 2000, 2100])
    
    # Emulator plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.contourf(Year, Lat, diff_vals, levels=levels, extend='both', cmap='PRGn')
    ax2.set_title("Emulator", weight="bold")
    ax2.set_yticks([])
    ax2.set_xticks([1900, 2000, 2100])
    ax2.set_xticklabels([1900, 2000, 2100])
    
    # Windspeed colorbar
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(c1, cax=cax, orientation="vertical", shrink=0.8, pad=0.05)
    cbar.ax.set_yticks([-0.5, 0, 0.5])
    cbar.set_label("Windspeed anomaly [m/s]")
    
    # Error-to-noise ratio plot
    ax3 = fig.add_subplot(gs[0, 5])
    c3 = ax3.contourf(Year, Lat, bnr, levels=bnrlevels, extend='max', cmap='RdPu')
    ax3.set_title("Error-to-noise ratio", weight="bold")
    ax3.set_yticks([])
    ax3.set_xticks([1900, 2000, 2100])
    ax3.set_xticklabels([1900, 2000, 2100])
    
    # Error-to-noise ratio colorbar
    cax2 = fig.add_subplot(gs[0, 7])
    cbar2 = fig.colorbar(c3, cax=cax2, orientation="vertical", shrink=0.8, pad=0.05)
    cbar2.ax.set_yticks([0, 1])
    cbar2.set_label("[1]", labelpad=-1)
    
    return fig


def main():
    """Main function to generate zonal wind plot."""
    fig = create_zonal_wind_plot()
    save_plot(fig, OUTPUT_DIR, 'westerlies.jpg', dpi=DPI)


if __name__ == "__main__":
    main()