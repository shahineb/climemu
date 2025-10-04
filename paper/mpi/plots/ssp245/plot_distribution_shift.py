import os
import sys
import regionmask
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.ssp245.utils import load_data, VARIABLES, setup_figure, save_plot
from paper.mpi.plots.piControl.utils import load_data as load_piControl_data


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/ssp245/files'
DPI = 300
WIDTH_MULTIPLIER = 2.2
HEIGHT_MULTIPLIER = 1.25
WSPACE = 0.01
HSPACE = 0.1


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def setup_regions():
    """Setup AR6 regions for distribution shift analysis."""
    ar6 = regionmask.defined_regions.ar6.all
    central_africa = ar6[[21, 22]]
    south_america = ar6[[10, 11, 12, 14]]
    arctic = ar6[[46]]
    india = ar6[[37]]
    
    domains = {'tas': central_africa,
               'hurs': india,
               'sfcWind': south_america,
               'pr': arctic}
    
    return ar6, domains


def process_region_data(data, domain_mask, var_name):
    """Process data for a specific region and variable."""
    region_data = data[var_name].where(domain_mask).values.flatten()
    region_data = region_data[~np.isnan(region_data)]
    return region_data


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds

climatology, piControl_diffusion, piControl_cmip6 = load_piControl_data(config, in_memory=False)
piControl_cmip6 = piControl_cmip6 - climatology

# Select time periods
ssp_cmip6 = target_data.sel(time=slice('2080-01', '2100-12'))
ssp_diffusion = pred_samples.sel(time=slice('2080-01', '2100-12'))

# Setup regions
ar6, domains = setup_regions()


# =============================================================================
# PLOTTING
# =============================================================================

def create_map_plot(fig, gs):
    """Create the map showing selected regions."""
    ax = fig.add_subplot(gs[0, -1], projection=ccrs.Robinson())
    ax.coastlines(linewidth=0.2, color="black")
    
    # Plot all regions used in the analysis
    regions_to_plot = [domains['tas'], domains['hurs'], domains['sfcWind'], domains['pr']]
    for region_group in regions_to_plot:
        for region in region_group:
            ax.add_geometries(
                [region.polygon],
                crs=ccrs.PlateCarree(),
                edgecolor="red",
                facecolor="none",
                linewidth=0.6,
                zorder=10,
            )
    return ax


def create_distribution_plots(fig, gs, piControl_cmip6, piControl_diffusion, ssp_cmip6, ssp_diffusion, domains):
    """Create distribution plots for each variable and region."""
    for idx, var_name in enumerate(VARIABLES.keys()):
        var_info = VARIABLES[var_name]
        unit = var_info['unit']
        domain = domains[var_name]
        domain_mask = ~np.isnan(domain.mask(target_data.lon, target_data.lat))

        # Process data for this region and variable
        piControl_data_cmip6 = process_region_data(piControl_cmip6, domain_mask, var_name)
        ssp_data_cmip6 = process_region_data(ssp_cmip6, domain_mask, var_name)
        piControl_data_diffusion = process_region_data(piControl_diffusion, domain_mask, var_name)
        ssp_data_diffusion = process_region_data(ssp_diffusion, domain_mask, var_name)

        # Create subplot for this variable
        subgs = gs[0, idx].subgridspec(2, 1, hspace=-0.75)

        # piControl plot
        ax1 = fig.add_subplot(subgs[0, 0])
        nbins = np.ceil(2 * len(piControl_data_cmip6) ** (1 / 3)).astype(int)
        sns.histplot(piControl_data_cmip6, ax=ax1, kde=False, stat="density", bins=nbins, 
                    color="dodgerblue", alpha=0.6, edgecolor=None, 
                    label=f"{config.data.model_name} piControl")
        sns.histplot(piControl_data_diffusion, ax=ax1, kde=False, stat="density", bins=nbins, 
                    color="tomato", alpha=0.4, edgecolor=None, 
                    label="Emulator piControl")
        ax1.xaxis.set_visible(False)

        # SSP2-4.5 plot
        ax2 = fig.add_subplot(subgs[1, 0])
        nbins = np.ceil(2 * len(ssp_data_cmip6) ** (1 / 3)).astype(int)
        sns.histplot(ssp_data_cmip6, ax=ax2, kde=False, stat="density", bins=nbins, 
                    color="dodgerblue", alpha=0.6, edgecolor=None, 
                    label=f"{config.data.model_name} SSP2-4.5")
        sns.histplot(ssp_data_diffusion, ax=ax2, kde=False, stat="density", bins=nbins, 
                    color="tomato", alpha=0.4, edgecolor=None, 
                    label="Emulator SSP2-4.5")
        ax2.tick_params(axis='x', labelsize=6)

        # Set common axis properties
        flat_values = np.concatenate([piControl_data_cmip6, piControl_data_diffusion, 
                                    ssp_data_diffusion, ssp_data_cmip6])
        vmax = np.quantile(flat_values, 0.99)
        vmin = np.quantile(flat_values, 0.01)
        for ax in [ax1, ax2]:
            ax.set_frame_on(False)
            ax.yaxis.set_visible(False)
            ax.set_xlabel(f"[{unit}]", fontsize=7)
            ax.set_xlim(vmin, vmax)

        # Add period labels (only for first variable)
        if idx == 0:
            ax1.text(0.15, 0.2, "piControl", transform=ax1.transAxes,
                    ha="right", va="top", weight="bold", fontsize=6)
            ax2.text(0.15, -0.1, "SSP2-4.5", transform=ax1.transAxes,
                    ha="right", va="top", weight="bold", fontsize=6)

        # Add variable name as title
        bbox = gs[0, idx].get_position(fig)
        fig.text(bbox.x0 + bbox.width/2, bbox.y1 + 0.06, var_info['name'],
                ha='center', va='top', weight="bold", fontsize=8)


def create_legend(fig):
    """Create the legend for the plot."""
    legend_elements = [
        Line2D([0], [0], color="dodgerblue", lw=6, alpha=0.6, label=config.data.model_name),
        Line2D([0], [0], color="tomato", lw=6, alpha=0.4, label="Emulator"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=1,
              frameon=False, bbox_to_anchor=(0.85, -0.1), fontsize=7)


def create_distribution_shift_plot():
    """Create the main distribution shift plot."""
    # Setup figure
    width_ratios = [1, 1, 1, 1, 0.66]
    height_ratios = [1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    # Create map plot
    create_map_plot(fig, gs)
    
    # Create distribution plots
    create_distribution_plots(fig, gs, piControl_cmip6, piControl_diffusion, ssp_cmip6, ssp_diffusion, domains)
    
    # Create legend
    create_legend(fig)
    
    return fig


def main():
    """Main function to generate distribution shift plot."""
    fig = create_distribution_shift_plot()
    save_plot(fig, OUTPUT_DIR, 'joy.jpg', dpi=DPI)


if __name__ == "__main__":
    main()