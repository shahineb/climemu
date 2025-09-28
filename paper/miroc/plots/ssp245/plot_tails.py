import os
import sys
import numpy as np
import seaborn as sns

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.utils import arrays
from experiments.miroc.config import Config
from experiments.miroc.plots.ssp245.utils import load_data, VARIABLES, setup_figure, save_plot
from experiments.miroc.plots.piControl.utils import load_data as load_piControl_data


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'experiments/miroc/plots/ssp245/files'
DPI = 300
WIDTH_MULTIPLIER = 4.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.1
HSPACE = 0.3


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def process_tail_data(data, var_name):
    """Process data for tail analysis."""
    data_values = data[var_name].values.ravel()
    
    # Convert temperature from Kelvin to Celsius
    if var_name == "tas":
        data_values = data_values - 273.15
    
    return data_values


def compute_tail_quantiles(data):
    """Compute tail quantiles for data."""
    return np.quantile(data, [0.99, 0.999, 0.9999, 0.99999])


def create_tail_histogram(ax, data, bins, color, alpha, label):
    """Create tail histogram plot."""
    sns.histplot(data, ax=ax, kde=False, element="step", stat="density", 
                fill=False, bins=bins, color=color, alpha=alpha, label=label)


def add_quantile_axes(ax, quantiles, color, label_prefix):
    """Add secondary x-axis showing quantile markers."""
    sec_axis = ax.secondary_xaxis('bottom')
    sec_axis.set_xticks(quantiles)
    sec_axis.set_xticklabels([f"{label_prefix}%", f"{label_prefix}.9%", 
                             f"{label_prefix}.99%", f"{label_prefix}.999%"], 
                            color=color, ha='left', fontsize=7, rotation=45)
    sec_axis.xaxis.set_ticks_position('top')
    sec_axis.spines['bottom'].set_color(color)
    sec_axis.tick_params(axis='x', colors=color, pad=0.1)


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds.sel(time=slice('2080-01', '2100-12'))
pred_samples = pred_samples.sel(time=slice('2080-01', '2100-12'))

climatology, _, piControl_cmip6 = load_piControl_data(config, in_memory=False)

# Add climatology and process data
target_data = arrays.groupby_month_and_year(target_data) + climatology
pred_samples = arrays.groupby_month_and_year(pred_samples) + climatology

target_data = target_data.compute()
pred_samples = pred_samples.compute()


# =============================================================================
# PLOTTING
# =============================================================================

def create_tail_plot():
    """Create the tail distribution plot."""
    # Setup figure
    width_ratios = [1, 1, 1, 1]
    height_ratios = [1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    for i, var in enumerate(VARIABLES.keys()):
        var_info = VARIABLES[var]
        var_name = var_info["name"]
        unit = var_info['unit']

        # Process data
        cmip6_data = process_tail_data(target_data, var)
        diffusion_data = process_tail_data(pred_samples, var)
        
        # Update unit for temperature
        if var == "tas":
            unit = "°C"

        # Compute tail quantiles and keep only data past 99th
        qcmip6 = compute_tail_quantiles(cmip6_data)
        qdiffusion = compute_tail_quantiles(diffusion_data)
        γ = min(qcmip6[0], qdiffusion[0]) - 0.1
        cmip6_tail = cmip6_data[cmip6_data >= γ]
        diffusion_tail = diffusion_data[diffusion_data >= γ]

        # Create bins for histplot
        bins_cmip6 = np.histogram_bin_edges(cmip6_tail, bins="fd")
        bins_diffusion = np.histogram_bin_edges(diffusion_tail, bins="fd")

        # Create the plot
        ax = fig.add_subplot(gs[0, i])
        create_tail_histogram(ax, cmip6_tail, bins_cmip6, "dodgerblue", 0.8, f"{config.data.model_name}")
        create_tail_histogram(ax, diffusion_tail, bins_diffusion, "tomato", 0.8, "Emulator")

        # Set plot properties
        ax.set_yscale('log')
        ax.set_xlabel(f"{var_name} [{unit}]")
        if i > 0:
            ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.margins(x=0, y=0)
        if i == 3:
            ax.legend(loc='upper right')

        # Add quantile axes
        add_quantile_axes(ax, qcmip6, "dodgerblue", "99")
        add_quantile_axes(ax, qdiffusion, "tomato", "99")
    
    return fig


def main():
    """Main function to generate tail distribution plot."""
    fig = create_tail_plot()
    save_plot(fig, OUTPUT_DIR, 'tails.jpg', dpi=DPI)


if __name__ == "__main__":
    main()