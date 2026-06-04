import os
import sys
import numpy as np
import jax.numpy as jnp
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.canesm.config import Config
from experiments.canesm.plots.piControl.utils import load_data, setup_figure, save_plot, wrap_lon

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'experiments/canesm/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 1.0
HEIGHT_MULTIPLIER = 1.1
WSPACE = 0.0
HSPACE = 0.0

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def normalize_data(x, μ, vmax, vmin):
    """Normalize data to [-1, 1] range."""
    x = np.asarray(x)
    return 2 * (x - μ) / (vmax - vmin)

def compute_emd_bulk3():
    """Bad EMD bulk (tas arctic july)."""
    lat_range = slice(70, 80)
    lon_range = slice(160, 220)
    data_cmip6 = piControl_cmip6['tas'].sel(lon=lon_range, lat=lat_range, month=7).values.ravel()
    data_diffusion = piControl_diffusion['tas'].sel(lon=lon_range, lat=lat_range, month=7).values.ravel()
    bulk3_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
    bulk3_diffusion = data_diffusion[~np.isnan(data_diffusion)]
    emd_bulk3 = wasserstein_distance(bulk3_cmip6, bulk3_diffusion)
    enr_bulk3 = emd_bulk3 / bulk3_cmip6.std()
    print(f"EMD bulk3: {enr_bulk3:.6f}")
    return bulk3_cmip6, bulk3_diffusion, enr_bulk3

def compute_emd_conc1():
    """Good EMD concentrated (pr pacific)."""
    lat_range = slice(-10, 10)
    lon_range = slice(160, 260)
    data_cmip6 = piControl_cmip6['pr'].sel(lon=lon_range, lat=lat_range, month=2).values.ravel()
    data_diffusion = piControl_diffusion['pr'].sel(lon=lon_range, lat=lat_range, month=2).values.ravel()
    conc1_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
    conc1_diffusion = data_diffusion[~np.isnan(data_diffusion)]
    emd_conc1 = wasserstein_distance(conc1_cmip6, conc1_diffusion)
    enr_conc1 = emd_conc1 / conc1_cmip6.std()
    print(f"EMD conc1: {enr_conc1:.6f}")
    return conc1_cmip6, conc1_diffusion, enr_conc1

def compute_emd_conc2():
    """Poor EMD concentrated (pr mediterranean)."""
    lat_range = slice(25, 41)
    lon_range = slice(-10, 15)
    data_cmip6 = wrap_lon(piControl_cmip6['pr']).sel(lon=lon_range, lat=lat_range, month=7).values.ravel()
    data_diffusion = wrap_lon(piControl_diffusion['pr']).sel(lon=lon_range, lat=lat_range, month=7).values.ravel()
    conc2_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
    conc2_diffusion = data_diffusion[~np.isnan(data_diffusion)]
    emd_conc2 = wasserstein_distance(conc2_cmip6, conc2_diffusion)
    enr_conc2 = emd_conc2 / conc2_cmip6.std()
    print(f"EMD conc2: {enr_conc2:.6f}")
    return conc2_cmip6, conc2_diffusion, enr_conc2

def compute_emd_conc3():
    """Very poor EMD concentrated (pr central africa)."""
    # lon_range = slice(-10, 45)
    # lat_range = slice(5, 15)
    lon_range = slice(-10, 38)
    lat_range = slice(5, 15)
    data_cmip6 = wrap_lon(piControl_cmip6['pr']).sel(lon=lon_range, lat=lat_range, month=12).values.ravel()
    data_diffusion = wrap_lon(piControl_diffusion['pr']).sel(lon=lon_range, lat=lat_range, month=12).values.ravel()
    conc3_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
    conc3_diffusion = data_diffusion[~np.isnan(data_diffusion)]
    emd_conc3 = wasserstein_distance(conc3_cmip6, conc3_diffusion)
    enr_conc3 = emd_conc3 / conc3_cmip6.std()
    print(f"EMD conc3: {enr_conc3:.6f}")
    return conc3_cmip6, conc3_diffusion, enr_conc3

def compute_all_emd_examples():
    """Compute all EMD examples for plotting."""
    return [
        compute_emd_bulk3(),
        compute_emd_conc1(),
        compute_emd_conc2(),
        compute_emd_conc3()
    ]

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
piControl_cmip6 = piControl_cmip6 - climatology

# Compute EMD examples
populations = compute_all_emd_examples()

# =============================================================================
# PLOTTING
# =============================================================================

def create_emd_examples_plot():
    """Create the EMD examples histogram plot."""
    width_ratios = [0.2, 1, 0.3, 1, 1, 1]
    height_ratios = [1, 0.1]

    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    # Add EMD-to-noise label
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(-0.1, -1, "EMD-to-noise =", va="center", ha="center", fontsize=6, weight="bold")

    # Define bins for different plot types
    linbins = np.linspace(-1.5, 1.5, 200)
    logbins = np.concatenate([-np.logspace(-6, 0.17, 100)[::-1], np.zeros(1), np.logspace(-6, 0.17, 100)])

    # columns: 1=tas bulk, then separator, 3=pr conc1, 4=pr conc2, 5=pr conc3
    col_indices = [1, 3, 4, 5]
    use_log = [False, True, True, True]
    labels = 'abcd'
    for idx, (cmip6, diffusion, emdtonoise) in enumerate(populations):
        col = col_indices[idx]

        # Plot histogram
        ax = fig.add_subplot(gs[0, col])
        μ = jnp.concatenate([cmip6, diffusion]).mean()
        vmax = jnp.quantile(jnp.concatenate([cmip6, diffusion]), 0.999)
        vmin = jnp.quantile(jnp.concatenate([cmip6, diffusion]), 0.001)

        if use_log[idx]:
            bins = logbins
            ax.set_ylim(0, 10)
        else:
            bins = linbins
            ax.set_ylim(0, 5)

        sns.histplot(normalize_data(cmip6, μ, vmax, vmin), ax=ax, kde=False, stat="density",
                    bins=bins, color="dodgerblue", edgecolor=None, label=f"{config.data.model_name}")
        sns.histplot(normalize_data(diffusion, μ, vmax, vmin), ax=ax, kde=False, stat="density",
                    bins=bins, color="tomato", edgecolor=None, alpha=0.5, label="Emulator")

        ax.set_frame_on(False)
        ax.set_xlim(-1.5, 1.5)
        ax.yaxis.set_visible(False)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(axis="x", labelsize=4, pad=0, length=0)

        # Add EMD value label
        ax = fig.add_subplot(gs[1, col])
        ax.axis("off")
        ax.text(0.5, -1, f"({labels[idx]}) {emdtonoise:.2f}", va="center", ha="center",
                fontsize=6, weight="bold")

    # Add legend
    legend_elements = [
        Line2D([0], [0], color="dodgerblue", lw=6, alpha=0.6, label=config.data.model_name),
        Line2D([0], [0], color="tomato", lw=6, alpha=0.4, label="Emulator"),
    ]
    fig.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.07, 0.9),
              ncol=1, frameon=False, fontsize=6)

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main function to generate EMD examples plot."""
    fig = create_emd_examples_plot()
    save_plot(fig, OUTPUT_DIR, 'emd_examples.jpg', dpi=DPI)

if __name__ == "__main__":
    main()