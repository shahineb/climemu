import os
import sys
import numpy as np
import regionmask
from tqdm import tqdm
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import colors
from scipy.stats import wasserstein_distance
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.plots.ssp245.utils import VARIABLES

from paper.mpi.config import Config as MPIConfig
from paper.miroc.config import Config as MIROCConfig
from paper.access.config import Config as ACCESSConfig

from paper.mpi.plots.ssp245.utils import load_data as load_mpi
from paper.miroc.plots.ssp245.utils import load_data as load_miroc
from paper.access.plots.ssp245.utils import load_data as load_access
from paper.intermodel.utils import setup_figure, save_plot, myRdPu


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/intermodel/files'
DPI = 300
WIDTH_MULTIPLIER = 0.25
HEIGHT_MULTIPLIER = 1.0
WSPACE = 0.2
HSPACE = 0.1


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def setup_regions():
    """Setup AR6 regions and domain groupings."""
    ar6 = regionmask.defined_regions.ar6.all
    north_america = ar6[[0, 1, 2, 3, 4, 5, 6, 8]]
    south_america = ar6[[7, 9, 10, 11, 12, 13, 14, 15]]
    europe = ar6[[16, 17, 18, 19]]
    africa = ar6[[20, 21, 22, 23, 24, 25, 26, 27]]
    asia = ar6[[28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]
    se_asia_oceania = ar6[[38, 39, 40, 41, 42, 43]]
    poles = ar6[[44, 45, 46]]
    oceans = ar6[[47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]]
    
    domains = [north_america, south_america, europe, africa, asia, se_asia_oceania, poles, oceans]
    domain_names = ["North America", "South America", "Europe", "Africa", "Asia", "Australasia", "Polar", "Oceans"]
    
    abbrevs = []
    for d in domains:
        abbrevs += list(map(lambda x: x.abbrev, d.regions.values()))
    
    return ar6, domains, domain_names, abbrevs


def load_all_model_data():
    """Load data for all three climate models."""
    config_miroc = MIROCConfig()
    config_mpi = MPIConfig()
    config_access = ACCESSConfig()

    target_data = {}
    pred_data = {}

    test_dataset, pred_data['MPI-ESM1-2-LR'], _, __ = load_mpi(config_mpi, in_memory=False)
    target_data['MPI-ESM1-2-LR'] = test_dataset['ssp245'].ds

    test_dataset, pred_data['MIROC6'], _, __ = load_miroc(config_miroc, in_memory=False)
    target_data['MIROC6'] = test_dataset['ssp245'].ds

    test_dataset, pred_data['ACCESS-ESM1-5'], _, __ = load_access(config_access, in_memory=False)
    target_data['ACCESS-ESM1-5'] = test_dataset['ssp245'].ds

    return target_data, pred_data


def compute_emd_data(target_data, pred_data, model_names, periods, seasons, domains, domain_names, ar6):
    """Compute EMD (Earth Mover's Distance) data for all models and periods."""
    n_iterations = len(model_names) * len(periods) * len(VARIABLES) * len(ar6) * len(seasons)
    portrait_data = {m: np.zeros((len(periods), len(VARIABLES), len(ar6), 4)) for m in model_names}
    
    with tqdm(total=n_iterations) as pbar:
        for m in model_names:
            for i, period in enumerate(periods):
                target_data_i = target_data[m].sel(time=period).load()
                pred_data_i = pred_data[m].sel(time=period).load()
                for j, var in enumerate(VARIABLES.keys()):
                    target_data_ij = target_data_i[var]
                    pred_data_ij = pred_data_i[var]
                    k = 0
                    for domain_idx, (domain, domain_name) in enumerate(zip(domains, domain_names)):
                        domain_mask = domain.mask(target_data[m].lon, target_data[m].lat)
                        for region_idx, region in domain.regions.items():
                            region_mask = domain_mask == region_idx
                            target_data_ijk = target_data_ij.where(region_mask)
                            pred_data_ijk = pred_data_ij.where(region_mask)
                            for season_idx, season in enumerate(seasons):
                                pbar.set_description(f"Processing {m} {period.start}-{period.stop}, {var}, {domain_name}, {region.name}, {season}")
                                target_data_ijkl = target_data_ijk.sel(time=target_data_ijk['time.season'] == season).values.flatten()
                                pred_data_ijkl = pred_data_ijk.sel(time=pred_data_ijk['time.season'] == season).values.flatten()
                                target_data_ijkl = target_data_ijkl[~np.isnan(target_data_ijkl)]
                                pred_data_ijkl = pred_data_ijkl[~np.isnan(pred_data_ijkl)]
                                emd = wasserstein_distance(target_data_ijkl, pred_data_ijkl).item()
                                σtarget = target_data_ijkl.std()
                                σ = max(σtarget, 0.1)
                                portrait_data[m][i, j, k, season_idx] = emd / σ
                                _ = pbar.update(1)
                            k += 1
    
    return portrait_data


# =============================================================================
# DATA LOADING
# =============================================================================

# Load data for all models
target_data, pred_data = load_all_model_data()

# Setup regions and parameters
ar6, domains, domain_names, abbrevs = setup_regions()
model_names = ['MPI-ESM1-2-LR', 'MIROC6', 'ACCESS-ESM1-5']
periods = [slice("2040-01", "2060-12"), slice("2080-01", "2100-12")]
seasons = ["DJF", "MAM", "JJA", "SON"]

# Compute EMD data
portrait_data = compute_emd_data(target_data, pred_data, model_names, periods, seasons, domains, domain_names, ar6)


# =============================================================================
# PLOTTING
# =============================================================================

def portrait_plot(fig, ax, seasonal_data, y_labels):
    """Create portrait plot with triangular seasonal cells."""
    n_rows, n_cols, n_seasons = seasonal_data.shape        
    patches = []
    colors_flat = []
    norm = colors.TwoSlopeNorm(vmin=0., vmax=1, vcenter=0.5)
    
    for i in range(n_rows):
        for j in range(n_cols):
            vals = seasonal_data[i, j]  # length 4
            
            # corners of the cell:
            x0, x1 = j,   j+1
            y0, y1 = i,   i+1
            xm, ym = (x0+x1)/2, (y0+y1)/2
            
            # triangles: 
            #   bottom (DJF)      = [(x0,y0),(x1,y0),(xm,ym)]
            #   right  (MAM)      = [(x1,y0),(x1,y1),(xm,ym)]
            #   top    (JJA)      = [(x1,y1),(x0,y1),(xm,ym)]
            #   left   (SON)      = [(x0,y1),(x0,y0),(xm,ym)]
            corners = [
                [(x0,y0),(x1,y0),(xm,ym)],  # DJF
                [(x1,y0),(x1,y1),(xm,ym)],  # MAM
                [(x1,y1),(x0,y1),(xm,ym)],  # JJA
                [(x0,y1),(x0,y0),(xm,ym)],  # SON
            ]
            
            for k, tri in enumerate(corners):
                patches.append(Polygon(tri))
                colors_flat.append(vals[k])
    
    coll = PatchCollection(patches, array=np.array(colors_flat),
                           cmap=myRdPu, norm=norm, edgecolor='grey', linewidth=0.2)
    ax.add_collection(coll)
    
    # set ticks halfway through each cell
    ax.set_yticks(np.arange(n_rows)+0.5)
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # invert y so first row is at top
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    return fig, ax


def create_season_legend(fig, cax):
    """Create season legend showing triangular cell layout."""
    bb = cax.get_position()   # Bbox(x0, y0, x1, y1) in figure coords
    width = bb.width * 3            # make our legend 80% as wide as the label
    height = 2 * width                    # square  
    x0 = bb.x0  + (bb.width - width) / 3 # center under the label
    y0 = bb.y0 - height*1.5           # drop just below it (10% gap)

    # ——— add a standalone Axes for the season‐legend ———
    ax_legend = fig.add_axes([x0, y0, width, height])
    ax_legend.axis("off")

    # 1) draw the square
    sq = Rectangle((0, 0), 1, 1, fill=False, lw=1.0,
                   transform=ax_legend.transAxes)
    ax_legend.add_patch(sq)

    ax_legend.plot([0,1], [0,1],
                   transform=ax_legend.transAxes, color='k', lw=0.5)
    ax_legend.plot([0,1], [1,0],
                   transform=ax_legend.transAxes, color='k', lw=0.5)
    ax_legend.text(0.5, 0.1, "DJF",
                   ha="center", va="bottom",
                   transform=ax_legend.transAxes, fontsize=6)
    ax_legend.text(0.58, 0.5, "MAM",
                   ha="left", va="center",
                   transform=ax_legend.transAxes, fontsize=6)
    ax_legend.text(0.5, 0.85, "JJA",
                   ha="center", va="top",
                   transform=ax_legend.transAxes, fontsize=6)
    ax_legend.text(0.4, 0.5, "SON",
                   ha="right", va="center",
                   transform=ax_legend.transAxes, fontsize=6)


def create_intermodel_portrait_plot():
    """Create the main intermodel portrait plot."""
    # Setup figure
    width_ratios = [1, 1] + [1] * len(abbrevs) + [1]
    height_ratios = [0.05,
                     1, 0.05, 1, 0.25,
                     1, 0.05, 1, 0.25,
                     1, 0.05, 1]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)
    
    x_labels = abbrevs
    y_labels = list(VARIABLES.keys())
    
    # Add domain headers
    idx = 3 + np.cumsum(list(map(len, domains)))
    idx = [4] + idx.tolist()
    for i, dname in enumerate(domain_names):
        ax = fig.add_subplot(gs[0, idx[i]:idx[i + 1]])
        ax.axis("off")
        ax.text(0.5, 0.5, dname, ha="center", fontsize=8, weight="bold")
    
    # Create plots for each model
    for i, m in enumerate(model_names):
        # Model name label
        ax = fig.add_subplot(gs[4 * i + 1:4 * i + 4, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, m, va="center", ha="center",
                rotation="vertical", fontsize=10, weight="bold")
        
        # Period labels
        ax = fig.add_subplot(gs[4 * i + 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, "2040-2060", va="center", ha="center",
                    rotation="vertical", fontsize=8, weight="bold")

        ax = fig.add_subplot(gs[4 * i + 3, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, "2080-2100", va="center", ha="center",
                    rotation="vertical", fontsize=8, weight="bold")

        # Portrait plots
        ax = fig.add_subplot(gs[4 * i + 1, 1:])
        fig, ax = portrait_plot(fig, ax, portrait_data[m][0], y_labels)
        ax.get_xaxis().set_visible(False)

        ax = fig.add_subplot(gs[4 * i + 3, 1:])
        fig, ax = portrait_plot(fig, ax, portrait_data[m][1], y_labels)
        if i == len(model_names) - 1:
            n_rows, n_cols, n_seasons = portrait_data[m][1].shape
            ax.set_xticks(np.arange(n_cols)+0.5)
            ax.set_xticklabels(x_labels, rotation=90, ha='center')
        else:
            ax.get_xaxis().set_visible(False)
    
    # Add colorbar
    cax = fig.add_subplot(gs[5:8, -2])
    cmap = myRdPu
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, extend='max')
    cbar.ax.set_yticks([0, 1])
    cbar.set_label('EMD-to-noise ratio [1]')
    
    # Add season legend
    create_season_legend(fig, cax)
    
    return fig


def main():
    """Main function to generate intermodel portrait plot."""
    fig = create_intermodel_portrait_plot()
    save_plot(fig, OUTPUT_DIR, 'portrait.jpg', dpi=DPI)


if __name__ == "__main__":
    main()