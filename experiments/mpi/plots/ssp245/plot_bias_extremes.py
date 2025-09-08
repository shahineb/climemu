# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import regionmask
from tqdm import tqdm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.utils import arrays
from experiments.mpi.config import Config
from experiments.mpi.plots.ssp245.utils import VARIABLES, load_data
from experiments.mpi.plots.piControl.utils import load_data as load_piControl_data



# %%
config = Config()

test_dataset, pred_data, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds
climatology, _, piControl_cmip6 = load_piControl_data(config, in_memory=False)


target_data = arrays.groupby_month_and_year(target_data) + climatology
pred_data = arrays.groupby_month_and_year(pred_data) + climatology

pred_data = pred_data.compute()
target_data = target_data.compute()

qlevels = [0.99, 0.999, 0.9999]
seasons = ["DJF", "MAM", "JJA", "SON"]
seasons_month = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
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


n_iterations = len(VARIABLES) * len(ar6) * len(seasons)
portrait_data = np.ones((len(qlevels), len(VARIABLES), len(ar6), 4))
portrait_data[:, 1] = 2

# %%
with tqdm(total=n_iterations) as pbar:  
    for j, var in enumerate(VARIABLES.keys()):
        var_info = VARIABLES[var]
        var_name = var_info['name']
        unit = var_info['unit']
        target_data_j = target_data[var]
        pred_data_j = pred_data[var]
        k = 0
        for domain_idx, (domain, domain_name) in enumerate(zip(domains, domain_names)):
            domain_mask = domain.mask(target_data.lon, target_data.lat)
            for region_idx, region in domain.regions.items():
                region_mask = domain_mask == region_idx
                target_data_jk = target_data_j.where(region_mask)
                pred_data_jk = pred_data_j.where(region_mask)
                for l, season in enumerate(seasons):
                    pbar.set_description(f"Processing {var}, {domain_name}, {region.name}, {season}")
                    target_data_jkl = target_data_jk.sel(month=seasons_month[season]).values.ravel()
                    pred_data_jkl = pred_data_jk.sel(month=seasons_month[season]).values.ravel()
                    target_data_jkl = target_data_jkl[~np.isnan(target_data_jkl)]
                    pred_data_jkl = pred_data_jkl[~np.isnan(pred_data_jkl)]
                    q_cmip6 = np.quantile(target_data_jkl, qlevels)
                    q_emulator = np.quantile(pred_data_jkl, qlevels)
                    relative_bias = (q_emulator - q_cmip6) / q_cmip6
                    portrait_data[:, j, k, l]  = relative_bias
                    _ = pbar.update(1)
                k += 1



# %%
abbrevs = []
for d in domains:
    abbrevs += list(map(lambda x: x.abbrev, d.regions.values()))


# %%
def portrait_plot(fig, ax, seasonal_data, vmax, y_labels):
    n_rows, n_cols, n_seasons = seasonal_data.shape        
    patches = []
    colors_flat = []
    norm = colors.TwoSlopeNorm(vmin=-vmax, vmax=vmax, vcenter=0.)
    
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
                           cmap='coolwarm', norm=norm, edgecolor='grey', linewidth=0.2)
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


width_ratios  = [1, 0.5] + [1] * len(abbrevs) + [1]
height_ratios = [0.05, 1, 1, 1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)

fig = plt.figure(figsize=(0.25 * ncoleff, 1 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.1,
              wspace=0.2)
x_labels = abbrevs
y_labels = list(VARIABLES.keys())

ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.text(0.5, 0.5, "99% \n quantile", va="center", ha="center",
            rotation="vertical", fontsize=8, weight="bold")


ax = fig.add_subplot(gs[2, 0])
ax.axis("off")
ax.text(0.5, 0.5, "99.9% \n quantile", va="center", ha="center",
            rotation="vertical", fontsize=8, weight="bold")

ax = fig.add_subplot(gs[3, 0])
ax.axis("off")
ax.text(0.5, 0.5, "99.99% \n quantile", va="center", ha="center",
            rotation="vertical", fontsize=8, weight="bold")


idx = 2 + np.cumsum(list(map(len, domains)))
idx = [4] + idx.tolist()
for i, dname in enumerate(domain_names):
    ax = fig.add_subplot(gs[0, idx[i]:idx[i + 1]])
    ax.axis("off")
    ax.text(0.5, 0.5, dname, ha="center", fontsize=8, weight="bold")

vmax = np.quantile(np.abs(portrait_data), 0.99)

ax = fig.add_subplot(gs[1, 2:])
fig, ax = portrait_plot(fig, ax, portrait_data[0], vmax, y_labels)
ax.get_xaxis().set_visible(False)

ax = fig.add_subplot(gs[2, 2:])
fig, ax = portrait_plot(fig, ax, portrait_data[1], vmax, y_labels)
ax.get_xaxis().set_visible(False)

ax = fig.add_subplot(gs[3, 2:])
fig, ax = portrait_plot(fig, ax, portrait_data[2], vmax, y_labels)
n_rows, n_cols, n_seasons = portrait_data[2].shape
ax.set_xticks(np.arange(n_cols)+0.5)
ax.set_xticklabels(x_labels, rotation=90, ha='center')


# colorbar
cax = fig.add_subplot(gs[1:, -2])
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=-100 * vmax, vmax=100 * vmax)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Relative bias [%]')


bb = cax.get_position()   # Bbox(x0, y0, x1, y1) in figure coords
width = bb.width * 3            # make our legend 80% as wide as the label
height = 5 * width                    # square  
x0 = bb.x0  + (bb.width - width) / 7 # center under the label
y0 = bb.y0 - height*1.2           # drop just below it (10% gap)

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

# plt.show()
plt.savefig("experiments/mpi/plots/ssp245/files/extremes_bias.jpg", dpi=300, bbox_inches='tight')