import os
import sys
import regionmask
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.ssp245.utils import load_data, VARIABLES
from experiments.miroc.plots.piControl.utils import load_data as load_piControl_data



config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds

climatology, piControl_diffusion, piControl_cmip6 = load_piControl_data(config, in_memory=False)
piControl_cmip6 = piControl_cmip6 - climatology


ssp_cmip6 = target_data.sel(time=slice('2080-01', '2100-12'))
ssp_diffusion = pred_samples.sel(time=slice('2080-01', '2100-12'))


ar6 = regionmask.defined_regions.ar6.all
africa = ar6[[20, 21, 22, 23, 24, 25, 26, 27]]
central_africa = ar6[[21, 22]]
europe = ar6[[16, 17, 18, 19]]
south_america = ar6[[10, 11, 12, 14]]
arctic = ar6[[46]]
india = ar6[[37]]

domains = {'tas': central_africa,
           'hurs': india,
           'sfcWind': south_america,
           'pr': arctic}


width_ratios  = [1, 1, 1, 1, 0.66]
height_ratios = [1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(2.2 * ncoleff, 1.25 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.1,
              wspace=0.01)

ax = fig.add_subplot(gs[0, -1], projection=ccrs.Robinson())
ax.coastlines(linewidth=0.2, color="black")
for _ in [central_africa, south_america, arctic, india]:
    for region in _:
        ax.add_geometries(
            [region.polygon],
            crs=ccrs.PlateCarree(),
            edgecolor="red",
            facecolor="none",
            linewidth=0.6,
            zorder=10,
        )


for idx, var_name in enumerate(VARIABLES.keys()):
    j = idx
    var_info = VARIABLES[var_name]
    unit = var_info['unit']
    domain = domains[var_name]
    domain_mask = ~np.isnan(domain.mask(target_data.lon, target_data.lat))

    piControl_data_cmip6 = piControl_cmip6[var_name].where(domain_mask).values.flatten()
    piControl_data_cmip6 = piControl_data_cmip6[~np.isnan(piControl_data_cmip6)]
    ssp_data_cmip6 = ssp_cmip6[var_name].where(domain_mask).values.flatten()
    ssp_data_cmip6 = ssp_data_cmip6[~np.isnan(ssp_data_cmip6)]

    piControl_data_diffusion = piControl_diffusion[var_name].where(domain_mask).values.flatten()
    piControl_data_diffusion = piControl_data_diffusion[~np.isnan(piControl_data_diffusion)]
    ssp_data_diffusion = ssp_diffusion[var_name].where(domain_mask).values.flatten()
    ssp_data_diffusion = ssp_data_diffusion[~np.isnan(ssp_data_diffusion)]


    subgs = gs[0, j].subgridspec(2, 1, hspace=-0.75)

    ax1 = fig.add_subplot(subgs[0, 0])
    nbins = np.ceil(2 * len(piControl_data_cmip6) ** (1 / 3)).astype(int)
    sns.histplot(piControl_data_cmip6, ax=ax1, kde=False, stat="density", bins=nbins, color="dodgerblue", alpha=0.6, edgecolor=None, label=f"{config.data.model_name} piControl")
    sns.histplot(piControl_data_diffusion, ax=ax1, kde=False, stat="density", bins=nbins, color="tomato", alpha=0.4, edgecolor=None, label="Emulator piControl")
    ax1.xaxis.set_visible(False)

    ax2 = fig.add_subplot(subgs[1, 0])
    nbins = np.ceil(2 * len(ssp_data_cmip6) ** (1 / 3)).astype(int)
    sns.histplot(ssp_data_cmip6, ax=ax2, kde=False, stat="density", bins=nbins, color="dodgerblue", alpha=0.6, edgecolor=None, label=f"{config.data.model_name} piControl")
    sns.histplot(ssp_data_diffusion, ax=ax2, kde=False, stat="density", bins=nbins, color="tomato", alpha=0.4, edgecolor=None, label="Emulator piControl")
    ax2.tick_params(axis='x', labelsize=6)

    flat_values = np.concatenate([piControl_data_cmip6, piControl_data_diffusion, ssp_data_diffusion, ssp_data_cmip6])
    vmax = np.quantile(flat_values, 0.99)
    vmin = np.quantile(flat_values, 0.01)
    for ax in [ax1, ax2]:
        ax.set_frame_on(False)
        ax.yaxis.set_visible(False)
        ax.set_xlabel(f"[{unit}]", fontsize=7)
        ax.set_xlim(vmin, vmax)

    if j == 0:
        ax1.text(
            0.15, 0.2,
            "piControl",
            transform=ax1.transAxes,
            ha="right", va="top", weight="bold",
            fontsize=6
        )
        ax2.text(
            0.15, -0.1,
            "SSP2-4.5",
            transform=ax1.transAxes,
            ha="right", va="top", weight="bold",
            fontsize=6
        )

    bbox = gs[0, j].get_position(fig)

    # place centered text just above it
    fig.text(
        bbox.x0 + bbox.width/2,
        bbox.y1 + 0.06,          
        var_info['name'],
        ha='center', va='top', weight="bold",
        fontsize=8
    )



legend_elements = [
    Line2D([0], [0], color="dodgerblue", lw=6, alpha=0.6, label=config.data.model_name),
    Line2D([0], [0], color="tomato",    lw=6, alpha=0.4, label="Emulator"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=1,
    frameon=False,
    bbox_to_anchor=(0.85, -0.1),
    fontsize=7
)
plt.savefig("experiments/miroc/plots/ssp245/files/joy.jpg", dpi=300, bbox_inches='tight')