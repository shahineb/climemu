# %%
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import itertools
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.access.config import Config
from experiments.access.plots.piControl.utils import load_data, VARIABLES


# %%
def pixelwise_correlation_maps(ds):
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


# %%
config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6_deseasonalized = piControl_cmip6 - climatology



# %%
corr_ds_cmip6 = pixelwise_correlation_maps(piControl_cmip6_deseasonalized)
corr_ds_diffusion = pixelwise_correlation_maps(piControl_diffusion)


# %%
width_ratios  = [0.05, 1, 1, 1, 1, 0.01, 0.05]
height_ratios = [0.05, 1, 1, 1, 1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)

fig = plt.figure(figsize=(5 * ncoleff, 3 * nroweff))

gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.01,
              wspace=0.01)


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


tri = mpatches.Polygon(
    triangleup,
    transform=fig.transFigure,
    facecolor='C1',
    zorder=0,
    alpha=0.1
)
fig.add_artist(tri)
tri = mpatches.Polygon(
    triangledown,
    transform=fig.transFigure,
    facecolor='C0',
    zorder=0,
    alpha=0.1
)
fig.add_artist(tri)


legend_handles = [
    Patch(color='C1', alpha=0.5, label='Emulator'),
    Patch(color='C0', alpha=0.5, label=config.data.model_name),
]
ax_leg = fig.add_subplot(gs[-1, -3])
ax_leg.axis('off')              # turn off ticks & frame
ax_leg.legend(
    handles=legend_handles,
    loc='center',               # center of this subplot
    frameon=False,
    facecolor='white',
    fontsize=20)



ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.text(0.5, 0.5, f"Temperature", va="center", ha="center",
            rotation="vertical", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.text(0.5, 0.5, f"Temperature", va="center", ha="center",
            rotation="horizontal", fontsize=16, weight="bold")

ax = fig.add_subplot(gs[2, 0])
ax.axis("off")
ax.text(0.5, 0.5, f"Precipitation", va="center", ha="center",
            rotation="vertical", fontsize=16, weight="bold")

ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.text(0.5, 0.5, f"Precipitation", va="center", ha="center",
            rotation="horizontal", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[0, 3])
ax.axis("off")
ax.text(0.5, 0.5, f"Relative humidity", va="center", ha="center",
            rotation="horizontal", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[3, 0])
ax.axis("off")
ax.text(0.5, 0.5, f"Relative humidity", va="center", ha="center",
            rotation="vertical", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[4, 0])
ax.axis("off")
ax.text(0.5, 0.5, f"Windspeed", va="center", ha="center",
            rotation="vertical", fontsize=16, weight="bold")

ax = fig.add_subplot(gs[0, 4])
ax.axis("off")
ax.text(0.5, 0.5, f"Windspeed", va="center", ha="center",
            rotation="horizontal", fontsize=16, weight="bold")


var_name = 'corr_tas_pr'
ax = fig.add_subplot(gs[1, 2], projection=ccrs.Robinson())
mesh = corr_ds_diffusion[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()


ax = fig.add_subplot(gs[2, 1], projection=ccrs.Robinson())
mesh = corr_ds_cmip6[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

var_name = 'corr_tas_hurs'
ax = fig.add_subplot(gs[1, 3], projection=ccrs.Robinson())
mesh = corr_ds_diffusion[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

ax = fig.add_subplot(gs[3, 1], projection=ccrs.Robinson())
mesh = corr_ds_cmip6[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

var_name = 'corr_tas_sfcWind'
ax = fig.add_subplot(gs[1, 4], projection=ccrs.Robinson())
mesh = corr_ds_diffusion[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

ax = fig.add_subplot(gs[4, 1], projection=ccrs.Robinson())
mesh = corr_ds_cmip6[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

var_name = 'corr_pr_hurs'
ax = fig.add_subplot(gs[2, 3], projection=ccrs.Robinson())
mesh = corr_ds_diffusion[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

ax = fig.add_subplot(gs[3, 2], projection=ccrs.Robinson())
mesh = corr_ds_cmip6[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()


var_name = 'corr_pr_sfcWind'
ax = fig.add_subplot(gs[2, 4], projection=ccrs.Robinson())
mesh = corr_ds_diffusion[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

ax = fig.add_subplot(gs[4, 2], projection=ccrs.Robinson())
mesh = corr_ds_cmip6[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()


var_name = 'corr_hurs_sfcWind'
ax = fig.add_subplot(gs[3, 4], projection=ccrs.Robinson())
mesh = corr_ds_diffusion[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()

ax = fig.add_subplot(gs[4, 3], projection=ccrs.Robinson())
mesh = corr_ds_cmip6[var_name].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
mesh.set_clim(0, 1)
ax.coastlines()


cax = fig.add_subplot(gs[2:-1, -1])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical')
cbar.ax.tick_params(labelsize=16)
cbar.ax.set_yticks([0, 1])
cbar.set_label(f"Correlation", labelpad=0, fontsize=16, weight="bold")

output_dir = 'experiments/access/plots/piControl/files'
filepath = os.path.join(output_dir, 'crosscorrelations.jpg')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(filepath, dpi=300, bbox_inches='tight')