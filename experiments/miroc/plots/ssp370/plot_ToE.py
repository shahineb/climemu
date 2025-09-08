import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec


# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.ssp370.utils import load_data, VARIABLES
from experiments.miroc.plots.piControl.utils import load_data as load_piControl_data


def compute_ToE(snr_ds, k=3):
    ToE = {}
    for var in VARIABLES:
        snr = snr_ds[var]
        mask = snr > 2
        mask_rolled = mask.rolling(year=k, center=False).construct("window_dim")
        window_all = mask_rolled.all(dim="window_dim")
        first_idx = window_all.values.argmax(axis=0)
        never_exceeds = ~window_all.values.any(axis=0)
        first_idx[never_exceeds] = -1
        years = snr['year'].values
        start_years = years[:window_all.sizes['year']]
        year_map = np.where(first_idx >= 0, start_years[first_idx], np.nan)
        ToE[var] = xr.DataArray(year_map, coords=[snr.lat, snr.lon], dims=['lat', 'lon'])
    return ToE


config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp370'].ds

climatology, _, piControl_cmip6 = load_piControl_data(config, in_memory=False)
piControl_cmip6 = piControl_cmip6 - climatology
σpiControl = piControl_cmip6.mean('month').std('year').compute()



μ_cmip6 = target_data.groupby('time.year').mean().mean('member')
snr_cmip6 = μ_cmip6 / σpiControl
snr_cmip6 = snr_cmip6.compute()


μ_diffusion = pred_samples.groupby('time.year').mean().mean('member')
snr_diffusion = μ_diffusion / σpiControl
snr_diffusion = snr_diffusion.compute()


ToE_cmip6 = compute_ToE(snr_cmip6)
ToE_diffusion = compute_ToE(snr_diffusion)


# plot
width_ratios  = [0.05, 1, 1, 1, 1, 0.05]
height_ratios = [0.05, 1, 1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(5 * ncoleff, 3.2 * nroweff))

gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.01,
              wspace=0.05)


ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.text(0.5, 0.5, config.data.model_name, va="center", ha="center",
        rotation="vertical", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[2, 0])
ax.axis("off")
ax.text(0.5, 0.5, "Emulator", va="center", ha="center",
        rotation="vertical", fontsize=16, weight="bold")


for i, var in enumerate(VARIABLES):
    var_name = VARIABLES[var]['name']
    ax = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson())
    mesh = ToE_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
    ax.coastlines()
    ax.set_title(f"{var_name}", fontsize=14, weight="bold")
    mesh.set_clim(2015, 2100)

    ax = fig.add_subplot(gs[2, i + 1], projection=ccrs.Robinson())
    mesh = ToE_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='Spectral_r', add_colorbar=False)
    ax.coastlines()
    mesh.set_clim(2015, 2100)

cax = fig.add_subplot(gs[1:, -1])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical')
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_yticks([2020, 2060, 2100])
cbar.set_label(f"Time of emergence", labelpad=0, fontsize=16, weight="bold")

filepath = f'experiments/miroc/plots/ssp370/files/ToE.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')