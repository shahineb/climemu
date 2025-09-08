import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs


# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.ssp370.utils import load_data
from experiments.miroc.plots.historical.utils import load_data as load_historical_data


config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp370'].ds

test_dataset_hist, pred_samples_hist, _, __ = load_historical_data(config, in_memory=False)
target_data_hist = test_dataset_hist['historical'].ds

da_cmip6 = xr.concat([target_data_hist[['tas', 'hurs']], target_data[['tas', 'hurs']]], dim='time')
da_diffusion = xr.concat([pred_samples_hist[['tas', 'hurs']], pred_samples[['tas', 'hurs']]], dim='time')



# Zones definition
land_fraction_filepath = "/home/shahineb/data/cmip6/raw/MIROC6/piControl/r1i1p1f1/sftlf/sftlf_fx_MIROC6_piControl_r1i1p1f1_gn.nc"
land_fraction_ds = xr.open_dataset(land_fraction_filepath)['sftlf']
land_fraction_ds = land_fraction_ds.interp(lat=da_cmip6.lat, lon=da_cmip6.lon, method='linear')
land_mask = land_fraction_ds > 25
land_mask = land_mask.where(land_mask.lat >= -60, 0)
ocean_mask = land_fraction_ds < 25
so_mask = ocean_mask.where(ocean_mask.lat < -55, 0)
to_mask = ocean_mask.where((ocean_mask.lat < 10) & (ocean_mask.lat > -10), 0)
wlat = np.cos(np.deg2rad(da_cmip6.lat))


# Southern Ocean
so_cmip6 = da_cmip6.where(so_mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()
so_ensemble_cmip6 = so_cmip6.groupby('time.year').mean()
μ_so_cmip6 = so_cmip6.mean('member').groupby('time.year').mean()
σ_so_cmip6 = so_cmip6.std('member').groupby('time.year').mean()
ub_so_cmip6 = μ_so_cmip6 + 2 * σ_so_cmip6
lb_so_cmip6 = μ_so_cmip6 - 2 * σ_so_cmip6


so_emulator = da_diffusion.where(so_mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()
μ_so_emulator = so_emulator.mean('member').groupby('time.year').mean()
σ_so_emulator = so_emulator.std('member').groupby('time.year').mean()
ub_so_emulator = μ_so_emulator + 2 * σ_so_emulator
lb_so_emulator = μ_so_emulator - 2 * σ_so_emulator


# Tropical Ocean
to_cmip6 = da_cmip6.where(to_mask == 1).mean(['lat', 'lon'], skipna=True).compute()
to_ensemble_cmip6 = to_cmip6.groupby('time.year').mean()
μ_to_cmip6 = to_cmip6.mean('member').groupby('time.year').mean()
σ_to_cmip6 = to_cmip6.std('member').groupby('time.year').mean()
ub_to_cmip6 = μ_to_cmip6 + 2 * σ_to_cmip6
lb_to_cmip6 = μ_to_cmip6 - 2 * σ_to_cmip6


to_emulator = da_diffusion.where(to_mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()
μ_to_emulator = to_emulator.mean('member').groupby('time.year').mean()
σ_to_emulator = to_emulator.std('member').groupby('time.year').mean()
ub_to_emulator = μ_to_emulator + 2 * σ_to_emulator
lb_to_emulator = μ_to_emulator - 2 * σ_to_emulator


# Arctic
arctic_cmip6 = da_cmip6.sel(lat=slice(80, 90)).mean(['lat', 'lon'], skipna=True).compute()
arctic_ensemble_cmip6 = arctic_cmip6.groupby('time.year').mean()
μ_arctic_cmip6 = arctic_cmip6.mean('member').groupby('time.year').mean()
σ_arctic_cmip6 = arctic_cmip6.std('member').groupby('time.year').mean()
ub_arctic_cmip6 = μ_arctic_cmip6 + 2 * σ_arctic_cmip6
lb_arctic_cmip6 = μ_arctic_cmip6 - 2 * σ_arctic_cmip6


arctic_emulator = da_diffusion.sel(lat=slice(80, 90)).mean(['lat', 'lon'], skipna=True).compute()
μ_arctic_emulator = arctic_emulator.mean('member').groupby('time.year').mean()
σ_arctic_emulator = arctic_emulator.std('member').groupby('time.year').mean()
ub_arctic_emulator = μ_arctic_emulator + 2 * σ_arctic_emulator
lb_arctic_emulator = μ_arctic_emulator - 2 * σ_arctic_emulator


# Land
land_cmip6 = da_cmip6.where(land_mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()
land_ensemble_cmip6 = land_cmip6.groupby('time.year').mean()
μ_land_cmip6 = land_cmip6.mean('member').groupby('time.year').mean()
σ_land_cmip6 = land_cmip6.std('member').groupby('time.year').mean()
ub_land_cmip6 = μ_land_cmip6 + 2 * σ_land_cmip6
lb_land_cmip6 = μ_land_cmip6 - 2 * σ_land_cmip6


land_emulator = da_diffusion.where(land_mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()
μ_land_emulator = land_emulator.mean('member').groupby('time.year').mean()
σ_land_emulator = land_emulator.std('member').groupby('time.year').mean()
ub_land_emulator = μ_land_emulator + 2 * σ_land_emulator
lb_land_emulator = μ_land_emulator - 2 * σ_land_emulator




# Plot
flat_values = xr.concat([ub_so_cmip6, lb_so_cmip6, ub_so_emulator, lb_so_emulator,
                         ub_arctic_cmip6, lb_arctic_cmip6, ub_arctic_emulator, lb_arctic_emulator,
                         ub_land_cmip6, lb_land_cmip6, ub_land_emulator, lb_land_emulator], dim='new', coords='minimal')
vmax = flat_values.quantile(q=0.99, dim=["new", "year"])
vmin = flat_values.quantile(q=0.01, dim=["new", "year"])


time = np.arange(1850, 2101)
land_color = "forestgreen"
arctic_color = "cornflowerblue"
so_color = "b"
to_color = "palevioletred"


width_ratios  = [1, 1, 1, 1, 0.05, 0.66]
height_ratios = [1, 1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(3 * ncoleff, 2 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.01,
              wspace=0.01)


for i, var in enumerate(['tas', 'hurs']):
    ax = fig.add_subplot(gs[i, 0])
    for ω in range(50):
        ax.plot(time, land_ensemble_cmip6[var].isel(member=ω).values, color="gray", lw=0.2, ls='--', alpha=0.2)
    ax.fill_between(time, lb_land_emulator[var], ub_land_emulator[var], color=land_color, alpha=0.2)
    ax.plot(time, μ_land_emulator[var], color=land_color, lw=1, alpha=1)
    if i == 0:
        ax.set_ylabel("Temperature [°C]")
        ax.set_yticks([0, 5, 10])
        ax.set_title("Land", weight="bold")
    elif i == 1:
        ax.set_yticks([-4, 0])
        ax.set_ylabel("Relative humidity [%]")
    ax.set_ylim(vmin[var], vmax[var])
    ax.set_xticks([1900, 2000])
    ax.margins(0)


    ax = fig.add_subplot(gs[i, 1])
    for ω in range(50):
        ax.plot(time, to_ensemble_cmip6[var].isel(member=ω).values, color="gray", lw=0.2, ls='--', alpha=0.2)
    ax.fill_between(time, lb_to_emulator[var], ub_to_emulator[var], color=to_color, alpha=0.2)
    ax.plot(time, μ_to_emulator[var], color=to_color, lw=1, alpha=1)
    ax.yaxis.set_visible(False)
    ax.set_ylim(vmin[var], vmax[var])
    ax.set_xticks([1900, 2000])
    ax.margins(0)
    if i == 0:
        ax.set_title("Tropical Ocean", weight="bold")


    ax = fig.add_subplot(gs[i, 2])
    for ω in range(50):
        ax.plot(time, so_ensemble_cmip6[var].isel(member=ω).values, color="gray", lw=0.2, ls='--', alpha=0.2)
    ax.fill_between(time, lb_so_emulator[var], ub_so_emulator[var], color=so_color, alpha=0.1)
    ax.plot(time, μ_so_emulator[var], color=so_color, lw=1, alpha=1)
    ax.yaxis.set_visible(False)
    ax.set_ylim(vmin[var], vmax[var])
    ax.set_xticks([1900, 2000])
    ax.margins(0)
    if i == 0:
        ax.set_title("Southern Ocean", weight="bold")


    ax = fig.add_subplot(gs[i, 3])
    for ω in range(50):
        ax.plot(time, arctic_ensemble_cmip6[var].isel(member=ω).values, color="gray", lw=0.2, ls='--', alpha=0.2)
    ax.fill_between(time, lb_arctic_emulator[var], ub_arctic_emulator[var], color=arctic_color, alpha=0.2)
    ax.plot(time, μ_arctic_emulator[var], color=arctic_color, lw=1, alpha=1)
    ax.yaxis.set_visible(False)
    ax.set_ylim(vmin[var], vmax[var])
    ax.set_xticks([1900, 2000])
    ax.margins(0)
    if i == 0:
        ax.set_title("Arctic", weight="bold")


proxy_samples = Line2D([0], [0],
                       color='0.3', ls='--', lw=0.5, alpha=0.5)
proxy_mean    = Line2D([0], [0],
                       color='0.01', lw=1)
proxy_fill    = Patch(facecolor='0.3', alpha=0.4)

fig.legend(
    handles=[proxy_samples, proxy_mean, proxy_fill],
    labels=[f"{config.data.model_name} members", "Emulated mean", "Emulator ±2σ"],
    loc='lower center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.45, -0.05)
)


ax = fig.add_subplot(gs[:, -1], projection=ccrs.Robinson())
ax.coastlines(linewidth=0.2, color="black")


arctic_mask = (land_mask.lat >= 80).astype(int).broadcast_like(land_mask)
masks  = [land_mask, so_mask, to_mask, arctic_mask]
colors = [land_color, so_color, to_color, arctic_color]

for mask, color in zip(masks, colors):
    da1 = mask.where(mask == 1)
    cmap = ListedColormap(['none', color])
    ax.pcolormesh(
        mask.lon,
        mask.lat,
        da1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.7,
        shading='auto'
    )


plt.savefig("experiments/miroc/plots/ssp370/files/tas_hurs_trends.jpg", dpi=300, bbox_inches='tight')