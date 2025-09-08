import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import seaborn as sns
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.piControl.utils import VARIABLES, load_data


def compute_emd(foo, bar):
    emd = xr.apply_ufunc(
        wasserstein_distance,
        foo,
        bar,
        input_core_dims=[['flat'], ['flat']],
        exclude_dims={'flat'},
        vectorize=True,
        output_dtypes=[float],
        dask="parallelized")
    return emd


def wrap_lon(ds):
    # assumes ds.lon runs 0…360
    lon360 = ds.lon.values
    lon180 = ((lon360 + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon180).sortby("lon")
    return ds


month_to_season = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3:  "MAM", 4:  "MAM", 5:  "MAM",
    6:  "JJA", 7:  "JJA", 8:  "JJA",
    9:  "SON", 10: "SON", 11: "SON"
}


# %%
config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
seasons = np.array([month_to_season[m] for m in piControl_diffusion['month'].values])
piControl_cmip6 = piControl_cmip6.assign_coords(season=("month", seasons))
piControl_diffusion = piControl_diffusion.assign_coords(season=("month", seasons))
piControl_diffusion = piControl_diffusion + climatology



# Compute max EMD to noise for sfcWind across seasons
def get_plot_data(piControl_cmip6, piControl_diffusion):
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        print(f"Computing EMD for sfcWind in {season}")
        emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
        esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
        σesm = esm_data.std('flat')
        σesm = σesm.where(σesm > 0.1, 0.1)
        emd[season] = compute_emd(emulator_data, esm_data) / σesm
    return emd

emd = get_plot_data(piControl_cmip6['sfcWind'] - climatology['sfcWind'], piControl_diffusion['sfcWind'])
max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")




# Compute sfcWind data over south american moonsoon region
lon_range = slice(-54, -38)
lat_range = slice(-13, -5)
piControl_ds = wrap_lon(piControl_cmip6["sfcWind"]).sel(lat=lat_range, lon=lon_range)
emulator_ds = wrap_lon(piControl_diffusion["sfcWind"]).sel(lat=lat_range, lon=lon_range)

sfcWind_data = piControl_ds.values.ravel()
sfcWind_emulator = emulator_ds.values.ravel()


# Find regions that have a strong concentration of zero windspeed signal
piControl_ds = piControl_cmip6["sfcWind"]
q10_ds = piControl_ds.quantile(0.1, dim=("year", "month")).compute()



# Plot
width_ratios  = [0.7, 0.1, 1, 0.05, 0.08, 1, 0.05]
height_ratios = [1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)

fig = plt.figure(figsize=(6 * ncoleff, 3.5 * nroweff))
gs = GridSpec(nrows=nrow,
            ncols=ncol,
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            hspace=0.05,
            wspace=0.05)


ax = fig.add_subplot(gs[0, 0])
nbins = 2 * np.ceil(2 * len(sfcWind_data) ** (1 / 3)).astype(int)
sns.histplot(sfcWind_data, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", alpha=0.6, edgecolor=None, label="MIROC6")
nbins = 2 * np.ceil(2 * len(sfcWind_emulator) ** (1 / 3)).astype(int)
sns.histplot(sfcWind_emulator, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", alpha=0.6, edgecolor=None, label="Emulator")
ax.set_yticks([])
ax.set_yscale('log')
ax.set_xlabel("[m/s]")
ax.legend()
ax.set_title("Windspeed distribution over \n South American Moonsoon region", weight="bold")

ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
mesh = q10_ds.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Greens", add_colorbar=False)
ax.coastlines()
ax.set_title("10th percentile of piControl windpseed", weight="bold")
mesh.set_clim(0, 2)

cax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical', extend='max')
cbar.set_label(f"[m/s]", labelpad=-1)
pos = cax.get_position()
new_width  = pos.width * 0.4    # thinner
new_height = pos.height * 0.7   # shorter
new_x0     = pos.x0 - 0.00      # move left
new_y0     = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
cax.set_position([new_x0, new_y0, new_width, new_height])


ax = fig.add_subplot(gs[0, 5], projection=ccrs.Robinson())
mesh = max_emd.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="RdPu", add_colorbar=False)
ax.coastlines()
ax.set_title("Max. EMD-to-noise across seasons", weight="bold")
mesh.set_clim(0, 1)

cax = fig.add_subplot(gs[0, 6])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical')
cbar.ax.set_yticks([0, 1])
cbar.set_label(f"[1]", labelpad=-1)
pos = cax.get_position()
new_width  = pos.width * 0.4    # thinner
new_height = pos.height * 0.7   # shorter
new_x0     = pos.x0 - 0.00      # move left
new_y0     = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
cax.set_position([new_x0, new_y0, new_width, new_height])


filepath = f'experiments/miroc/plots/piControl/files/windspeed_discrepancy.jpg'
plt.savefig(filepath, dpi=350, bbox_inches='tight')
plt.close()