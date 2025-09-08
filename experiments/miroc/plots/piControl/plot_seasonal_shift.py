# %%
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.piControl.utils import VARIABLES, load_data


# %%
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


# %%
# Compute max EMD to noise for pr across seasons
def get_plot_data(piControl_cmip6, piControl_diffusion):
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        print(f"Computing EMD for pr in {season}")
        emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
        esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
        σesm = esm_data.std('flat')
        σesm = σesm.where(σesm > 0.1, 0.1)
        emd[season] = compute_emd(emulator_data, esm_data) / σesm
    return emd

emd = get_plot_data(piControl_cmip6['pr'] - climatology["pr"], piControl_diffusion['pr'])
max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")



# %%
# Compute pr data for each month over 3 regions
lat_range = {"Central Africa": slice(5, 18),
             "India": slice(15, 30),
             "Northern Australia": slice(-20, -5)}

lon_range = {"Central Africa": slice(-20, 40),
             "India": slice(65, 80),
             "Northern Australia": slice(115, 145)}


pr_data = {}
for region in lat_range:
    piControl_ds = wrap_lon(piControl_cmip6["pr"]).sel(lat=lat_range[region], lon=lon_range[region])
    pr_data[region] = []
    for month in range(1, 13):
        pr_data[region].append(piControl_ds.sel(month=month).values.ravel())
    pr_data[region] = np.asarray(pr_data[region])


# %%
# Find regions that oscillate between wet and dry seasonally
piControl_data = {}
piControl_ds = piControl_cmip6["pr"]
q95_ds = piControl_ds.quantile(0.95, dim=("year")).compute()

maxq95 = q95_ds.max(dim="month")
minq95 = q95_ds.min(dim="month")
minq95 = minq95.where(minq95 > 0.1, 0.1)
precipratio = maxq95 / minq95


# %%
# Plot precipratio and max EMD to noise
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
for region in pr_data:
    q95 = np.quantile(pr_data[region], 0.95, axis=1)
    m = list(range(12))
    ax.plot(m, q95, marker="o", ms=3, lw=2, label=region)
ax.margins(0.03)
ax.set_xticks(m)
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("[mm/day]")
ax.legend()
ax.set_title("95th percentile of \n monthly precipitations", weight="bold")


ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
mesh = precipratio.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Spectral_r", add_colorbar=False)
ax.coastlines()
ax.set_title("95th percentile max-min ratio", weight="bold")
mesh.set_clim(0, 100)

cax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical')
cbar.ax.set_yticks([0, 50, 100])
cbar.set_label(f"[1]", labelpad=-1)
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


filepath = f'experiments/miroc/plots/piControl/files/precip_discrepancy.jpg'
plt.savefig(filepath, dpi=500, bbox_inches='tight')
plt.close()