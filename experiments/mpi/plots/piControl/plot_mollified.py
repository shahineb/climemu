import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import seaborn as sns
from shapely.geometry import box


# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import load_data


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


config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
seasons = np.array([month_to_season[m] for m in piControl_diffusion['month'].values])
piControl_cmip6 = piControl_cmip6.assign_coords(season=("month", seasons)) - climatology
piControl_diffusion = piControl_diffusion.assign_coords(season=("month", seasons))



# Extract data from overmollified regions
esm_data = {}
emulator_data = {}

# DJF hurs central Africa
lat_range = slice(8, 13)
lon_range = slice(14, 33)
piControl_ds = wrap_lon(piControl_cmip6["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
esm_data["DJF hurs CAF"] = piControl_ds.where(piControl_ds.season == "DJF", drop=True).values.ravel()
emulator_data["DJF hurs CAF"] = emulator_ds.where(emulator_ds.season == "DJF", drop=True).values.ravel()

# JJA tas arctic
lat_range = slice(70, 80)
lon_range = slice(160, 220)
piControl_ds = wrap_lon(piControl_cmip6["tas"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["tas"]).sel(lat=lat_range, lon=lon_range).compute()
esm_data["JJA tas arctic"] = piControl_ds.where(piControl_ds.season == "JJA", drop=True).values.ravel()
emulator_data["JJA tas arctic"] = emulator_ds.where(emulator_ds.season == "JJA", drop=True).values.ravel()

# MAM pr india
lat_range = slice(16.768440, 31.441990)
lon_range = slice(67.456055, 79.672852)
piControl_ds = wrap_lon(piControl_cmip6["pr"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["pr"]).sel(lat=lat_range, lon=lon_range).compute()
esm_data["MAM pr india"] = piControl_ds.where(piControl_ds.season == "MAM", drop=True).values.ravel()
emulator_data["MAM pr india"] = emulator_ds.where(emulator_ds.season == "MAM", drop=True).values.ravel()




# Plot
width_ratios  = [1, 1, 1, 0.1, 0.66]
height_ratios = [1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)

fig = plt.figure(figsize=(4 * ncoleff, 2.4 * nroweff))
gs = GridSpec(nrows=nrow,
            ncols=ncol,
            figure=fig,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            hspace=0.05,
            wspace=0.05)

ax = fig.add_subplot(gs[0, 0])
data1 = esm_data["DJF hurs CAF"]
nbins = 3 * np.ceil(2 * len(data1) ** (1 / 3)).astype(int)
sns.histplot(data1, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
data2 = emulator_data["DJF hurs CAF"]
nbins = 3 * np.ceil(2 * len(data2) ** (1 / 3)).astype(int)
sns.histplot(data2, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
ax.set_frame_on(False)
ax.yaxis.set_visible(False)
ax.set_xlabel("[%]")
data12 = np.concatenate([data1, data2])
vmin, vmax = np.quantile(data12, [0.001, 0.999])
ax.set_xlim(vmin, vmax)
ax.set_title("(a)", weight="bold")



ax = fig.add_subplot(gs[0, 1])
data1 = esm_data["JJA tas arctic"]
nbins = 3 * np.ceil(2 * len(data1) ** (1 / 3)).astype(int)
sns.histplot(data1, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
data2 = emulator_data["JJA tas arctic"]
nbins = 3 * np.ceil(2 * len(data2) ** (1 / 3)).astype(int)
sns.histplot(data2, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
ax.set_frame_on(False)
ax.yaxis.set_visible(False)
ax.set_xlabel("[°C]")
data12 = np.concatenate([data1, data2])
vmin, vmax = np.quantile(data12, [0.001, 0.999])
ax.set_xlim(vmin, vmax)
ax.set_title("(b)", weight="bold")



ax = fig.add_subplot(gs[0, 2])
data1 = esm_data["MAM pr india"]
nbins = 3 * np.ceil(2 * len(data1) ** (1 / 3)).astype(int)
vmin, vmax = np.quantile(data1, [0.01, 0.99])
logbins = np.concatenate([-np.logspace(-6, np.log(-vmin), nbins // 2)[::-1], np.zeros(1), np.logspace(-6, np.log(vmax), nbins // 2)])
sns.histplot(data1, ax=ax, kde=False, stat="density", bins=logbins, color="dodgerblue", label="MPI-ESM1-2-LR", edgecolor=None, alpha=0.5)
data2 = emulator_data["MAM pr india"]
nbins = 3 * np.ceil(2 * len(data2) ** (1 / 3)).astype(int)
vmin, vmax = np.quantile(data2, [0.01, 0.99])
logbins = np.concatenate([-np.logspace(-6, np.log(-vmin), nbins // 2)[::-1], np.zeros(1), np.logspace(-6, np.log(vmax), nbins // 2)])
sns.histplot(data2, ax=ax, kde=False, stat="density", bins=logbins, color="tomato", label="Emulator", edgecolor=None, alpha=0.5)
ax.set_frame_on(False)
ax.yaxis.set_visible(False)
ax.set_xlabel("[mm/day]")
data12 = np.concatenate([data1, data2])
vmin, vmax = np.quantile(data12, [0.01, 0.99])
ax.set_xlim(vmin, vmax)
ax.set_title("(c)", weight="bold")
ax.legend(frameon=False)


ax_map = fig.add_subplot(gs[:, -1], projection=ccrs.Robinson())
ax_map.coastlines(linewidth=0.5, color="black")

regions = {
    "DJF hurs CAF": dict(lat=[8, 13], lon=[14, 33], color="red"),
    "JJA tas arctic": dict(lat=[70, 80], lon=[160, 220], color="red"),
    "MAM pr india": dict(lat=[16.768440, 31.441990], lon=[67.456055, 79.672852], color="red"),
}
for r in regions.values():
    lon0, lon1 = r["lon"]
    lat0, lat1 = r["lat"]
    geom = box(lon0, lat0, lon1, lat1)  # makes a rectangle polygon
    ax_map.add_geometries(
        [geom],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=r["color"],
        linewidth=1,
    )


filepath = f'experiments/mpi/plots/piControl/files/mollified.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()