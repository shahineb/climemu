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

from experiments.access.config import Config
from experiments.access.plots.piControl.utils import load_data



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


config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
seasons = np.array([month_to_season[m] for m in piControl_diffusion['month'].values])
piControl_cmip6 = piControl_cmip6.assign_coords(season=("month", seasons))
piControl_diffusion = piControl_diffusion.assign_coords(season=("month", seasons))


# Compute max EMD to noise for hurs across seasons
def get_plot_data(piControl_cmip6, piControl_diffusion):
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for season in ["DJF", "MAM", "JJA", "SON"]:
        print(f"Computing EMD for hurs in {season}")
        emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
        esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
        σesm = esm_data.std('flat')
        σesm = σesm.where(σesm > 0.1, 0.1)
        emd[season] = compute_emd(emulator_data, esm_data) / σesm
    return emd
emd = get_plot_data(piControl_cmip6['hurs'] - climatology["hurs"], piControl_diffusion['hurs'])
max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")




# Compute entropy of regions
piControl_data = {}
da = piControl_cmip6["hurs"] - climatology["hurs"]
nbins = np.ceil(2 * (3 * len(da.year)) ** (1 / 3)).astype(int).item()
vmin = da.quantile(0.001).values.item()
vmax = da.quantile(0.999).values.item()

def entropy(samples):
    hist, _ = np.histogram(samples, bins=nbins, range=(vmin, vmax), density=True)
    hist = hist[hist > 0]
    H = -np.sum(hist * np.log2(hist))
    return H

H = xr.apply_ufunc(
    entropy,
    da.groupby('season'),
    input_core_dims=[["year", "month"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
)
Hmin = H.min('season')


# Extract SON data from atlantic segment and colombian coast
hurs_data = {}
hurs_emulator = {}

lat_range = slice(-6, 9)
lon_range = slice(-81, -73)
piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
hurs_data["Colombian coast"] = piControl_ds.where(piControl_ds.season == "SON", drop=True).values.ravel()
hurs_emulator["Colombian coast"] = emulator_ds.where(emulator_ds.season == "SON", drop=True).values.ravel()


lat_range = slice(2, 7)
lon_range = slice(-34, -10)
piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
hurs_data["Equatoral Atlantic Ocean"] = piControl_ds.where(piControl_ds.season == "SON", drop=True).values.ravel()
hurs_emulator["Equatoral Atlantic Ocean"] = emulator_ds.where(emulator_ds.season == "SON", drop=True).values.ravel()


lat_range = slice(-22, -8)
lon_range = slice(-56, -39)
piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
hurs_data["Southern Brasil"] = piControl_ds.where(piControl_ds.season == "SON", drop=True).values.ravel()
hurs_emulator["Southern Brasil"] = emulator_ds.where(emulator_ds.season == "SON", drop=True).values.ravel()


# Plot
width_ratios  = [0.1, 0.7, 0.1, 1, 0.05, 0.08, 1, 0.05]
height_ratios = [0.5, 0.1, 0.5]
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
ax.axis("off")
ax.text(0.5, 0.5, "Colombian coast", va="center", ha="center", rotation="vertical", weight="bold")
ax = fig.add_subplot(gs[0, 1])
data = hurs_data["Colombian coast"]
nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
logbins = np.concatenate([-np.logspace(-3, np.log(10), nbins // 2)[::-1], np.zeros(1), np.logspace(-3, np.log(10), nbins // 2)])
sns.histplot(data, ax=ax, kde=False, stat="density", bins=logbins, color="dodgerblue", edgecolor=None, alpha=0.5)
data = hurs_emulator["Colombian coast"]
sns.histplot(data, ax=ax, kde=False, stat="density", bins=logbins, color="tomato", edgecolor=None, alpha=0.5)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-20, 20)
ax.set_ylim(0, 1)
ax.set_ylabel("")
ax.set_xlabel("[%]")
ax.set_frame_on(False)
ax.set_title("SON relative humidity anomaly", weight="bold")


ax = fig.add_subplot(gs[2, 0])
ax.axis("off")
ax.text(0.5, 0.5, "South Brasil", va="center", ha="center", rotation="vertical", weight="bold")
ax = fig.add_subplot(gs[2, 1])
data = hurs_data["Southern Brasil"]
nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
sns.histplot(data, ax=ax, kde=False, stat="density", bins=nbins, color="dodgerblue", label="ACCESS-ESM1-5", edgecolor=None, alpha=0.5)
data = hurs_emulator["Southern Brasil"]
sns.histplot(data, ax=ax, kde=False, stat="density", bins=nbins, color="tomato", label="Emulator", edgecolor=None, alpha=0.5)
ax.set_yticks([])
ax.set_xlim(-20, 20)
ax.set_ylim(0, 1)
ax.set_ylabel("")
ax.set_xlabel("[%]")
ax.legend(frameon=False)
ax.set_frame_on(False)



ax = fig.add_subplot(gs[:, 3], projection=ccrs.Robinson())
mesh = Hmin.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Spectral", add_colorbar=False)
ax.coastlines()
ax.set_title("Shannon Entropy", weight="bold")
mesh.set_clim(1., 2.)

cax = fig.add_subplot(gs[:, 4])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical',
                    extend="both")
cbar.ax.set_yticks([1, 2])
cbar.set_label(f"[bits]", labelpad=1)
pos = cax.get_position()
new_width  = pos.width * 0.4    # thinner
new_height = pos.height * 0.7   # shorter
new_x0     = pos.x0 - 0.00      # move left
new_y0     = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
cax.set_position([new_x0, new_y0, new_width, new_height])


ax = fig.add_subplot(gs[:, 6], projection=ccrs.Robinson())
mesh = max_emd.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="RdPu", add_colorbar=False)
ax.coastlines()
ax.set_title("Max. EMD-to-noise across seasons", weight="bold")
mesh.set_clim(0, 1)


cax = fig.add_subplot(gs[:, 7])
cbar = fig.colorbar(mesh,
                    cax=cax,
                    orientation='vertical',
                    extend="max")
cbar.ax.set_yticks([0, 1])
cbar.set_label(f"[1]", labelpad=-1)
pos = cax.get_position()
new_width  = pos.width * 0.4    # thinner
new_height = pos.height * 0.7   # shorter
new_x0     = pos.x0 - 0.00      # move left
new_y0     = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
cax.set_position([new_x0, new_y0, new_width, new_height])


filepath = f'experiments/access/plots/piControl/files/hurs_discrepancy.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()




# # %%
# Plot pr distribution for each season over colombian coast
# lat_range = slice(-3, 12)
# lon_range = slice(-81, -60)
lat_range = slice(-6, 9)
lon_range = slice(-81, -73)


piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()

hurs_data = {}
hurs_emulator = {}
for season in ["DJF", "MAM", "JJA", "SON"]:
    data = piControl_ds.where(piControl_ds.season == season, drop=True)
    hurs_data[season] = data.values.ravel()

    data = emulator_ds.where(emulator_ds.season == season, drop=True)
    hurs_emulator[season] = data.values.ravel()


nrows, ncols = 1, 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2), sharey=True)
for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
    data = hurs_data[season]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data = hurs_emulator[season]
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    # ax[i].axvline(np.quantile(data, 0.5), color="red", linestyle="--", linewidth=1)
    ax[i].set_xlim(-20, 20)
    ax[i].set_ylabel("")
    # ax[i].set_yscale('log')
    ax[i].set_yticks([])
    ax[i].set_xlabel("[%]")
    ax[i].set_title(f"{season}")  
    
filepath = f'experiments/access/plots/piControl/files/hursstuff_colombia.jpg'  # SON
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()



# Plot pr distribution for each season over europe
lat_range = slice(37, 60)
lon_range = slice(-12, 32)
piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()


hurs_data = {}
hurs_emulator = {}
for season in ["DJF", "MAM", "JJA", "SON"]:
    data = piControl_ds.where(piControl_ds.season == season, drop=True)
    hurs_data[season] = data.values.ravel()

    data = emulator_ds.where(emulator_ds.season == season, drop=True)
    hurs_emulator[season] = data.values.ravel()


nrows, ncols = 1, 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2), sharey=True)
for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
    data = hurs_data[season]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data = hurs_emulator[season]
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    # ax[i].axvline(np.quantile(data, 0.5), color="red", linestyle="--", linewidth=1)
    ax[i].set_xlim(-20, 20)
    ax[i].set_ylabel("")
    # ax[i].set_yscale('log')
    ax[i].set_yticks([])
    ax[i].set_xlabel("[%]")
    ax[i].set_title(f"{season}")  
    
filepath = f'experiments/access/plots/piControl/files/hursstuff.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()




# Plot hurs distribution for each season over central Africa
lat_range = slice(-3, 7)
lon_range = slice(9, 32)
piControl_ds = wrap_lon(piControl_cmip6["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"] + climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()


hurs_data = {}
hurs_emulator = {}
for season in ["DJF", "MAM", "JJA", "SON"]:
    data = piControl_ds.where(piControl_ds.season == season, drop=True)
    hurs_data[season] = data.values.ravel()

    data = emulator_ds.where(emulator_ds.season == season, drop=True)
    hurs_emulator[season] = data.values.ravel()


nrows, ncols = 1, 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2), sharey=True)
for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
    data = hurs_data[season]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data = hurs_emulator[season]
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    # ax[i].axvline(np.quantile(data, 0.5), color="red", linestyle="--", linewidth=1)
    ax[i].set_xlim(20, 100)
    ax[i].set_ylabel("")
    # ax[i].set_yscale('log')
    ax[i].set_yticks([])
    ax[i].set_xlabel("[%]")
    ax[i].set_title(f"{season}")  
    
filepath = f'experiments/access/plots/piControl/files/hurs_discrepancy.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()




# Plot hurs distribution for each season over norther Russia
# lat_range = slice(51, 59)
# lon_range = slice(73, 116)

# lat_range = slice(43, 50)
# lon_range = slice(79, 130)

# lat_range = slice(60, 83)
# lon_range = slice(-66, -14)

# lat_range = slice(10, 30)
# lon_range = slice(92, 117)

lat_range = slice(48, 60)
lon_range = slice(45, 83)



piControl_ds = wrap_lon(piControl_cmip6["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"] + climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()


hurs_data = {}
hurs_emulator = {}
for season in ["DJF", "MAM", "JJA", "SON"]:
    data = piControl_ds.where(piControl_ds.season == season, drop=True)
    hurs_data[season] = data.values.ravel()

    data = emulator_ds.where(emulator_ds.season == season, drop=True)
    hurs_emulator[season] = data.values.ravel()


nrows, ncols = 1, 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2), sharey=True)
for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
    data = hurs_data[season]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data = hurs_emulator[season]
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    # ax[i].axvline(np.quantile(data, 0.5), color="red", linestyle="--", linewidth=1)
    ax[i].set_xlim(20, 100)
    ax[i].set_ylabel("")
    # ax[i].set_yscale('log')
    ax[i].set_yticks([])
    ax[i].set_xlabel("[%]")
    ax[i].set_title(f"{season}")  
    
filepath = f'experiments/access/plots/piControl/files/hursstuff.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()





# Plot hurs distribution for each season over Europe
lat_range = slice(37, 58)
lon_range = slice(-6, 33)


# Alabama/Arkansas
lat_range = slice(30, 37)
lon_range = slice(-95, -84)


# Atlantic
lat_range = slice(2, 7)
lon_range = slice(-34, -10)


# Arctic
# lat_range = slice(82, 85)
# lon_range = slice(-170, 4)

piControl_ds = wrap_lon(piControl_cmip6["hurs"] - climatology["hurs"]).sel(lat=lat_range, lon=lon_range).compute()
emulator_ds = wrap_lon(piControl_diffusion["hurs"]).sel(lat=lat_range, lon=lon_range).compute()


hurs_data = {}
hurs_emulator = {}
for season in ["DJF", "MAM", "JJA", "SON"]:
    data = piControl_ds.where(piControl_ds.season == season, drop=True)
    hurs_data[season] = data.values.ravel()

    data = emulator_ds.where(emulator_ds.season == season, drop=True)
    hurs_emulator[season] = data.values.ravel()


nrows, ncols = 1, 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2), sharey=True)
for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
    data = hurs_data[season]
    nbins = 2 * np.ceil(2 * len(data) ** (1 / 3)).astype(int)
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="dodgerblue", edgecolor=None, alpha=0.5)
    data = hurs_emulator[season]
    sns.histplot(data, ax=ax[i], kde=False, stat="density", bins=nbins, color="tomato", edgecolor=None, alpha=0.5)
    # ax[i].axvline(np.quantile(data, 0.5), color="red", linestyle="--", linewidth=1)
    ax[i].set_xlim(-20, 20)
    ax[i].set_ylabel("")
    # ax[i].set_yscale('log')
    ax[i].set_yticks([])
    ax[i].set_xlabel("[%]")
    ax[i].set_title(f"{season}")  
    
filepath = f'experiments/access/plots/piControl/files/hursstuff_atlantic.jpg'   # Atlantic SON
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()






# # Get pr statistics
# piControl_data = {}
# piControl_ds = piControl_cmip6["pr"]

# for season in ["DJF", "MAM", "JJA", "SON"]:
#     esm_data = piControl_ds.where(piControl_ds.season == season, drop=True)
#     stack = esm_data.stack(flat=('year', 'month'))
#     piControl_data[season] = stack.quantile(0.5, dim='flat').compute()


# # Plot
# cmap = plt.get_cmap("BrBG")
# cmap_pos = mcolors.LinearSegmentedColormap.from_list("BrBG_pos", cmap(np.linspace(0.5, 1, 256)))

# width_ratios  = [1, 1, 1, 1, 0.05]
# height_ratios = [1]
# nrow = len(height_ratios)
# ncol = len(width_ratios)
# nroweff = sum(height_ratios)
# ncoleff = sum(width_ratios)


# fig = plt.figure(figsize=(5 * ncoleff, 3 * nroweff))
# gs = GridSpec(nrows=nrow,
#               ncols=ncol,
#               figure=fig,
#               width_ratios=width_ratios,
#               height_ratios=height_ratios,
#               hspace=0.05,
#               wspace=0.05)


# flatvalues = []
# meshes = []

# for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
#     ax = fig.add_subplot(gs[0, i], projection=ccrs.Robinson())
#     mesh = piControl_data[season].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Blues", add_colorbar=False)
#     ax.coastlines()
#     flatvalues.append(piControl_data[season].values.ravel())
#     meshes.append(mesh)

# vmax = np.quantile(np.concatenate(flatvalues), 0.999)
# log_norm = mcolors.LogNorm(vmin=1e-1, vmax=vmax)
# for mesh in meshes:
#     mesh.set_norm(log_norm)

# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(mesh,
#                     cax=cax,
#                     orientation='vertical')
# cbar.set_label(f"[mm/day]", labelpad=4)

# filepath = f'experiments/mpi/plots/piControl/files/precips.jpg'
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()






# # Plot EMD to noise with ITCZ lines overlaid
# def get_plot_data(piControl_cmip6, piControl_diffusion):
#     emd = dict()
#     piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
#     piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
#     for season in ["DJF", "MAM", "JJA", "SON"]:
#         print(f"Computing EMD for pr in {season}")
#         emulator_data = piControl_diffusion_flat.where(piControl_diffusion_flat.season == season, drop=True)
#         esm_data = piControl_cmip6_flat.where(piControl_cmip6_flat.season == season, drop=True)
#         σesm = esm_data.std('flat')
#         σesm = σesm.where(σesm > 0.1, 0.1)
#         emd[season] = compute_emd(emulator_data, esm_data) / σesm
#     return emd

# emd = get_plot_data(piControl_cmip6['pr'] - climatology["pr"], piControl_diffusion['pr'])
# max_emd = xr.concat(list(emd.values()), dim="season").max(dim="season")


# jan_itcz_lat = climatology["pr"].sel(month=[1]).mean('month').idxmax(dim="lat")
# jan_itcz_lat = gaussian_filter1d(jan_itcz_lat, sigma=3, mode="wrap")

# july_itcz_lat = climatology["pr"].sel(month=[7]).mean('month').idxmax(dim="lat")
# july_itcz_lat = gaussian_filter1d(july_itcz_lat, sigma=3, mode="wrap")


# piControl_data = {}
# piControl_ds = piControl_cmip6["pr"]
# q95_ds = piControl_ds.quantile(0.95, dim=("year")).compute()
# q05_ds = piControl_ds.quantile(0.05, dim=("year")).compute()


# maxdiff_ds = q95_ds.max(dim="month") / q05_ds.min(dim="month")
# maxdiff_ds = maxdiff_ds.where(maxdiff_ds > 10)



# width_ratios  = [1, 0.05, 1, 0.05]
# height_ratios = [1]
# nrow = len(height_ratios)
# ncol = len(width_ratios)
# nroweff = sum(height_ratios)
# ncoleff = sum(width_ratios)

# fig = plt.figure(figsize=(5 * ncoleff, 3 * nroweff))
# gs = GridSpec(nrows=nrow,
#             ncols=ncol,
#             figure=fig,
#             width_ratios=width_ratios,
#             height_ratios=height_ratios,
#             hspace=0.05,
#             wspace=0.05)


# ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
# mesh = maxdiff_ds.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="Blues", add_colorbar=False)
# ax.coastlines()
# mesh.set_clim(0, 20)


# ax.plot(piControl_ds.lon, july_itcz_lat,
#         transform=ccrs.PlateCarree(), color="red", lw=1, alpha=0.4, label="July ITCZ")
# ax.plot(piControl_ds.lon, jan_itcz_lat,
#         transform=ccrs.PlateCarree(), color="blue", lw=1, alpha=0.4, label="Jan ITCZ")

# cax = fig.add_subplot(gs[0, 1])
# cbar = fig.colorbar(mesh,
#                     cax=cax,
#                     orientation='vertical')
# cbar.set_label(f"[mm/day]", labelpad=4)


# ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
# mesh = max_emd.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="RdPu", add_colorbar=False)
# ax.coastlines()
# mesh.set_clim(0, 1)


# ax.plot(piControl_ds.lon, july_itcz_lat,
#         transform=ccrs.PlateCarree(), color="red", lw=1, alpha=0.4, label="July ITCZ")
# ax.plot(piControl_ds.lon, jan_itcz_lat,
#         transform=ccrs.PlateCarree(), color="blue", lw=1, alpha=0.4, label="Jan ITCZ")

# cax = fig.add_subplot(gs[0, 3])
# cbar = fig.colorbar(mesh,
#                     cax=cax,
#                     orientation='vertical')
# cbar.set_label(f"[1]", labelpad=4)

# filepath = f'experiments/mpi/plots/piControl/files/precips.jpg'
# plt.savefig(filepath, dpi=300, bbox_inches='tight')
# plt.close()