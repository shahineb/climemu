# %%
import os
import numpy as np
import xarray as xr
import pandas as pd
import regionmask
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from src.datasets import CMIP6Data
import cartopy.crs as ccrs


# %%

CLIMATOLOGY_ROOT = "/home/shahineb/data/products/cmip6/processed"
CLIMATOLOGY_MODEL = 'MPI-ESM1-2-LR'
CLIMATOLOGY_MEMBER = 'r1i1p1f1'

months = np.arange(1, 13).astype('int64')
base_path = os.path.join(CLIMATOLOGY_ROOT, CLIMATOLOGY_MODEL, 'piControl', CLIMATOLOGY_MEMBER)
climatology_tas = xr.open_dataset(os.path.join(base_path, f"tas_climatology/Amon/tas_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc"))
climatology_tas = climatology_tas.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time') - 273.15
climatology_pr = xr.open_dataset(os.path.join(base_path, f"pr_climatology/Amon/pr_Amon_{CLIMATOLOGY_MODEL}_piControl_{CLIMATOLOGY_MEMBER}_monthly_climatology.nc"))
climatology_pr = climatology_pr.assign_coords(month=('time', months)).swap_dims({'time': 'month'}).drop_vars('time') * 86400
climatology = xr.merge([climatology_tas, climatology_pr])


root = "/orcd/data/raffaele/001/shahineb/products/cmip6/processed"
cmip6data = CMIP6Data(root, "MPI-ESM1-2-LR", ["ssp126", "ssp585"], ["tas", "pr"], {"time": slice("2100-01", "2100-12")})
cmip6data.load()
ssp126 = cmip6data["ssp126"].ds + climatology.sel(month=cmip6data["ssp126"].time.dt.month)
ssp585 = cmip6data["ssp585"].ds + climatology.sel(month=cmip6data["ssp585"].time.dt.month)

cmip7 = xr.open_dataset("/home/shahineb/data/emulated/climemu-private/cmip7_medium-extension/2100.nc")[["tas", "pr"]]
cmip7.load()
cmip7 = cmip7 + climatology


ar6 = regionmask.defined_regions.ar6.all
region_idx = 10
region = ar6[[region_idx]]
print(region)
region_mask = ~np.isnan(region.mask(cmip7.lon, cmip7.lat))
ssp126_region = ssp126.where(region_mask)
ssp585_region = ssp585.where(region_mask)
cmip7_region = cmip7.where(region_mask)

ssp126_tas = ssp126_region['tas'].values.ravel()
nan_mask = np.isnan(ssp126_tas)
ssp126_tas = ssp126_tas[~nan_mask]
ssp126_pr = ssp126_region['pr'].values.ravel()[~nan_mask]
ssp585_tas = ssp585_region['tas'].values.ravel()[~nan_mask]
ssp585_pr = ssp585_region['pr'].values.ravel()[~nan_mask]

cmip7_tas = cmip7_region['tas'].values.ravel()
nan_mask = np.isnan(cmip7_tas)
cmip7_tas = cmip7_tas[~nan_mask]
cmip7_pr = cmip7_region['pr'].values.ravel()[~nan_mask].clip(min=0)

ssp126_logpr = np.log1p(ssp126_pr)
ssp585_logpr = np.log1p(ssp585_pr)
cmip7_logpr = np.log1p(cmip7_pr)

N = 10000
np.random.seed(5)
subset_idx = np.random.choice(len(ssp126_tas), N, replace=False)
ssp126_tas, ssp126_logpr = ssp126_tas[subset_idx], ssp126_logpr[subset_idx]
ssp585_tas, ssp585_logpr = ssp585_tas[subset_idx], ssp585_logpr[subset_idx]
subset_idx = np.random.choice(len(cmip7_tas), N, replace=False)
cmip7_tas, cmip7_logpr = cmip7_tas[subset_idx],  cmip7_logpr[subset_idx]

# df = pd.DataFrame(data=np.stack([ssp126_tas, ssp126_logpr,
#                         ssp585_tas, ssp585_logpr,
#                         cmip7_tas, cmip7_logpr], axis=-1),
#                   columns=["ssp126_tas", "ssp126_logpr",
#                            "ssp585_tas", "ssp585_logpr",
#                            "cmip7_tas",  "cmip7_logpr"])
# df.to_csv("tas_pr.csv")

# %%
df = pd.read_csv("cmip7_extensions_1750-2500-2.csv")
df = df.loc[df.scenario == "medium-extension"]
df = df.loc[df.variable == "CO2 FFI"]
years = pd.to_numeric(df.columns, errors="coerce")
year_mask = (years >= 1850.5) & (years <= 2100.5)
cmip7_co2 = df.loc[:, year_mask].values.flatten()
hist_co2 = xr.open_dataset("inputs_historical.nc")["CO2"].values
ssp126_co2 = xr.open_dataset("inputs_ssp126.nc")["CO2"].values
ssp585_co2 = xr.open_dataset("inputs_ssp585.nc")["CO2"].values
ssp126_co2 = np.concatenate([hist_co2, ssp126_co2])
ssp585_co2 = np.concatenate([hist_co2, ssp585_co2])
ssp126_co2 = np.diff(ssp126_co2, prepend=0)
ssp585_co2 = np.diff(ssp585_co2, prepend=0)
years = list(range(1850, 2101))



# %%
df = pd.read_csv("tas_pr.csv")
ssp126_tas = df["ssp126_tas"].values
ssp126_logpr = df["ssp126_logpr"].values
ssp585_tas = df["ssp585_tas"].values
ssp585_logpr = df["ssp585_logpr"].values
cmip7_tas = df["cmip7_tas"].values
cmip7_logpr = df["cmip7_logpr"].values


# %%
tasmin = min(ssp126_tas.min(), ssp585_tas.min())
tasmax = max(ssp126_tas.max(), ssp585_tas.max())
logprmin = min(ssp126_logpr.min(), ssp585_logpr.min())
logprmax = max(ssp126_logpr.max(), ssp585_logpr.max())

x_grid = np.linspace(tasmin, tasmax, 300)
ylog_grid = np.linspace(logprmin, logprmax, 300)

XX, YY = np.meshgrid(x_grid, ylog_grid)
levels = [0.05, 0.25, 0.5, 0.75, 0.95]

def kde_mass_contours(x, y, masses=(0.5, 0.75, 0.9)):
    kde = gaussian_kde(np.vstack([x, y]))
    zz = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    z_sorted = np.sort(zz.ravel())[::-1]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    levels = np.sort([z_sorted[np.searchsorted(cdf, m)] for m in masses])
    return zz, levels

# %%
zz1, zlev1 = kde_mass_contours(ssp126_tas, ssp126_logpr, levels)
zz2, zlev2 = kde_mass_contours(ssp585_tas, ssp585_logpr, levels)
zz3, zlev3 = kde_mass_contours(cmip7_tas,  cmip7_logpr,  levels)


# %%
height_ratios = [1]
width_ratios = [1, 0.3, 1.5, 0.2, 0.5]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)
width_multiplier = 5.
height_multiplier = 5.0
hspace = 0.01
wspace = 0.01

fig = plt.figure(figsize=(width_multiplier * ncoleff, height_multiplier * nroweff))
gs = gridspec.GridSpec(
    nrows=nrow,
    ncols=ncol,
    figure=fig,
    width_ratios=width_ratios,
    height_ratios=height_ratios,
    hspace=hspace,
    wspace=wspace
)

# fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5, 0.5]})

ax = fig.add_subplot(gs[0, 0])
ax.plot(years, ssp126_co2, label="SSP1-2.6", color="cornflowerblue", lw=4, alpha=0.8)
ax.plot(years, ssp585_co2, label="SSP5-8.5", color="salmon", lw=4, alpha=0.8)
ax.plot(years, cmip7_co2, label="M-Ext", color="k", lw=2, ls="--")
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_ylabel("CO$_2$ emissions (GtCO$_2$/yr)", fontsize=14)
ax.legend(frameon=False, prop={"size": 14, "weight": "bold"})
ax.margins(0.01)
ax.spines['top'].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)
ax.tick_params(axis="both", which="major", labelsize=14) 
ax.set_xlim(1950, 2105)
ax.set_xticks([1900, 2000, 2100])
ax.set_title("(a)", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[0, 2])
cs = ax.contourf(
    XX, np.expm1(YY), zz1,
    levels=20,
    cmap="Blues",
    alpha=1.,
    zorder=0,
)

cs = ax.contourf(
    XX, np.expm1(YY), zz2,
    levels=20,
    cmap="Reds",
    alpha=0.4,
    zorder=0,
)

cs = ax.contour(
    XX, np.expm1(YY), zz3,
    levels=zlev3,
    colors="k",
    linewidths=1,
    linestyles="--",
    zorder=5,
)
fmt = {lev: f"{int(p*100)}%" for lev, p in zip(zlev3, levels)}
ax.clabel(cs, fmt=fmt, fontsize=9)


legend_handles = [
    Line2D([0], [0], color="cornflowerblue",   lw=4, ls="-",  alpha=0.5, label="MPI-ESM1-2-LR SSP1-2.6"),
    Line2D([0], [0], color="salmon", lw=4, ls="-", alpha=0.5, label="MPI-ESM1-2-LR SSP5-8.5"),
    Line2D([0], [0], color="k",  lw=1, ls="--",  label="Emulated M-Ext")
]
ax.legend(handles=legend_handles, frameon=False, fontsize=18, prop={"size": 14})
ax.set_xlabel("Near-surface temperature (°C)", fontsize=16)
ax.set_ylabel("Precipitation (mm/day)", fontsize=16)
ax.set_ylim(0, 18)
ax.set_xlim(20, 40)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.spines['top'].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(True)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.set_title("(b)", fontsize=16, weight="bold")


ax = fig.add_subplot(gs[0, 4], projection=ccrs.Robinson())
ax.coastlines(linewidth=0.2, color="black")
ax.add_geometries(
            [ar6[10].polygon],
            crs=ccrs.PlateCarree(),
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            zorder=10,
        )
plt.tight_layout()
plt.savefig("m-ext.jpg", dpi=300, bbox_inches="tight")

# %%
