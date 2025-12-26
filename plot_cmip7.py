import os
import numpy as np
import xarray as xr
import pandas as pd
import regionmask
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from src.datasets import CMIP6Data

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

df = pd.DataFrame(data=np.stack([ssp126_tas, ssp126_logpr,
                        ssp585_tas, ssp585_logpr,
                        cmip7_tas, cmip7_logpr], axis=-1),
                  columns=["ssp126_tas", "ssp126_logpr",
                           "ssp585_tas", "ssp585_logpr",
                           "cmip7_tas",  "cmip7_logpr"])
df.to_csv("tas_pr.csv")


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


zz1, zlev1 = kde_mass_contours(ssp126_tas, ssp126_logpr, levels)
zz2, zlev2 = kde_mass_contours(ssp585_tas, ssp585_logpr, levels)
zz3, zlev3 = kde_mass_contours(cmip7_tas,  cmip7_logpr,  levels)


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

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
    Line2D([0], [0], color="cornflowerblue",   lw=4, ls="-",  alpha=0.5, label="SSP1-2.6"),
    Line2D([0], [0], color="salmon", lw=4, ls="-", alpha=0.5, label="SSP5-8.5"),
    Line2D([0], [0], color="k",  lw=1, ls="--",  label="M-Ext")
]
ax.legend(handles=legend_handles, frameon=False, fontsize=18)

ax.set_xlabel("Near-surface temperature (°C)", fontsize=16)
ax.set_ylabel("Precipitation (mm/day)", fontsize=16)

ax.set_ylim(0, 18)
ax.set_xlim(20, 40)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
plt.tight_layout()
plt.savefig("bar.jpg", dpi=300, bbox_inches="tight")




# ax.contour(
#     xx2, np.expm1(yy2), zz2,
#     levels=zlev2,
#     colors="tab:orange",
#     linewidths=2,
#     linestyles="--",
#     label="SSP5-8.5"
# )

# ax.contour(
#     xx3, np.expm1(yy3), zz3,
#     levels=zlev3,
#     colors="tab:green",
#     linewidths=2,
#     linestyles=":",
#     label="CMIP7"
# )

# ax.set_xlabel("Temperature (tas)")
# ax.set_ylabel("Precipitation (pr)")
# ax.set_title("Joint tas–pr distribution (probability contours)")


# ax.set_ylim(bottom=0)
# ax.legend(frameon=False)
# plt.tight_layout()
# plt.savefig("bar.jpg", dpi=300)




# # ssp126_taspr = np.stack([ssp126_tas, ssp126_pr], axis=-1)
# # ssp585_taspr = np.stack([ssp585_tas, ssp585_pr], axis=-1)
# # cmip7_taspr = np.stack([cmip7_tas, cmip7_pr], axis=-1)

# df_ssp126 = pd.DataFrame({"tas": ssp126_tas, "pr":  ssp126_logpr, "scenario": "SSP1-2.6"})
# df_ssp585 = pd.DataFrame({"tas": ssp585_tas, "pr":  ssp585_logpr, "scenario": "SSP5-8.5"})
# df_cmip7 = pd.DataFrame({"tas": cmip7_tas, "pr":  cmip7_logpr, "scenario": "CMIP7 HM"})
# df = pd.concat([df_ssp126, df_ssp585, df_cmip7], ignore_index=True).sample(100000)


# fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# ax.scatter(ssp126_tas, ssp126_pr, s=1, alpha=0.5, zorder=0)
# ax.scatter(ssp585_tas, ssp585_pr, s=1, alpha=0.5, zorder=0)
# ax.scatter(cmip7_tas, cmip7_pr, s=1, alpha=0.01)
# plt.xlabel("Temperature (tas)")
# plt.ylabel("Precipitation (pr)")
# plt.title("Joint KDE of tas–pr")
# plt.tight_layout()
# plt.savefig("foo.jpg")
