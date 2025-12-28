import os
import sys
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xesmf as xe
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.ssp245.utils import load_data, VARIABLES, setup_figure, save_plot


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/ssp245/files'
DPI = 300
WIDTH_MULTIPLIER = 5
HEIGHT_MULTIPLIER = 3
WSPACE = 0.01
HSPACE = 0.1


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds
time, lat, lon = target_data.time, target_data.lat, target_data.lon

eof_tas_mean = xr.open_dataarray("/home/shahineb/code/repos/climemu-private/eof/eof_emulator_T_del.nc").isel(lon=slice(None, -1))
eof_tas_stddev = xr.open_dataarray("/home/shahineb/code/repos/climemu-private/eof/eof_emulator_T_std_dev.nc").isel(lon=slice(None, -1))
eof_pr_mean = xr.open_dataarray("/home/shahineb/code/repos/climemu-private/eof/eof_emulator_precip_del.nc").isel(lon=slice(None, -1)) * 86400
eof_pr_stddev = xr.open_dataarray("/home/shahineb/code/repos/climemu-private/eof/eof_emulator_precip_std_dev.nc").isel(lon=slice(None, -1)) * 86400
eof_mean = xr.Dataset({"tas": eof_tas_mean, "pr": eof_pr_mean})
eof_stddev = xr.Dataset({"tas": eof_tas_stddev, "pr": eof_pr_stddev})

lon360 = (eof_mean.lon + 360) % 360
eof_mean = eof_mean.assign_coords(time=time, lat=lat, lon=lon360).sortby('lon')
eof_stddev = eof_stddev.assign_coords(time=time, lat=lat, lon=lon360).sortby('lon')
regridder = xe.Regridder(eof_mean, target_data, method='bilinear')
eof_mean = regridder(eof_mean, keep_attrs=True)
eof_stddev = regridder(eof_stddev, keep_attrs=True)


time_slice = slice("2080-01", "2100-12")
eof_mean = eof_mean.sel(time=time_slice)
eof_stddev = eof_stddev.sel(time=time_slice)
target_mean = target_data[["tas", "pr"]].sel(time=time_slice).mean('member').compute()
target_stddev = target_data[["tas", "pr"]].sel(time=time_slice).std('member').compute()
pred_mean = pred_samples[["tas", "pr"]].sel(time=time_slice).mean('member').compute()
pred_stddev = pred_samples[["tas", "pr"]].sel(time=time_slice).std('member').compute()



err_eof_mean = np.abs(eof_mean - target_mean).mean('time')
err_eof_stddev = np.abs(eof_stddev - target_stddev).mean('time')
err_pred_mean = np.abs(pred_mean - target_mean).mean('time')
err_pred_stddev = np.abs(pred_stddev - target_stddev).mean('time')

rel_error_mean = err_pred_mean - err_eof_mean
rel_error_stddev = err_pred_stddev - err_eof_stddev



width_ratios = [0.1, 1, 1, 0.01, 0.03, 0.15, 1, 0.01, 0.03]
height_ratios = [1, 1]
fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

var = "pr"
unit = VARIABLES[var]["unit"]
name = VARIABLES[var]["name"]

ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.text(0.5, 0.5, f"{name} \n mean", va="center", ha="center",
        rotation="vertical", fontsize=16, weight="bold")

ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.text(0.5, 0.5, f"{name} \n stddev", va="center", ha="center",
        rotation="vertical", fontsize=16, weight="bold")

# Means
flatvalues = []
flatvalues.append(err_eof_mean[var].values.ravel())
ax = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
mesh1 = err_eof_mean[var].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="viridis", add_colorbar=False)
ax.coastlines()
ax.set_title("Error EOF emulator", weight="bold", fontsize=16)

flatvalues.append(err_pred_mean[var].values.ravel())
ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
mesh2 = err_pred_mean[var].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="viridis", add_colorbar=False)
ax.coastlines()
ax.set_title("Error diffusion emulator", weight="bold", fontsize=16)

vmax = np.quantile(np.concatenate(flatvalues), 0.999)
mesh1.set_clim(0, vmax)
mesh2.set_clim(0, vmax)

cax = fig.add_subplot(gs[0, 4])
cbar = fig.colorbar(mesh1,
                    cax=cax,
                    extend='max')
cbar.ax.tick_params(labelsize=8)
cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
pos = cax.get_position()
new_width = pos.width * 1
new_height = pos.height * 0.5
new_y0 = pos.y0 + (pos.height - new_height) / 2
cax.set_position([pos.x0, new_y0, new_width, new_height])


ax = fig.add_subplot(gs[0, 6], projection=ccrs.Robinson())
mesh3 = rel_error_mean[var].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="RdBu_r", add_colorbar=False)
ax.coastlines()
vmin, vmax = np.quantile(rel_error_mean[var].values.ravel(), [0.01, 0.99])
vmax = max(vmax, -vmin)
mesh3.set_clim(-vmax, vmax)
ax.set_title("Error(Diffusion) - Error(EOF)", weight="bold", fontsize=16)


cax = fig.add_subplot(gs[0, 8])
cbar = fig.colorbar(mesh3,
                    cax=cax,
                    extend='both')
cbar.ax.tick_params(labelsize=8)
cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
pos = cax.get_position()
new_width = pos.width * 1
new_height = pos.height * 0.5
new_y0 = pos.y0 + (pos.height - new_height) / 2
cax.set_position([pos.x0, new_y0, new_width, new_height])



# Stddev
flatvalues = []
flatvalues.append(err_eof_stddev[var].values.ravel())
ax = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
mesh1 = err_eof_stddev[var].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="viridis", add_colorbar=False)
ax.coastlines()

flatvalues.append(err_pred_stddev[var].values.ravel())
ax = fig.add_subplot(gs[1, 2], projection=ccrs.Robinson())
mesh2 = err_pred_stddev[var].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="viridis", add_colorbar=False)
ax.coastlines()
vmax = np.quantile(np.concatenate(flatvalues), 0.999)
mesh1.set_clim(0, vmax)
mesh2.set_clim(0, vmax)

cax = fig.add_subplot(gs[1, 4])
cbar = fig.colorbar(mesh1,
                    cax=cax,
                    extend='max')
cbar.ax.tick_params(labelsize=8)
cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
pos = cax.get_position()
new_width = pos.width * 1
new_height = pos.height * 0.5
new_y0 = pos.y0 + (pos.height - new_height) / 2
cax.set_position([pos.x0, new_y0, new_width, new_height])


ax = fig.add_subplot(gs[1, 6], projection=ccrs.Robinson())
mesh3 = rel_error_stddev[var].plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="RdBu_r", add_colorbar=False)
ax.coastlines()
vmin, vmax = np.quantile(rel_error_stddev[var].values.ravel(), [0.01, 0.99])
vmax = max(vmax, -vmin)
mesh3.set_clim(-vmax, vmax)

cax = fig.add_subplot(gs[1, 8])
cbar = fig.colorbar(mesh3,
                    cax=cax,
                    extend='both')
cbar.ax.tick_params(labelsize=8)
cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
pos = cax.get_position()
new_width = pos.width * 1
new_height = pos.height * 0.5
new_y0 = pos.y0 + (pos.height - new_height) / 2
cax.set_position([pos.x0, new_y0, new_width, new_height])


save_plot(fig, "./", f"eof_comparison_{var}.jpg")