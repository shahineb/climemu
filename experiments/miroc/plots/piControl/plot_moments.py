# %%
import os
import sys
import numpy as np
from scipy.stats import skew
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs


# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.piControl.utils import load_data, VARIABLES


# %%
config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology


# %%
stack = piControl_diffusion.stack(flat=('year', 'month'))
μ_diffusion = stack.mean(['flat'])
σ_diffusion = stack.std(['flat'])
γ1_diffusion = xr.apply_ufunc(skew, stack, input_core_dims=[['flat']], kwargs={'axis': -1, 'nan_policy': 'omit'}, output_dtypes=[float])


stack = piControl_cmip6.stack(flat=('year', 'month'))
μ_cmip6 = stack.mean(['flat'])
σ_cmip6 = stack.std(['flat'])
γ1_cmip6 = xr.apply_ufunc(skew, stack, input_core_dims=[['flat']], kwargs={'axis': -1, 'nan_policy': 'omit'}, output_dtypes=[float])


# %%
def plot_variable(fig, gs, var, i):
    var_info = VARIABLES[var]
    var_name = var_info['name']
    unit = var_info['unit']
    cmap = plt.get_cmap(var_info['cmap'])

    ax = fig.add_subplot(gs[i:i + 3, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, var_name, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, config.data.model_name, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")
    ax = fig.add_subplot(gs[i + 2, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, "Emulator", va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")


    # Plot mean
    flatvalues = []
    flatvalues.append(μ_cmip6[var].values.ravel())
    ax = fig.add_subplot(gs[i, 2], projection=ccrs.Robinson())
    mesh1 = μ_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap, add_colorbar=False)
    ax.coastlines()
    if i == 0:
        ax.set_title(f"Mean", fontsize=16, weight="bold")


    flatvalues.append(μ_diffusion[var].values.ravel())
    ax = fig.add_subplot(gs[i + 2, 2], projection=ccrs.Robinson())
    mesh2 = μ_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap, add_colorbar=False)
    ax.coastlines()


    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    vmin = np.quantile(np.concatenate(flatvalues), 0.001)
    vmax = max(np.abs(vmax), np.abs(vmin))
    mesh1.set_clim(-vmax, vmax)
    mesh2.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[i + 1, 2])
    cbar = fig.colorbar(mesh1,
                        cax=cax,
                        orientation='horizontal')
    cbar.locator = ticker.MaxNLocator(nbins=3, integer=True)
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
    pos = cax.get_position()
    new_width = pos.width * 0.4
    new_height = pos.height * 0.6
    new_x0  = pos.x0 + (pos.width - new_width)/2
    cax.set_position([new_x0, pos.y0, new_width, new_height])


    # Plot stddev
    cmap_upper = mcolors.LinearSegmentedColormap.from_list("upper_half", cmap(np.linspace(0.5, 1.0, 256)))
    flatvalues = []
    flatvalues.append(σ_cmip6[var].values.ravel())
    ax = fig.add_subplot(gs[i, 3], projection=ccrs.Robinson())
    mesh1 = σ_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap_upper, add_colorbar=False)
    ax.coastlines()
    if i == 0:
        ax.set_title(f"Standard deviation", fontsize=16, weight="bold")


    flatvalues.append(σ_diffusion[var].values.ravel())
    ax = fig.add_subplot(gs[i + 2, 3], projection=ccrs.Robinson())
    mesh2 = σ_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap=cmap_upper, add_colorbar=False)
    ax.coastlines()


    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    mesh1.set_clim(0, vmax)
    mesh2.set_clim(0, vmax)
    cax = fig.add_subplot(gs[i + 1, 3])
    cbar = fig.colorbar(mesh1,
                        cax=cax,
                        orientation='horizontal')
    cbar.locator = ticker.MaxNLocator(nbins=3, integer=True)
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"[{unit}]", labelpad=4, fontsize=8)
    pos = cax.get_position()
    new_width = pos.width * 0.4
    new_height = pos.height * 0.6
    new_x0    = pos.x0 + (pos.width - new_width)/2
    cax.set_position([new_x0, pos.y0, new_width, new_height])



    # Plot skew
    flatvalues = []
    flatvalues.append(γ1_cmip6[var].values.ravel())
    ax = fig.add_subplot(gs[i, 4], projection=ccrs.Robinson())
    mesh1 = γ1_cmip6[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='PiYG', add_colorbar=False)
    ax.coastlines()
    if i == 0:
        ax.set_title(f"Skewness", fontsize=16, weight="bold")


    flatvalues.append(γ1_diffusion[var].values.ravel())
    ax = fig.add_subplot(gs[i + 2, 4], projection=ccrs.Robinson())
    mesh2 = γ1_diffusion[var].plot.pcolormesh(
                    ax=ax, transform=ccrs.PlateCarree(),
                    cmap='PiYG', add_colorbar=False)
    ax.coastlines()


    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    vmin = np.quantile(np.concatenate(flatvalues), 0.001)
    vmax = max(np.abs(vmax), np.abs(vmin))
    mesh1.set_clim(-vmax, vmax)
    mesh2.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[i + 1, 4])
    cbar = fig.colorbar(mesh1,
                        cax=cax,
                        orientation='horizontal')
    cbar.locator = ticker.MaxNLocator(nbins=3, integer=True)
    cbar.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("[1]", fontsize=8, labelpad=4)
    pos = cax.get_position()
    new_width = pos.width * 0.4
    new_height = pos.height * 0.6
    new_x0    = pos.x0 + (pos.width - new_width)/2
    cax.set_position([new_x0, pos.y0, new_width, new_height])



# %%
# plot
width_ratios  = [0.06, 0.05, 1, 1, 1]
height_ratios = [1, 0.05, 1, 0.1] * 4
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(6 * ncoleff, 2.9 * nroweff))

gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.2,
              wspace=0.05)


plot_variable(fig, gs, 'tas', 0)
plot_variable(fig, gs, 'pr', 4)
plot_variable(fig, gs, 'hurs', 8)
plot_variable(fig, gs, 'sfcWind', 12)

filepath = f'experiments/miroc/plots/piControl/files/moments.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()