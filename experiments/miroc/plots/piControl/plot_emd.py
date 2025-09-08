# %%
import os
import sys
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.piControl.utils import load_data, VARIABLES

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


# %%
config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology


# %%
month_to_season = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3:  "MAM", 4:  "MAM", 5:  "MAM",
    6:  "JJA", 7:  "JJA", 8:  "JJA",
    9:  "SON", 10: "SON", 11: "SON"
}
seasons = np.array([month_to_season[m] for m in piControl_diffusion['month'].values])
piControl_diffusion = piControl_diffusion.assign_coords(season=("month", seasons))
piControl_cmip6 = piControl_cmip6.assign_coords(season=("month", seasons))



# %%
ε = {'tas': 0.1,
     'pr': 0.1,
     'hurs': 0.1,
     'sfcWind': 0.1}

def get_plot_data(piControl_cmip6, piControl_diffusion):
    emd = dict()
    piControl_diffusion_flat = piControl_diffusion.stack(flat=('year', 'month'))
    piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))
    for var in VARIABLES.keys():
        emd_var = dict()
        piControl_diffusion_flat_var = piControl_diffusion_flat[var]
        piControl_cmip6_flat_var = piControl_cmip6_flat[var]
        for season in ["DJF", "MAM", "JJA", "SON"]:
            print(f"Computing EMD for {var} in {season}")
            emulator_data = piControl_diffusion_flat_var.where(piControl_diffusion_flat_var.season == season, drop=True)
            esm_data = piControl_cmip6_flat_var.where(piControl_cmip6_flat_var.season == season, drop=True)
            σesm = esm_data.std('flat')
            σesm = σesm.where(σesm > 0.1, 0.1)
            emd_var[season] = compute_emd(emulator_data, esm_data) / σesm
        emd[var] = emd_var
    return emd


def plot_variable(fig, gs, var, i):
    var_info = VARIABLES[var]
    var_name = var_info['name']

    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, var_name, va="center", ha="center",
                rotation="vertical", fontsize=16, weight="bold")

    # flatvalues = []
    meshes = []
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        ax = fig.add_subplot(gs[i, j + 1], projection=ccrs.Robinson())
        mesh = emd[var][season].plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap='RdPu', add_colorbar=False
        )
        ax.coastlines()
        # flatvalues.append(emd[var][season].values.ravel())
        meshes.append(mesh)
        if i == 0:
            ax.set_title(f"{season}", fontsize=16, weight="bold")
    # vmax = np.quantile(np.concatenate(flatvalues), 0.99)
    for mesh in meshes:
        mesh.set_clim(0, 1)
    if i == 0:
        cax = fig.add_subplot(gs[1:-1, 5])
        cbar = fig.colorbar(mesh,
                            cax=cax,
                            orientation='vertical')
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticks([0, 0.5, 1])
        cbar.set_label(f"EMD-to-noise ratio", labelpad=4, fontsize=16, weight="bold")

# %%
emd = get_plot_data(piControl_cmip6, piControl_diffusion)



# %%
width_ratios  = [0.05, 1, 1, 1, 1, 0.05]
height_ratios = [1, 1, 1, 1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(5 * ncoleff, 3 * nroweff))

gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.05,
              wspace=0.05)


plot_variable(fig, gs, 'tas', 0)
plot_variable(fig, gs, 'pr', 1)
plot_variable(fig, gs, 'hurs', 2)
plot_variable(fig, gs, 'sfcWind', 3)
output_dir = 'experiments/miroc/plots/piControl/files'
filepath = os.path.join(output_dir, 'emd.jpg')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(filepath, dpi=500, bbox_inches='tight')