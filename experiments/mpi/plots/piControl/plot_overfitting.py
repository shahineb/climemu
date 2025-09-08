import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import seaborn as sns

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import VARIABLES, load_data
from experiments.mpi.data import load_dataset


def wrap_lon(ds):
    # assumes ds.lon runs 0…360
    lon360 = ds.lon.values
    lon180 = ((lon360 + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon180).sortby("lon")
    return ds


config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
piControl_cmip6 = piControl_cmip6 - climatology


β = jnp.load(config.data.pattern_scaling_path)
stats = jnp.load(config.data.norm_stats_path)
μ_train, σ_train = jnp.array(stats['μ']), jnp.array(stats['σ'])
test_dataset = load_dataset(
    root=config.data.root_dir,
    model=config.data.model_name,
    experiments=["historical"],
    variables=config.data.variables,
    in_memory=False,
    external_β=β)


# Get sfcWind mean
lon_range = slice(-130, -55)
lat_range = slice(20, 65)
time_range = slice("1850-01", "1900-01")
emulator = {}
piControl = {}
historical = {}
for var in ["sfcWind", "hurs"]:
    emulator[var] = wrap_lon(piControl_diffusion[var]).sel(lat=lat_range, lon=lon_range).mean(['year', 'month']).compute()
    piControl[var] = wrap_lon(piControl_cmip6[var]).sel(lat=lat_range, lon=lon_range).mean(['year', 'month']).compute()
    historical[var] = wrap_lon(test_dataset['historical'][var]).sel(time=time_range, lat=lat_range, lon=lon_range).mean(['time', 'member']).compute()


# Get gmst distributions
gmst_historical = test_dataset.gmst['historical']['tas'].sel(time=time_range).values.ravel()
piControl_dataset = load_dataset(root=config.data.root_dir,
                            model=config.data.model_name,
                            experiments=["piControl"],
                            variables=["tas"],
                            in_memory=False,
                            external_β=β)
gmst_piControl = piControl_dataset.gmst['piControl']['tas'].values.ravel()
σpiControl = gmst_piControl.std()
np.random.seed(5)
gmst_emulator = σpiControl * np.random.randn(gmst_piControl.shape[0])
# df = pd.DataFrame({
#     "GMST": np.concatenate([gmst_piControl, gmst_emulator, gmst_historical]),
#     "Dataset": ([f"{config.data.model_name} \n piControl"] * len(gmst_piControl)
#               + [r"$\mathcal{N}(0, \sigma_{PI}^2)$"] * len(gmst_emulator)
#               + [f"{config.data.model_name} \n historical (1850-1900)"] * len(gmst_historical))
# })

df = pd.DataFrame({
    "GMST": np.concatenate([gmst_piControl, gmst_emulator, gmst_historical]),
    "Dataset": (["(a)"] * len(gmst_piControl)
              + ["(b)"] * len(gmst_emulator)
              + ["(c)"] * len(gmst_historical))
})



# Plot
width_ratios  = [0.6, 0.02, 1, 1, 1, 0.05]
height_ratios = [1, 1]
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


ax = fig.add_subplot(gs[:, 0])
sns.boxplot(data=df, x="Dataset", y="GMST", ax=ax, fill=False, showfliers=True, color=".2", fliersize=1, legend=False)
sns.despine()
ax.set_xlabel("")
ax.set_ylabel("GMST anomaly [°C]", weight="bold", fontsize=14)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(14)



for i, var in enumerate(["sfcWind", "hurs"]):
    var_info = VARIABLES[var]
    var_name = var_info['name']
    unit = var_info['unit']
    cmap = var_info['cmap']
    flatvalues = []
    meshes = []

    ax = fig.add_subplot(gs[i, -4], projection=ccrs.PlateCarree())
    mesh = piControl[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.coastlines()
    flatvalues.append(piControl[var].values.ravel())
    meshes.append(mesh)
    if i == 0:
        ax.set_title(f"(a) {config.data.model_name} piControl", fontsize=16, weight="bold")

    ax = fig.add_subplot(gs[i, -3], projection=ccrs.PlateCarree())
    mesh = emulator[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.coastlines()
    flatvalues.append(emulator[var].values.ravel())
    meshes.append(mesh)
    if i == 0:
        ax.set_title(f"(b) Emulator", fontsize=16, weight="bold")

    ax = fig.add_subplot(gs[i, -2], projection=ccrs.PlateCarree())
    mesh = historical[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    ax.coastlines()
    flatvalues.append(historical[var].values.ravel())
    meshes.append(mesh)
    if i == 0:
        ax.set_title(f"(c) {config.data.model_name} \n early historical (1850-1900)", fontsize=16, weight="bold")

    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    vmin = np.quantile(np.concatenate(flatvalues), 0.001)
    vmax = max(np.abs(vmax), np.abs(vmin))
    for mesh in meshes:
        mesh.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[i, -1])
    cbar = fig.colorbar(mesh,
                        cax=cax,
                        orientation='vertical')
    cbar.set_label(f"[{unit}]", labelpad=4)

filepath = f'experiments/mpi/plots/piControl/files/overfitting.jpg'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()