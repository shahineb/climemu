import os
import sys
import numpy as np
import jax.numpy as jnp
import pyshtools as pysh
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.miroc.config import Config
from experiments.miroc.plots.piControl.utils import load_data, VARIABLES


def compute_Cl(da):
    da = da - da.mean()
    grid = pysh.SHGrid.from_xarray(da, grid='GLQ')
    clm = grid.expand()
    Cl = clm.spectrum()
    return Cl


def get_plot_data(pred_da, cmip6_da):
    diffusion_Cl = []
    cmip6_Cl = []
    n_year = len(cmip6_da.year)
    nt = 12 * n_year
    with tqdm(total=nt) as pbar:
        for yr in range(n_year):
            for m in range(12):
                Cl = compute_Cl(pred_da.isel(year=yr, month=m, drop=True))
                diffusion_Cl.append(Cl)
                Cl = compute_Cl(cmip6_da.isel(year=yr, month=m, drop=True))
                cmip6_Cl.append(Cl)
                _ = pbar.update(1)
    diffusion_Cl = jnp.stack(diffusion_Cl)
    cmip6_Cl = jnp.stack(cmip6_Cl)
    diffusion = {'median': jnp.quantile(diffusion_Cl, 0.5, axis=0),
                 'lb': jnp.quantile(diffusion_Cl, 0.05, axis=0),
                 'ub': jnp.quantile(diffusion_Cl, 0.95, axis=0)}
    cmip6 = {'median': jnp.quantile(cmip6_Cl, 0.5, axis=0),
             'lb': jnp.quantile(cmip6_Cl, 0.05, axis=0),
             'ub': jnp.quantile(cmip6_Cl, 0.95, axis=0)}
    return diffusion, cmip6


# %%
config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology

# %%
R = 6371  # km
ell = np.arange(piControl_diffusion.sizes['lat'])
k = np.where(ell > 0, ell, np.nan) / (2 * np.pi * R)


# %%
np.random.seed(5)
sample_year_idx = np.random.randint(len(piControl_cmip6.year))
sample_month_idx = np.random.randint(12)
print(f"Sample for month {sample_month_idx + 1}")

emulator = dict()
cmip6 = dict()

for i, var in enumerate(VARIABLES):
    emulator_psd_data, cmip6_psd_data = get_plot_data(piControl_diffusion[var],
                                                        piControl_cmip6[var])
    emulator_psd_data['sample'] = piControl_diffusion[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    cmip6_psd_data['sample'] = piControl_cmip6[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    emulator[var] = emulator_psd_data
    cmip6[var] = cmip6_psd_data



width_ratios  = [0.1, 1, 0.1, 0.25, 1, 0.1, 0.25, 1, 0.1, 0.25, 1, 0.1, 0.25]
height_ratios = [0.4, 0.4, 1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(4 * ncoleff, 3.5 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.05,
              wspace=0.01)

ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.text(2.2, 0.5, f"{config.data.model_name} \n realization", va="center", ha="right", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.text(2.2, 0.5, "Emulator \n sample", va="center", ha="right", fontsize=8)


for i, var in enumerate(VARIABLES):
    var_info = VARIABLES[var]
    unit = var_info['unit']
    cmap = var_info['cmap']
    i = 3 * i + 1
    flatvalues = []

    ax = fig.add_subplot(gs[0, i], projection=ccrs.Robinson())
    mesh1 = cmip6[var]['sample'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    flatvalues.append(cmip6[var]['sample'].values.ravel())
    ax.coastlines(lw=0.5, alpha=0.1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)
    ax.set_title(var_info['name'], fontsize=16, weight='bold')

    ax = fig.add_subplot(gs[1, i], projection=ccrs.Robinson())
    mesh2 = emulator[var]['sample'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    flatvalues.append(emulator[var]['sample'].values.ravel())
    ax.coastlines(lw=0.5, alpha=0.1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

    vmax = np.quantile(np.concatenate(flatvalues), 0.999)
    vmin = np.quantile(np.concatenate(flatvalues), 0.001)
    vmax = max(np.abs(vmax), np.abs(vmin))
    mesh1.set_clim(-vmax, vmax)
    mesh2.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[:2, i + 1])
    cbar = fig.colorbar(mesh1,
                        cax=cax,
                        orientation='vertical')
    cbar.set_label(f"[{unit}]", labelpad=4)
    pos = cax.get_position()
    new_width  = pos.width * 0.3    # thinner
    new_height = pos.height * 0.5   # shorter
    new_x0     = pos.x0 - 0.02      # move left
    new_y0     = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
    cax.set_position([new_x0, new_y0, new_width, new_height])

    ax = fig.add_subplot(gs[2, i:i+2])
    ax.fill_between(k, emulator[var]['lb'], emulator[var]['ub'], color='tomato', alpha=0.2)
    ax.fill_between(k, cmip6[var]['lb'], cmip6[var]['ub'], color='cornflowerblue', alpha=0.2)
    ax.plot(k, cmip6[var]['median'], label=config.data.model_name, color='cornflowerblue')
    ax.plot(k, emulator[var]['median'], label='Emulator', color='tomato')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.margins(x=0, y=0)
    ax.set_xlabel('Inverse wavelength [km⁻¹]', fontsize=12)
    ax.set_ylabel(f'[({unit})²⋅km]', fontsize=12)
    if i == 1:
        ax.legend(fontsize=14)

output_dir = 'experiments/miroc/plots/piControl/files'
filepath = os.path.join(output_dir, 'psd.jpg')
plt.savefig(filepath, dpi=300, bbox_inches='tight')