import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.ssp370.utils import load_data
from experiments.mpi.plots.historical.utils import load_data as load_historical_data


config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp370'].ds

test_dataset_hist, pred_samples_hist, _, __ = load_historical_data(config, in_memory=False)
target_data_hist = test_dataset_hist['historical'].ds

da_cmip6 = xr.concat([target_data_hist['sfcWind'], target_data['sfcWind']], dim='time')
da_diffusion = xr.concat([pred_samples_hist['sfcWind'], pred_samples['sfcWind']], dim='time')

σ_cmip6 = da_cmip6.mean('lon').std('member').groupby('time.year').mean().compute()

da_cmip6 = da_cmip6.mean(['member', 'lon']).groupby('time.year').mean().compute()
da_diffusion = da_diffusion.mean(['member', 'lon']).groupby('time.year').mean().compute()

cmip6_vals = da_cmip6.values.T
diff_vals = da_diffusion.values.T
σ_cmip6_vals = σ_cmip6.where(σ_cmip6 > 0.1, 0.1).values.T
bnr = np.abs(diff_vals - cmip6_vals) / σ_cmip6_vals



lats = da_cmip6['lat'].values
times = da_cmip6['year'].values
years = times




Year, Lat = np.meshgrid(years, lats)

width_ratios  = [1, 1, 0.02, 0.05, 0.2, 1, 0.02, 0.05]
height_ratios = [1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


flat_values = np.concatenate([cmip6_vals, diff_vals])
vmax = np.quantile(flat_values, 0.99)
vmin = np.quantile(flat_values, 0.01)
vmax = max(np.abs(vmax), np.abs(vmin))
locator = ticker.MaxNLocator(nbins=29, prune=None)
levels = locator.tick_values(-vmax, vmax)


bnrmax = max(1, np.quantile(bnr, 0.99))
locator = ticker.MaxNLocator(nbins=14, prune=None)
bnrlevels = locator.tick_values(0, bnrmax)



yticks = [-60, -30, 0, 30, 60]
ytick_labels = ["60S", "30S", "0", "30N", "60N"]

fig = plt.figure(figsize=(5 * ncoleff, 2.5 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.01,
              wspace=0.01)


ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.contourf(Year, Lat, cmip6_vals, levels=levels, extend='both', cmap='PRGn')
ax1.set_title(config.data.model_name, weight="bold")
ax1.set_ylabel("Latitude")
ax1.set_yticks(yticks)
ax1.set_yticklabels(ytick_labels)
ax1.set_xticks([1900, 2000, 2100])
ax1.set_xticklabels([1900, 2000, 2100])


ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.contourf(Year, Lat, diff_vals, levels=levels, extend='both', cmap='PRGn')
ax2.set_title("Emulator", weight="bold")
ax2.set_yticks([])
ax2.set_xticks([1900, 2000, 2100])
ax2.set_xticklabels([1900, 2000, 2100])

cax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(c2, cax=cax, orientation="vertical", shrink=0.8, pad=0.05)
cbar.ax.set_yticks([-0.5, 0, 0.5])
cbar.set_label("Windspeed anomaly [m/s]")


ax3 = fig.add_subplot(gs[0, 5])
c3 = ax3.contourf(Year, Lat, bnr, levels=bnrlevels, extend='max', cmap='RdPu')
ax3.set_title("Error-to-noise ratio", weight="bold")
ax3.set_yticks([])
ax3.set_xticks([1900, 2000, 2100])
ax3.set_xticklabels([1900, 2000, 2100])

cax = fig.add_subplot(gs[0, 7])
cbar = fig.colorbar(c3, cax=cax, orientation="vertical", shrink=0.8, pad=0.05)
cbar.ax.set_yticks([0, 1])
cbar.set_label("[1]", labelpad=-1)

plt.savefig("experiments/mpi/plots/ssp370/files/westerlies.jpg", dpi=300, bbox_inches='tight')