import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.access.config import Config
from experiments.access.plots.ssp370.utils import load_data
from experiments.access.plots.historical.utils import load_data as load_historical_data


config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp370'].ds

test_dataset_hist, pred_samples_hist, _, __ = load_historical_data(config, in_memory=False)
target_data_hist = test_dataset_hist['historical'].ds

da_cmip6 = xr.concat([target_data_hist['pr'], target_data['pr']], dim='time')
da_diffusion = xr.concat([pred_samples_hist['pr'], pred_samples['pr']], dim='time')

σ_cmip6 = da_cmip6.sel(lat=slice(-30, 30)).mean('lat').std('member').groupby('time.year').mean().compute()

da_cmip6 = da_cmip6.sel(lat=slice(-30, 30)).mean(['member', 'lat']).groupby('time.year').mean().compute()
da_diffusion = da_diffusion.sel(lat=slice(-30, 30)).mean(['member', 'lat']).groupby('time.year').mean().compute()



lons = da_cmip6['lon'].values
lons = ((lons + 180) % 360) - 180
sort_idx = np.argsort(lons)
lons = lons[sort_idx]
years = da_cmip6['year'].values

cmip6_vals = da_cmip6.values[:, sort_idx]
diff_vals = da_diffusion.values[:, sort_idx]
σ_cmip6_vals = σ_cmip6.where(σ_cmip6 > 0.1, 0.1).values[:, sort_idx]
bnr = np.abs(diff_vals - cmip6_vals) / σ_cmip6_vals


Lon, Year = np.meshgrid(lons, years)

width_ratios  = [1, 1, 0.05, 0.3, 1, 0.05]
height_ratios = [1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)

flat_values = np.concatenate([cmip6_vals, diff_vals])
vmax = np.quantile(flat_values, 0.99)
vmin = np.quantile(flat_values, 0.01)
vmax = max(np.abs(vmax), np.abs(vmin))
locator = ticker.MaxNLocator(nbins=14, prune=None)
levels = locator.tick_values(-vmax, vmax)


bnrmax = max(1, np.quantile(bnr, 0.99))
locator = ticker.MaxNLocator(nbins=14, prune=None)
bnrlevels = locator.tick_values(0, bnrmax)


xticks = [-120, -60, 0, 60, 120]
xtick_labels = ["120W", "60W", "0", "60E", "120E"]

fig = plt.figure(figsize=(4 * ncoleff, 3 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.01,
              wspace=0.1)


ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.contourf(Lon, Year, cmip6_vals, levels=levels, extend='both', cmap='BrBG')
ax1.set_title(config.data.model_name, weight="bold")
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)
ax1.set_yticks([1900, 2000, 2100])


ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.contourf(Lon, Year, diff_vals, levels=levels, extend='both', cmap='BrBG')
ax2.set_title("Emulator", weight="bold")
ax2.set_yticklabels([]) 
ax2.set_xticks(xticks)
ax2.set_xticklabels(xtick_labels)
ax2.yaxis.set_visible(False)



cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(c1, cax=cax)
cb.set_ticks([-0.25, 0, 0.25])
cb.set_label("Precipitation anomaly [mm/day]")



ax3 = fig.add_subplot(gs[0, 4])
c3 = ax3.contourf(Lon, Year, bnr, levels=bnrlevels, extend='max', cmap='RdPu')
ax3.set_title("Error-to-noise ratio", weight="bold")
ax3.set_yticklabels([])
ax3.set_xticks(xticks)
ax3.set_xticklabels(xtick_labels)
ax3.yaxis.set_visible(False)


cax2 = fig.add_subplot(gs[0, 5])
cb = fig.colorbar(c3, cax=cax2)
cb.ax.set_yticks([0, 1])
cb.set_label("[1]")
 


plt.savefig("experiments/access/plots/ssp370/files/hovmoller.jpg", dpi=300, bbox_inches='tight')