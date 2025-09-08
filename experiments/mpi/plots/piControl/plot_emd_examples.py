import os
import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import wasserstein_distance


# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import VARIABLES, load_data



def wrap_lon(ds):
    # assumes ds.lon runs 0…360
    lon360 = ds.lon.values
    lon180 = ((lon360 + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon180).sortby("lon")
    return ds

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
piControl_cmip6 = piControl_cmip6 - climatology


###################################

# Low EMD bulk (sfcWind mediterranean nov)
lat_range = slice(28, 45)
lon_range = slice(-15, 35)

data_cmip6 = wrap_lon(piControl_cmip6['sfcWind']).sel(lon=lon_range, lat=lat_range, month=11).values.ravel()
data_diffusion = wrap_lon(piControl_diffusion['sfcWind']).sel(lon=lon_range, lat=lat_range, month=11).values.ravel()

bulk1_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
bulk1_diffusion = data_diffusion[~np.isnan(data_diffusion)]

emd_bulk1 = wasserstein_distance(bulk1_cmip6, bulk1_diffusion)
enr_bulk1 = emd_bulk1 / bulk1_cmip6.std()
print(enr_bulk1)  # 0.05863974655338643



# Medium EMD bulk (hurs ivory coast october)
lon_range = slice(-12, -2)
lat_range = slice(2, 10)

data_cmip6 = wrap_lon(piControl_cmip6['hurs']).sel(lon=lon_range, lat=lat_range, month=10).values.ravel()
data_diffusion = wrap_lon(piControl_diffusion['hurs']).sel(lon=lon_range, lat=lat_range,  month=10).values.ravel()

bulk2_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
bulk2_diffusion = data_diffusion[~np.isnan(data_diffusion)]

emd_bulk2 = wasserstein_distance(bulk2_cmip6, bulk2_diffusion)
enr_bulk2 = emd_bulk2 / bulk2_cmip6.std()
print(enr_bulk2)  # 0.3871811574505722



# Bad EMD bulk (tas arctic july)
lat_range = slice(70, 80)
lon_range = slice(160, 220)

data_cmip6 = piControl_cmip6['tas'].sel(lon=lon_range, lat=lat_range, month=7).values.ravel()
data_diffusion = piControl_diffusion['tas'].sel(lon=lon_range, lat=lat_range,  month=7).values.ravel()

bulk3_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
bulk3_diffusion = data_diffusion[~np.isnan(data_diffusion)]

emd_bulk3 = wasserstein_distance(bulk3_cmip6, bulk3_diffusion)
enr_bulk3 = emd_bulk3 / bulk3_cmip6.std()
print(enr_bulk3)  # 0.826843902270992





# Good EMD concentrated (pr pacific)
lat_range = slice(-10, 10)
lon_range = slice(160, 260)

data_cmip6 = piControl_cmip6['pr'].sel(lon=lon_range, lat=lat_range, month=2).values.ravel()
data_diffusion = piControl_diffusion['pr'].sel(lon=lon_range, lat=lat_range, month=2).values.ravel()

conc1_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
conc1_diffusion = data_diffusion[~np.isnan(data_diffusion)]

emd_conc1 = wasserstein_distance(conc1_cmip6, conc1_diffusion)
enr_conc1 = emd_conc1 / conc1_cmip6.std()
print(enr_conc1)  # 0.06419806572933695



# Poor EMD concentrated (pr mediterranean)
lat_range = slice(28, 45)
lon_range = slice(-10, 15)

data_cmip6 = wrap_lon(piControl_cmip6['pr']).sel(lon=lon_range, lat=lat_range, month=7).values.ravel()
data_diffusion = wrap_lon(piControl_diffusion['pr']).sel(lon=lon_range, lat=lat_range, month=7).values.ravel()

conc2_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
conc2_diffusion = data_diffusion[~np.isnan(data_diffusion)]

emd_conc2 = wasserstein_distance(conc2_cmip6, conc2_diffusion)
enr_conc2 = emd_conc2 / conc2_cmip6.std()
print(enr_conc2)  # 0.33291723109473154


# Very poor EMD concentrated (pr central africa)
lon_range = slice(-10, 45)
lat_range = slice(5, 15)

data_cmip6 = wrap_lon(piControl_cmip6['pr']).sel(lon=lon_range, lat=lat_range, month=12).values.ravel()
data_diffusion = wrap_lon(piControl_diffusion['pr']).sel(lon=lon_range, lat=lat_range, month=12).values.ravel()

conc3_cmip6 = data_cmip6[~np.isnan(data_cmip6)]
conc3_diffusion = data_diffusion[~np.isnan(data_diffusion)]

emd_conc3 = wasserstein_distance(conc3_cmip6, conc3_diffusion)
enr_conc3 = emd_conc3 / conc3_cmip6.std()
print(enr_conc3)  # 0.7977949897513181




# %%
# plot
def z(x, μ, vmax, vmin):
    x = np.asarray(x)
    return (x - μ) / (vmax - vmin)

populations = [
    (bulk1_cmip6, bulk1_diffusion, enr_bulk1),
    (bulk2_cmip6, bulk2_diffusion, enr_bulk2),
    (bulk3_cmip6, bulk3_diffusion, enr_bulk3),
    (conc1_cmip6, conc1_diffusion, enr_conc1),
    (conc2_cmip6, conc2_diffusion, enr_conc2),
    (conc3_cmip6, conc3_diffusion, enr_conc3)
]
linbins = np.linspace(-0.5, 0.5, 200)
logbins = np.concatenate([-np.logspace(-6, 0, 100)[::-1], np.zeros(1), np.logspace(-6, 0, 100)])

width_ratios  = [0.2, 1, 1, 1, 0.3, 1, 1, 1]
height_ratios = [1, 0.1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(1. * ncoleff, 1.1 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.0,
              wspace=0.0)

ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.text(-0.1, -1, "EMD-to-noise =", va="center", ha="center", fontsize=6, weight="bold")

labels = 'abc-def'
for i, (cmip6, diffusion, emdtonoise) in enumerate(populations):
    if i >= 3:
        i = i + 1
    ax = fig.add_subplot(gs[0, i + 1])
    μ = jnp.concatenate([cmip6, diffusion]).mean()
    vmax = jnp.concatenate([cmip6, diffusion]).max()
    vmin = jnp.concatenate([cmip6, diffusion]).min()
    if i >= 3:
        bins = logbins
        ax.set_ylim(0, 45)
    else:
        bins = linbins
        ax.set_ylim(0, 15)
    sns.histplot(z(cmip6, μ, vmax, vmin), ax=ax, kde=False, stat="density", bins=bins, color="dodgerblue", edgecolor=None, label=f"{config.data.model_name}")
    sns.histplot(z(diffusion, μ, vmax, vmin), ax=ax, kde=False, stat="density", bins=bins, color="tomato", edgecolor=None, alpha=0.5, label="Emulator")
    ax.set_xlabel("[1]", fontsize=5, labelpad=1)
    ax.set_frame_on(False)
    ax.set_xlim(-0.3, 0.3)
    ax.yaxis.set_visible(False)
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1, i + 1])
    ax.axis("off")
    ax.text(0.5, -1, f"({labels[i]}) {emdtonoise:.2f}", va="center", ha="center", fontsize=6, weight="bold")


legend_elements = [
    Line2D([0], [0], color="dodgerblue", lw=6, alpha=0.6, label=config.data.model_name),
    Line2D([0], [0], color="tomato",    lw=6, alpha=0.4, label="Emulator"),
]
fig.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(0.07, 0.9),
    ncol=1,
    frameon=False,
    fontsize=6
)
output_dir = "experiments/mpi/plots/piControl/files"
filepath = os.path.join(output_dir, "emd_examples.jpg")
plt.savefig(filepath, dpi=500, bbox_inches='tight', pad_inches=0.02)
plt.close()
# plt.show()