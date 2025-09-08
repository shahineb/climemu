import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import beta



# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.utils import arrays
from experiments.access.config import Config
from experiments.access.plots.ssp245.utils import load_data, VARIABLES
from experiments.access.plots.piControl.utils import load_data as load_piControl_data


def quantile_ci(sorted_data, probs, alpha=0.05):
    n = len(sorted_data)
    k = np.floor(probs * n).astype(int)

    # vectorized beta-based proportion CIs
    ci_low = beta.ppf(alpha/2, k, n-k+1)
    ci_high = beta.ppf(1-alpha/2, k+1, n-k)

    # convert from proportion -> sample quantiles via interpolation
    qs = np.linspace(0, 1, n)
    lb = np.interp(ci_low, qs, sorted_data)
    ub = np.interp(ci_high, qs, sorted_data)
    return lb, ub



config = Config()
test_dataset, pred_samples, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds.sel(time=slice('2080-01', '2100-12'))
pred_samples = pred_samples.sel(time=slice('2080-01', '2100-12'))

climatology, _, piControl_cmip6 = load_piControl_data(config, in_memory=False)

target_data = arrays.groupby_month_and_year(target_data) + climatology
pred_samples = arrays.groupby_month_and_year(pred_samples) + climatology

target_data = target_data.compute()
pred_samples = pred_samples.compute()




width_ratios  = [1, 1, 1, 1]
height_ratios = [1]
nrow = len(height_ratios)
ncol = len(width_ratios)
nroweff = sum(height_ratios)
ncoleff = sum(width_ratios)


fig = plt.figure(figsize=(4 * ncoleff, 3 * nroweff))
gs = GridSpec(nrows=nrow,
              ncols=ncol,
              figure=fig,
              width_ratios=width_ratios,
              height_ratios=height_ratios,
              hspace=0.3,
              wspace=0.1)



for i, var in enumerate(VARIABLES.keys()):
    var_info = VARIABLES[var]
    var_name = var_info["name"]
    unit = var_info['unit']

    cmip6_data = target_data[var].values.ravel()
    diffusion_data = pred_samples[var].values.ravel()
    if var == "tas":
        unit = "°C"
        cmip6_data = cmip6_data - 273.15
        diffusion_data = diffusion_data - 273.15


    # Compute tail quantiles and keep only data past 99th
    qcmip6 = np.quantile(cmip6_data, [0.99, 0.999, 0.9999, 0.99999])
    qdiffusion = np.quantile(diffusion_data, [0.99, 0.999, 0.9999, 0.99999])
    γ = min(qcmip6[0], qdiffusion[0]) - 0.1
    cmip6_tail = cmip6_data[cmip6_data >= γ]
    diffusion_tail = diffusion_data[diffusion_data >= γ]

    # Create bins for histplot
    bins_cmip6 = np.histogram_bin_edges(cmip6_tail, bins="fd")
    bins_diffusion = np.histogram_bin_edges(diffusion_tail, bins="fd")

    # Make the plots
    ax = fig.add_subplot(gs[0, i])
    sns.histplot(cmip6_tail, ax=ax, kde=False, element="step", stat="density", fill=False, bins=bins_cmip6, color="dodgerblue", alpha=0.8, label=f"{config.data.model_name}")
    sns.histplot(diffusion_tail, ax=ax, kde=False, element="step", stat="density", fill=False, bins=bins_diffusion, color="tomato", alpha=0.8, label="Emulator")

    ax.set_yscale('log')
    ax.set_xlabel(f"{var_name} [{unit}]")
    if i > 0:
        ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.margins(x=0, y=0)
    if i == 3:
        ax.legend(loc='upper right')

    sec_top = ax.secondary_xaxis('bottom')
    sec_top.set_xticks(qcmip6)
    sec_top.set_xticklabels(["99%", "99.9%", "99.99%", "99.999%"], color="dodgerblue", ha='left', fontsize=7, rotation=45)
    sec_top.xaxis.set_ticks_position('top')
    sec_top.spines['bottom'].set_color("dodgerblue")
    sec_top.tick_params(axis='x', colors='dodgerblue', pad=0.1)

    sec_bot = ax.secondary_xaxis('bottom')
    sec_bot.set_xticks(qdiffusion)
    sec_bot.set_xticklabels(["99%", "99.9%", "99.99%", "99.999%"], color="tomato", ha='left', fontsize=7, rotation=45)
    sec_bot.xaxis.set_ticks_position('top')
    sec_bot.spines['bottom'].set_color("tomato")
    sec_bot.tick_params(axis='x', colors='tomato', pad=0.1)

plt.savefig("experiments/access/plots/ssp245/files/tails.jpg", dpi=300, bbox_inches='tight')
plt.close()