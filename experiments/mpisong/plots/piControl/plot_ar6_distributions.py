import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
import jax.numpy as jnp
import matplotlib.pyplot as plt
import regionmask
import seaborn as sns
from tqdm import tqdm


# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import load_data, VARIABLES, assign_month_and_season_from_doy, setup_figure, save_plot, myRdPu


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'experiments/mpi/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.05
HSPACE = 0.05


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def adaptive_quantiles(num_quantiles, p_min=0.00001, p_max=0.99999):
    # Convert boundaries to logit space
    logit_min = np.log(p_min / (1 - p_min))
    logit_max = np.log(p_max / (1 - p_max))
    
    # Uniform spacing in logit space
    logits = np.linspace(logit_min, logit_max, num=num_quantiles)
    
    # Convert back to probability space using the logistic function
    quantiles = 1 / (1 + np.exp(-logits))
    return quantiles

def qq_plot(data1, data2, ax, num_quantiles=400):
    data1 = np.sort(np.asarray(data1))
    data2 = np.sort(np.asarray(data2))
    quantiles = adaptive_quantiles(num_quantiles)
    q1 = np.quantile(data1, quantiles)
    q2 = np.quantile(data2, quantiles)
    lims = [min(q1.min(), q2.min()), max(q1.max(), q2.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.scatter(q1, q2, s=2)
    return ax

def subsample_years(ds, n_years):
    unique_years = np.unique(ds.time.dt.year.values)
    idx = np.linspace(0, len(unique_years) - 1, n_years).round().astype(int)
    years_to_keep = unique_years[idx]
    return ds.sel(time=ds.time.dt.year.isin(years_to_keep))



config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=False)
climatology = climatology.load()

piControl_diffusion = piControl_diffusion + climatology
piControl_cmip6 = subsample_years(piControl_cmip6, piControl_diffusion.sizes['sample'])
doy = piControl_cmip6["time"].dt.dayofyear
piControl_cmip6 = piControl_cmip6 + climatology.sel(dayofyear=doy)
with ProgressBar():
    piControl_diffusion = piControl_diffusion.compute()
    piControl_cmip6 = piControl_cmip6.compute()
piControl_diffusion['pr'] = piControl_diffusion['pr'].clip(min=0)
piControl_diffusion['hurs'] = piControl_diffusion['hurs'].clip(min=0, max=100)


piControl_diffusion = assign_month_and_season_from_doy(piControl_diffusion, dim="dayofyear")
piControl_cmip6 = assign_month_and_season_from_doy(piControl_cmip6, dim="time")

piControl_diffusion_flat = piControl_diffusion.stack(flat=('dayofyear', 'sample'))
piControl_cmip6_flat = piControl_cmip6.stack(flat=('time',))


# AR6 regions
ar6 = regionmask.defined_regions.ar6.all
north_america = ar6[[0, 1, 2, 3, 4, 5, 6, 8]]
south_america = ar6[[7, 9, 10, 11, 12, 13, 14, 15]]
europe = ar6[[16, 17, 18, 19]]
africa = ar6[[20, 21, 22, 23, 24, 25, 26, 27]]
asia = ar6[[28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]
se_asia_oceania = ar6[[38, 39, 40, 41, 42, 43]]
poles = ar6[[44, 45, 46]]
oceans = ar6[[47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]]
domains = [north_america, south_america, europe, africa, asia, se_asia_oceania, poles, oceans]
domain_names = ["North America", "South America", "Europe", "Africa", "Asia", "Southeast Asia & Oceania", "Poles", "Oceans"]

# Create output directory if it doesn't exist
output_dir = "experiments/mpi/plots/piControl/files/densityplots"
os.makedirs(output_dir, exist_ok=True)

# Iterate over each variable
n_iterations = len(VARIABLES) * len(ar6) * 4
with tqdm(total=n_iterations) as pbar:
    for var_name in VARIABLES.keys():
        var_info = VARIABLES[var_name]
        unit = var_info['unit']
        
        # Iterate over each domain
        for domain_idx, (domain, domain_name) in enumerate(zip(domains, domain_names)):
            pbar.set_description(f"Processing {var_name} for {domain_name}")
            
            domain_mask = domain.mask(climatology.lon, climatology.lat)
            nrows = len(domain)
            ncols = 8
            
            # Skip empty domains or domains with very few regions
            if nrows == 0:
                print(f"Skipping empty domain: {domain_name}")
                continue
                
            fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
            
            # Handle single region case
            if nrows == 1:
                ax = np.array([[ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7]]])
            
            for i, (region_idx, region) in enumerate(domain.regions.items()):
                region_mask = domain_mask == region_idx

                region_data_target_all = piControl_cmip6_flat[var_name].where(region_mask)
                region_data_pred_all = piControl_diffusion_flat[var_name].where(region_mask)

                for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
                    season_mask = piControl_cmip6_flat.season == season
                    region_data_target = region_data_target_all.where(season_mask, drop=True).values.ravel()
                    season_mask = piControl_diffusion_flat.season == season
                    region_data_pred = region_data_pred_all.where(season_mask, drop=True).values.ravel()

                    region_data_target = region_data_target[~np.isnan(region_data_target)]
                    region_data_pred = region_data_pred[~np.isnan(region_data_pred)]

                    nbins = np.ceil(2 * len(region_data_target) ** (1 / 3)).astype(int)
                    sns.histplot(region_data_target, ax=ax[i, 2*j], kde=False, stat="density", bins=nbins, color="skyblue", edgecolor=None, label=f"{config.data.model_name}")
                    sns.histplot(region_data_pred, ax=ax[i, 2*j], kde=False, stat="density", bins=nbins, color="orange", edgecolor=None, alpha=0.5, label="Diffusion")
                    ax_cdf = ax[i, 2*j].twinx()
                    sns.histplot(region_data_target, ax=ax_cdf, bins=nbins, stat="density", cumulative=True, element="step", fill=False, linewidth=1.5, color="skyblue")
                    sns.histplot(region_data_pred, ax=ax_cdf, bins=nbins, stat="density", cumulative=True, element="step", fill=False, linewidth=1.5, color="orange")
                    qq_plot(region_data_pred, region_data_target, ax=ax[i, 2*j+1])
                    ax[i, 2*j+1].set_xlabel(f"Diffusion [{unit}]")
                    ax[i, 2*j+1].set_ylabel(f"{config.data.model_name} [{unit}]")
                    ax[i, 2*j].set_title(f"{season} - {region_idx} - {region.name}")
                    ax[i, 2*j].set_xlabel(f"[{unit}]")
                    if var_name == "pr":
                        ax[i, 2*j].set_yscale('log')
                    _ = pbar.update(1)
                
            ax[0, 0].legend()
            plt.suptitle(f"{var_name} Distribution: {domain_name}", fontsize=16)
            
            # Save the figure
            filename = f"distribution_{var_name}_{domain_name.replace(' ', '_').lower()}.jpg"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

    print("Finished generating all plots.")
