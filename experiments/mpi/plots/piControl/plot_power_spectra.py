import os
import sys
import numpy as np
import jax.numpy as jnp
import pyshtools as pysh
import cartopy.crs as ccrs
from tqdm import tqdm

# Add base directory to path if not already added
base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from experiments.mpi.config import Config
from experiments.mpi.plots.piControl.utils import load_data, VARIABLES, setup_figure, save_plot

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'experiments/mpi/plots/piControl/files'
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.05
HSPACE = 0.05
RANDOM_SEED = 5  # For reproducible sample selection


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================


def compute_Cl(da):
    """Compute power spectrum using spherical harmonics."""
    da = da - da.mean()
    grid = pysh.SHGrid.from_xarray(da, grid='GLQ')
    clm = grid.expand()
    Cl = clm.spectrum()
    return Cl

def get_plot_data(pred_da, cmip6_da):
    """Compute power spectra ensemble statistics for both emulator and CMIP6 data."""
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


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

config = Config()
climatology, piControl_diffusion, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology

# Setup wavenumber array for power spectra
R = 6371  # km
ell = np.arange(piControl_diffusion.sizes['lat'])
k = np.where(ell > 0, ell, np.nan) / (2 * np.pi * R)

# Select random sample for visualization
np.random.seed(RANDOM_SEED)
sample_year_idx = np.random.randint(len(piControl_diffusion.year))
sample_month_idx = np.random.randint(12)
print(f"Sample for month {sample_month_idx + 1}")

# Compute power spectra for all variables
emulator = dict()
cmip6 = dict()

for i, var in enumerate(VARIABLES):
    emulator_psd_data, cmip6_psd_data = get_plot_data(piControl_diffusion[var],
                                                        piControl_cmip6[var])
    emulator_psd_data['sample'] = piControl_diffusion[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    cmip6_psd_data['sample'] = piControl_cmip6[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    emulator[var] = emulator_psd_data
    cmip6[var] = cmip6_psd_data



# =============================================================================
# PLOTTING
# =============================================================================

def create_power_spectra_plot():
    """Create the power spectra comparison plot."""
    width_ratios = [0.1, 1, 0.1, 0.25, 1, 0.1, 0.25, 1, 0.1, 0.25, 1, 0.1, 0.25]
    height_ratios = [0.8, 0.8, 1.5]
    
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    # Add labels
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0.25, 0.5, f"{config.data.model_name} \n realization", va="center", ha="center", rotation="vertical", fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0.25, 0.5, "Emulator \n sample", va="center", ha="center", rotation="vertical", fontsize=12)

    # Plot each variable
    for i, var in enumerate(VARIABLES):
        var_info = VARIABLES[var]
        unit = var_info['unit']
        cmap = var_info['cmap']
        i = 3 * i + 1
        flatvalues = []

        # CMIP6 sample
        ax = fig.add_subplot(gs[0, i], projection=ccrs.Robinson())
        mesh1 = cmip6[var]['sample'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        flatvalues.append(cmip6[var]['sample'].values.ravel())
        ax.coastlines(lw=0.5, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)
        ax.set_title(var_info['name'], fontsize=16, weight='bold')

        # Emulator sample
        ax = fig.add_subplot(gs[1, i], projection=ccrs.Robinson())
        mesh2 = emulator[var]['sample'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        flatvalues.append(emulator[var]['sample'].values.ravel())
        ax.coastlines(lw=0.5, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)

        # Set consistent color limits
        vmax = np.quantile(np.concatenate(flatvalues), 0.999)
        vmin = np.quantile(np.concatenate(flatvalues), 0.001)
        vmax = max(np.abs(vmax), np.abs(vmin))
        mesh1.set_clim(-vmax, vmax)
        mesh2.set_clim(-vmax, vmax)
        
        # Colorbar
        cax = fig.add_subplot(gs[:2, i + 1])
        cbar = fig.colorbar(mesh1, cax=cax, orientation='vertical', extend='both')
        cbar.set_label(f"[{unit}]", labelpad=4)
        pos = cax.get_position()
        new_width = pos.width * 0.3    # thinner
        new_height = pos.height * 0.3   # shorter
        new_x0 = pos.x0 + 0.00      # move right
        new_y0 = pos.y0 + (pos.height - new_height) / 2  # recenter vertically
        cax.set_position([new_x0, new_y0, new_width, new_height])

        # Power spectra plot
        ax = fig.add_subplot(gs[2, i:i+2])
        ax.fill_between(k, emulator[var]['lb'], emulator[var]['ub'], color='tomato', alpha=0.2)
        ax.fill_between(k, cmip6[var]['lb'], cmip6[var]['ub'], color='cornflowerblue', alpha=0.2)
        ax.plot(k, cmip6[var]['median'], label=config.data.model_name, color='cornflowerblue')
        ax.plot(k, emulator[var]['median'], label='Emulator', color='tomato')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.margins(x=0, y=0)
        ax.set_xlabel('Inverse wavelength [km⁻¹]', fontsize=14)
        ax.set_ylabel(f'[({unit})²⋅km]', fontsize=14)
        if i == 1:
            ax.legend(fontsize=16, frameon=False)
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to generate power spectra plot."""
    fig = create_power_spectra_plot(emulator, cmip6, k, config)
    save_plot(fig, OUTPUT_DIR, 'psd.jpg', dpi=DPI)

if __name__ == "__main__":
    main()