# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs

# Constants
base_dir = os.path.dirname(__file__)
REGION_IDX = 10

def load_plot_data():
    """Load CMIP6 and CMIP7 emulated data."""
    df = pd.read_csv(os.path.join(base_dir, "outputs/tas_pr_2100_region.csv"))
    ssp126_tas = df["ssp126_tas"].values
    ssp126_logpr = df["ssp126_logpr"].values
    ssp585_tas = df["ssp585_tas"].values
    ssp585_logpr = df["ssp585_logpr"].values
    cmip7_tas = df["cmip7_tas"].values
    cmip7_logpr = df["cmip7_logpr"].values
    return ssp126_tas, ssp126_logpr, ssp585_tas, ssp585_logpr, cmip7_tas, cmip7_logpr

def load_emission_data():
    dirpath = os.path.join(base_dir, "data")
    df = pd.read_csv(os.path.join(dirpath, "extensions_1750-2500.csv"))
    df = df.loc[df.scenario == "medium-extension"]
    df = df.loc[df.variable == "CO2 FFI"]
    years = pd.to_numeric(df.columns, errors="coerce")
    year_mask = (years >= 1850.5) & (years <= 2100.5)
    cmip7_co2 = df.loc[:, year_mask].values.flatten()
    hist_co2 = xr.open_dataset(os.path.join(dirpath, "emissions_historical.nc"))["CO2"].values
    ssp126_co2 = xr.open_dataset(os.path.join(dirpath, "emissions_ssp126.nc"))["CO2"].values
    ssp585_co2 = xr.open_dataset(os.path.join(dirpath, "emissions_ssp585.nc"))["CO2"].values
    ssp126_co2 = np.concatenate([hist_co2, ssp126_co2])
    ssp585_co2 = np.concatenate([hist_co2, ssp585_co2])
    ssp126_co2 = np.diff(ssp126_co2, prepend=0)
    ssp585_co2 = np.diff(ssp585_co2, prepend=0)
    years = list(range(1850, 2101))
    return years, ssp126_co2, ssp585_co2, cmip7_co2


def setup_kde_grid(ssp126_tas, ssp585_tas, ssp126_logpr, ssp585_logpr, grid_size=300):
    """Setup grid for KDE contour plotting."""
    tasmin = min(ssp126_tas.min(), ssp585_tas.min())
    tasmax = max(ssp126_tas.max(), ssp585_tas.max())
    logprmin = min(ssp126_logpr.min(), ssp585_logpr.min())
    logprmax = max(ssp126_logpr.max(), ssp585_logpr.max())
    
    x_grid = np.linspace(tasmin, tasmax, grid_size)
    ylog_grid = np.linspace(logprmin, logprmax, grid_size)
    
    return np.meshgrid(x_grid, ylog_grid)

def kde_mass_contours(x, y, XX, YY, masses=(0.5, 0.75, 0.9)):
    """Compute KDE and return density and mass contour levels."""
    kde = gaussian_kde(np.vstack([x, y]))
    zz = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    z_sorted = np.sort(zz.ravel())[::-1]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]
    levels = np.sort([z_sorted[np.searchsorted(cdf, m)] for m in masses])
    return zz, levels


def create_figure():
    """Create figure with custom grid layout."""
    height_ratios = [0.2, 1]
    width_ratios = [1, 0.3, 1.5, 0.2, 0.01, 0.5]
    width_multiplier = 5.0
    height_multiplier = 5.0
    
    fig = plt.figure(figsize=(width_multiplier * sum(width_ratios), 
                              height_multiplier * sum(height_ratios)))
    gs = gridspec.GridSpec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.0
    )
    return fig, gs


def plot_main_contours(ax, YY, XX, zz1, zz2, zz3, zlev3, levels):
    """Plot main contour plot with SSP126, SSP585, and CMIP7 data."""
    ax.contourf(np.expm1(YY), XX, zz2, levels=20, cmap="Reds", alpha=0.6, zorder=1)
    ax.contourf(np.expm1(YY), XX, zz1, levels=20, cmap="Blues", alpha=0.4, zorder=2)
    cs = ax.contour(np.expm1(YY), XX, zz3, levels=zlev3, colors="k", linewidths=1, 
                    linestyles="--", zorder=5)
    fmt = {lev: f"{int(p*100)}%" for lev, p in zip(zlev3, levels)}
    ax.clabel(cs, fmt=fmt, fontsize=9)
    
    legend_handles = [
        Line2D([0], [0], color="cornflowerblue", lw=4, ls="-", alpha=0.5, 
               label="MPI-ESM1-2-LR SSP1-2.6"),
        Line2D([0], [0], color="salmon", lw=4, ls="-", alpha=0.5,
               label="MPI-ESM1-2-LR SSP5-8.5"),
        Line2D([0], [0], color="k", lw=1, ls="--", label="Emulated M")
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=18, prop={"size": 14})
    ax.set_ylabel("Near-surface temperature (°C)", fontsize=16)
    ax.set_xlabel("Precipitation (mm/day)", fontsize=16)
    ax.set_ylim(22, 39)
    ax.set_xlim(0, 18)
    ax.spines['top'].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def plot_marginals(ax_top, ax_right, YY, XX, zz1, zz2, zz3):
    """Plot marginal distributions."""
    x = XX[0, :]
    y = np.expm1(YY)[:, 0]
    mx1 = np.trapezoid(zz1, y, axis=0)
    my1 = np.trapezoid(zz1, x, axis=1)
    mx2 = np.trapezoid(zz2, y, axis=0)
    my2 = np.trapezoid(zz2, x, axis=1)
    mx3 = np.trapezoid(zz3, y, axis=0)
    my3 = np.trapezoid(zz3, x, axis=1)
    
    ax_top.plot(y, my1, color="cornflowerblue", lw=1, alpha=0.5)
    ax_top.plot(y, my2, color="tomato", lw=1, alpha=0.5)
    ax_top.plot(y, my3, color="k", lw=1, ls="--")
    ax_top.set_title("(b)", fontsize=16, weight="bold")
    ax_top.axis("off")
    
    ax_right.plot(mx1, x, color="cornflowerblue", lw=1, alpha=0.5)
    ax_right.plot(mx2, x, color="tomato", lw=1, alpha=0.5)
    ax_right.plot(mx3, x, color="k", lw=1, ls="--")
    ax_right.axis("off")


def plot_region_map(ax, region):
    """Plot region map."""
    ax.coastlines(linewidth=0.2, color="black")
    ax.add_geometries(
        [region.polygon],
        crs=ccrs.PlateCarree(),
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        zorder=10,
    )


def plot_emissions(years, ssp126_co2, ssp585_co2, cmip7_co2):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(years, ssp126_co2, label="SSP1-2.6", color="cornflowerblue", lw=4, alpha=0.8)
    ax.plot(years, ssp585_co2, label="SSP5-8.5", color="salmon", lw=4, alpha=0.8)
    ax.plot(years, cmip7_co2, label="M", color="k", lw=2, ls="--")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("CO$_2$ emissions (GtCO$_2$/yr)", fontsize=14)
    ax.legend(frameon=False, prop={"size": 14, "weight": "bold"})
    ax.margins(0.01)
    ax.spines['top'].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", which="major", labelsize=14) 
    ax.set_xlim(1950, 2105)
    ax.set_xticks([1900, 2000, 2100])
    ax.set_title("(a)", fontsize=16, weight="bold")
    plt.savefig(os.path.join(base_dir, "outputs/co2.jpg"), dpi=300, bbox_inches="tight")


def main():
    """Main plotting function."""
    # Load emission data and plot
    years, ssp126_co2, ssp585_co2, cmip7_co2 = load_emission_data()
    plot_emissions(years, ssp126_co2, ssp585_co2, cmip7_co2)

    # Load data
    ssp126_tas, ssp126_logpr, ssp585_tas, ssp585_logpr, cmip7_tas, cmip7_logpr = load_plot_data()

    # Setup KDE
    XX, YY = setup_kde_grid(ssp126_tas, ssp585_tas, ssp126_logpr, ssp585_logpr)
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    zz1, _ = kde_mass_contours(ssp126_tas, ssp126_logpr, XX, YY, levels)
    zz2, _ = kde_mass_contours(ssp585_tas, ssp585_logpr, XX, YY, levels)
    zz3, zlev3 = kde_mass_contours(cmip7_tas, cmip7_logpr, XX, YY, levels)
    
    # Create plot
    fig, gs = create_figure()
    
    # Main contour plot
    ax = fig.add_subplot(gs[1, 2])
    plot_main_contours(ax, YY, XX, zz1, zz2, zz3, zlev3, levels)
    
    # Marginal plots
    ax_top = fig.add_subplot(gs[0, 2], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 3], sharey=ax)
    plot_marginals(ax_top, ax_right, YY, XX, zz1, zz2, zz3)
    
    # Region map
    ax_map = fig.add_subplot(gs[1, -1], projection=ccrs.Robinson())
    ar6 = regionmask.defined_regions.ar6.all
    region = ar6[REGION_IDX]
    plot_region_map(ax_map, region)
    
    # Save
    plt.savefig(os.path.join(base_dir, "outputs/m-tp.jpg"), dpi=300, bbox_inches="tight")


# if __name__ == "__main__":
#     main()

# %%
main()


# %%
import seaborn as sns
ssp126_tas, ssp126_logpr, ssp585_tas, ssp585_logpr, cmip7_tas, cmip7_logpr = load_plot_data()


# %%
x = np.expm1(ssp126_logpr).clip(min=1e-6)
p0 = np.mean(x <= 0)
xp = x
yp = np.log10(xp)
kde_y = gaussian_kde(yp)

grid = np.linspace(1e-3, xp.max(), 600)
dens_pos = kde_y(np.log10(grid)) / grid
dens = (1 - p0) * dens_pos


# %%
fig, ax = plt.subplots(figsize=(7, 4))

# Histogram on positives only (density=True => integrates to 1 over positives)
# ax.hist(xp, bins=40, density=True, alpha=0.35, edgecolor="none")

# Mixed-model density on x>0 (this integrates to 1-p0 over (0,inf))
ax.plot(grid, dens, lw=2)

# # Spike at 0 to represent point mass p0
# ymax = max(ax.get_ylim()[1], dens.max() * 1.05)
# ax.vlines(0, 0, ymax * min(1.0, 5 * p0), lw=3)  # scaled spike for visibility
# ax.annotate(f"p0={p0:.2f}", xy=(0, ymax * 0.9), xytext=(0.02 * grid.max(), ymax * 0.9))

ax.set_xlabel("precip")
ax.set_ylabel("density (continuous part) + spike at 0")
ax.set_xlim(0, 1)
# ax.set_xscale("log")


plt.tight_layout()
plt.show()
# %%
