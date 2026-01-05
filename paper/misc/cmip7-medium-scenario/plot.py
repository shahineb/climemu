# %%
import os
import numpy as np
import pandas as pd
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
    # df = pd.read_csv(os.path.join(base_dir, "data/tas_pr.csv"))
    df = pd.read_csv(os.path.join(base_dir, "outputs/tas_pr_2100_region.csv"))
    ssp126_tas = df["ssp126_tas"].values
    ssp126_logpr = df["ssp126_logpr"].values
    ssp585_tas = df["ssp585_tas"].values
    ssp585_logpr = df["ssp585_logpr"].values
    cmip7_tas = df["cmip7_tas"].values
    cmip7_logpr = df["cmip7_logpr"].values
    return ssp126_tas, ssp126_logpr, ssp585_tas, ssp585_logpr, cmip7_tas, cmip7_logpr

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
    ax.contourf(np.expm1(YY), XX, zz1, levels=20, cmap="Blues", alpha=1.0, zorder=0)
    ax.contourf(np.expm1(YY), XX, zz2, levels=20, cmap="Reds", alpha=0.4, zorder=0)
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
    y = YY[:, 0]
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

# %%
ssp126_tas, ssp126_logpr, ssp585_tas, ssp585_logpr, cmip7_tas, cmip7_logpr = load_plot_data()
XX, YY = setup_kde_grid(ssp126_tas, ssp585_tas, ssp126_logpr, ssp585_logpr)
levels = [0.05, 0.25, 0.5, 0.75, 0.95]
zz1, _ = kde_mass_contours(ssp126_tas, ssp126_logpr, XX, YY, levels)
zz2, _ = kde_mass_contours(ssp585_tas, ssp585_logpr, XX, YY, levels)
zz3, zlev3 = kde_mass_contours(cmip7_tas, cmip7_logpr, XX, YY, levels)


# %%
# Create plot
fig, gs = create_figure()

# Main contour plot
ax = fig.add_subplot(gs[1, 2])
plot_main_contours(ax, YY, XX, zz1, zz2, zz3, zlev3, levels)

# # Marginal plots
# ax_top = fig.add_subplot(gs[0, 2], sharex=ax)
# ax_right = fig.add_subplot(gs[1, 3], sharey=ax)
# plot_marginals(ax_top, ax_right, YY, XX, zz1, zz2, zz3)

# # Region map
# ax_map = fig.add_subplot(gs[1, -1], projection=ccrs.Robinson())
# ar6 = regionmask.defined_regions.ar6.all
# region = ar6[REGION_IDX]
# plot_region_map(ax_map, region)
    


# %%

def main():
    """Main plotting function."""
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
