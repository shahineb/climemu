import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import seaborn as sns



def plot_mean_and_distribution_over_us(combined_ds):
    lon_range = slice(230, 305)
    lat_range = slice(20, 65)

    tas_ds = combined_ds['tas'].sel(lon=lon_range, lat=lat_range)
    pr_ds = combined_ds['pr'].sel(lon=lon_range, lat=lat_range)
    hurs_ds = combined_ds['hurs'].sel(lon=lon_range, lat=lat_range)
    sfcWind_ds = combined_ds['sfcWind'].sel(lon=lon_range, lat=lat_range)

    tas_data = tas_ds.values.ravel()
    pr_data = pr_ds.values.ravel()
    hurs_data = hurs_ds.values.ravel()
    sfcWind_data = sfcWind_ds.values.ravel()

    mean_tas = tas_ds.mean('member')
    mean_pr = pr_ds.mean('member')
    mean_hurs = hurs_ds.mean('member')
    mean_sfcWind = sfcWind_ds.mean('member')

    width_ratios  = [1, 0.05, 0.2, 1]
    height_ratios = [1, 1, 1, 1]
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
                hspace=0.2,
                wspace=0.05)


    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    mesh = mean_tas.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', add_colorbar=False)
    ax.coastlines()
    vmax = np.abs(mean_tas.values).max().item()
    mesh.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(mesh,
                        cax=cax,
                        orientation='vertical')
    cbar.set_label(f"Mean tas anomaly (°C)", labelpad=4)

    ax = fig.add_subplot(gs[0, 3])
    sns.histplot(tas_data, ax=ax, kde=False, stat="density", bins=100, color="cornflowerblue", edgecolor=None)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[°C]")


    ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    mesh = mean_pr.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='BrBG', add_colorbar=False)
    ax.coastlines()
    vmax = np.abs(mean_pr.values).max().item()
    mesh.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(mesh,
                        cax=cax,
                        orientation='vertical')
    cbar.set_label(f"Mean pr anomaly (mm/day)", labelpad=4)

    ax = fig.add_subplot(gs[1, 3])
    sns.histplot(pr_data, ax=ax, kde=False, stat="density", bins=100, color="cornflowerblue", edgecolor=None)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[mm/day]")



    ax = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    mesh = mean_hurs.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='BrBG', add_colorbar=False)
    ax.coastlines()
    vmax = np.abs(mean_hurs.values).max().item()
    mesh.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[2, 1])
    cbar = fig.colorbar(mesh,
                        cax=cax,
                        orientation='vertical')
    cbar.set_label(f"Mean hurs anomaly (%)", labelpad=4)

    ax = fig.add_subplot(gs[2, 3])
    sns.histplot(hurs_data, ax=ax, kde=False, stat="density", bins=100, color="cornflowerblue", edgecolor=None)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[%]")


    ax = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
    mesh = mean_sfcWind.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='PRGn', add_colorbar=False)
    ax.coastlines()
    vmax = np.abs(mean_sfcWind.values).max().item()
    mesh.set_clim(-vmax, vmax)
    cax = fig.add_subplot(gs[3, 1])
    cbar = fig.colorbar(mesh,
                        cax=cax,
                        orientation='vertical')
    cbar.set_label(f"Mean sfcWind anomaly (m/s)", labelpad=4)

    ax = fig.add_subplot(gs[3, 3])
    sns.histplot(sfcWind_data, ax=ax, kde=False, stat="density", bins=100, color="cornflowerblue", edgecolor=None)
    ax.set_frame_on(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("[m/s]")
    plt.show()