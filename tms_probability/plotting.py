from tms_probability.probability_analysis import get_thresholds, get_results, aggregate_threshold_probabilities
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from math import ceil, floor
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import classic_formatter
from mpl_toolkits.basemap import Basemap
import matplotlib.tri as tri
from scipy.ndimage import gaussian_filter
import warnings

def plot_io_curves(fname_list, efields_list, title, legend, morphs=[1, 2, 3, 4, 5]):
    probabilities_list = []
    for fname, efields in zip(fname_list, efields_list):
        thresholds = get_thresholds(fname)
        probabilities = [aggregate_threshold_probabilities(thresholds, efield, morphs) for efield in efields]
        probabilities_list.append(probabilities)

    matplotlib.rc('figure', figsize=(12, 6))
    for probabilities, efields in zip(probabilities_list, efields_list):
        plt.plot(efields, probabilities)
    plt.title(title, fontsize=20)
    plt.xlabel('Electric Field Amplitude (V/m)', fontsize=15)
    plt.ylabel('Recruitment Probability', fontsize=15)
    plt.legend(legend, fontsize=15)
    plt.show()


def get_threshold_heatmap_by_cell_old(thresholds):
    projected_data_by_cell = []
    for cell, polar_dict in thresholds.items():
        # Project threshold data by onto a spherical map by E-field direction
        projected_data = []
        for azimuthal_dict in polar_dict['Polar'].values():
            plot_azimuth_band = []
            idx = 0
            for azimuthal, threshold in azimuthal_dict['Azimuthal'].items():
                azimuthal = float(azimuthal)
                
                if type(threshold) == dict:
                    threshold = threshold['threshold']
                
                try:
                    next_azimuthal = float(list(azimuthal_dict['Azimuthal'].keys())[idx+1])
                except IndexError:
                    next_azimuthal = 360
                plot_azimuths = list(range(floor(azimuthal), floor(next_azimuthal)))

                plot_azimuth_band.extend([threshold for _ in range(len(plot_azimuths))])
                idx += 1
            projected_data.append(plot_azimuth_band)
        projected_data_by_cell.append(projected_data)
    return projected_data_by_cell

def get_threshold_map_by_cell(thresholds):
    data_by_cell = {}
    for cell, polar_dict in thresholds.items():
        # Project threshold data onto a spherical map by E-field direction
        data = []
        polars = []
        azimuthals = []
        for polar, azimuthal_dict in polar_dict['Polar'].items():
            polar = float(polar)
            for azimuthal, threshold in azimuthal_dict['Azimuthal'].items():
                azimuthal = float(azimuthal)
                
                if type(threshold) == dict:
                    threshold = threshold['threshold']
                
                if azimuthal == 0.0:
                    thresh_zero_az = threshold # Store the threshold at azimuthal=0 to use for azimuthal=360 
                
                data.append(threshold)
                polars.append(polar)
                azimuthals.append(azimuthal)
            data.append(thresh_zero_az)
            polars.append(polar)
            azimuthals.append(360)
        data_by_cell[cell] = [data, polars, azimuthals]
    return data_by_cell


def get_threshold_diff_by_cell(data_by_cell_0, data_by_cell_1):
    thresh_diff_by_cell = {}
    for (name0, [thresh0, polars0, azimuthals0]), (name1, [thresh1, polars1, azimuthals1]) in zip(data_by_cell_0.items(), data_by_cell_1.items()):
        thresh_diff_by_cell[name0] = [list(np.array(thresh0)-np.array(thresh1)), polars0, azimuthals0]
    return thresh_diff_by_cell

def plot_threshold_heatmap_by_cell_old(fname):
    thresholds = get_thresholds(fname)
    projected_data_by_cell = get_threshold_heatmap_by_cell_old(thresholds)
    plotting_azimuths = list(range(360))

    for projected_data, [cell, polar_dict] in zip(projected_data_by_cell, thresholds.items()):
        polar_list = [float(polar) for polar in polar_dict['Polar'].keys()]
        plot_projection_old(data=projected_data, 
                        azimuths=np.array(plotting_azimuths)-180, 
                        polars=np.array(polar_list)-90, 
                        cmap='inferno_r',
                        vmax=np.max(projected_data), 
                        vmin=np.min(projected_data), 
                        title=cell, 
                        cbar_label="TMS E-field Threshold",
                        grid=False,
                        )


def plot_threshold_heatmap_by_cell(fname, grid=False):
    thresholds = get_thresholds(fname)
    data_by_cell = get_threshold_map_by_cell(thresholds)

    fig, axs = plt.subplots(5, 1, figsize=(100, 20))
    for (cell, [data, polars, azimuthals]), ax in zip(data_by_cell.items(), axs):
        plot_projection(data=data, 
                        azimuthals=azimuthals, 
                        polars=polars, 
                        cmap='inferno_r',
                        title=cell, 
                        cbar_label="TMS E-field Threshold",
                        grid=grid,
                        ax=ax,
                        )


def plot_threshold_diff_heatmap_by_cell(fnames):
    thresholds_0 = get_thresholds(fnames[0])
    thresholds_1 = get_thresholds(fnames[1])
    projected_data_by_cell_0 = get_threshold_map_by_cell(thresholds_0)
    projected_data_by_cell_1 = get_threshold_map_by_cell(thresholds_1)
    projected_differences_by_cell = np.array(projected_data_by_cell_0)-np.array(projected_data_by_cell_1)
    plotting_azimuths = list(range(360))

    for projected_differences, [cell, polar_dict] in zip(projected_differences_by_cell, thresholds_0.items()):
        polar_list = [float(polar) for polar in polar_dict['Polar'].keys()]
        vmax = np.max(np.abs(projected_differences))
        plot_projection_old(data=projected_differences, 
                        azimuths=np.array(plotting_azimuths)-180, 
                        polars=np.array(polar_list)-90, 
                        cmap='bwr', 
                        vmax=vmax, 
                        vmin=-vmax, 
                        title=cell, 
                        cbar_label="TMS E-field Threshold Difference",
                        grid=False,
                        )

def plot_projection_old(data, azimuths, polars, cmap, vmax, vmin, title, cbar_label, grid):
    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='mollweide')

    ax.pcolormesh(
        np.radians(azimuths), np.radians(polars), 
        data,
        cmap=cmap, 
        shading='auto',
        vmin=vmin, vmax=vmax
    )
    ax.set_title(title)
    ax.grid(visible=grid, axis='both')

    # add the colorbar
    cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap            
            ),
            ax=plt.gca()
        )
    cbar.set_label(cbar_label)

def plot_projection(data, azimuthals, polars, cmap, title, cbar_label, grid, ax):
    # Plot figure
    m = Basemap(projection='moll', lon_0=0, ax=ax)
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-90, 90, 181)

    triang = tri.Triangulation(np.array(azimuthals)-180, -np.array(polars)+90)
    interpolator = tri.LinearTriInterpolator(triang, data)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    datai = interpolator(lon_grid, lat_grid)
    data_smooth = gaussian_filter(datai, sigma=0)

    # a, p = m(ax_grid, pol_grid)
    # print(help(ax.pcolormesh))
    # print(lon_grid)
    # print(lat_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # m.pcolormesh throws a warning about the inputs being non-monotonic
        heatmap = m.pcolormesh(lon_grid, lat_grid, data_smooth, cmap=cmap, shading='auto', latlon=True)
    cbar = m.colorbar(heatmap, pad=0.4)
    cbar.set_label(cbar_label)

    if grid:
        num_parallels = 12
        num_meridians = 24
        m.drawparallels(np.linspace(-90, 90, num_parallels+1))
        m.drawmeridians(np.linspace(-180, 180, num_meridians+1))

        # merids = np.linspace(-180, 180, num_meridians+1)
        # front = [merid for merid in merids if merid>=-90 and merid<=90]
        # back = [merid for merid in merids if merid<-90 or merid>90]
        # m.drawmeridians(front, color=np.array([1, 1, 1])*.6)
        # m.drawmeridians(back, color=np.array([1, 1, 1])*0)

def plot_projection_cartopy(data, azimuths, polars, cmap, vmax, vmin, title, cbar_label):
    # WIP; TODO
    # Plot figure
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide())

    ax.pcolormesh(
        azimuths, polars, 
        data,
        cmap=cmap, 
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(title)
    gl = ax.gridlines(draw_labels=True)
    # gl.xformatter = classic_formatter
    # gl.yformatter = classic_formatter
    # gl.n_steps = 10
    # gl.xlabels_top = False

    # add the colorbar
    cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap            
            ),
            ax=plt.gca()
        )
    cbar.set_label(cbar_label)