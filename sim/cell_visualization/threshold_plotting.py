import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)

from tms_thresholds.recruitment_analysis import get_thresholds_dict, get_results, get_cell_type_recruitment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import classic_formatter
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
import warnings


def get_threshold_map_by_cell(thresholds):
    data_by_cell = {}
    for cell, polar_dict in thresholds.items():
        # Project threshold data onto a spherical map by E-field direction
        data = []
        polars = []
        azimuthals = []
        for polar, azimuthal_dict in polar_dict['Polar'].items():
            polar = float(polar)
            thresh_zero_az = None

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


def get_threshold_diff_map_by_cell(thresholds_0, thresholds_1):
    # Project threshold difference data onto a spherical map by E-field direction
    data_by_cell = {}
    for (cell_0, polar_dict_0), (cell_1, polar_dict_1) in zip(thresholds_0.items(), thresholds_1.items()):
        if cell_0 != cell_1:
            raise ValueError("Cells in the two threshold dictionaries do not match")
        data = []
        polars = []
        azimuthals = []
        for (polar_0, azimuthal_dict_0), (polar_1, azimuthal_dict_1) in zip(polar_dict_0['Polar'].items(), polar_dict_1['Polar'].items()):
            polar_0 = float(polar_0)
            polar_1 = float(polar_1)
            if polar_0 != polar_1:
                raise ValueError("Polar angles in the two threshold dictionaries do not match")
            for (azimuthal_0, threshold_0), (azimuthal_1, threshold_1) in zip(azimuthal_dict_0['Azimuthal'].items(), azimuthal_dict_1['Azimuthal'].items()):
                azimuthal_0 = float(azimuthal_0)
                azimuthal_1 = float(azimuthal_1)
                if azimuthal_0 != azimuthal_1:
                    raise ValueError("Azimuthal angles in the two threshold dictionaries do not match")

                if type(threshold_0) == dict:
                    threshold_0 = threshold_0['threshold']
                if type(threshold_1) == dict:
                    threshold_1 = threshold_1['threshold']

                threshold_diff = threshold_0 - threshold_1
                
                if azimuthal_0 == 0.0:
                    thresh_diff_zero_az = threshold_diff # Store the threshold at azimuthal=0 to use for azimuthal=360 
                
                data.append(threshold_diff)
                polars.append(polar_0)
                azimuthals.append(azimuthal_0)
            data.append(thresh_diff_zero_az)
            polars.append(polar_0)
            azimuthals.append(360)
        data_by_cell[cell_0] = [data, polars, azimuthals]
    return data_by_cell


# old version of get_threshold_diff_map_by_cell
# requires inputs to be the results from get_threshold_map_by_cell instead of the raw threshold dictionaries
def get_threshold_diff_from_data_by_cell(data_by_cell_0, data_by_cell_1):
    thresh_diff_by_cell = {}
    for (name0, [thresh0, polars0, azimuthals0]), (name1, [thresh1, polars1, azimuthals1]) in zip(data_by_cell_0.items(), data_by_cell_1.items()):
        thresh_diff_by_cell[name0] = [list(np.array(thresh0)-np.array(thresh1)), polars0, azimuthals0]
    return thresh_diff_by_cell


def plot_threshold_heatmap_by_cell(fname, axs=None, fontsizes=[60, 40, 40], grid=False):
    thresholds = get_thresholds_dict(fname)
    data_by_cell = get_threshold_map_by_cell(thresholds)

    if axs is None:
        fig, axs = plt.subplots(len(data_by_cell), 1, figsize=(100, 20))
    for (cell, [data, polars, azimuthals]), ax in zip(data_by_cell.items(), axs):
        plot_projection(data=data, 
                        azimuthals=azimuthals, 
                        polars=polars, 
                        cmap='inferno_r',
                        title=cell, 
                        cbar_label="TMS E-field Threshold",
                        grid=grid,
                        ax=ax,
                        fontsizes=fontsizes,
                        )


def plot_threshold_diff_heatmap_by_cell(fnames, axs=None, fontsizes=[60, 40, 40], grid=False):
    thresholds_0 = get_thresholds_dict(fnames[0])
    thresholds_1 = get_thresholds_dict(fnames[1])
    data_by_cell = get_threshold_diff_map_by_cell(thresholds_0, thresholds_1)

    if axs is None:
        fig, axs = plt.subplots(len(data_by_cell), 1, figsize=(100, 20))
    for (cell, [data, polars, azimuthals]), ax in zip(data_by_cell.items(), axs):
        vmax = np.max(np.abs(data))
        plot_projection(data=data, 
                        azimuthals=azimuthals, 
                        polars=polars, 
                        cmap='bwr',
                        title=cell, 
                        cbar_label="TMS E-field Threshold Difference",
                        grid=grid,
                        ax=ax,
                        vmax=vmax, 
                        vmin=-vmax, 
                        fontsizes=fontsizes,
                        )

def plot_cell_threshold_comparison(fnames, figsize=None, fontsizes=[32, 24, 24], consistent_scale=False, grid=False, lim_max=None):
    num_conditions = len(fnames)
    num_diffs = (num_conditions-1)*num_conditions//2
    num_columns = num_conditions+num_diffs

    if figsize is None:
        figsize = (num_columns*12, 32)
    
    fig, axs = plt.subplots(5, num_columns, figsize=figsize)

    thresh_dicts = [get_thresholds_dict(fname) for fname in fnames]
    thresh_maps = [get_threshold_map_by_cell(t) for t in thresh_dicts]
    thresh_diffs = [get_threshold_diff_map_by_cell(thresh_dicts[i], thresh_dicts[j]) for i in range(num_conditions) for j in range(i+1, num_conditions)]

    data_mins = np.min([[np.min(thresh) for [thresh, polars, azimuthals] in map.values()] for map in thresh_maps], axis=0)
    data_maxs = np.max([[np.max(thresh) for [thresh, polars, azimuthals] in map.values()] for map in thresh_maps], axis=0)
    thresh_lims = np.array([data_mins, data_maxs]).T

    for thresh_map, ax_col in zip(thresh_maps, axs.T[:num_conditions]):
        for (cell, [thresh, polars, azimuthals]), thresh_lim, ax in zip(thresh_map.items(), thresh_lims, ax_col):
            if consistent_scale:
                vmin, vmax = thresh_lim
            else:
                vmin, vmax = (None, None)
            if lim_max is not None:
                if max(np.abs(thresh)) > lim_max:
                    vmax = lim_max
            plot_projection(data=thresh, 
                            azimuthals=azimuthals, 
                            polars=polars, 
                            cmap='inferno_r',
                            cbar_label="TMS Threshold (V/m)",
                            grid=grid,
                            ax=ax,
                            vmin=vmin,
                            vmax=vmax,
                            fontsizes=fontsizes,
                            )
    
    for thresh_diff, ax_col in zip(thresh_diffs, axs.T[num_conditions:]):
        for (cell, [thresh_maps, polars, azimuthals]), ax in zip(thresh_diff.items(), ax_col):
            vmax = max(np.abs(thresh_maps))
            plot_projection(data=thresh_maps, 
                            azimuthals=azimuthals, 
                            polars=polars, 
                            cmap='bwr',
                            cbar_label="TMS Thres Diff (V/m)",
                            grid=grid,
                            ax=ax,
                            vmin=-vmax, 
                            vmax=vmax, 
                            fontsizes=fontsizes,
                            )


def plot_projection(data, azimuthals, polars, cmap, cbar_label, grid, ax, title=None, vmin=None, vmax=None, fontsizes=[60, 40, 40]):
    # Interpolate data onto a grid of lon-lat
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-90, 90, 181)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = np.array([np.array(azimuthals)-180, -np.array(polars)+90]).T
    interp_grid = griddata(points, data, (lon_grid, lat_grid), method='cubic')# method='nearest', 'linear', or 'cubic'

    # Plot figure
    m = Basemap(projection='moll', lon_0=0, ax=ax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # m.pcolormesh throws a warning about the inputs being non-monotonic
        heatmap = m.pcolormesh(lon_grid, lat_grid, interp_grid, cmap=cmap, shading='auto', latlon=True, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title, fontsize=fontsizes[0])
    cbar = m.colorbar(heatmap, pad=0.4)
    cbar.set_label(cbar_label, fontsize=fontsizes[1])
    cbar.ax.tick_params(labelsize=fontsizes[2])


    if grid:
        num_parallels = 12
        num_meridians = 24
        m.drawparallels(np.linspace(-90, 90, num_parallels+1))
        m.drawmeridians(np.linspace(-180, 180, num_meridians+1))

        # TODO: make gridlines appear similar to those Aberra's plots
        # merids = np.linspace(-180, 180, num_meridians+1)
        # front = [merid for merid in merids if merid>=-90 and merid<=90]
        # back = [merid for merid in merids if merid<-90 or merid>90]
        # m.drawmeridians(front, color=np.array([1, 1, 1])*.6)
        # m.drawmeridians(back, color=np.array([1, 1, 1])*0)
