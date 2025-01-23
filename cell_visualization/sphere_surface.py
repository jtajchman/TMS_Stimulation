import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
import matplotlib.cm as cm

from file_management import reach_out_for_func

from tms_probability.probability_analysis import get_thresholds
from tms_probability.plotting import get_threshold_map_by_cell, get_threshold_diff_by_cell
from extracellular_stim_tools.spherical_cartesian_conversion import norm_cartesian_to_spherical, norm_spherical_to_cartesian
from extracellular_stim_tools.coord_rotations import get_rotation_to_pos_z, rotate_coords


def angular_distance(polar1, azim1, polar2, azim2):
    """Calculates the angular distance between two points on a sphere.
    """

    polar1, azim1, polar2, azim2 = map(math.radians, [polar1, azim1, polar2, azim2])

    # Calculate the dot product of the two unit vectors
    dot_product = np.sin(polar1) * np.sin(polar2) * np.cos(azim1 - azim2) + np.cos(polar1) * np.cos(polar2)

    # Clamp the dot product to the range [-1, 1] to handle numerical precision issues
    dot_product = np.clip(dot_product, -1, 1)

    # Calculate the angular distance
    return math.degrees(np.arccos(dot_product))


def nearest_data(data, cell_name, r, x, y, z):
    [cell_data, cell_polars, cell_azimuthals] = data[cell_name]
    sphere_polar, sphere_azimuthal = norm_cartesian_to_spherical(x, y, z)
    distances = [angular_distance(cell_polar, cell_azimuthal, sphere_polar, sphere_azimuthal) 
                                  for cell_polar, cell_azimuthal in zip(cell_polars, cell_azimuthals)]
    nearest_idx = np.argmin(distances)
    return cell_data[nearest_idx]


def nearest_spherical_coords(data, cell_name, input_polar, input_azim):
    [cell_data, cell_polars, cell_azimuthals] = data[cell_name]
    distances = [angular_distance(cell_polar, cell_azimuthal, input_polar, input_azim) 
                                  for cell_polar, cell_azimuthal in zip(cell_polars, cell_azimuthals)]
    nearest_idx = np.argmin(distances)
    return cell_polars[nearest_idx], cell_azimuthals(nearest_idx)


def sphere_vertices(radius, angular_resolution):
    # Generate the sphere mesh
    points_per_pi_radians = int(np.ceil(180/angular_resolution))
    azim = np.linspace(0, 360, 2*points_per_pi_radians+1)
    polar = np.linspace(0, 180, points_per_pi_radians+1)

    x = np.outer(np.cos(np.radians(azim)), np.sin(np.radians(polar)))*radius
    y = np.outer(np.sin(np.radians(azim)), np.sin(np.radians(polar)))*radius
    z = np.outer(np.ones(np.size(azim)), np.cos(np.radians(polar)))*radius
    return [x, y, z]


def plot_data(data_map_by_cell, cell_name, cmap_name, cmap_min, cmap_max, ax: Axes3D, radius, angular_resolution, title, cbar_title):
    coords = sphere_vertices(radius, angular_resolution)
    r = get_rotation_to_pos_z([0, 1, 0])
    
    cmap = colormaps[cmap_name]
    color = lambda t: cmap((t-cmap_min)/(cmap_max-cmap_min))

    data = []
    colors = []
    [x, y, z] = coords
    for x1, y1, z1 in zip(x, y, z):
        data1 = []
        color1 = []
        for x2, y2, z2 in zip(x1, y1, z1):
            data2 = nearest_data(data_map_by_cell, cell_name, r, x2, y2, z2)
            color2 = color(data2)
            data1.append(data2)
            color1.append(color2)
        data.append(data1)
        colors.append(color1)

    # min_row_idx, min_col_idx = np.unravel_index(np.argmin(data), np.array(data).shape)
    # colors[min_row_idx][min_col_idx] = colormaps['Greens'](1.)

    # Plot the sphere
    cb = cm.ScalarMappable(cmap=cmap)
    cb.set_clim(cmap_min, cmap_max)
    ax.plot_surface(x, y, z, alpha=0.6, facecolors=colors, linewidth=0)

    cbar = plt.colorbar(cb, ax=ax)
    cbar.ax.set_ylabel(cbar_title)

    # Set aspect
    ax.set_aspect('equal')

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_threshold_sphere_map(thresh_map_by_cell, cell_name, ax, radius, angular_resolution=10, condition=None):
    data = thresh_map_by_cell[cell_name][0]

    cmap_name = 'inferno_r'
    cmap_min = min(data)
    cmap_max = max(data)
    title = f'Threshold Map for {cell_name}'
    if condition != None:
        title += f' for {condition} Pulse'
    cbar_title = 'TMS E-field Threshold (V/m)'

    plot_data(thresh_map_by_cell, cell_name, cmap_name, cmap_min, cmap_max, ax, radius, angular_resolution, title, cbar_title)
    
def plot_threshold_diff_sphere_map(thresh_diff_map_by_cell, cell_name, ax, radius, angular_resolution=10, conditions=None):
    data = thresh_diff_map_by_cell[cell_name][0]

    cmap_name = 'bwr'
    cmap_lim = max([abs(min(data)), abs(max(data))])
    cmap_min = -cmap_lim
    cmap_max = cmap_lim
    title = f'Threshold Difference Map for {cell_name}'
    cbar_title = 'TMS E-field Threshold Difference (V/m)'
    if conditions != None:
        title += f'\nShowing {conditions[0]} - {conditions[1]}'
        cbar_title += f'\nBlue = Lower Thresholds for {conditions[0]}' + \
                      f'\nRed = Lower Thresholds for {conditions[1]}'

    plot_data(thresh_diff_map_by_cell, cell_name, cmap_name, cmap_min, cmap_max, ax, radius, angular_resolution, title, cbar_title)

def get_thresh_map_from_fname(threshold_fname):
    cell_data = reach_out_for_func(get_thresholds, threshold_fname)
    thresh_map_by_cell = get_threshold_map_by_cell(cell_data)
    return thresh_map_by_cell

def get_thresh_diff_map_from_fnames(threshold_fnames):
    cell_data_0 = reach_out_for_func(get_thresholds, threshold_fnames[0])
    cell_data_1 = reach_out_for_func(get_thresholds, threshold_fnames[1])
    data_by_cell_0 = get_threshold_map_by_cell(cell_data_0)
    data_by_cell_1 = get_threshold_map_by_cell(cell_data_1)
    thresh_diff_map_by_cell = get_threshold_diff_by_cell(data_by_cell_0, data_by_cell_1)
    return thresh_diff_map_by_cell

def get_thresh_diff_map(cell_data_0, cell_data_1):
    data_by_cell_0 = get_threshold_map_by_cell(cell_data_0)
    data_by_cell_1 = get_threshold_map_by_cell(cell_data_1)
    thresh_diff_map_by_cell = get_threshold_diff_by_cell(data_by_cell_0, data_by_cell_1)
    return thresh_diff_map_by_cell