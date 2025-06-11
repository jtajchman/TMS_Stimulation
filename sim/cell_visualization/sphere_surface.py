import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)

import numpy as np
from numpy import pi
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
import matplotlib.cm as cm

import pyvista as pv
from pyvista.plotting.plotter import Plotter

from tms_thresholds.recruitment_analysis import get_thresholds_dict
from cell_visualization.threshold_plotting import get_threshold_map_by_cell, get_threshold_diff_map_by_cell
from extracellular_stim_tools.spherical_cartesian_conversion import norm_cartesian_to_spherical, norm_spherical_to_cartesian
from extracellular_stim_tools.coord_rotations import get_rotation_from_axis_to_pos_z, get_rotation_from_pos_z_to_axis, get_inverted_rotation, rotate_coords


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


def nearest_data(data, x, y, z, max_a=360):
    '''
    Finds the closest datapoint to the provided cartesian coordinates
    '''
    [cell_data, cell_polars, cell_azimuthals] = data
    s_pol, s_azm = norm_cartesian_to_spherical(x, y, z)
    
    max_a_pol = max_a
    max_a_azm = np.clip(max_a / np.sin(np.radians(s_pol)), 0, 180)
    
    distances = []
    for c_pol, c_azm in zip(cell_polars, cell_azimuthals):
        if c_pol>=s_pol-max_a_pol and c_pol<=s_pol+max_a_pol and \
           c_azm>=s_azm-max_a_azm and c_azm<=s_azm+max_a_azm:       # Roughly narrows the search area for the nearest point based on how restricting max_a is
            distances.append(angular_distance(c_pol, c_azm, s_pol, s_azm))
        else: distances.append(360)
    nearest_idx = np.argmin(distances)
    return cell_data[nearest_idx]


def nearest_spherical_coords(data, input_polar, input_azim):
    '''
    Finds the closest sample point coordinates in the data to the provided spherical coordinates
    '''
    [cell_data, cell_polars, cell_azimuthals] = data
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


def sphere_vertices_pl(radius, angular_resolution):
    # Generate the sphere mesh for plotly
    points_per_pi_radians = int(np.ceil(180/angular_resolution))
    azim = np.linspace(0, 2*pi, 2*points_per_pi_radians+1)
    polar = np.linspace(-pi/2, pi/2, points_per_pi_radians+1)
    azim, polar = np.meshgrid(azim, polar)
    x = radius * np.cos(azim) * np.cos(polar)
    y = radius * np.cos(azim) * np.sin(polar)
    z = radius * np.sin(azim)
    return [x, y, z]


def plot_sphere_pv(thresh_map, cmap_name, radius, angular_resolution=10, plotter=None, notebook=True, show=True, export=False, opacity=1, toggle=True):
    coords = sphere_vertices(radius, angular_resolution/2)
    sphere = pv.StructuredGrid(*coords)
    
    centers = sphere.cell_centers()
    thresholds = [nearest_data(thresh_map, *center, max_a=10) for center in centers.points]
    sphere.cell_data['thresholds'] = thresholds
    cmap = colormaps[cmap_name]

    if cmap_name == 'bwr':
        data = thresh_map[0]
        cmap_lim = max([abs(min(data)), abs(max(data))])
        cmap_min = -cmap_lim
        cmap_max = cmap_lim
        clim = [cmap_min, cmap_max]
    else: clim = None

    if plotter is None:
        plotter = Plotter(notebook=notebook)
    sphere_actor = plotter.add_mesh(sphere, scalars='thresholds', opacity=opacity, cmap=cmap, clim=clim)

    def toggle_sphere(flag):
        sphere_actor.SetVisibility(flag)
    if toggle:
        plotter.add_checkbox_button_widget(toggle_sphere, value=True,)

    plotter.enable_terrain_style()
    if export:
        plotter.export_html('sphere.html')
    if show:
        plotter.show()#jupyter_backend='client' # trame, client, server
    return plotter


def anim_sphere_test(thresh_map, cmap_name, radius, angular_resolution=10, plotter=None, notebook=True, show=True, export=False, opacity=1, toggle=True):
    coords = sphere_vertices(radius, angular_resolution/2)
    sphere = pv.StructuredGrid(*coords)
    
    centers = sphere.cell_centers()
    thresholds = [nearest_data(thresh_map, *center, max_a=10) for center in centers.points]
    sphere.cell_data['thresholds'] = thresholds
    cmap = colormaps[cmap_name]

    if cmap_name == 'bwr':
        data = thresh_map[0]
        cmap_lim = max([abs(min(data)), abs(max(data))])
        cmap_min = -cmap_lim
        cmap_max = cmap_lim
        clim = [cmap_min, cmap_max]
    else: clim = None

    if plotter is None:
        plotter = Plotter(notebook=notebook)
    sphere_actor = plotter.add_mesh(sphere, scalars='thresholds', opacity=opacity, cmap=cmap, clim=clim)

    def toggle_sphere(flag):
        sphere_actor.SetVisibility(flag)
    if toggle:
        plotter.add_checkbox_button_widget(toggle_sphere, value=True,)

    plotter.enable_terrain_style()

    # plotter.open_gif("sphere_test.gif")
    nframe = 20
    pts = sphere.points.copy()
    # plotter.write_frame()
    # for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    def callback(step=0):
        plotter.update_coordinates(pts * np.cos(step), render=True)
        # plotter.write_frame()
    # plotter.close()
    # plotter.add_timer_event(max_steps=200, duration=500, callback=callback)
    plotter.add_slider_widget(callback=callback, rng=(0, np.pi/2), value=0, interaction_event="always", style="modern")


    if export:
        plotter.export_html('sphere.html')
    if show:
        plotter.show()#jupyter_backend='client' # trame, client, server
    return plotter


def get_thresh_map_from_fname(threshold_fname):
    cell_data = get_thresholds_dict(threshold_fname)
    thresh_map_by_cell = get_threshold_map_by_cell(cell_data)
    return thresh_map_by_cell


def get_thresh_diff_map_from_fnames(threshold_fnames):
    cell_data_0 = get_thresholds_dict(threshold_fnames[0])
    cell_data_1 = get_thresholds_dict(threshold_fnames[1])
    data_by_cell_0 = get_threshold_map_by_cell(cell_data_0)
    data_by_cell_1 = get_threshold_map_by_cell(cell_data_1)
    thresh_diff_map_by_cell = get_threshold_diff_map_by_cell(data_by_cell_0, data_by_cell_1)
    return thresh_diff_map_by_cell


def get_thresh_diff_map(cell_data_0, cell_data_1):
    data_by_cell_0 = get_threshold_map_by_cell(cell_data_0)
    data_by_cell_1 = get_threshold_map_by_cell(cell_data_1)
    thresh_diff_map_by_cell = get_threshold_diff_map_by_cell(data_by_cell_0, data_by_cell_1)
    return thresh_diff_map_by_cell