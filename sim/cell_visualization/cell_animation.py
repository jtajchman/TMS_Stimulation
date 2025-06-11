import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath('')).parent)
sys.path.append(rootFolder)

from file_management import set_cwd, suppress_stdout

import numpy as np
from netpyne import sim, specs
import pyvista as pv
from pyvista.plotting.plotter import Plotter

from extracellular_stim_tools.netpyne_extracellular import flattenLL, calculate_segments_pts, get_section_list_NetPyNE
from extracellular_stim_tools.coord_rotations import get_rotation_from_axis_to_pos_z, rotate_coords
from cell_plotting import get_cell_points_by_type


def plot_cell_3D_w_voltages(cell_name_ID: str, title=None, plotter=None, notebook=True, show=True, export=False):
    """ Plot a 3D representation of a cell using PyVista.
    Parameters:
    - cell_name_ID: str, the name and ID of the cell.
    - plotter: pv.Plotter, optional, a PyVista plotter object to use for plotting.
    - notebook: bool, optional, whether to use the notebook backend for PyVista.
    - show: bool, optional, whether to show the plot immediately.
    - export: bool, optional, whether to export the plot as an HTML file.
    """

    if plotter is None:
        plotter = Plotter(notebook=notebook, title=title)
    cell_points_by_type, cell_diams_by_type = get_cell_points_by_type(cell_name_ID)
    sec_types = [  'Soma',      'Dend',       'Apic',      'Axon',      'Myelin',    'Unmyelin',  'Node']
    colors =    ["#FF00FF", '#00FF00', '#0000FF', '#FFFF00', '#FF0000', '#00FFFF', "#000000"]

    splines_by_type = [[pv.Spline(sec_points) for sec_points in type_points] for type_points in cell_points_by_type]
    for type_splines, type_diams, type_points in zip(splines_by_type, cell_diams_by_type, cell_points_by_type):
        for spline, diam, sec_points in zip(type_splines, type_diams, type_points):
            if len(diam) > len(spline.points):      # If diameters are longer than points, truncate
                diam = diam[:len(spline.points)]
            elif len(diam) < len(spline.points):    # If diameters are shorter than points, extend
                diam = np.r_[diam, [diam[-1] for _ in range(len(spline.points) - len(diam))]]
            diam = np.clip(diam, 0, 20)             # Clip diameters to a maximum of 20
            spline["radius"] = diam * 5/2             # Set radius for tube rendering and scale for plotting

    tubes_by_type = [[spline.tube(scalars="radius", absolute=True) for spline in type_splines] for type_splines in splines_by_type]
    type_tubes = [pv.MultiBlock(tube) for tube in tubes_by_type]

    for type_tube, color, sec_type in zip(type_tubes, colors, sec_types):
        plotter.add_mesh(type_tube, line_width=5, color=color, label=sec_type)
    plotter.enable_terrain_style()
    plotter.view_xz()
    plotter.add_legend()
    
    if export:
        plotter.export_html('sphere.html')
    if show:
        plotter.show()#jupyter_backend='client' # trame, client, server
    return plotter