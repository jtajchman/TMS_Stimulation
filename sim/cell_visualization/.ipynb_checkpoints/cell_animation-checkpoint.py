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
import pickle
import copy

from extracellular_stim_tools.netpyne_extracellular import flattenLL, calculate_segments_pts, get_section_list_NetPyNE, SingleExtracellular
from extracellular_stim_tools.coord_rotations import get_rotation_from_axis_to_pos_z, rotate_coords
from cell_plotting import get_cell_points, load_cell_netpyne
from cell_plotting import plot_cell_3D, plot_cell_3D_w_init_site


def plot_cell_3D_w_voltages(cell_name_ID: str, tms_params: dict = None, sim_results: str = None, title=None, plotter=None, notebook=True, show=True, export=False):
    """ Plot a 3D representation of a cell using PyVista.
    Parameters:
    - cell_name_ID: str, the name and ID of the cell.
    - plotter: pv.Plotter, optional, a PyVista plotter object to use for plotting.
    - notebook: bool, optional, whether to use the notebook backend for PyVista.
    - show: bool, optional, whether to show the plot immediately.
    - export: bool, optional, whether to export the plot as an HTML file.
    """

    if tms_params is not None and sim_results is None:
        cell = load_cell_netpyne(cell_name_ID)
        ecs = SingleExtracellular(cell=cell, v_recording=True, **tms_params)
        ecs.run_simulation()
        ecs.save_v_and_spikes()
        sim_results = f"{ecs.stim_cell.cell.__repr__()}_sim_results.pkl"
    if sim_results is not None:
        with open(sim_results, 'rb') as f:
            [t, voltages, action_potentials, action_potentials_recording_ids] = pickle.load(f)
            t = np.round(t, 3)
            mean_sec_voltages = np.array([np.mean(sec_voltages, axis=0) for sec_voltages in voltages])
            voltages = np.array(flattenLL(voltages))
    else:
        raise ValueError("tms_params or sim_results must be defined")

    if plotter is None:
        plotter = Plotter(notebook=notebook, title=title)
    cell_points, cell_diams = get_cell_points(cell_name_ID)
    seg_points, seg_diams = get_seg_pts(cell_points, cell_diams)
    splines = [pv.Spline(sec_points) for sec_points in seg_points]

    for spline, v_time_course in zip(splines, voltages):
        spline["voltage"] = v_time_course[0]
    multisplines = pv.MultiBlock(splines)

    actor = plotter.add_mesh(multisplines, render_lines_as_tubes=False, line_width=2, scalars="voltage", clim=[-80, 40], scalar_bar_args={"title": "Membrane Potential (mV)"})
    plotter.enable_terrain_style()
    plotter.view_xz()
    # plotter.add_legend()

    
    def callback(time):
        t_ind = list(t).index(float(time))
        for spline, v_time_course in zip(splines, mean_sec_voltages):
            spline["voltage"] = v_time_course[t_ind]
    # plotter.add_timer_event(max_steps=200, duration=500, callback=callback)
    # plotter.add_slider_widget(callback=callback, rng=(0, 1), value=0, interaction_event="always", style="modern")
    tmax = 5
    slider_dt = 0.005
    t_str = [str(t_elem) for t_elem in t if t_elem <= tmax and (round(float(t_elem)%slider_dt, 9)==0 or round(float(t_elem)%slider_dt, 9)==slider_dt)]
    plotter.add_text_slider_widget(callback=callback, data=t_str, value=0, interaction_event="always", style="modern")
    
    if export:
        plotter.export_html('sphere.html')
    if show:
        plotter.show()#jupyter_backend='client' # trame, client, server
    return plotter, sim_results


def get_seg_pts(cell_points, cell_diams):
    # Get flattened lists of the points and diameters of every segment
    seg_points = np.array([[sec[i-1], seg] for sec in cell_points for i, seg in enumerate(sec) if i != 0])
    seg_diams = np.array([[sec[i-1], seg] for sec in cell_diams for i, seg in enumerate(sec) if i != 0])
    return seg_points, seg_diams


def plot_cell_all(cell_name_ID: str, tms_params: dict = None, sim_results: str = None, title=None, notebook=True, show=True):
        plotter = Plotter(notebook=notebook, shape="1|2")
        plotter.subplot(0)
        _, sim_results = plot_cell_3D_w_voltages(cell_name_ID, tms_params=tms_params, sim_results=sim_results, title=f"{title} Voltage Trace", plotter=plotter, show=False)
        plotter.subplot(1)
        plot_cell_3D(cell_name_ID, title="Section Types", plotter=plotter, show=False)
        plotter.subplot(2)
        plot_cell_3D_w_init_site(cell_name_ID, tms_params=tms_params, sim_results=sim_results, title="AP Init Site", plotter=plotter, show=False)
        plotter.link_views()

        if show:
            plotter.show(jupyter_backend='trame')#jupyter_backend='client' # trame, client, server