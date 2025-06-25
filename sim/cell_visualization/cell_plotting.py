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

from extracellular_stim_tools.netpyne_extracellular import flattenLL, calculate_segments_pts, get_section_list_NetPyNE, SingleExtracellular
from extracellular_stim_tools.coord_rotations import get_rotation_from_axis_to_pos_z, rotate_coords


def load_cell_netpyne(cellName):
    cfg = specs.SimConfig()
    netParams = specs.NetParams()

    netParams.popParams[cellName] = {'cellType': cellName, 'cellModel': 'HH_full', 'numCells': 1}
    with set_cwd("sim"):
        netParams.loadCellParamsRule(label = cellName, fileName = f'cells/{cellName}_cellParams.json') 
    netParams.cellParams[cellName]['conds']['cellType'] = cellName

    with suppress_stdout():
        sim.initialize(
            simConfig = cfg, 	
            netParams = netParams)  				# create network object and set cfg and net params

        sim.net.createPops()               			# instantiate network populations
        sim.net.createCells();              		# instantiate network cells based on defined populations
        sim.net.connectCells()            			# create connections between cells based on params
        sim.net.addStims() 							# add network stimulation
        sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
        sim.net.defineCellShapes()
    return sim.net.cells[0]


# def plot_potentials(
#         cellName: str, 
#         decay_rate_percent_per_mm: float,
#         E_field_dir: dict,
#         decay_dir: dict,
#         ref_point_um: list[float],
#         somatodendritic_axis: list[float],
#     ):

#     cell = load_cell_netpyne(cellName)

#     # Calculations of cell geometry and E-field coupling
#     section_list = get_section_list_NetPyNE(cell)
#     centers = calculate_segments_centers(section_list)
#     quasipotentials = set_E_field(
#         section_list=section_list,
#         decay_rate_percent_per_mm=decay_rate_percent_per_mm,
#         E_field_dir=E_field_dir,
#         decay_dir=decay_dir,
#         ref_point_um=ref_point_um,
#         somatodendritic_axis=somatodendritic_axis,
#     )
    
#     # Ensure flattened lists
#     centers = flattenLL(centers)

#     # Plotting
#     [xs, ys, zs] = np.array(centers).T

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     scatter = ax.scatter(xs, zs, ys, c=quasipotentials, cmap='bwr') 
#     ax.set_aspect('equal')
#     ax.set_title("Extracellular Quasipotentials\n(Positive field points from positive to negative potential)") 
#     ax.set_xlabel('x-axis') 
#     ax.set_ylabel('z-axis') 
#     ax.set_zlabel('y-axis') 
#     ax.invert_yaxis()
#     cbar = plt.colorbar(scatter)
#     cbar.ax.set_ylabel('Quasipotentials (mV)')

#     # if saveplot:
#     #     import mpld3
#     #     mpld3.save_html(fig, saveplot)

#     #     # import plotly.io as pio
#     #     # pio.write_html(fig, file=saveplot)


def plot_cell_3D(cell_name_ID: str, title=None, plotter=None, notebook=True, show=True):
    """ Plot a 3D representation of a cell using PyVista.
    Parameters:
    - cell_name_ID: str, the name and ID of the cell.
    - plotter: pv.Plotter, optional, a PyVista plotter object to use for plotting.
    - notebook: bool, optional, whether to use the notebook backend for PyVista.
    - show: bool, optional, whether to show the plot immediately.
    """

    if plotter is None:
        plotter = Plotter(notebook=notebook, title=title)
    cell_points_by_type, cell_diams_by_type = get_cell_points_by_type(cell_name_ID)
    sec_types = [  'Soma',      'Dend',       'Apic',      'Axon',      'Myelin',    'Unmyelin',  'Node']
    colors =    ["#FF00FF", '#00FF00', '#0000FF', '#FFFF00', '#FF0000', '#00FFFF', "#000000"]

    splines_by_type = [[pv.Spline(sec_points) for sec_points in type_points] for type_points in cell_points_by_type]
    for type_splines, type_diams in zip(splines_by_type, cell_diams_by_type):
        for spline, diam in zip(type_splines, type_diams):
            if len(spline.points) == 3 and len(diam) == 2:  # Happens for single-segment sections; splines have minimum 3 points
                diam = np.insert(diam, 1, np.mean(diam))    # Insert mean diameter between defined points
            rad = np.clip(diam, 0, 20)/2            # Clip radii to a maximum of 10
            spline["radius"] = rad * 5              # Set radius for tube rendering and scale for plotting

    tubes_by_type = [[spline.tube(scalars="radius", absolute=True) for spline in type_splines] for type_splines in splines_by_type]
    type_tubes = [pv.MultiBlock(tube) for tube in tubes_by_type]

    for type_tube, color, sec_type in zip(type_tubes, colors, sec_types):
        plotter.add_mesh(type_tube, color=color, label=sec_type)
    plotter.enable_terrain_style()
    plotter.view_xz()
    plotter.add_legend()
    
    if show:
        plotter.show()#jupyter_backend='client' # trame, client, server
    return plotter


def plot_cell_3D_w_init_site(cell_name_ID: str, tms_params: dict = None, sim_results: str = None, title=None, plotter=None, notebook=True, show=True):
    if tms_params is not None and sim_results is None:
        cell = load_cell_netpyne(cell_name_ID)
        ecs = SingleExtracellular(cell=cell, v_recording=True, **tms_params)
        ecs.run_simulation()
        ecs.save_v_and_spikes()
        sim_results = f"{ecs.stim_cell.cell.__repr__()}_sim_results.pkl"
    if sim_results is not None:
        with open(sim_results, 'rb') as f:
            [t, voltages, action_potentials, action_potentials_recording_ids] = pickle.load(f)
    else:
        raise ValueError("tms_params or sim_results must be defined")

    if plotter is None:
        plotter = Plotter(notebook=notebook, title=title)
    seg_pts, seg_diams = get_cell_points(cell_name_ID)
    init_seg_pts, init_seg_diams, ninit_seg_pts, ninit_seg_diams = get_init_points(seg_pts, seg_diams, action_potentials_recording_ids)

    init_splines = [pv.Spline(sec_points) for sec_points in init_seg_pts]
    ninit_splines = [pv.Spline(sec_points) for sec_points in ninit_seg_pts]
    for type_splines, type_diams in zip([init_splines, ninit_splines], [init_seg_diams, ninit_seg_diams]):
        for spline, diam in zip(type_splines, type_diams):
            if len(spline.points) == 3 and len(diam) == 2:  # Happens for single-segment sections; splines have minimum 3 points
                diam = np.insert(diam, 1, np.mean(diam))    # Insert mean diameter between defined points
            rad = np.clip(diam, 0, 20)/2            # Clip radii to a maximum of 10
            spline["radius"] = rad * 5              # Set radius for tube rendering and scale for plotting

    init_tubes = [spline.tube(scalars="radius", absolute=True) for spline in init_splines]
    ninit_tubes = [spline.tube(scalars="radius", absolute=True) for spline in ninit_splines]
    init_tubes_mb = pv.MultiBlock(init_tubes)
    ninit_tubes_mb = pv.MultiBlock(ninit_tubes)

    plotter.add_mesh(init_tubes_mb, color="#FF0000", label='AP Init Site')
    ninit_actor = plotter.add_mesh(ninit_tubes_mb, color='#000000', label='Neuron')
    plotter.enable_terrain_style()
    plotter.view_xz()
    plotter.add_legend()

    def toggle_sphere(flag):
        ninit_actor.SetVisibility(flag)
    plotter.add_checkbox_button_widget(toggle_sphere, value=True,)
    
    if show:
        plotter.show()#jupyter_backend='client' # trame, client, server
    return plotter


def get_cell_points_by_type(cell_name_ID):
    """
    Get the cell points for a given cell name and ID, sorted by section type.
    
    Parameters:
    - cell_name_ID: str, the name and ID of the cell.
    
    Returns:
    - cell_points: np.ndarray, the rotated coordinates of the cell segments.
    """
    # Load the cell using NetPyNE
    cell = load_cell_netpyne(cell_name_ID)
    sec_list = get_section_list_NetPyNE(cell)
    R = get_rotation_from_axis_to_pos_z([0, 1, 0])
    seg_centers, seg_diams = calculate_segments_pts(sec_list, centers=False, diams=True)

    sec_types = ['soma', 'dend', 'apic', 'axon', 'Myelin', 'Unmyelin', 'Node']
    cell_points_by_type = []
    cell_diams_by_type = []
    for type in sec_types:
        seg_centers_type = [seg_centers[i] for i, sec in enumerate(sec_list) if type in sec.name().split('.')[1]]
        seg_diams_type = [seg_diams[i] for i, sec in enumerate(sec_list) if type in sec.name().split('.')[1]]
        # Rotate the coordinates from the somatodendritic axis to the positive z axis
        cell_points_by_type.append(rotate_coords(R, seg_centers_type))
        cell_diams_by_type.append(seg_diams_type)
    return cell_points_by_type, cell_diams_by_type


def get_cell_points(cell_name_ID):
    """
    Get the cell points for a given cell name and ID.
    
    Parameters:
    - cell_name_ID: str, the name and ID of the cell.
    
    Returns:
    - cell_points: np.ndarray, the rotated coordinates of the cell segments.
    """
    # Load the cell using NetPyNE
    cell = load_cell_netpyne(cell_name_ID)
    sec_list = get_section_list_NetPyNE(cell)
    R = get_rotation_from_axis_to_pos_z([0, 1, 0])
    seg_pts, seg_diams = calculate_segments_pts(sec_list, centers=False, diams=True)
    return rotate_coords(R, seg_pts), seg_diams


def get_init_points(seg_pts, seg_diams, action_potentials_recording_ids):
    init_ids = list(action_potentials_recording_ids)[:3]
    init_seg_pts = []
    init_seg_diams = []
    ninit_seg_pts = []
    ninit_seg_diams = []
    
    i = 0
    # print(init_ids)
    for sec_pts, sec_diams in zip(seg_pts, seg_diams):
        j = np.array(range(len(sec_pts)-1))
        init_j = sorted([l for l in j if l+i in init_ids])

        if init_j == []: # No segments in this section initialized spiking
            ninit_seg_pts.append(sec_pts)
            ninit_seg_diams.append(sec_diams)
        else:
            # print(init_j, np.array(init_j)+i)
            # print(sec_idxs)
            init_bounds, ninit_bounds = get_sec_bounds_idx(init_j, len(sec_pts)-1)
            for bound in init_bounds:
                init_seg_pts.append(sec_pts[bound[0]:bound[1]])
                init_seg_diams.append(sec_diams[bound[0]:bound[1]])
            for bound in ninit_bounds:
                ninit_seg_pts.append(sec_pts[bound[0]:bound[1]])
                ninit_seg_diams.append(sec_diams[bound[0]:bound[1]])
        i += len(sec_pts)-1

    return init_seg_pts, init_seg_diams, ninit_seg_pts, ninit_seg_diams


def get_sec_bounds_idx(k, max_idx):
    """
    Get the bounds of indexes in the segment-in-section list to include in the init vs. exclude in the ninit lists
    """
    incl_bounds = []
    excl_bounds = []
    for i, j in enumerate(k):
        if i > 0 and j == k[i-1]+1: # Current bound extends the previous set
            incl_bounds[-1][1] = j+2
        else: # General case: add new set of bounds
            bound = [j, j+2]
            incl_bounds.append(bound)
    for i, incl_bound in enumerate(incl_bounds):
        if i == 0 and incl_bound[0] > 0: # If there are idxs before first included lower bound
            excl_bounds.append([0, incl_bound[0]+1]) # Exclude them
        if i < len(incl_bounds)-1: # If there are more components to the included bounds
            excl_bounds.append([incl_bound[1]-1, incl_bounds[i+1][0]+1]) # Exclude the space between bounds
        elif incl_bound[1] != max_idx+1: # If there are idxs after last included upper bound
            excl_bounds.append([incl_bound[1]-1, max_idx+1]) # Exclude them
    return incl_bounds, excl_bounds