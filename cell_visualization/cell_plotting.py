from extracellular_stim_tools.netpyne_extracellular import flattenLL, calculate_segments_centers, get_section_list_NetPyNE# set_E_field, get_section_list_NetPyNE, calculate_segments_centers, 
from extracellular_stim_tools.coord_rotations import get_rotation_to_pos_z, rotate_coords
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from netpyne import sim, specs
import os

def load_cell(cellName):
    cfg = specs.SimConfig()
    netParams = specs.NetParams()

    curr_dir = os.getcwd()
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(dir_path)
    netParams.popParams[cellName] = {'cellType': cellName, 'cellModel': 'HH_full', 'numCells': 1}
    netParams.loadCellParamsRule(label = cellName, fileName = f'cells/{cellName}_cellParams.json') 
    netParams.cellParams[cellName]['conds']['cellType'] = cellName
    os.chdir(curr_dir)

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

#     cell = load_cell(cellName)

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

def plot_cell(cellName: str, ax: Axes3D):
    cell = load_cell(cellName)
    section_list = get_section_list_NetPyNE(cell)
    centers = flattenLL(calculate_segments_centers(section_list))

    # Plotting
    r = get_rotation_to_pos_z([0, 1, 0])
    [xs, ys, zs] = np.array([rotate_coords(r, [x, y, z]) for [x, y, z] in centers]).T

    ax.scatter(xs, ys, zs) 
    ax.set_aspect('equal')
    if ax.get_title() == '':
        ax.set_title(cellName) 
    ax.set_xlabel('x-axis') 
    ax.set_ylabel('y-axis') 
    ax.set_zlabel('z-axis')

def plot_cell_with_init_site(non_init_sec_pts: list[float], init_sec_pts: list[list[float]], ax: Axes3D):
    # Plotting
    r = get_rotation_to_pos_z([0, 1, 0])
    non_init_sec_pts = flattenLL(non_init_sec_pts)
    [xns, yns, zns] = np.array([rotate_coords(r, [x, y, z]) for [x, y, z] in non_init_sec_pts]).T
    ax.scatter(xns, yns, zns, c='b') 

    for pts in init_sec_pts:
        print(pts)
        [xs, ys, zs] = np.array([rotate_coords(r, [x, y, z]) for [x, y, z] in pts]).T
        print([xs, ys, zs])
        ax.scatter(xs, ys, zs, c='r') 

    ax.set_aspect('equal')
    ax.set_xlabel('x-axis') 
    ax.set_ylabel('y-axis') 
    ax.set_zlabel('z-axis')