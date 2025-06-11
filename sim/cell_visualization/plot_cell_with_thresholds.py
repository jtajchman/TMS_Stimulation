import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)


from tms_thresholds.recruitment_analysis import get_thresholds_dict, get_results
from cell_visualization.threshold_plotting import get_threshold_map_by_cell, get_threshold_diff_map_by_cell
from tms_thresholds.sim_control import TMS_sim
from tms_thresholds.threshold_sim import detect_spike
from extracellular_stim_tools.netpyne_extracellular import SingleExtracellular, calculate_segments_pts
from sphere_surface import plot_threshold_sphere_map, plot_threshold_diff_sphere_map, get_thresh_diff_map, nearest_spherical_coords
from cell_plotting import plot_cell, plot_cell_with_init_site# plot_potentials, 
import matplotlib.pyplot as plt
import numpy as np


def get_threshold(cell_data, cell_name, polar, azimuthal):
    return cell_data[cell_name]['Polar'][str(polar)]['Azimuthal'][str(azimuthal)]['threshold']

def overlay_cell_and_thresholds(threshold_fname, cell_name, radius, condition=None, init_site_direction=None, thresh_buffer=0):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if type(threshold_fname) == str:
        results = get_results(threshold_fname)
        cell_data = results['threshold_map']
        data_map_by_cell = get_threshold_map_by_cell(cell_data)
        plot_threshold_sphere_map(data_map_by_cell[cell_name], cell_name, ax, radius=radius, condition=condition)
        results_list = [results]
    elif type(threshold_fname) == list and len(threshold_fname) == 2:
        if condition != None: assert type(condition) == list and len(condition) == 2
        results_0 = get_results(threshold_fname[0])
        results_1 = get_results(threshold_fname[1])
        cell_data_0 = results_0['threshold_map']
        cell_data_1 = results_1['threshold_map']
        data_map_by_cell = get_thresh_diff_map(cell_data_0, cell_data_1)
        plot_threshold_diff_sphere_map(data_map_by_cell[cell_name], cell_name, ax, radius=radius, conditions=condition)
        results_list = [results_0, results_1]
    else: raise ValueError('threshold_fname must be a str (for threshold map) or a list of 2 str (for thresh diff map) \n condition must match this or be None')

    if init_site_direction == None: plot_cell(cell_name, ax)
    else:
        sim_direction = None
        if type(init_site_direction) == tuple and len(init_site_direction) == 2:
            sim_direction = nearest_spherical_coords(data_map_by_cell, cell_name, init_site_direction[0], init_site_direction[1])
        else:
            [data, polars, azimuthals] = data_map_by_cell[cell_name]
            direction_idx = None
            if init_site_direction == 'lowest_threshold': 
                direction_idx = np.argmin(data)
            elif init_site_direction == 'highest_threshold': 
                direction_idx = np.argmax(data)
            if direction_idx != None:
                sim_direction = (polars[direction_idx], azimuthals[direction_idx])
            
        if sim_direction != None:
            for res in results_list:
                cdata = res['threshold_map']
                tms_params = res['tms_params']
                tms_params['E_field_dir'] = {'Coord_type': 'Spherical', 'Polar': sim_direction[0], 'Azimuthal': sim_direction[1]}
                # print(get_threshold(cdata, cell_name, sim_direction[0], sim_direction[1]))
                tms_params['efield_amplitude_V_per_m'] = get_threshold(cdata, cell_name, sim_direction[0], sim_direction[1]) + thresh_buffer
                ecs = TMS_sim(cell_name, tms_params, clear_ecs_data=False)
                if detect_spike(ecs):
                    init_sec_id = int(ecs.action_potentials_recording_ids[0])
                    print(ecs.stim_cell.section_list[init_sec_id].name())
                    non_init_sec_pts = list(calculate_segments_pts(ecs.stim_cell.section_list))
                    init_sec_pts = non_init_sec_pts.pop(init_sec_id)
                else: raise RuntimeError('Threshold simulation did not elicit a spike')

            plot_cell_with_init_site(non_init_sec_pts, [init_sec_pts], ax)
            return sim_direction

        
    # Select field direction
    # If single threshold map:
        # Get tms parameters from threshold_fname
        # Run simulation at threshold
        # Find site of AP initiation
        # Plot cell with site highlighted ##
    # If threshold diff map:
        # Get tms parameters from both threshold_fnames
        # Run simulations at threshold
        # Find sites of AP initiation
        # Plot cell with sites highlighted (one color for each condition if sites are different (blue, red), another if they are the same (purple))
