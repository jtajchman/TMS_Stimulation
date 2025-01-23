from extracellular_stim_tools import get_angles, SingleExtracellular
from .sim_control import TMS_sim, save_results
from multiprocessing import Pool
from copy import deepcopy
import os

def estimate_cell_threshold(cell_name_ID, starting_E, search_factor, search_precision, tms_params, syn_params):
    # Find the TMS electric field amplitude threshold of a cell with a given set of TMS parameters
    # Currently only works with 0 background activity

    # print(tms_params)

    EF_amp = starting_E
    EF_upper = None
    EF_lower = None

    threshold_found = False

    bounding = 1
    num_sim_bounding = 0
    num_sim_refining = 0

    while not threshold_found:
        num_sim_bounding += bounding
        num_sim_refining += 1-bounding

        tms_params['efield_amplitude_V_per_m'] = EF_amp

        ecs = TMS_sim(cell_name_ID, tms_params, syn_params)

        if detect_spike(ecs):
            EF_upper = EF_amp
            print(f'spike detected, EF_upper now {EF_upper}')
        else:
            EF_lower = EF_amp
            print(f'spike not detected, EF_lower now {EF_lower}')

        if EF_upper == None:
            EF_amp = EF_lower*search_factor
        elif EF_lower == None:
            EF_amp = EF_upper/search_factor
        else:
            bounding = 0
            EF_amp = (EF_upper+EF_lower)/2
            threshold_found = EF_upper-EF_lower <= search_precision
    
    threshold = EF_upper    # Return the smallest amplitude that causes a spike instead of the midpoint of the bounds (EF_amp)
                            # Ensures that a simulation run at the returned threshold will cause a spike
    return threshold, num_sim_bounding, num_sim_refining


def detect_spike(ecs: SingleExtracellular):
    return len(ecs.action_potentials) >= 3


def cell_type_threshold_map(cell_name, morphIDs, starting_E, search_factor, search_precision, angular_resolution, num_cores, tms_params, syn_params):
    # Iterates over E-field angles and amplitudes to produce a map of E-field amplidude thresholds at each angle
    polar_resolution = angular_resolution
    azimuthal_resolution = angular_resolution
    binned_angles, angles = get_angles(polar_resolution, azimuthal_resolution)

    cell_names_ID = [f'{cell_name}_{morphID}' for morphID in morphIDs]

    threshold_map = {}

    params_list = []
    cell_angles_list = []

    for cell_name_ID in cell_names_ID:
        threshold_map[cell_name_ID] = {'Polar': {}}
        for angle in binned_angles:
            polar = angle[0]
            threshold_map[cell_name_ID]['Polar'][polar] = {'Azimuthal': {}}
            for azimuthal in angle[1]:
                copy_tms_params = deepcopy(tms_params)
                copy_tms_params['E_field_dir'] = {
                        'Coord_type': 'Spherical',
                        'Polar': polar,
                        'Azimuthal': azimuthal,
                    }
                params_list.append((
                        cell_name_ID,
                        starting_E,
                        search_factor,
                        search_precision,
                        copy_tms_params,
                        syn_params
                    ))
                cell_angles_list.append([cell_name_ID, polar, azimuthal])

    with Pool(num_cores) as pool:
        results = pool.starmap(estimate_cell_threshold, params_list)

    for [cell_name_ID, polar, azimuthal], [threshold, num_sim_bounding, num_sim_refining] in zip(cell_angles_list, results):
        threshold_map[cell_name_ID]['Polar'][polar]['Azimuthal'][azimuthal] = {'threshold': threshold,
                                                                                'num_sim_bounding': num_sim_bounding,
                                                                                'num_sim_refining': num_sim_refining,
                                                                                'num_sim_total': num_sim_bounding+num_sim_refining,}
    
    # Save parameters & probabilities
    save_data = dict(
            threshold_map=threshold_map,
            cell_name=cell_name,
            morphIDs=morphIDs,
            starting_E=starting_E,
            search_factor=search_factor,
            search_precision=search_precision,
            polar_resolution=polar_resolution,
            azimuthal_resolution=azimuthal_resolution,
            binned_angles=binned_angles,
            tms_params=tms_params,
        )
    save_results(save_data, 'data/tms_thresholds', cell_name)

    return threshold_map