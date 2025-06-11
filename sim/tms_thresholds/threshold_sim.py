from extracellular_stim_tools import get_angles, SingleExtracellular
from .sim_control import TMS_sim, save_data
from file_management import set_cwd
from multiprocessing import Pool
import tqdm
from copy import deepcopy
import os, sys
import numpy as np
from netpyne import sim
from netpyne.analysis.tools import getSpktSpkid

def estimate_cell_threshold(cell_name_ID, starting_E, search_factor, search_precision, tms_params, syn_params, ecs_spike_recording=True):
    # Find the TMS electric field amplitude threshold of a cell with a given set of TMS parameters
    # Currently only works with 0 background activity

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

        tms_params["efield_amplitude_V_per_m"] = EF_amp

        ecs = TMS_sim(cell_name_ID, tms_params, syn_params, ecs_spike_recording=ecs_spike_recording)

        if detect_spike(ecs, ecs_spike_recording):
            EF_upper = EF_amp
        else:
            EF_lower = EF_amp

        if EF_upper == None:
            EF_amp = EF_lower*search_factor
        elif EF_lower == None:
            EF_amp = EF_upper/search_factor
        else:
            bounding = 0
            EF_amp = (EF_upper+EF_lower)/2
            threshold_found = (EF_upper-EF_lower)/EF_lower <= search_precision  # Check if the relative uncertainty of the threshold is <= the search precision
                                                                                # Relative to the lower bound to be more restrictive
    
    threshold = EF_upper    # Return the lowest amplitude that causes a spike instead of the midpoint of the bounds (EF_amp)
                            # Ensures that a simulation run at the returned threshold will cause a spike
    return threshold, num_sim_bounding, num_sim_refining, cell_name_ID, tms_params


def detect_spike(ecs: SingleExtracellular, ecs_spike_recording=True):
    # Returns whether there are any spikes at all in the simulation (useful for single cell sim)
    if ecs_spike_recording:
        # Use the spike detection function in the ExtracellularStim class
        # sys.stdout.write("\n" + str(list(ecs.action_potentials)))
        return len(ecs.action_potentials) >= 3
    else:
        # Use the spike detection function from NetPyNE
        spike_times_by_id = [getSpktSpkid(cellGids=[cell.gid])[1] for cell in sim.net.cells]
        # sys.stdout.write("\n" + str(spike_times_by_id))
        return any(spike_times_by_id)


def estimate_cell_threshold_wrapped(args):
    return estimate_cell_threshold(*args)


def sort_results(results, cell_angles_list):
    # Sorts the results of the simulations by cell name and E-field angle
    sorted_results = np.empty(len(results)).tolist() # Preallocate memory
    for i, (threshold, num_sim_bounding, num_sim_refining, cell_name_ID, tms_params) in enumerate(results):
        polar = tms_params["E_field_dir"]["Polar"]
        azimuthal = tms_params["E_field_dir"]["Azimuthal"]
        sorted_results[cell_angles_list.index([cell_name_ID, polar, azimuthal])] = results[i]
    return sorted_results


def cell_type_threshold_map(cell_name, morphIDs, starting_E, search_factor, search_precision, angular_resolution, 
                            num_cores, tms_params, syn_params, ecs_spike_recording=True, save_results=False):
    # Iterates over E-field angles and amplitudes to produce a map of E-field amplidude thresholds at each angle
    polar_resolution = angular_resolution
    azimuthal_resolution = angular_resolution
    angles = get_angles(polar_resolution, azimuthal_resolution)

    cell_names_ID = [f"{cell_name}_{morphID}" for morphID in morphIDs]

    threshold_map = {}

    params_list = []
    cell_angles_list = []

    for cell_name_ID in cell_names_ID:
        threshold_map[cell_name_ID] = {"Polar": {}}
        for angle in angles:
            polar = angle[0]
            threshold_map[cell_name_ID]["Polar"][polar] = {"Azimuthal": {}}
            for azimuthal in angle[1]:
                copy_tms_params = deepcopy(tms_params)
                copy_tms_params["E_field_dir"] = {
                        "Coord_type": "Spherical",
                        "Polar": polar,
                        "Azimuthal": azimuthal,
                    }
                params_list.append((
                        cell_name_ID,
                        starting_E,
                        search_factor,
                        search_precision,
                        copy_tms_params,
                        syn_params,
                        ecs_spike_recording,
                    ))
                cell_angles_list.append([cell_name_ID, polar, azimuthal])

    with set_cwd("sim"):
        with Pool(num_cores) as pool:
            results = []
            for result in tqdm.tqdm(pool.imap_unordered(estimate_cell_threshold_wrapped, params_list), total=len(params_list)):
                results.append(result)
            results = sort_results(results, cell_angles_list)

    # start = time.time()
    # with Pool(num_cores) as pool:
    #     results = pool.starmap(estimate_cell_threshold, params_list)
    # print(f"Time taken for {len(params_list)} simulations: {time.time()-start} seconds")

    for threshold, num_sim_bounding, num_sim_refining, cell_name_ID, res_tms_params in results:
        polar = res_tms_params["E_field_dir"]["Polar"]
        azimuthal = res_tms_params["E_field_dir"]["Azimuthal"]
        threshold_map[cell_name_ID]["Polar"][polar]["Azimuthal"][azimuthal] = {"threshold": threshold,
                                                                                "num_sim_bounding": num_sim_bounding,
                                                                                "num_sim_refining": num_sim_refining,
                                                                                "num_sim_total": num_sim_bounding+num_sim_refining,}
        
    if save_results:
        # Save parameters & probabilities
        data = dict(
                threshold_map=threshold_map,
                cell_name=cell_name,
                morphIDs=morphIDs,
                starting_E=starting_E,
                search_factor=search_factor,
                search_precision=search_precision,
                polar_resolution=polar_resolution,
                azimuthal_resolution=azimuthal_resolution,
                angles=angles,
                tms_params=tms_params,
            )
        if type(save_results) == str:
            fname = save_results
        else:
            fname = cell_name
        save_data(data, "data/tms_thresholds", fname)

    return threshold_map