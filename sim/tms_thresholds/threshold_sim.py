from extracellular_stim_tools import SingleExtracellular
from extracellular_stim_tools.spherical_cartesian_conversion import norm_spherical_to_cartesian
from extracellular_stim_tools.angular_grid import get_angles, unpack_angles
from .sim_control import TMS_sim, save_data
from file_management import set_cwd
from multiprocessing import Pool
import tqdm
from copy import deepcopy
import os, sys, math
import numpy as np
from netpyne import sim
from netpyne.analysis.tools import getSpktSpkid
from time import perf_counter

thresh_keys = dict(
    morphID=0,
    polar=1,
    azimuthal=2,
    threshold=3,
    num_sim_bounding=4,
    num_sim_refining=5,
    num_sim_total=6,
    starting_E=7,
    search_factor=8,
    search_precision=9,
)

def estimate_cell_threshold(cell_name_ID, starting_E, search_factor, search_precision, tms_params, ecs_spike_recording=True):
    # Find the TMS electric field amplitude threshold of a cell with a given set of TMS parameters
    # Operates using exponential bracketing followed by bisection on a monotonic predicate
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

        ecs = TMS_sim(cell_name_ID, tms_params, ecs_spike_recording=ecs_spike_recording)

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


def threshold_map_fast_serial(cell_name, morphIDs, init_starting_E, search_precision, angular_resolution, 
                              tms_params, ecs_spike_recording=True, save_results=False):
    # Iterates over E-field angles and amplitudes to produce a map of E-field amplidude thresholds at each angle
    # Uses a faster serial method instead of multiprocessing
    polar_resolution = angular_resolution
    azimuthal_resolution = angular_resolution
    angles = unpack_angles(get_angles(polar_resolution, azimuthal_resolution))
    n_sims = len(morphIDs) * len(angles)
    init_search_factor = 2 # Baseline search factor if no neighbor data is available

    cell_names_ID = [f"{cell_name}_{morphID}" for morphID in morphIDs]
    threshold_matrix = np.empty((n_sims, 10), dtype=float)  # Preallocate memory for threshold matrix

    # Put all simulation iterations in a single loop for tqdm progress bar
    iteration_params = [[cell_name_ID, morphID, polar, azimuthal] for cell_name_ID, morphID in zip(cell_names_ID, morphIDs) for [polar, azimuthal] in angles]

    tstart = perf_counter()

    for i, [cell_name_ID, morphID, polar, azimuthal] in enumerate(tqdm.tqdm(iteration_params, desc=f"Estimating thresholds for {cell_names_ID}")):
        copy_tms_params = deepcopy(tms_params)
        copy_tms_params["E_field_dir"] = {
                "Coord_type": "Spherical",
                "Polar": polar,
                "Azimuthal": azimuthal,
            }
        # Find starting_E and search_factor based on previous simulations to speed up search
        neighbor_range = 10  # degrees
        upd_E, upd_search_factor = get_search_params_from_neighbors(threshold_matrix, morphID, polar, azimuthal, neighbor_range, init_starting_E, init_search_factor)
        # Perform threshold estimation
        threshold, num_sim_bounding, num_sim_refining, _, _ = estimate_cell_threshold(
                cell_name_ID,
                upd_E,
                upd_search_factor,
                search_precision,
                copy_tms_params,
                ecs_spike_recording,
            )
        threshold_matrix[i] = [
            morphID,
            polar,
            azimuthal,
            threshold,
            num_sim_bounding,
            num_sim_refining,
            num_sim_bounding+num_sim_refining,
            upd_E,
            upd_search_factor,
            search_precision,
        ]
        
    sim_time_s = perf_counter() - tstart

    if save_results:
        # Ensure no items are np arrays for JSON serialization
        threshold_matrix_l = threshold_matrix.tolist()
        if type(angles) == np.ndarray:
            angles = angles.tolist()
        # Save parameters & probabilities
        data = dict(
                thresh_keys=thresh_keys,
                threshold_matrix=threshold_matrix_l,
                cell_name=cell_name,
                morphIDs=list(morphIDs),
                starting_E=init_starting_E,
                init_search_factor=init_search_factor,
                search_precision=search_precision,
                polar_resolution=polar_resolution,
                azimuthal_resolution=azimuthal_resolution,
                angles=angles,
                tms_params=tms_params,
                ecs_spike_recording=ecs_spike_recording,
                sim_time_s=sim_time_s,
                n_sims=n_sims,
            )
        print(data)
        if type(save_results) == str:
            fname = save_results
        else:
            fname = cell_name
        save_data(data, "data/tms_thresholds", fname)

    return threshold_matrix


def get_search_params_from_neighbors(threshold_matrix, morphID, polar, azimuthal, neighbor_range=10, starting_E=100, search_factor=2):
    '''Check neighboring points for previous threshold estimates to inform starting_E for current point using moving least squares (MLS)
    Uses first order Taylor expansion on tangent plane of sphere with gradients estimated via distance weighted least squares
    Ideally would also estimate optimal search_factor based on MLS, but this is not yet implemented'''
    
    # Include only entries for the given cell
    if threshold_matrix.dtype == object:
        reduced_matrix = threshold_matrix[np.where(threshold_matrix[:,0] == morphID)] # In this case morphID is a string (cell_name_ID)
    else:
        reduced_matrix = threshold_matrix[np.where(threshold_matrix[:,0] == float(morphID))]
    # Isolate spherical coordinates of points
    sphere_angles = reduced_matrix[:, 1:3].astype(float)

    # Convert to cartesian coordinates
    thetas, phis = np.hsplit(sphere_angles, 2)
    xi = norm_spherical_to_cartesian(thetas.flatten(), phis.flatten()) # All previous samples
    x0 = norm_spherical_to_cartesian(polar, azimuthal) # Target sample

    # Get neighboring sample indeces within neighbor_range degrees
    neighbor_indices = neighbor_indeces(xi, x0, neighbor_range)
    n_neighbors = len(neighbor_indices)
    # If no neighbors (first sample point), return default starting_E and search_factor
    if n_neighbors == 0:
        return starting_E, search_factor

    # Get thresholds and coordinates of neighboring samples
    ui_n = reduced_matrix[neighbor_indices, 3].astype(float) # Thresholds of neighboring samples
    xi_n = xi[neighbor_indices] # Coordinates of neighboring samples

    ## Predict target threshold using first order Taylor expansion on the tangent plane of the sphere
    ## with gradients estimated via distance weighted least squares from neighboring samples
    # Get tangent plane basis vectors at x0
    v = tangent_basis(polar, azimuthal)
    # Project neighboring coordinates onto tangent plane
    di = xi_n - x0 # Displacement vectors from x0 to neighbors
    eta = di @ v  # Coordinates of neighbors in tangent plane basis
    # Construct design matrix for least squares
    A = np.hstack((np.ones((n_neighbors, 1)), eta))
    # Weighting matrix based on distance
    alpha = np.linalg.norm(di, axis=1)
    sigma = np.percentile(alpha, 50)
    W = np.sqrt(np.exp(-(alpha / sigma) ** 2))

    Aw = A * W[:, None]
    bw = ui_n * W

    coeffs, residuals, rank, _ = np.linalg.lstsq(Aw, bw)
    u0, u_theta, u_phi = coeffs

    # Attempt to use residuals to estimate variance of estimate at x0 (doesn't work)
    # if len(residuals) == 1:
    #     residuals = residuals[0]
    # elif len(residuals) == 0:
    #     residuals = -1  # Underdetermined system (not enough neighbors)

    # Attempt to use residuals to estimate variance of estimate at x0 (doesn't work)
    # sigma2 = np.sum((bw - A @ coeffs) ** 2) / (n_neighbors - 3)
    # # Calculate variance of estimate at x0
    # ATA = A.T @ np.diag(W) @ A
    # if n_neighbors > 3:
    #     cov = sigma2 * np.linalg.inv(ATA)
    #     u0_variance = cov[0, 0]
    # else:
    #     u0_variance = -1.  # Underdetermined system (not enough neighbors)

    # Return starting_E and search_factor based on previous simulations
    return u0, 1.01  # Estimated threshold and search factor


def neighbor_indeces(xi, x0, neighbor_range):
    # Returns the indeces of neighboring angles in the threshold matrix
    a = np.cos(np.radians(neighbor_range))
    neighbor_mask = xi @ x0 >= a
    neighbor_indices = np.where(neighbor_mask)[0]

    return neighbor_indices


def tangent_basis(theta, phi):
    e_theta = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])

    e_phi = np.array([
        -np.sin(phi),
        np.cos(phi),
        0.0
    ])

    return np.vstack((e_theta, e_phi)).T


def threshold_map_to_matrix(threshold_map):
    '''Converts threshold_map dict into a matrix for easier processing
    NOTE: When slicing the resulting matrix, one should use .astype(float) on the relevant columns
    to convert from object dtype to float dtype for numerical operations
    '''
    n_entries = sum(
        len(azimuthal_dict['Azimuthal'])
        for cell_name_ID, polar_dict in threshold_map.items()
        for polar, azimuthal_dict in polar_dict['Polar'].items()
    )
    threshold_matrix = np.empty((n_entries, 10), dtype=object)  # Preallocate memory
    i = 0
    for cell_name_ID, polar_dict in threshold_map.items():
        for polar, azimuthal_dict in polar_dict['Polar'].items():
            polar = float(polar)
            for azimuthal, threshold_data in azimuthal_dict['Azimuthal'].items():
                azimuthal = float(azimuthal)
                try: 
                    threshold_matrix[i] = [
                        cell_name_ID,
                        polar,
                        azimuthal,
                        threshold_data['threshold'],
                        threshold_data['num_sim_bounding'],
                        threshold_data['num_sim_refining'],
                        threshold_data['num_sim_total'],
                        threshold_data['starting_E'],
                        threshold_data['search_factor'],
                        threshold_data['search_precision'],
                    ]
                except KeyError:
                    threshold_matrix[i] = [
                        cell_name_ID,
                        polar,
                        azimuthal,
                        threshold_data['threshold'],
                        threshold_data['num_sim_bounding'],
                        threshold_data['num_sim_refining'],
                        threshold_data['num_sim_total'],
                        None,
                        None,
                        None,
                    ]
                i += 1
    return threshold_matrix


def num_sims(rel_error, search_factor, search_precision=0.01):
    # Calculate number of simulations required for a given relative error and search factor
    n_sims_bounding = int(1 + np.ceil(np.log(rel_error) / np.log(search_factor)))
    n_sims_refining = int(np.ceil(-np.log2(search_precision/(search_factor-1)))) # Unable to catch some edge cases, but good approximation
    return n_sims_bounding, n_sims_refining


def calc_rel_errors(threshold_matrix, cell_name_ID):
    # Calculates the relative errors of how MLS estimates thresholds compared to simulated thresholds
    rel_errors = np.empty(threshold_matrix.shape[0])
    for i, row in enumerate(threshold_matrix):
        polar = row[1]
        azimuthal = row[2]
        test_matrix = threshold_matrix[:i]
        est_thresh, _ = get_search_params_from_neighbors(test_matrix, cell_name_ID, polar, azimuthal, neighbor_range=10, starting_E=200)
        sim_thresh = row[3]
        rel_errors[i] = np.exp(np.abs(np.log(est_thresh/sim_thresh))) # relative error of MLS estimate
    return rel_errors


def estimate_optimal_search_factor(rel_errors):
    pass