import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)

from file_management import set_cwd_to_root_for_func

import json
from scipy.stats import beta
import numpy as np
import bisect

def get_results(results_fname):
    with set_cwd_to_root_for_func(open, results_fname, 'r') as f:
        return json.load(f)


def get_thresholds_dict(results_fname):
    return get_results(results_fname)['threshold_map']


def thresholds_list(thresholds_dict, morphs):
    thresholds_list = []
    for idx, [cell, polar_dict] in enumerate(thresholds_dict.items()):
        if idx+1 in morphs:
            for polar, azimuthal_dict in polar_dict['Polar'].items():
                polar = float(polar)
                num_azimuth = len(azimuthal_dict['Azimuthal'])
                for azimuthal, threshold in azimuthal_dict['Azimuthal'].items():
                    azimuthal = float(azimuthal)
                    thresholds_list.append([threshold, polar, azimuthal, num_azimuth, cell])
    return thresholds_list


def thresholds_list_from_file(results_fname, morphs):
    return thresholds_list(get_thresholds_dict(results_fname), morphs)


def population_weight(x, polar_angular_range, azimuthal_angular_range):
    '''
    Beta distribution approximation of polar angles found in TMS simulation of a realistic head model by Weise et. al. (2023)
    https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00036/118104/Directional-sensitivity-of-cortical-neurons
    Weights represent the proportion of cells in a population experiencing TMS at a given angle
    '''
    dist_min = 0
    dist_max = 180
    dist = beta(1.51, 1.56)

    cdf = lambda x: dist.cdf((x-dist_min)/(dist_max-dist_min))

    # Bounds of the polar angle band that x represents
    polar_upper = np.clip(x+polar_angular_range/2, 0, 180)
    polar_lower = np.clip(x-polar_angular_range/2, 0, 180)

    # Divided by the proportion of the azimuthal angles that x represents so that the weights are normalized 
    return (cdf(polar_upper) - cdf(polar_lower)) * (azimuthal_angular_range/360)


def sort_thresholds_with_weights(thresholds_list):
    # Sort thresholds_list by threshold values
    thresholds = np.array(thresholds_list).T[0]
    idx = np.argsort(thresholds)
    sorted_thresholds = np.array(thresholds_list)[idx]
    # Insert weights into the sorted list
    return [[threshold, population_weight(polar, polar_angular_range=10, azimuthal_angular_range=360/num_azimuth), polar, azimuthal, num_azimuth, cell] 
            for [threshold, polar, azimuthal, num_azimuth, cell] in sorted_thresholds]
    

def get_probability_map(thresholds_dict, ef_amp, morphs):
    data_map = {}
    for idx, [cell, polar_dict] in enumerate(thresholds_dict.items()):
        if idx+1 in morphs:
            for polar, azimuthal_dict in polar_dict['Polar'].items():
                polar = float(polar)
                azimuthal_data = []
                for azimuthal, threshold in azimuthal_dict['Azimuthal'].items():
                    azimuthal = float(azimuthal)

                    if type(threshold) == dict:
                        threshold = threshold['threshold']
                        
                    # Calculate whether ef_amp is above threshold
                    thresh_reached = ef_amp >= threshold
                    # Add azimuthal data to list
                    azimuthal_data.append(thresh_reached)
                # Aggregate the azimuthal data
                num_fired = sum(azimuthal_data)
                num_azimuth = len(azimuthal_data)

                if polar in [data for data in data_map.keys()]:
                    data_map[polar][0] += num_fired
                    data_map[polar][1] += num_azimuth
                else:
                    data_map[polar] = [num_fired, num_azimuth]

    # Aggregate polar data between cells
    probability_map = [[polar, num_fired/num_samples, num_samples] for polar, [num_fired, num_samples] in data_map.items()]
    return probability_map


def get_probability_map_from_file(results_fname, ef_amp, morphs):
    thresholds = get_thresholds_dict(results_fname)
    return get_probability_map(thresholds, ef_amp, morphs)


# Groups thresholds by polar angle
def aggregate_thresholds(thresholds_dict, morphs) -> dict:
    threshold_map = {}
    for idx, (cell, polar_dict) in enumerate(thresholds_dict.items()):
        if idx+1 in morphs:
            for polar, azimuthal_dict in polar_dict['Polar'].items():
                polar = float(polar)
                azimuthal_data = np.empty(len(azimuthal_dict['Azimuthal'])).tolist() # Preallocate memory
                for i, (azimuthal, threshold) in enumerate(azimuthal_dict['Azimuthal'].items()):
                    if type(threshold) == dict:
                        threshold = threshold['threshold']
                    # Add azimuthal data to list
                    azimuthal_data[i] = threshold
                # Aggregate the azimuthal data
                if polar in [data for data in threshold_map.keys()]:
                    threshold_map[polar].extend(azimuthal_data)
                else:
                    threshold_map[polar] = azimuthal_data
    return threshold_map


def aggregate_thresholds_from_file(results_fname, morphs):
    return aggregate_thresholds(get_thresholds_dict(results_fname), morphs)


def calc_polar_probabilities(aggregated_thresholds, ef_amp):
    polar_probabilities = np.empty(len(aggregated_thresholds)).tolist() # Preallocate memory
    for i, (polar, thresholds) in enumerate(aggregated_thresholds.items()):
        num_fired = sum(ef_amp >= np.array(thresholds))
        num_thresholds = len(thresholds)
        polar_probabilities[i] = [polar, num_fired/num_thresholds, num_thresholds]
    return polar_probabilities


def calc_polars_and_nsamples(aggregated_thresholds):
    return aggregated_thresholds.keys(), [len(thresholds) for thresholds in aggregated_thresholds.values()]


def calc_probs(aggregated_thresholds, ef_amp):
    return [np.mean(ef_amp >= np.array(thresholds)) for thresholds in aggregated_thresholds.values()]


def calc_pop_distribution_weights(polars, nums_samples):
    return [population_weight(polar, polar_angular_range=10, azimuthal_angular_range=360/num_samples) for polar, num_samples in zip(polars, nums_samples)]


def calc_recruitment(probs, distribution_weights):
    # Calculates recruitment over all polar angles
    return np.average(probs, weights=distribution_weights)


def get_dist_weights(aggregated_thresholds):
    polars, nums_samples = calc_polars_and_nsamples(aggregated_thresholds)
    return calc_pop_distribution_weights(polars, nums_samples)


def get_cell_type_recruitment(aggregated_thresholds, ef_amp, distribution_weights):
    # Calculate probabilities for each polar angle
    probs = calc_probs(aggregated_thresholds, ef_amp)
    # Calculate recruitment
    return calc_recruitment(probs, distribution_weights)


def get_cell_type_recruitment_from_thresholds(thresholds_dict, ef_amp, morphs):
    # Aggregate thresholds
    aggregated_thresholds = aggregate_thresholds(thresholds_dict, morphs)
    # Extract polar angles and number of samples for each polar angle
    polars, nums_samples = calc_polars_and_nsamples(aggregated_thresholds)
    # Calculate distribution weights
    distribution_weights = calc_pop_distribution_weights(polars, nums_samples)
    # Calculate probabilities for each polar angle
    probs = calc_probs(aggregated_thresholds, ef_amp)
    # Calculate recruitment
    return calc_recruitment(probs, distribution_weights)


# Returns a function that calculates recruitment for a given ef_amp
# Uses a known set of polar angles
def get_recruitment_func_with_known_polars(thresholds_dict, morphs, known_polars):
    # Aggregate thresholds
    aggregated_thresholds = aggregate_thresholds(thresholds_dict, morphs)
    # Polar angles available in the threshold map
    map_polars = aggregated_thresholds.keys()
    # Dist weights for the known polar angles
    distribution_weights = polar_weights(map_polars, known_polars)

    def recruitment_func(ef_amp):
        # Calculate probabilities for each polar angle
        probs = calc_probs(aggregated_thresholds, ef_amp)
        # Calculate recruitment
        return calc_recruitment(probs, distribution_weights)

    return recruitment_func


def polar_weights(ref_polars, target_polars):
    weights = np.zeros_like(ref_polars)
    for target_polar in target_polars:
        w = interpolation_weights(ref_polars, target_polar)
        for i, weight in w.items():
            weights[i] += weight
    return weights


def interpolation_weights(x, q):
    if not (x[0] <= q <= x[-1]):
        raise ValueError("q is outside the interpolation range")

    i = bisect.bisect_right(x, q) - 1

    if i == len(x) - 1:
        return {i: 1.0}

    x0, x1 = x[i], x[i + 1]

    w1 = float((q - x0) / (x1 - x0))
    w0 = 1.0 - w1

    return {i: w0, i + 1: w1}