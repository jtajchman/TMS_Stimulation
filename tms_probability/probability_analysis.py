import json


def get_results(results_fname):
    with open(results_fname, 'r') as f:
        return json.load(f)


def get_thresholds(results_fname):
    return get_results(results_fname)['threshold_map']
    

def thresholds_to_probability_from_file(results_fname, ef_amp, morphs):
    thresholds = get_thresholds(results_fname)
    return thresholds_to_probability(thresholds, ef_amp, morphs)


def thresholds_to_probability(thresholds, ef_amp, morphs):
    data_map = {}
    for idx, [cell, polar_dict] in enumerate(thresholds.items()):
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


def aggregate_probabilities(probability_map):
    '''
    Aggregates probabilities of one map over all polar angles according to the beta distribution 
    of polar angles found in TMS simulation of a realistic head model by Weise et. al. (2023)
    https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00036/118104/Directional-sensitivity-of-cortical-neurons
    '''
    # TODO: change how the distribution is applied; i.e. polar angles at the max or min are completely unrepresented at the end
    # Weights should represent the spherical surface area that each polar angle band is meant to fill instead of applying the beta distribution to the polar angle itself
    from scipy.stats import beta
    import numpy as np

    dist_min = 0
    dist_max = 180
    b = lambda x: beta.pdf((x-dist_min)/(dist_max-dist_min), 1.51, 1.56)

    polars = [polar for [polar, prob, num_samples] in probability_map]
    probs = [prob for [polar, prob, num_samples] in probability_map]
    distribution_weights = [b(polar) for polar in polars]
    return np.average(probs, weights=distribution_weights)


def aggregate_threshold_probabilities(results, ef_amp, morphs):
    return aggregate_probabilities(thresholds_to_probability(results, ef_amp, morphs))


def aggregate_threshold_probabilities_from_file(results_fname, ef_amp, morphs):
    return aggregate_probabilities(thresholds_to_probability_from_file(results_fname, ef_amp, morphs))

