import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath('')).parent)
sys.path.append(rootFolder)

from tms_thresholds.recruitment_analysis import (get_thresholds_dict, 
                                                 get_cell_type_recruitment, 
                                                 aggregate_thresholds,
                                                 get_dist_weights,
                                                 get_cell_type_recruitment)
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import make_interp_spline


# Takes threshold data from fnames and returns a function that turns an efield value to a recruitment value
def get_recruitment_func(thresholds, morphs):
    if type(thresholds) == str: # If thresholds is a file name
        thresholds = get_thresholds_dict(thresholds)

    aggregated_thresholds = aggregate_thresholds(thresholds, morphs)
    distribution_weights = get_dist_weights(aggregated_thresholds)
    def recruitment_func(efields):
        try: # If efields is a list
            iter(efields)
            return [get_cell_type_recruitment(aggregated_thresholds, efield, distribution_weights) for efield in efields]
        except TypeError: # If efields is a single value
            return get_cell_type_recruitment(aggregated_thresholds, efields, distribution_weights)
    return recruitment_func


# From https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot by user "TomSelleck"
def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


def plot_threshold_distributions(thresholds_list, efield_min, efield_max, num_efield_points, legend, morphs_list="all", title=None, check_prop_sum=False, ax=None):
    if morphs_list == "all":
        morphs_list = [[1, 2, 3, 4, 5] for _ in range(len(thresholds_list))]
    recruitment_funcs = [get_recruitment_func(thresholds, morphs) for thresholds, morphs in zip(thresholds_list, morphs_list)]
    efields = np.linspace(efield_min, efield_max, num_efield_points)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    for i, recruitment_func in enumerate(recruitment_funcs):
        recruitments = recruitment_func(efields)
        proportions = [recruitments[j]-recruitments[j-1] if j > 0 else recruitments[0] for j in range(len(recruitments))]
        if check_prop_sum:
            # Prints the sum of the proportions to check if the sum is 1
            print(f"{legend[i]} sum = {sum(proportions)}")
        # ax.plot(efields, proportions)
        ax.plot(efields, smooth(proportions, 0.7))
    if title is not None:
        ax.set_title(title, fontsize=30)
    ax.set_xlabel('Electric Field Amplitude (V/m)', fontsize=20)
    ax.set_ylabel('Proportion of Thresholds', fontsize=20)
    ax.legend(legend, fontsize=20)
    try:
        return fig, ax
    except UnboundLocalError:
        return ax

def plot_recruitment_curves(thresholds_list, efield_min, efield_max, num_efield_points, legend, morphs_list="all", title=None, ax=None):
    if morphs_list == "all":
        morphs_list = [[1, 2, 3, 4, 5] for _ in range(len(thresholds_list))]
    recruitment_funcs = [get_recruitment_func(thresholds, morphs) for thresholds, morphs in zip(thresholds_list, morphs_list)]
    efields = np.linspace(efield_min, efield_max, num_efield_points)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    for recruitment_func in recruitment_funcs:
        recruitments = recruitment_func(efields)
        ax.plot(efields, recruitments)
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel('Electric Field Amplitude (V/m)', fontsize=15)
    ax.set_ylabel('Recruitment Probability', fontsize=15)
    ax.legend(legend, fontsize=15)
    try:
        return fig, ax
    except UnboundLocalError:
        return ax

def plot_parametric_recruitment(thresholds_lists, efield_min, efield_max, num_efield_points, legend, xlabel, ylabel, morphs_lists="all", title=None, efield_profile=None, ax=None):
    # thresholds_lists = [[thresholds1_1, thresholds1_2], [thresholds2_1, thresholds2_2], ...]
    # efield_profile = efield value at which to plot recruitment profile
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    if morphs_lists == "all":
        shape = np.array(thresholds_lists).shape
        morphs_lists = np.empty((shape[0], shape[1], 5), dtype=object)
        morphs_lists[:, :] = [1, 2, 3, 4, 5]

    efields = np.linspace(efield_min, efield_max, num_efield_points)

    profiles = []
    for thresholds_list, morphs_list, label in zip(thresholds_lists, morphs_lists, legend):
        recruitment_func_0 = get_recruitment_func(thresholds_list[0], morphs_list[0])
        recruitment_func_1 = get_recruitment_func(thresholds_list[1], morphs_list[1])
        probabilities1 = recruitment_func_0(efields)
        probabilities2 = recruitment_func_1(efields)
        ax.plot(probabilities1, probabilities2, label=f"{label} Curve")
        if efield_profile is not None:
            prof_0 = recruitment_func_0(efield_profile)
            prof_1 = recruitment_func_1(efield_profile)
            print(f"{label} Profile @ {efield_profile} V/m: ({prof_0:.3f}, {prof_1:.3f})")
            profiles.append((prof_0, prof_1, label))
    
    if efield_profile is not None:
        for prof_0, prof_1, label in profiles:
            ax.plot(prof_0, prof_1, marker='o', markersize=10) #, label=f"{label} Profile @ {efield_profile} V/m: ({prof_0:.3f}, {prof_1:.3f})"

    ax.plot([0, 1], [0, 1], 'k--')
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(fontsize=15)
    try:
        return fig, ax
    except UnboundLocalError:
        return ax


# Returns the recruitment profile at the given efield value, assuming the simulations represented in fnames use the same pulse parameters
def get_recruitment_profile(thresholds_list, efield, morphs_list="all"):
    if morphs_list == "all":
        morphs_list = [[1, 2, 3, 4, 5] for _ in range(len(thresholds_list))]
    recruitment_funcs = [get_recruitment_func(thresholds, morphs) for thresholds, morphs in zip(thresholds_list, morphs_list)]
    return [recruitment_func([efield])[0] for recruitment_func in recruitment_funcs]


def plot_recruitment_profile(prof, ax, label):
    ax.plot(prof[0], prof[1], label=label)