# -*- coding: utf-8 -*-

# Based on StimCell.py in the "tms-effects" package written by Vitor V. Cuziol
# Edited & expanded for use in NetPyNE by Jacob Tajchman

# TODO:
# Test Netpyne interval function to force time points and change time step
# Test parallelization of coord_rotations, as it uses warnings.catch_warnings(), which is undefined for multi-threaded applications
# Test E-field waveform generation
# Experimentally measured TMS waveforms
# Implement num_time_steps_in_pulse as alternative to active_dt
# Revise input parameter formats (combine degenerate parameters into one)
    # Instead of "define parameter A = ~ or parameter B = ~"
    # Change to "define parameter A_or_B = ('A', ~) or ('B', ~)"
# Arbitrary number of E-field sources (superposition of fields)

# math
import numpy as np
from .spherical_cartesian_conversion import (
    cartesian_to_spherical, 
    spherical_to_cartesian, 
    norm_cartesian_to_spherical, 
    norm_spherical_to_cartesian
    )
from .coord_rotations import rotate_coords_from_axis, get_rotation, rotate_coords

# NEURON
from neuron import h

from .extracellular_efield import get_efield

from netpyne.network import Network
from netpyne.cell import CompartCell
from .units import mm, um


# From tmseffects/tms_networks/core/TMSSimulation.py
def build_v_ext(v_seg_values, time_course):
    v_ext = np.zeros((len(v_seg_values), len(time_course)))

    for i in range(len(v_seg_values)):
        # v_seg_values contains normalized quasipotentials for each segment (units um)
        # time_course contains E-field values over time (units mV/um)
        v_ext[i, :] = [v_seg_values[i] * vt for vt in time_course]  # mV

    return v_ext


# From tmseffects/tms_networks/core/codes/LFPyCell.py
def insert_v_ext(cell, v_ext, t_ext, section_list):
    """Set external extracellular potential around cell.

    Playback of some extracellular potential v_ext on each cell.totnseg
    compartments. Assumes that the "extracellular"-mechanism is inserted
    on each compartment.
    Can be used to study ephaptic effects and similar
    The inputs will be copied and attached to the cell object as
    cell.v_ext, cell.t_ext, and converted
    to (list of) neuron.h.Vector types, to allow playback into each
    compartment e_extracellular reference.
    Can not be deleted prior to running cell.simulate()

    Parameters
    ----------
    v_ext : ndarray
        Numpy array of size cell.totnsegs x t_ext.size, unit mV
    t_ext : ndarray
        Time vector of v_ext
    """

    # create list of extracellular potentials on each segment, time vector
    cell.t_ext = h.Vector(t_ext)
    cell.v_ext = []
    for v in v_ext:
        cell.v_ext.append(h.Vector(v))

    # play v_ext into e_extracellular reference
    i = 0
    for sec in section_list:  # v
        for seg in sec:
            cell.v_ext[i].play(seg._ref_e_extracellular, cell.t_ext, True)
            # print(seg is sec(seg.x))
            i += 1


def normalize_vector(vector):
    vector = np.array(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm != 0:
        vector = vector / norm
    return vector


def get_direction_vector(direction, somatodendritic_axis):
    if direction['Coord_type'] == 'Cartesian':
        vector = normalize_vector([direction['X'], direction['Y'], direction['Z']])
    elif direction['Coord_type'] == 'Spherical':
        vector = rotate_coords_from_axis(somatodendritic_axis, norm_spherical_to_cartesian(direction['Polar'], direction['Azimuthal']))
    else:
        raise ValueError('Direction must have a defined Coord_type of "Cartesian" or "Spherical"')
    return np.array(vector)


def get_section_list_NetPyNE(NetPyNE_cell):
    soma = NetPyNE_cell.secs["soma_0"]["hObj"]
    return soma.wholetree()


# From tmseffects/tms_networks/core/codes/list_routines.py
def flattenLL(LL):
    """Flattens a list of lists: each element of each list
    becomes an element of a single 1-D list."""
    return [x for L0 in LL for x in L0]


def plot_quasipotentials(quasipotentials, centers):
    from mpl_toolkits.mplot3d import Axes3D 
    import matplotlib.pyplot as plt 
    
    quasipotentials = flattenLL(quasipotentials)
    centers = flattenLL(centers)

    xs = np.array([center[0] for center in centers])
    ys = np.array([center[1] for center in centers])
    zs = np.array([center[2] for center in centers])

    ax = plt.figure().add_subplot(projection='3d')

    scatter = ax.scatter(xs, zs, ys, c=quasipotentials, cmap='bwr') 
    ax.set_aspect('equal')
    ax.set_title("Cell Quasipotentials\n(Field points from positive to negative potential)") 
    ax.set_xlabel('x-axis') 
    ax.set_ylabel('z-axis') 
    ax.set_zlabel('y-axis') 
    ax.invert_yaxis()
    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel('Quasipotentials (mV)')

    # scatter = ax.scatter(xs, ys, zs, c=quasipotentials, cmap='bwr') 
    # ax.set_aspect('equal')
    # ax.set_title("Cell Quasipotentials\n(Field points from positive to negative potential)") 
    # ax.set_xlabel('x-axis') 
    # ax.set_ylabel('y-axis') 
    # ax.set_zlabel('z-axis') 
    # cbar = plt.colorbar(scatter)
    # cbar.ax.set_ylabel('Quasipotentials (mV)')


def calculate_segments_centers(section_list):
    # This is an adaptation of the "grindaway" function from
    # "extracellular_stim_and_rec".

    segments_centers = []

    for sec in section_list:
        section_seg_centers = []

        # get data for the section
        # nn = sec_num_pts
        sec_num_pts = int(sec.n3d())  # number of 3d points for the current section
        xx = []
        yy = []
        zz = []
        length = []

        # for each point, get x,y,z coordinates and arc length position.
        for ii in range(0, sec_num_pts):
            xx.append(sec.x3d(ii))
            yy.append(sec.y3d(ii))
            zz.append(sec.z3d(ii))
            length.append(sec.arc3d(ii))

        length = np.array(length)
        if int(length[-1]) != 0:
            length = length / (length[-1])

        # initialize the x-coordinates at which to evaluate
        # the interpolated values.
        rangev = []
        rangev_step = 1.0 / sec.nseg
        rangev_length = sec.nseg + 2
        rangev = np.linspace(
            0, 0 + (rangev_step * rangev_length), rangev_length, endpoint=False
        )
        rangev = rangev - 1.0 / (2.0 * sec.nseg)
        rangev = rangev[1:-1]

        # numpy interp function: y = interp(x, xp, fp), where
        # y are the interpolated values.
        xint = np.interp(rangev, length, xx)
        yint = np.interp(rangev, length, yy)
        zint = np.interp(rangev, length, zz)

        # stores the segment centers separately, by section.
        for ii in range(sec.nseg):
            section_seg_centers.append([xint[ii], yint[ii], zint[ii]])

        segments_centers.append(np.array(section_seg_centers))

    # returns centers grouped by section.
    return np.array(segments_centers, dtype=object)


def calculate_cell_quasipotentials(E_vectors, centers, section_list):
    """
    Calculate quasipotentials by numerical integration of a given
    eletric field's values,
    following the order of segments given by 'self.section_list'.

    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6035313/
    """

    segment_index = 0
    quasipotentials = []  # list of lists
    sec_names = []

    for sec in section_list:
        segment_in_section_index = 0
        section_quasipotentials = []
        for seg in sec:
            # if the segment is the root segment (first segment of soma):
            if segment_index == 0:
                section_quasipotentials.append(0.0)  # phi_soma = 0
                segment_in_section_index += 1
                segment_index += 1
                continue
            # if the segment is the first of the section:
            elif segment_in_section_index == 0:
                # get previous section's id, for use as
                # index with E_field and centers
                previous_sec_name = sec.parentseg().sec.name()
                previous_sec_id = sec_names.index(previous_sec_name)

                # displacement vector
                seg_disp = (
                    centers[len(sec_names)][segment_in_section_index]
                    - centers[previous_sec_id][-1]
                )  # Center of current segment - center of last segment of previous section (displacement)
                E_p = E_vectors[previous_sec_id][
                    -1
                ]  # Normalized E-field at last segment of previous section

                # get the quasipotential of the
                # previous section's last segment.
                phi_p = quasipotentials[previous_sec_id][
                    -1
                ]  # Quasipotential at last segment of previous section
            # if the segment is other than the first of the section:
            else:
                seg_disp = (
                    centers[len(sec_names)][segment_in_section_index]
                    - centers[len(sec_names)][segment_in_section_index - 1]
                )
                E_p = E_vectors[len(sec_names)][segment_in_section_index - 1]

                phi_p = section_quasipotentials[-1]

            E_c = E_vectors[len(sec_names)][
                segment_in_section_index
            ]  # Normalized E-field at current segment

            # converts from micrometers to milimeters, so that E-field
            # of unit mV/mm (which is equivalent to V/m) can be used.
            # NOTE: NEURON simulation uses um, so this should not be changed
            # seg_disp = seg_disp * 1e-3

            # NOTE: CALCULATION OF NORMALIZED QUASIPOTENTIAL; any non-logic errors stem from this calculation or that of its component variables
            # Due to normalized E-field vectors, the quasipotential
            phi_c = phi_p - 0.5 * np.dot((E_c + E_p), seg_disp)

            section_quasipotentials.append(phi_c)

            segment_in_section_index += 1
            segment_index += 1

        assert len(list(sec)) == len(section_quasipotentials)  # debug

        sec_names.append(sec.name())
        quasipotentials.append(section_quasipotentials)

    return quasipotentials  # Important: E-field Normalized - varaible technically has units of um; scaling will occur in build_v_ext()


def set_E_field(
        section_list: list,
        decay_rate_percent_per_mm: float,
        E_field_dir: dict,
        decay_dir: dict,
        ref_point_um: list[float],
        somatodendritic_axis: list[float],
        plot: bool,
    ):
    """
    Sets electric field vectors defining the stimulus over this cell,
    and calculates quasipotentials (i.e., electric potential under
    the quasistatic assumption).
    """

    E_field_dir_vector = get_direction_vector(E_field_dir, somatodendritic_axis)
    decay_dir_vector = get_direction_vector(decay_dir, somatodendritic_axis)
    ref_point_um = np.array(ref_point_um, dtype=float)

    segments_centers = calculate_segments_centers(section_list)

    # Construct E_vectors with the same section & segment structure as segments_centers
    # such that E_vector is scaled according to the field experienced at each center
    E_vectors = []
    for sec_segs in segments_centers:
        sec_ef = [] # List of electric field vectors at each segment in the section
        for seg_center in sec_segs:
            ext_field_scalar = (1 - decay_rate_percent_per_mm/100) ** np.dot(
                (np.array(seg_center) - ref_point_um) * um / mm, # Convert from um to mm because decay rate is defined in mm
                decay_dir_vector
            )
            sec_ef.append(E_field_dir_vector * ext_field_scalar)
        E_vectors.append(sec_ef)

    # check
    for i in range(len(E_vectors)):
        assert len(E_vectors[i]) == len(segments_centers[i])

    # quasipotentials are calculated using the E-field at each segment's center.
    quasipotentials = calculate_cell_quasipotentials(
        E_vectors, segments_centers, section_list
    )

    if plot:
        plot_quasipotentials(quasipotentials, segments_centers)

    # values of electric potential at segments centers, but flattened,
    # i.e., not grouped by sections.
    # (in 'self.quasipotentials', they are grouped by section.)
    return np.array(flattenLL(quasipotentials))


def set_stimulation(
        cell: CompartCell, 
        decay_rate_percent_per_mm: float,
        E_field_dir: dict,
        decay_dir: dict,
        ref_point_um: list[float],
        somatodendritic_axis: list[float],
        plot: bool,
        wav: list,
        time: list,
    ):


    # Populate section_list in order of parent-child structure (depth-first)
    section_list = get_section_list_NetPyNE(cell)

    for sec in section_list:
        # Set extracellular mechanism on all sections
        sec.insert("extracellular")

    v_segments = set_E_field(
        section_list=section_list,
        decay_rate_percent_per_mm=decay_rate_percent_per_mm,
        E_field_dir=E_field_dir,
        decay_dir=decay_dir,
        ref_point_um=ref_point_um,
        somatodendritic_axis=somatodendritic_axis,
        plot=plot,
    )  # Returns a list of quasipotentials corresponding to each segment in the cell

    # generate v_ext matrix
    v_ext = build_v_ext(v_segments, wav)
    t_ext = np.array(time)

    # insert extracellular potentials
    insert_v_ext(cell, np.array(v_ext), np.array(t_ext), section_list)

    return v_segments

def apply_extracellular_stim(
        cells_list: list, 
        stim_type: str,
        decay_rate_percent_per_mm: float,
        E_field_dir: dict,
        decay_dir: dict,
        somatodendritic_axis: list[float],
        ref_point_um: list[float] = [0, 0, 0],
        plot: bool = False,
        **waveform_params,
    ):
    """
    net: NetPyNE network object
    freq_Hz: Frequency of TMS pulses in Hz
    duration_ms: Duration of simulation in ms
    pulse_resolution_ms: Temporal resolution of pulses in ms (independent of simulation dt)
    stim_start_ms: Time of first pulse in ms
    stim_end_ms: Time when stimulation ends in ms
    ef_amp_V_per_m: Amplitude of pulse in V/m
    width_ms: Period of waveform in ms
    pshape: Qualitative description of waveform ("Sine" and "Square" are the only currently supported options)
    decay_rate_percent_per_mm: Rate of exponential decay of electric field in %(V/m)/mm; Valid for (1, 0] (exclusive, inclusive bounds)
    E_field_dir: Direction of electric field; vector does not need to be normalized
    decay_dir: Direction along which the decay of the electric field occurs; vector does not need to be normalized
    ref_point_um: Point in um at which the E-field magnitude = specified amplitude (technically defines a plane normal to decay_dir intersecting this point)
    """
    print(f"Applying extracellular stim ({stim_type}) to network...")

    wav, time, interval_func = get_efield(
            stim_type=stim_type,
            **waveform_params,
        )
    
    for cell in cells_list:
        set_stimulation(
            cell, 
            decay_rate_percent_per_mm,
            E_field_dir,
            decay_dir,
            ref_point_um,
            somatodendritic_axis,
            plot,
            wav, 
            time,
        )
    
    return interval_func