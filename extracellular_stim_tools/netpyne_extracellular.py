# -*- coding: utf-8 -*-

# Based on StimCell.py in the "tms-effects" package written by Vitor V. Cuziol
# Edited & expanded for use in NetPyNE by Jacob Tajchman

# TODO:
# Further test Netpyne interval function to force time points and change time step
# Test E-field waveform generation
# Experimentally measured TMS waveforms - cTMS
# Revise input parameter formats (combine degenerate parameters into one)
    # Instead of "define parameter A = ~ or parameter B = ~"
    # Change to "define parameter A_or_B = ('A', ~) or ('B', ~)"
# Arbitrary number of E-field sources (superposition of fields)

# math
import numpy as np
from .spherical_cartesian_conversion import norm_spherical_to_cartesian
from .coord_rotations import rotate_coords_from_pos_z_to_axis

# NEURON
from neuron import h

from .extracellular_efield import get_efield

from netpyne.network import Network
from netpyne.cell import CompartCell
from .units import mm, um

# From tmseffects/tms_networks/core/codes/list_routines.py
def flattenLL(LL):
    """Flattens a list of lists: each element of each list
    becomes an element of a single 1-D list."""
    return [x for L0 in LL for x in L0]


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
        vector = rotate_coords_from_pos_z_to_axis(somatodendritic_axis, norm_spherical_to_cartesian(direction['Polar'], direction['Azimuthal']))
    else:
        raise ValueError('Direction must have a defined Coord_type of "Cartesian" or "Spherical"')
    return np.array(vector)


def get_section_list_NetPyNE(cell):
    # Get section_list from cell assuming the cell is a NetPyNE CompartCell
    return cell.secs["soma_0"]["hObj"].wholetree()


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

    # centers grouped by section.
    return np.array(segments_centers, dtype=object)


class StimCell():
    def __init__(self, cell: CompartCell):
        self.cell = cell
        self.section_list = []
        self.segments_centers = []
        self.E_vectors = []
        self.norm_quasipotentials = []
        self.extracellular_quasipotentials = []

class ExtracellularStim():
    def __init__(self,
                 cells_list: list[CompartCell], 
                 stim_type: str,
                 decay_rate_percent_per_mm: float,
                 E_field_dir: dict | list[dict],
                 decay_dir: dict,
                 somatodendritic_axis: list[float],
                 ref_point_um: list[float] = [0, 0, 0],
                 **waveform_params,):

        self.cells_list = [StimCell(cell) for cell in cells_list]
        self.stim_cell = None
        self.stim_type = stim_type
        self.decay_rate_percent_per_mm = decay_rate_percent_per_mm
        self.E_field_dir = E_field_dir
        self.decay_dir = decay_dir
        self.somatodendritic_axis = somatodendritic_axis
        self.ref_point_um = np.array(ref_point_um, dtype=float)

        self.wav, self.time, self.interval_func = get_efield(
                stim_type=self.stim_type,
                **waveform_params,
            )

    def set_stimulation(self, stim_cell: StimCell | None = None):
        if stim_cell == None:
            stim_cell = self.cells_list[0]
        self.stim_cell = stim_cell
        # Populate section_list in order of parent-child structure (depth-first)
        self.stim_cell.section_list = get_section_list_NetPyNE(self.stim_cell.cell)

        for sec in self.stim_cell.section_list:
            # Set extracellular mechanism on all sections
            sec.insert("extracellular")

        self.set_E_field()  # Stores a list of normalized quasipotentials corresponding to each segment in the cell

        # insert extracellular potentials
        self.insert_extracellular_quasipotentials()


    def set_E_field(self):
        """
        Sets electric field vectors defining the stimulus over this cell,
        and calculates normalized quasipotentials (i.e., electric potential under
        the quasistatic assumption of a normalized electric field).
        """
        self.stim_cell.segments_centers = calculate_segments_centers(self.stim_cell.section_list)
        self.calculate_E_vectors()
        self.calculate_norm_quasipotentials()
    

    def calculate_E_vectors(self):
        # Get direction vectors in normalized cartesian coordinates
        E_field_dir_vector = get_direction_vector(self.E_field_dir, self.somatodendritic_axis)
        decay_dir_vector = get_direction_vector(self.decay_dir, self.somatodendritic_axis)

        # Construct E_vectors with the same section & segment structure as segments_centers
        # such that E_vector is scaled according to the field experienced at each center
        self.stim_cell.E_vectors = []
        for sec_segs in self.stim_cell.segments_centers: # Separate into lists of segments grouped by section
            sec_ef = [] # List of electric field vectors at each segment in the section
            for seg_center in sec_segs:
                ext_field_scalar = (1 - self.decay_rate_percent_per_mm/100) ** np.dot(
                    (np.array(seg_center) - self.ref_point_um) * um / mm, # Convert from um to mm because decay rate is defined in mm
                    decay_dir_vector
                )
                sec_ef.append(E_field_dir_vector * ext_field_scalar)
            self.stim_cell.E_vectors.append(sec_ef)


    def calculate_norm_quasipotentials(self):
        """
        Calculate quasipotentials by numerical integration of a given
        eletric field's values,
        following the order of segments given by 'self.section_list'.

        Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6035313/
        """
        # quasipotentials are calculated using the E-field at each segment's center.

        segment_index = 0
        norm_quasipotentials = []  # Normalized quasipotentials; list of lists
        sec_names = []

        for sec in self.stim_cell.section_list:
            segment_in_section_index = 0
            section_norm_quasipotentials = []
            for seg in sec:
                # if the segment is the root segment (first segment of soma):
                if segment_index == 0:
                    section_norm_quasipotentials.append(0.0)  # phi_soma = 0
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
                        self.stim_cell.segments_centers[len(sec_names)][segment_in_section_index]
                        - self.stim_cell.segments_centers[previous_sec_id][-1]
                    )  # Center of current segment - center of last segment of previous section (displacement)
                    E_p = self.stim_cell.E_vectors[previous_sec_id][
                        -1
                    ]  # Normalized E-field at last segment of previous section

                    # get the normalized quasipotential of the
                    # previous section's last segment.
                    phi_p = norm_quasipotentials[previous_sec_id][
                        -1
                    ]  # Quasipotential at last segment of previous section
                # if the segment is other than the first of the section:
                else:
                    seg_disp = (
                        self.stim_cell.segments_centers[len(sec_names)][segment_in_section_index]
                        - self.stim_cell.segments_centers[len(sec_names)][segment_in_section_index - 1]
                    )
                    E_p = self.stim_cell.E_vectors[len(sec_names)][segment_in_section_index - 1]

                    phi_p = section_norm_quasipotentials[-1]

                E_c = self.stim_cell.E_vectors[len(sec_names)][
                    segment_in_section_index
                ]  # Normalized E-field at current segment

                # Normalized quasipotential of current segment
                phi_c = phi_p - 0.5 * np.dot((E_c + E_p), seg_disp)

                section_norm_quasipotentials.append(phi_c)

                segment_in_section_index += 1
                segment_index += 1

            sec_names.append(sec.name())
            norm_quasipotentials.append(section_norm_quasipotentials)

        self.stim_cell.norm_quasipotentials = norm_quasipotentials  # Important: E-field Normalized - varaible technically has units of um; scaling will occur in build_v_ext()
    

    # From tmseffects/tms_networks/core/codes/LFPyCell.py/insert_v_ext()
    def insert_extracellular_quasipotentials(self):
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

        # generate v_ext matrix
        self.calculate_extracellular_quasipotentials()

        # create list of extracellular potentials on each segment, time vector
        self.stim_cell.cell.t_ext = h.Vector(np.array(self.time))
        self.stim_cell.cell.v_ext = []
        for v in self.stim_cell.extracellular_quasipotentials:
            self.stim_cell.cell.v_ext.append(h.Vector(v))

        # play v_ext into e_extracellular reference
        i = 0
        for sec in self.stim_cell.section_list:  # v
            for seg in sec:
                self.stim_cell.cell.v_ext[i].play(seg._ref_e_extracellular, self.stim_cell.cell.t_ext, True)
                i += 1


    # From tmseffects/tms_networks/core/TMSSimulation.py/build_v_ext()
    def calculate_extracellular_quasipotentials(self):
        norm_quasipotentials_list = flattenLL(self.stim_cell.norm_quasipotentials)
        v_ext = np.zeros((len(norm_quasipotentials_list), len(self.wav)))

        for i in range(len(norm_quasipotentials_list)):
            # norm_quasipotentials_list contains normalized quasipotentials for each segment (units um)
            # self.wav contains E-field values over time (units mV/um) to scale the normalized quasipotentials
            v_ext[i, :] = [norm_quasipotentials_list[i] * vt for vt in self.wav]  # mV

        self.stim_cell.extracellular_quasipotentials = v_ext


    def clear_stim_data(self):
        # Sometimes necessary to clear this clutter from memory
        for item in [self.cells_list, self.stim_cell, self.stim_type, self.decay_rate_percent_per_mm, self.E_field_dir, 
                     self.decay_dir, self.somatodendritic_axis, self.ref_point_um, self.wav, self.time]:
            del item


class SingleExtracellular(ExtracellularStim):
    def __init__(self, cell, stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um = [0, 0, 0], **waveform_params):
        super().__init__([cell], stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um, **waveform_params)
        from netpyne import sim
        if sim.rank == 0:
            print(f"Applying extracellular stim ({stim_type}) to network...")
        
        self.set_stimulation()
        self.netcons = []
        self.action_potentials = h.Vector()
        self.action_potentials_recording_ids = h.Vector()


    # From tmsneurosim/nrn/simulation/simulation.py/Simulation._init_spike_recording()
    def init_spike_recording(self):
        """Initializes spike recording for every segment of the neuron."""
        self.netcons = []
        for i, section in enumerate(self.stim_cell.section_list):
            for segment in section:
                recording_netcon = h.NetCon(segment._ref_v, None, sec=section)
                recording_netcon.threshold = 0
                recording_netcon.record(
                    self.action_potentials, self.action_potentials_recording_ids, i
                )
                self.netcons.append(recording_netcon)
    
    def clear_spike_data(self):
        # Sometimes necessary to clear this clutter from memory
        for item in [self.netcons, self.action_potentials, self.action_potentials_recording_ids]:
            del item


class MultiExtracellular(ExtracellularStim):
    def __init__(self, cells_list, stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um = [0, 0, 0], **waveform_params):
        super().__init__(cells_list, stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um, **waveform_params)
        from netpyne import sim
        if sim.rank == 0:
            print(f"Applying extracellular stim ({stim_type}) to network...")
            
        for stim_cell in self.cells_list:
            self.set_stimulation(stim_cell)