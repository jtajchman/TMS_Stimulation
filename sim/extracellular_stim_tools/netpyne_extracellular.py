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
from .netpyne_custom_run import runSimWithIntervalFunc

from netpyne import sim
from netpyne.network import Network
from netpyne.cell import CompartCell
from .units import mm, um

import pickle

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
    try:
        if direction['Coord_type'] == 'Cartesian':
            vector = normalize_vector([direction['X'], direction['Y'], direction['Z']])
        elif direction['Coord_type'] == 'Spherical':
            # Rotate spherical coordinates to the somatodendritic axis and convert to cartesian coordinates
            vector = rotate_coords_from_pos_z_to_axis(somatodendritic_axis, norm_spherical_to_cartesian(direction['Polar'], direction['Azimuthal']))
        else:
            raise KeyError('Direction must have a defined Coord_type of "Cartesian" or "Spherical"')
    except KeyError:
        raise KeyError('Direction must have a defined Coord_type of "Cartesian" or "Spherical"')

    return np.array(vector)


def get_section_list_NetPyNE(cell: CompartCell) -> list:
    # Get section_list from cell assuming the cell is a NetPyNE CompartCell
    return cell.secs["soma_0"]["hObj"].wholetree()


def calculate_segments_pts(section_list, centers=True, diams=False):
    """
    Calculate the 3D points of each segment in the given sections.
    Calculated by interpolating the 3D coordinates of the section's points
    and returning the points of each segment in a list.
    Parameters
    ----------
    section_list : list
        List of sections to calculate the segment centers for.
    centers : bool, optional
        If True, calculate the centers of the segments. Otherwise, calculate the vertices bounding the segments. Default is True.
    diams : bool, optional
        If True, also return the diameters of the segments. Default is False.
    Returns
    -------
    segments_pts : list
        List of segment points for each section.
        Each element is a numpy array of shape (nseg, 3) where nseg is the number of segments in the section.
    """
    
    # This is an adaptation of the "grindaway" function from
    # "extracellular_stim_and_rec".

    segments_pts = []
    segments_diams = []

    for sec in section_list:
        section_seg_pts = []

        # get data for the section
        # nn = sec_num_pts
        sec_num_pts = int(sec.n3d())  # number of 3d points for the current section
        xx = []
        yy = []
        zz = []
        length = []
        dd = []

        # for each point, get x,y,z coordinates and arc length position.
        for ii in range(0, sec_num_pts):
            xx.append(sec.x3d(ii))
            yy.append(sec.y3d(ii))
            zz.append(sec.z3d(ii))
            length.append(sec.arc3d(ii))
            dd.append(sec.diam3d(ii))

        length = np.array(length)
        if length[-1] != 0.0:
            length = length / (length[-1])

        # initialize the x-coordinates at which to evaluate
        # the interpolated values.
        rangev = []
        rangev_step = 1.0 / sec.nseg
        rangev_length = sec.nseg + 2
        rangev = np.linspace(
            0, 0 + (rangev_step * rangev_length), rangev_length, endpoint=False
        )
        if centers:
            rangev = rangev - rangev_step / 2.0
            rangev = rangev[1:-1]           # Remove the first and last points to include only internal sections
        else:
            rangev = rangev[:-1]
    

        # numpy interp function: y = interp(x, xp, fp), where
        # y are the interpolated values.
        xint = np.interp(rangev, length, xx)
        yint = np.interp(rangev, length, yy)
        zint = np.interp(rangev, length, zz)
        dint = np.interp(rangev, length, dd)

        # stores the segment centers separately, by section.
        for ii in range(len(rangev)):
            section_seg_pts.append([xint[ii], yint[ii], zint[ii]])
            
        # centers grouped by section.
        segments_pts.append(np.array(section_seg_pts))
        segments_diams.append(dint)

    if diams:
        # If diams is True, return both segment centers and diameters
        return np.array(segments_pts, dtype=object), np.array(segments_diams, dtype=object)
    else:
        return np.array(segments_pts, dtype=object)


class StimCell():
    def __init__(self, cell: CompartCell):
        self.cell = cell
        self.section_list = []
        self.segments_centers = []
        self.E_vectors = []
        self.neqs = []
        self.eqs = []


class ExtracellularStim():
    def __init__(self,
                 cells_list: list[CompartCell], 
                 stim_type: str,
                 decay_rate_percent_per_mm: float,
                 E_field_dir: dict,
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

        self.set_norm_E_field()  # Calculates how the E-field will result in extracellular potentials at each segment

        # insert extracellular potentials
        self.insert_extracellular_quasipotentials()


    def set_norm_E_field(self):
        """
        Sets normalized (time-static) electric field vectors defining the field distribution over this cell,
        and calculates normalized extracellular quasipotentials
        """
        self.stim_cell.segments_centers = calculate_segments_pts(self.stim_cell.section_list)
        self.calculate_E_vectors()
        self.calculate_neqs()
    

    def calculate_E_vectors(self):
        # Calculate the electric field vector at each segment center

        # Get direction vectors in normalized cartesian coordinates
        E_field_dir_vector = get_direction_vector(self.E_field_dir, self.somatodendritic_axis)
        decay_dir_vector = get_direction_vector(self.decay_dir, self.somatodendritic_axis)

        # Construct E_vectors with the same section & segment structure as segments_centers
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


    def calculate_neqs(self):
        """
        Calculate the normalized extracellular quasipotentials (neqs) following the order of segments given by 'self.section_list'.

        'Quasi'-potential because the electric field is non-conservative (in TMS), and traditional electric potential is undefined.
        However, values analagous to electric potential (quasipotentials) can be defined across small distances (e.g., within a cell).
        Therefore, at the micro-scale of cells, magnetically-induced quasipotentials are mechanistically interchangeable with electric potentials.

        'Normalized' because these represent scaling factors for the electric field amplitude. I.e., the actual extracellular quasipotential
        for a particular segment at a particular time is the product of the normalized quasipotential of that segment and the 
        electric field at that time. These normalized quasipotentials represent how much extracellular quasipotential the segment would experience
        for a given electric field and have units mV/(mV/um). 

        Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6035313/
        """
        # quasipotentials are calculated using the E-field at each segment's center.

        segment_index = 0
        neqs = []  # Normalized quasipotentials; list of lists
        sec_names = []

        for sec in self.stim_cell.section_list:
            segment_in_section_index = 0
            section_neqs = []
            for seg in sec:
                # if the segment is the root segment (first segment of soma):
                if segment_index == 0:
                    section_neqs.append(0.0)  # phi_soma = 0
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
                    seg_disp = (self.stim_cell.segments_centers[len(sec_names)][segment_in_section_index]
                                - self.stim_cell.segments_centers[previous_sec_id][-1])  
                                # Center of current segment - center of last segment of previous section (displacement)
                    E_p = self.stim_cell.E_vectors[previous_sec_id][-1]  # Normalized E-field at last segment of previous section

                    # get the NEQ of the
                    # previous section's last segment.
                    phi_p = neqs[previous_sec_id][-1]  # NEQ at last segment of previous section
                # if the segment is other than the first of the section:
                else:
                    seg_disp = (self.stim_cell.segments_centers[len(sec_names)][segment_in_section_index]
                                - self.stim_cell.segments_centers[len(sec_names)][segment_in_section_index - 1])
                    E_p = self.stim_cell.E_vectors[len(sec_names)][segment_in_section_index - 1]

                    phi_p = section_neqs[-1]

                E_c = self.stim_cell.E_vectors[len(sec_names)][segment_in_section_index]  # Normalized E-field at current segment

                # NEQ of current segment
                phi_c = phi_p - 0.5 * np.dot((E_c + E_p), seg_disp)

                section_neqs.append(phi_c)

                segment_in_section_index += 1
                segment_index += 1

            sec_names.append(sec.name())
            neqs.append(section_neqs)

        self.stim_cell.neqs = neqs
    

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
        for v in self.stim_cell.eqs:
            self.stim_cell.cell.v_ext.append(h.Vector(v))

        # play v_ext into e_extracellular reference
        i = 0
        for sec in self.stim_cell.section_list:  # v
            for seg in sec:
                self.stim_cell.cell.v_ext[i].play(seg._ref_e_extracellular, self.stim_cell.cell.t_ext, True)
                i += 1


    # From tmseffects/tms_networks/core/TMSSimulation.py/build_v_ext()
    def calculate_extracellular_quasipotentials(self):
        norm_quasipotentials_list = flattenLL(self.stim_cell.neqs)
        v_ext = np.zeros((len(norm_quasipotentials_list), len(self.wav)))

        for i in range(len(norm_quasipotentials_list)):
            # norm_quasipotentials_list contains normalized quasipotentials for each segment (units um)
            # self.wav contains E-field values over time (units mV/um) to scale the normalized quasipotentials
            v_ext[i, :] = [norm_quasipotentials_list[i] * vt for vt in self.wav]  # mV

        self.stim_cell.eqs = v_ext # extracellular quasipotentials (units mV)


    def clear_stim_data(self):
        # Sometimes necessary to clear this clutter from memory to save resources and runtime
        del self.cells_list
        del self.stim_cell
        del self.stim_type
        del self.decay_rate_percent_per_mm
        del self.E_field_dir
        del self.decay_dir
        del self.somatodendritic_axis
        del self.ref_point_um
        del self.wav
        del self.time


class SingleExtracellular(ExtracellularStim):
    def __init__(self, cell, stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um = [0, 0, 0], 
                 v_recording=False, ecs_spike_recording=True, clear_ecs_data=False, save=False, **waveform_params):
        super().__init__([cell], stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um, **waveform_params)
        if sim.rank == 0:
            print(f"Applying extracellular stim ({stim_type}) to cell...")
        
        self.set_stimulation()

        # Voltage recording
        self.t = []
        self.voltages = []

        # Spike recording
        self.netcons = []
        self.action_potentials = h.Vector()
        self.action_potentials_recording_ids = h.Vector()

        if v_recording:
            self.init_v_recording()
        if ecs_spike_recording:
            self.init_spike_recording()
        if clear_ecs_data:
            self.clear_stim_data()


    def init_v_recording(self):
        """Initializes membrane potential recording for every segment of the neuron"""
        self.t = h.Vector().record(h._ref_t)
        self.voltages = []
        # Voltage traces organized by section
        for section in self.stim_cell.section_list:
            v_vec_sec = []
            for segment in section:
                v_vec_seg = h.Vector().record(segment._ref_v)
                v_vec_sec.append(v_vec_seg)
            self.voltages.append(v_vec_sec)

    
    def clear_v_data(self):
        del self.t
        del self.voltages


    # From tmsneurosim/nrn/simulation/simulation.py/Simulation._init_spike_recording()
    def init_spike_recording(self):
        """Initializes spike recording for every segment of the neuron."""
        self.netcons = []
        i = 0
        for section in self.stim_cell.section_list:
            for segment in section:
                recording_netcon = h.NetCon(segment._ref_v, None, sec=section)
                recording_netcon.threshold = 0
                recording_netcon.record(
                    self.action_potentials, self.action_potentials_recording_ids, i
                )
                self.netcons.append(recording_netcon)
                i += 1
    

    def clear_spike_data(self):
        # Sometimes necessary to clear this clutter from memory
        del self.netcons, 
        del self.action_potentials, 
        del self.action_potentials_recording_ids
    

    def run_simulation(self):
        runSimWithIntervalFunc('dt', self.interval_func, func_first=True)
        sim.gatherData()

    
    def save_v_and_spikes(self, save_name=None):
        if type(save_name) is not str:
            save_name = self.stim_cell.cell.__repr__()
        with open(f"{save_name}_sim_results.pkl", 'wb') as f:
            pickle.dump([self.t, self.voltages, self.action_potentials, self.action_potentials_recording_ids], f)



class MultiExtracellular(ExtracellularStim):
    """WARNING: Untested"""
    def __init__(self, cells_list, stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um = [0, 0, 0], **waveform_params):
        super().__init__(cells_list, stim_type, decay_rate_percent_per_mm, E_field_dir, decay_dir, somatodendritic_axis, ref_point_um, **waveform_params)
        from netpyne import sim
        if sim.rank == 0:
            print(f"Applying extracellular stim ({stim_type}) to network...")
            
        for stim_cell in self.cells_list:
            self.set_stimulation(stim_cell)