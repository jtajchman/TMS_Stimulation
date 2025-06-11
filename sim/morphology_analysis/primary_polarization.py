import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)

from file_management import set_cwd, suppress_stdout

from netpyne.cell import CompartCell
from neuron.units import cm
import numpy as np
from extracellular_stim_tools.netpyne_extracellular import get_section_list_NetPyNE, flattenLL, calculate_segments_pts
from tms_thresholds.sim_control import setup

class ConductiveSegment:
    """
    Class representing the conductive properties of a segment in a neuronal morphology.
    Caclulates the polarization potential of the segment in a nonuniform, but unidirectional electric field.
    Attributes:
        segment: The segment object from NEURON.
        idx: Index of the segment in the cell.
        parent_segment: The parent segment of this segment.
        child_segments: List of child segments connected to this segment.
        sister_segments: List of sister segments connected to this segment.
        segment_center: Center coordinates of the segment.
        ri: Axial resistance of the segment to its direct parent (Mohm).
        external_ri: Axial resistance of the segment's distal external parent to that segment's internal parent (if applicable).
        sac_vectors: List of specific axial conductance vectors to connected segments.
        connected_indices: List of indices of connected segments.
    
        coupling_vector: Vector representing the coupling of the segment to a potential field.
        ppp: Primary polarization potential of the segment in the field (mV).
    """
    
    def __init__(self, segment, segment_center: np.ndarray, segment_length: float):
        self.segment = segment
        self.idx: int = segment.node_index()
        self.segment_center: np.ndarray = segment_center
        self.L: float = segment_length
        self.c = segment.cm * self.L * np.pi * segment.diam / cm**2  # Capacitance of the segment (uF)
        self.ri: float = self.segment.ri()
        self.external_ri: float = 0.
        self.parent_segment = None
        self.child_segments: list = []
        self.sister_segments: list = []
        self.sac_vectors: list = []
        self.connected_indices: list = []
        self.coupling_vector: np.ndarray | None = None
        self.ppp: float | None = None

    def calc_sac_vectors(self):
        # Calculate the sac vector from the parent segment
        self.sac_vectors.append((self.segment_center - self.parent_segment.segment_center)/(self.ri + self.external_ri))
        self.connected_indices.append(self.parent_segment.idx)
        # Calculate the sac vectors from the child segments
        for seg in self.child_segments: 
            self.sac_vectors.append((self.segment_center - seg.segment_center)/(seg.ri + seg.external_ri))
            self.connected_indices.append(seg.idx)
        # Calculate the sac vectors from the sister segments
        for seg in self.sister_segments: 
            # If both segments have an external parent (the same one), the resistance between them bypasses the parent
            if self.external_ri != 0 and seg.external_ri != 0: sister_ri = self.ri + seg.ri
            # Else, the resistance of the external parent is included
            else: sister_ri = self.ri + seg.ri + self.external_ri + seg.external_ri
            self.sac_vectors.append((self.segment_center - seg.segment_center)/sister_ri)
            self.connected_indices.append(seg.idx)

    def calc_coupling_vector(self, family_field_strengths):
        self.coupling_vector = np.dot(family_field_strengths, self.sac_vectors)

    def calc_primary_polarization_potential(self, field_direction):
        self.ppp = np.dot(self.coupling_vector, field_direction)


def build_conductive_segments(cell: CompartCell) -> list[list[ConductiveSegment]]:
    section_list = get_section_list_NetPyNE(cell)
    sec_lengths = [sec.L for sec in section_list]
    seg_lengths = [[sec_len/sec.nseg for seg in sec] for sec_len, sec in zip(sec_lengths, section_list)]
    centers = calculate_segments_pts(section_list)
    conductive_segments = [[ConductiveSegment(seg, center, seg_len)] for sec, centers_list, seg_len_list in zip(section_list, centers, seg_lengths)
                            for seg, center, seg_len in zip(sec, centers_list, seg_len_list)]
    # idxs = [seg.idx for sec in conductive_segments for seg in sec]
    return conductive_segments


def attach_segments(conductive_segments: list[list[ConductiveSegment]]) -> None:
    """
    Attach segments to their parent, child, and sister segments.
    
    Parameters:
    - conductive_segments: List of lists of ConductiveSegment objects.
    """
    for conductive_sec in conductive_segments:
        # Attach segments within the section
        seg_counter = 0
        sec = conductive_sec[0].segment.sec     # All segments in conductive_sec belong to the same section
        children = sec.children()
        parent_seg = sec.parentseg()
        for seg in conductive_sec:
            if seg_counter == 0:  # Proximal internal node
                if parent_seg.x == 1: # Distal external node
                    seg.parent_segment = None # TODO
                seg.parent_segment = parent_seg
                seg.external_ri
            else:  # Internal or distal external node
                seg.parent_segment = get_parent_internal_segment(seg.segment)
            # Get the child segments
            seg.child_segments = [ConductiveSegment(child_seg, child_seg.x, child_seg.diam) for child_seg in seg.segment.children()]
            # Get the sister segments
            seg.sister_segments = [ConductiveSegment(sister_seg, sister_seg.x, sister_seg.diam) for sister_seg in seg.segment.siblings()]
            # Calculate sac vectors
            seg.calc_sac_vectors()


def find_segment_idx_by_node_idx(conductive_segments: list[list[ConductiveSegment]], idx: int) -> tuple | None:
    """
    Find the segment index by the node index.
    
    Parameters:
    - conductive_segments: List of lists of ConductiveSegment objects.
    - idx: Node index to search for.
    
    Returns:
    - Tuple containing the section index and segment index, or None if not found.
    """
    for sec_idx, sec in enumerate(conductive_segments):
        for seg_idx, seg in enumerate(sec):
            if seg.idx == idx:
                return (sec_idx, seg_idx)
    return None


def sac_vectors(cell: CompartCell) -> np.ndarray:
    section_list = get_section_list_NetPyNE(cell)
    segments = [list(sec.allseg()) for sec in section_list]
    sec_lengths = [sec.L for sec in section_list]
    seg_lengths = [[sec_len/sec.nseg for seg in sec] for sec_len, sec in zip(sec_lengths, section_list)]
    centers = calculate_segments_pts(section_list)

    # Get the conductivity vectors between each segment and its parent
    # Displacement vectors reflect distance between segment centers to respect how quasipotentials are calculated
    # Conductance values reflect the path traversed by current to respect how current is calculated in simulation




    test_seg = segments[0][0]
    test_sec = section_list[0]
    # for seg in test_sec: get_parent_segment(seg) 
    print(dir(test_sec))
    print(test_sec.parentseg())

    # print(test_seg.cm)                  # Specific membrane capacitance of the segment (uF/cm2)
    # print(test_seg.sec)
    # print(test_seg.x)
    # print(test_seg.diam)                # Diameter of the segment (um)
    # print(test_seg.ri())                # Axial resistance of the segment (Mohm)
    # print(seg_lengths[1][2])         # Length of the segment (um)
    # print(test_seg.ri()*(np.pi*(test_seg.diam/2)**2)/seg_lengths[1][2] * 1e6 * 1e-4)
    # print(f"pas: {test_seg.pas.e}, {test_seg.pas.g}, {test_seg.pas.i}, {test_seg.pas.is_ion()}, {test_seg.pas.name()}, {test_seg.pas.segment()}")
    
    # Calculate conductivity vectors
    # conductivity_vectors = np.array([seg.conductivity_vector for seg in segments])
    
    # return conductivity_vectors, centers


def get_parent_internal_segment(seg):
    """
    Get the parent internal segment of a given segment (x>0 & x<1).
    
    Parameters:
    - seg: A segment object from NetPyNE
    
    Returns:
    - The parent segment of the given segment
    """
    x = seg.x    
    sec = seg.sec
    nseg = sec.nseg
    x_range_step = 1.0 / nseg
    # seg_index = int(x / x_range_step - 0.5)
    # print(seg_index)
    # Cases:
    # If the segment is the proximal external node of the section
    if x == 0:
        parent_seg = sec.parentseg()
        # If the parent of the external node is the proximal external node of the parent section
        if parent_seg.x == 0:
            parent_seg = parent_seg.sec.parentseg()
            # Return the parent of that node
        # If the parent of the external node is an internal node
            # Return that internal node
        # If the parent of the external node is the distal external node of the parent section
            # Return the parent of that node
        # If the segment does not have a parent (root node)
            # Return None
    # If the segment's parent is the proximal external node of the section
    elif x == x_range_step*0.5:
        parent_seg = sec.sec(0)
    # If the segment's parent is an internal node
    else:
        parent_seg = sec(x - x_range_step)
        # ri = sec.ri() # Axial resistance of the section (Mohm)
    return parent_seg


if __name__ == "__main__":
    sim_params = {
        'simulation_duration_ms': 1000,  # Duration of the simulation in milliseconds
        'default_dt': 0.025,  # Default time step in milliseconds
    }
    with suppress_stdout():    
        setup("L5_TTPC2_cADpyr_1", sim_params, syn_params=None, savestate=None)
    from netpyne import sim
    cell = sim.net.cells[0]
    # build_conductive_segments(cell)
    sac_vectors(cell)