# noinspection PyUnresolvedReferences
import math

# noinspection PyUnresolvedReferences
import pathlib

# noinspection PyUnresolvedReferences
from neuron import h

# noinspection PyUnresolvedReferences
import tmsneurosim
from tmsneurosim.nrn.cells.cell_modification_parameters.cell_modification_parameters import (
    CellModificationParameters,
)
from tmsneurosim.nrn.cells.neuron_cell import NeuronCell


class L6_DBC_bAC(NeuronCell):
    def __init__(
        self,
        morphology_id,
        modification_parameters: CellModificationParameters = None,
        variation_seed=None,
    ):
        super().__init__(
            str(
                list(
                    pathlib.Path(str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath('cells/cells_hoc/L6_DBC_bAC/morphology/').resolve()))
                    .joinpath(f"{morphology_id}")
                    .iterdir()
                )[0]
            ),
            morphology_id,
            modification_parameters,
            variation_seed,
        )

    @staticmethod
    def get_morphology_ids():
        return sorted(
            [
                int(f.name)
                for f in pathlib.Path(str(pathlib.Path(tmsneurosim.nrn.__file__).parent.joinpath('cells/cells_hoc/L6_DBC_bAC/morphology/').resolve())).iterdir()
                if f.is_dir()
            ]
        )

    def apply_biophysics(self):
        for soma_section in self.soma:
            soma_section.insert('NaTs2_t')
            soma_section.insert('SKv3_1')
            soma_section.insert('Ca')
            soma_section.insert('SK_E2')
            soma_section.insert('Ca_LVAst')
            soma_section.insert('Nap_Et2')
            soma_section.insert('Im')
            soma_section.insert('K_Pst')
            soma_section.insert('K_Tst')
            soma_section.insert('CaDynamics_E2')
            soma_section.ena = 50 
            soma_section.ek = -85 
        
        for all_section in self.all:
            all_section.insert('pas')
            all_section.Ra = 100.0 
            all_section.cm = 1.0 
            all_section.e_pas = -75.300257 
        
        for apic_section in self.apic:
            apic_section.insert('NaTs2_t')
            apic_section.insert('SKv3_1')
            apic_section.insert('Nap_Et2')
            apic_section.insert('Ih')
            apic_section.insert('Im')
            apic_section.insert('K_Pst')
            apic_section.insert('K_Tst')
        
        for axon_section in self.axon:
            axon_section.insert('SKv3_1')
            axon_section.insert('Ca')
            axon_section.insert('SK_E2')
            axon_section.insert('CaDynamics_E2')
            axon_section.insert('Nap_Et2')
            axon_section.insert('Im')
            axon_section.insert('K_Pst')
            axon_section.insert('K_Tst')
            axon_section.insert('Ca_LVAst')
            axon_section.insert('NaTa_t')
            axon_section.ena = 50 
            axon_section.ek = -85 
        
        for dend_section in self.dend:
            dend_section.insert('NaTs2_t')
            dend_section.insert('SKv3_1')
            dend_section.insert('Nap_Et2')
            dend_section.insert('Ih')
            dend_section.insert('Im')
            dend_section.insert('K_Pst')
            dend_section.insert('K_Tst')
        
        for dend_section in self.dend:
            for dend_segment in dend_section.allseg():
                dend_segment_distance = h.distance(self.soma[0](0), dend_segment)
                dend_segment.gK_Tstbar_K_Tst = (-0.869600 + 2.087000*math.exp((dend_segment_distance--1000.000000)*-0.003000))*0.002940
                dend_segment.gSKv3_1bar_SKv3_1 = (0.0 * dend_segment_distance + 1.0)*0.005450
                dend_segment.gNap_Et2bar_Nap_Et2 = (0.0 * dend_segment_distance + 1.0)*0.000001
                dend_segment.gNaTs2_tbar_NaTs2_t = (0.0 * dend_segment_distance + 1.0)*0.004119
                dend_segment.gIhbar_Ih = (-0.869600 + 2.087000*math.exp((dend_segment_distance-0.000000)*0.003000))*0.000051
                dend_segment.e_pas = (0.0 * dend_segment_distance + 1.0)*-60.065872
                dend_segment.g_pas = (0.0 * dend_segment_distance + 1.0)*0.000006
                dend_segment.gImbar_Im = (-0.869600 + 2.087000*math.exp((dend_segment_distance--1000.000000)*-0.003000))*0.000371
        for apic_section in self.apic:
            for apic_segment in apic_section.allseg():
                apic_segment_distance = h.distance(self.soma[0](0), apic_segment)
                apic_segment.gK_Tstbar_K_Tst = (-0.869600 + 2.087000*math.exp((apic_segment_distance--1000.000000)*-0.003000))*0.002940
                apic_segment.gSKv3_1bar_SKv3_1 = (0.0 * apic_segment_distance + 1.0)*0.005450
                apic_segment.gNap_Et2bar_Nap_Et2 = (0.0 * apic_segment_distance + 1.0)*0.000001
                apic_segment.gNaTs2_tbar_NaTs2_t = (0.0 * apic_segment_distance + 1.0)*0.004119
                apic_segment.gIhbar_Ih = (-0.869600 + 2.087000*math.exp((apic_segment_distance-0.000000)*0.003000))*0.000051
                apic_segment.e_pas = (0.0 * apic_segment_distance + 1.0)*-60.065872
                apic_segment.g_pas = (0.0 * apic_segment_distance + 1.0)*0.000006
                apic_segment.gImbar_Im = (-0.869600 + 2.087000*math.exp((apic_segment_distance--1000.000000)*-0.003000))*0.000371
        for axon_section in self.axon:
            for axon_segment in axon_section.allseg():
                axon_segment_distance = h.distance(self.soma[0](0), axon_segment)
                axon_segment.gNaTa_tbar_NaTa_t = (0.0 * axon_segment_distance + 1.0)*3.418459
                axon_segment.gK_Tstbar_K_Tst = (0.0 * axon_segment_distance + 1.0)*0.026009
                axon_segment.gamma_CaDynamics_E2 = (0.0 * axon_segment_distance + 1.0)*0.003923
                axon_segment.gNap_Et2bar_Nap_Et2 = (0.0 * axon_segment_distance + 1.0)*0.000001
                axon_segment.gCa_LVAstbar_Ca_LVAst = (0.0 * axon_segment_distance + 1.0)*0.002256
                axon_segment.gSK_E2bar_SK_E2 = (0.0 * axon_segment_distance + 1.0)*0.000009
                axon_segment.gK_Pstbar_K_Pst = (0.0 * axon_segment_distance + 1.0)*0.025854
                axon_segment.gSKv3_1bar_SKv3_1 = (0.0 * axon_segment_distance + 1.0)*0.196957
                axon_segment.decay_CaDynamics_E2 = (0.0 * axon_segment_distance + 1.0)*20.715642
                axon_segment.e_pas = (0.0 * axon_segment_distance + 1.0)*-60.250899
                axon_segment.g_pas = (0.0 * axon_segment_distance + 1.0)*0.000002
                axon_segment.gImbar_Im = (0.0 * axon_segment_distance + 1.0)*0.000599
                axon_segment.gCabar_Ca = (0.0 * axon_segment_distance + 1.0)*0.000138
        for soma_section in self.soma:
            for soma_segment in soma_section.allseg():
                soma_segment_distance = h.distance(self.soma[0](0), soma_segment)
                soma_segment.gK_Tstbar_K_Tst = (0.0 * soma_segment_distance + 1.0)*0.008343
                soma_segment.gamma_CaDynamics_E2 = (0.0 * soma_segment_distance + 1.0)*0.000893
                soma_segment.gNap_Et2bar_Nap_Et2 = (0.0 * soma_segment_distance + 1.0)*0.000001
                soma_segment.gCa_LVAstbar_Ca_LVAst = (0.0 * soma_segment_distance + 1.0)*0.005574
                soma_segment.gSK_E2bar_SK_E2 = (0.0 * soma_segment_distance + 1.0)*0.026900
                soma_segment.gK_Pstbar_K_Pst = (0.0 * soma_segment_distance + 1.0)*0.502333
                soma_segment.gSKv3_1bar_SKv3_1 = (0.0 * soma_segment_distance + 1.0)*0.653374
                soma_segment.decay_CaDynamics_E2 = (0.0 * soma_segment_distance + 1.0)*605.033222
                soma_segment.e_pas = (0.0 * soma_segment_distance + 1.0)*-76.498119
                soma_segment.g_pas = (0.0 * soma_segment_distance + 1.0)*0.000057
                soma_segment.gImbar_Im = (0.0 * soma_segment_distance + 1.0)*0.000784
                soma_segment.gNaTs2_tbar_NaTs2_t = (0.0 * soma_segment_distance + 1.0)*0.300054
                soma_segment.gCabar_Ca = (0.0 * soma_segment_distance + 1.0)*0.000792
        
