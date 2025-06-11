from netpyne.cell import CompartCell
from extracellular_stim_tools.netpyne_extracellular import get_section_list_NetPyNE

class CellState():
    def __init__(self, cell: CompartCell):
        self.cell = cell
        self.section_list = get_section_list_NetPyNE(cell)
        self.record_state()


    def record_state(self):
        # WIP
        mechs = [mech for mech in self.section_list[0](0.5)]
        for mech in mechs:
            print(mech, dir(mech))
        print([seg for seg in self.section_list[0].allseg()])
        print(dir(self.section_list[0](0.5)))