# ['L1_NGC-DA', 'L23_PC', 'L23_SBC', 'L4_LBC_cAC', 'L4_LBC_cNAC', 'L4_MC', 'L4_SS', 'L5_LBC', 'L5_TTPC2', 'L6_TPC']
cell_choice = [
    "L23_PC",
    "L23_SBC",
    "L4_LBC_cNAC",
    "L4_MC",
    "L4_SS",
    "L5_LBC",
    "L5_TTPC2",
]
cell_id = 1
reduced = 1

inh_cells = ["L1_NGC_DA", "L23_SBC", "L4_LBC_cAC", "L4_LBC_cNAC", "L4_MC", "L5_LBC"]
exc_cells = ["L23_PC", "L4_SS", "L5_TTPC2", "L6_TPC"]

allpops = ["L2_PV", "L3_P", "L3_SS", "L4_SST", "L4_SS", "L4_PV", "L5_P", "L5_PV"]
cell_types = [
    "L23_SBC",
    "L23_PC",
    "L4_SS",
    "L4_MC",
    "L4_SS",
    "L4_LBC_cNAC",
    "L5_TTPC2",
    "L5_LBC",
]
if reduced:
    allpops = ["L5_P"]
    cell_types = ["L5_TTPC2"]
    cell_choice = cell_types
pop_to_cell = {pop: cell for pop, cell in zip(allpops, cell_types)}