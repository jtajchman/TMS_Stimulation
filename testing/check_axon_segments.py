import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath('')).parent)
sys.path.append(rootFolder)

from netpyne import specs, sim

cell_name = 'L5_TTPC2_cADpyr'
# L23_PC_cADpyr, L5_TTPC2_cADpyr, L23_SBC_bNAC, L5_LBC_cNAC
# etc
morphIDs = [1, 2, 3, 4, 5]
cell_names_ID = [f'{cell_name}_{morphID}' for morphID in morphIDs]
netParams = specs.NetParams()

for cell_name_ID in cell_names_ID:
    netParams.loadCellParamsRule(label = cell_name_ID, fileName = f'cells/{cell_name_ID}_cellParams.json') 
    netParams.cellParams[cell_name_ID]['conds']['cellType'] = cell_name_ID
    secLists = netParams.cellParams[cell_name_ID]["secLists"]
    print(
                f"Cell sections ({cell_name_ID}) -\t"
              + f"Axon: {len(secLists['axonal'])}\t"
              + f"All: {len(secLists['all'])}"
            )