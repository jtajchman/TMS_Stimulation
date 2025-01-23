import numpy as np
from netpyne import specs

from network_cell_choice import cell_choice, cell_id, allpops, pop_to_cell

netParams = (
    specs.NetParams()
)  # object of class NetParams to store the network parameters

# ------------------------------------------------------------------------------
# Cell parameters
# ------------------------------------------------------------------------------
# StochKv_deterministic.mod

for cellName in cell_choice:
    cellRuleLabel = cellName + "_rule"
    cellRule = netParams.importCellParams(
        label=cellRuleLabel,
        somaAtOrigin=False,
        conds={"cellType": cellName, "cellModel": "HH_full"},
        fileName="cellwrapper3.py",
        cellName="loadCell_tmsneurosim",
        cellInstance=True,
        cellArgs={"cellName": cellName, "id": cell_id},
    )

    cellSecLists = cellRule["secLists"]
    cellSecs = cellRule["secs"]

    cellSecLists["all"] = list(cellSecs.keys())
    cellSecLists["somatic"] = [sec for sec in list(cellSecs.keys()) if "soma" in sec]
    cellSecLists["apical"] = [sec for sec in list(cellSecs.keys()) if "apic" in sec]
    cellSecLists["basal"] = [sec for sec in list(cellSecs.keys()) if "dend" in sec]
    cellSecLists["axonal"] = [
        sec
        for sec in list(cellSecs.keys())
        if "Node" in sec or "axon" in sec or "y" in sec
    ]

    for sec in cellSecs.values():
        # sec["mechs"]["extracellular"] = {}
        del sec.mechs.xtra
        if sec["geom"]["diam"] > 10:
            sec["geom"]["diam"] = 1.0
            sec["geom"]["pt3d"] = [
                (pt[0], pt[1], pt[2], 1.0) for pt in sec["geom"]["pt3d"]
            ]

    print(
        f"Axon sections ({cellName}): "
        + str(len(netParams.cellParams[cellRuleLabel]["secLists"]["axonal"]))
    )

# ------------------------------------------------------------------------------
# Population parameters
# ------------------------------------------------------------------------------
"""
From 'Large-scale biophysically detailed model of somatosensory thalamocortical circuits in NetPyNE'
https://www.frontiersin.org/articles/10.3389/fninf.2022.884245/full

Layer	height (um)	height (normal)	from	to
L1	    165		    0.079		    0.000	0.079
L2	    149		    0.072		    0.079	0.151
L3	    353		    0.170		    0.151	0.320
L4	    190		    0.091		    0.320	0.412
L5	    525		    0.252		    0.412	0.664
L6	    700		    0.336		    0.664	1.000
L23	    502		    0.241		    0.079	0.320
All	    2082	    1.000
"""

# Primary axis of neurons & neural column is in the z-direction
netParams.sizeZ = 2082  # um
norm_layer_z_ranges = {
    "L1": [0.0, 0.079],
    "L2": [0.079, 0.151],
    "L3": [0.151, 0.320],
    "L23": [0.079, 0.320],
    "L4": [0.320, 0.412],
    "L5": [0.412, 0.664],
    "L6": [0.664, 1.0],
    "longS1": [2.2, 2.3],
    "longS2": [2.3, 2.4],
}  # normalized layer boundaries
# layer_y_ranges = {'L1': [0, 165], 'L23': [165, 667], 'L4': [667, 857], 'L5': [857, 1382], 'L6': [1382, 2082]}


def scale_layer_norm_ranges(
    norm_ranges: dict[str, list[float]]
) -> dict[str, list[float]]:
    """
    norm_ranges should be a dict of the form {'layer': [low_bound, up_bound], ...}
    """
    return {
        layer: [val * netParams.sizeZ for val in values]
        for layer, values in norm_ranges.items()
    }


layer_z_ranges = scale_layer_norm_ranges(norm_layer_z_ranges)


def pop_to_layer(pop: str) -> str:
    return pop[: pop.find("_")]


# positions = [[{'x': x-857 , 'y': x , 'z': 0}] for x in range(857, 1382, 250)]
for pop in allpops:
    cellName = pop_to_cell[pop]
    position = [{"x": 0, "y": 0, "z": -1*np.mean(layer_z_ranges[pop_to_layer(pop)])}]
    print(pop, position)
    netParams.popParams[pop] = {
        "cellType": cellName,
        "cellModel": "HH_full",
        "cellsList": position,
    }

# Network connections WORK IN PROGRESS
## Synaptic mechanism parameters
netParams.synMechParams["exc"] = {
    "mod": "Exp2Syn",
    "tau1": 0.2,
    "tau2": 5.0,
    "e": 0,
}  # excitatory synaptic mechanism

netParams.synMechParams["inh"] = {
    "mod": "Exp2Syn",
    "tau1": 0.2,
    "tau2": 5.0,
    "e": -70,
}  # inhibitory synaptic mechanism

# Stimulation parameters
netParams.stimSourceParams["bkg"] = {"type": "NetStim", "rate": 5, "noise": 1.0}  # hz

netParams.stimTargetParams["bkg->all"] = {
    "source": "bkg",
    "conds": {"pop": allpops},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

## Cell connectivity rules
netParams.connParams["L2_PV->L2_PV"] = {
    "preConds": {"pop": "L2_PV"},
    "postConds": {"pop": "L2_PV"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

netParams.connParams["L2_PV->L3_P"] = {
    "preConds": {"pop": "L2_PV"},
    "postConds": {"pop": "L3_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

netParams.connParams["L3_P->L2_PV"] = {
    "preConds": {"pop": "L3_P"},
    "postConds": {"pop": "L2_PV"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L3_P->L3_P"] = {
    "preConds": {"pop": "L3_P"},
    "postConds": {"pop": "L3_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L3_P->L5_P"] = {
    "preConds": {"pop": "L3_P"},
    "postConds": {"pop": "L5_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L3_SS->L5_P"] = {
    "preConds": {"pop": "L3_SS"},
    "postConds": {"pop": "L5_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L4_SST->L5_P"] = {
    "preConds": {"pop": "L4_SST"},
    "postConds": {"pop": "L5_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

netParams.connParams["L4_SS->L5_P"] = {
    "preConds": {"pop": "L4_SS"},
    "postConds": {"pop": "L5_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L4_PV->L4_PV"] = {
    "preConds": {"pop": "L4_PV"},
    "postConds": {"pop": "L4_PV"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

netParams.connParams["L4_PV->L5_P"] = {
    "preConds": {"pop": "L4_PV"},
    "postConds": {"pop": "L5_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

netParams.connParams["L5_P->L3_SS"] = {
    "preConds": {"pop": "L5_P"},
    "postConds": {"pop": "L3_SS"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L5_P->L4_SST"] = {
    "preConds": {"pop": "L5_P"},
    "postConds": {"pop": "L4_SST"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L5_P->L4_SS"] = {
    "preConds": {"pop": "L5_P"},
    "postConds": {"pop": "L4_SS"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "exc",
}

netParams.connParams["L5_PV->L5_P"] = {
    "preConds": {"pop": "L5_PV"},
    "postConds": {"pop": "L5_P"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

netParams.connParams["L5_PV->L5_PV"] = {
    "preConds": {"pop": "L5_PV"},
    "postConds": {"pop": "L5_PV"},
    "weight": 0.05,
    "delay": 5,
    "synMech": "inh",
}

# netParams.connParams['->'] = {
#     'preConds': {'pop': }, 
#     'postConds': {'pop': }, 
#     'weight': 0.05, 
#     'delay': 5,
#     'synMech': 'exc'}

# netParams.connParams['->'] = {
#     'preConds': {'pop': }, 
#     'postConds': {'pop': }, 
#     'weight': 0.05, 
#     'delay': 5,
#     'synMech': 'inh'}