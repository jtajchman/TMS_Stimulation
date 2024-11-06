import json
import numpy as np
import os
import sys
from netpyne import sim, specs
from neuron import h
from matplotlib import pyplot as plt
import pickle
import tmsneurosim
from tmsneurosim.nrn import cells
from tmsneurosim.nrn.cells import NeuronCell

cfg = specs.SimConfig()

cfg.duration = 20  ## Duration of the sim, in ms
cfg.dt = 0.025
cfg.seeds = {"conn": 4321, "stim": 1234, "loc": 4321}
cfg.hParams = {"celsius": 34, "v_init": -70}
cfg.verbose = False
cfg.createNEURONObj = True
cfg.createPyStruct = True
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.1

cfg.includeParamsLabel = False
cfg.printPopAvgRates = True
cfg.checkErrors = False

from network_cell_choice import allpops

cfg.recordCells = allpops  # which cells to record from
## Dict with traces to record
cfg.recordTraces = {
    "V_soma": {"sec": "soma_0", "loc": 0.5, "var": "v"},
    "V_axon_0": {"sec": "axon_0", "loc": 0.5, "var": "v"},
    "V_axon_1": {"sec": "Myelin_0", "loc": 0.5, "var": "v"},
    # 'V_apic_0': {'sec':'apic_0', 'loc':0.5, 'var':'v'},
    "V_apic_5": {"sec": "apic_5", "loc": 0.5, "var": "v"},
    "V_apic_95": {"sec": "apic_95", "loc": 0.5, "var": "v"},
    # 'V_apic_100': {'sec':'apic_100', 'loc':0.5, 'var':'v'},
    # 'V_dend_8_1': {'sec':'dend_8', 'loc':0.1, 'var':'v'},
    # 'V_dend_8_3': {'sec':'dend_8', 'loc':0.3, 'var':'v'},
    # 'V_dend_8_5': {'sec':'dend_8', 'loc':0.5, 'var':'v'},
    # 'V_dend_8_7': {'sec':'dend_8', 'loc':0.7, 'var':'v'},
    # 'V_dend_8_9': {'sec':'dend_8', 'loc':0.9, 'var':'v'},
    "V_dend_5": {"sec": "dend_5", "loc": 0.5, "var": "v"},
    "V_dend_65": {"sec": "dend_65", "loc": 0.5, "var": "v"},
    # 'V_dend_70': {'sec':'dend_70', 'loc':0.5, 'var':'v'},
}

cfg.recordStim = True
cfg.recordTime = True
cfg.recordStep = 0.025

cfg.recordLFP = [[0, y, 0] for y in [-400]]  # 1 elec on skull

cfg.simLabel = "Real_Net_Prototype"
cfg.saveFolder = "."
cfg.savePickle = False  ## Save pkl file
cfg.saveJson = False  ## Save json file
cfg.saveDataInclude = [
    "simConfig",
    "netParams",
]  ## 'simData' , 'simConfig', 'netParams'
cfg.backupCfgFile = None  ##
cfg.gatherOnlySimData = False  ##
cfg.saveCellSecs = False  ##
cfg.saveCellConns = False  ##

cfg.analysis["plotTraces"] = {
    "include": allpops,
    "figSize": (12, 4),
    "saveFig": False,
    "overlay": True,
    "oneFigPer": "cell",
}  # Plot recorded traces for this list of cells