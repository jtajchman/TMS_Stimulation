import json
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from neuron import h

rootFolder = os.getcwd()
os.chdir(rootFolder)
print(rootFolder)
folder = os.listdir("WeiseEtAl2023/cells/")
folder = sorted(folder)

savedata = 1  # Save Netpyne and BBP soma_voltage


def loadTemplateName(cellnumber):
    f = open(outFolder + "/template.hoc", "r")
    for line in f.readlines():
        if "begintemplate" in line:
            templatename = str(line)
    templatename = templatename[14:-1]
    return templatename


cellnumber = 30
cellName = folder[cellnumber]
outFolder = rootFolder + "/WeiseEtAl2023/cells/" + folder[cellnumber]
cellTemplateName = loadTemplateName(cellnumber)
print("CellNumber = %d" % cellnumber)
print("CellName = %s" % cellName)
print("TemplateName = %s" % cellTemplateName)

with open(outFolder + "/current_amps.dat") as current_file:
    current_content = current_file.read()

holding_current, step1_current, step2_current, step3_current = [
    float(x) for x in current_content.split()
]
# step1_current = step2_current
print("load step1_current from current_amps.dat = %s" % step1_current)

step1_current = holding_current + step1_current / 1.25

# os.chdir(rootFolder)
# !nrnivmodl WeiseEtAl2023/mechanisms/

from netpyne import sim, specs
import pickle

plt.switch_backend("TkAgg")

cfg = specs.SimConfig()

cfg.duration = 1000  ## Duration of the sim, in ms
cfg.dt = 0.025
cfg.seeds = {"conn": 4321, "stim": 1234, "loc": 4321}
cfg.hParams = {"celsius": 34, "v_init": -70}
cfg.verbose = False
cfg.createNEURONObj = True
cfg.createPyStruct = True
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.5

cfg.includeParamsLabel = False
cfg.printPopAvgRates = True
cfg.checkErrors = False

allpops = ["L5_TTPC"]

cfg.recordCells = allpops  # which cells to record from
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


## Dict with traces to record
cfg.recordStim = True
cfg.recordTime = True
cfg.recordStep = 0.025


cfg.recordLFP = [[0, y, 0] for y in [-400]]  # 1 elec on skull


cfg.simLabel = "S1_L5_TTPC"
cfg.saveFolder = "."
cfg.savePickle = False  ## Save pkl file
cfg.saveJson = False  ## Save json file
cfg.saveDataInclude = [
    "simConfig",
    "netParams",
]  ## 'simData' , 'simConfig', 'netParams'
cfg.backupCfgFile = None  ##
cfg.gatherOnlySimData = False  ##
cfg.saveCellSecs = True  ##
cfg.saveCellConns = False  ##

cfg.analysis["plotTraces"] = {
    "include": ["L5_TTPC"],
    "figSize": (12, 4),
    "saveFig": False,
    "overlay": True,
    "oneFigPer": "cell",
}  # Plot recorded traces for this list of cells


# ------------------------------------------------------------------------------
# Current inputs
# ------------------------------------------------------------------------------
cfg.addIClamp = 1

cfg.IClamp1 = {
    "pop": "L5_TTPC",
    "sec": "soma_0",
    "loc": 0.5,
    "start": 1000,
    "dur": 1000,
    "amp": step1_current,
}
# cfg.IClamp2 = {'pop': 'L5_TTPC', 'sec': 'apic_98', 'loc': 0.5, 'start': 700, 'dur': 25, 'amp': step1_current}
# cfg.IClamp3 = {'pop': 'L5_TTPC', 'sec': 'apic_99', 'loc': 0.5, 'start': 900, 'dur': 25, 'amp': step1_current}
# cfg.IClamp4 = {'pop': 'L5_TTPC', 'sec': 'apic_100', 'loc': 0.5, 'start': 1100, 'dur': 25, 'amp': step1_current}


netParams = (
    specs.NetParams()
)  # object of class NetParams to store the network parameters

# ------------------------------------------------------------------------------
# Cell parameters
# ------------------------------------------------------------------------------
# StochKv_deterministic.mod
cellName = folder[cellnumber]
cellTemplateName = loadTemplateName(cellnumber)
cellNumber = 1
cellRule = netParams.importCellParams(
    label=cellName + "_rule",
    somaAtOrigin=False,
    conds={"cellType": cellName, "cellModel": "HH_full"},
    fileName="cellwrapper3.py",
    cellName="loadCell_L5_TTPC2_cADpyr",
    cellInstance=True,
    cellArgs={"cellNumber": cellNumber},
)

# ------------------------------------------------------------------------------
# Population parameters
# ------------------------------------------------------------------------------

## Population parameters
# pyr_positions = [[x , y , 0] for x in range(100,1100,300) for y in range(100,1100,300)]
# pyr_positions = [[x-857 , x , 0] for x in range(857,1382,250)]
pyr_positions = [[x, 857, 210] for x in range(210, 1382, 2500)]
cellsList = [{"x": x, "y": y, "z": z} for x, y, z in pyr_positions]
netParams.popParams["L5_TTPC"] = {
    "cellType": cellName,
    "cellModel": "HH_full",
    "cellsList": cellsList,
}


# #------------------------------------------------------------------------------
# #  extracellular mechs
# #------------------------------------------------------------------------------
# for celltyp in netParams.cellParams.keys():
#     label = []
#     for secname in netParams.cellParams[celltyp]['secs'].keys():
#         netParams.cellParams[celltyp]['secs'][secname]['mechs']['extracellular'] = {}

# -----------------------------------------------------------------------------------#
# for cellMe in netParams.cellParams.keys():
#     axon_pt3d_x, axon_pt3d_y, axon_pt3d_z, soma_pt3d_diam =  netParams.cellParams[cellMe]['secs']['soma_0']['geom']['pt3d'][-1]
#     axon_pt3d_diam =  netParams.cellParams[cellMe]['secs']['axon_0']['geom']['diam']
#     axon_pt3d_L =  netParams.cellParams[cellMe]['secs']['axon_0']['geom']['L']

#     netParams.cellParams[cellMe]['secs']['axon_0']['geom']['pt3d'] = [(axon_pt3d_x, axon_pt3d_y, axon_pt3d_z, axon_pt3d_diam),
#                                                                       (axon_pt3d_x, axon_pt3d_y+axon_pt3d_L/2.0, axon_pt3d_z, axon_pt3d_diam),
#                                                                       (axon_pt3d_x, axon_pt3d_y+axon_pt3d_L, axon_pt3d_z, axon_pt3d_diam)]

#     axon1_pt3d_x, axon1_pt3d_y, axon1_pt3d_z, soma_pt3d_diam =  netParams.cellParams[cellMe]['secs']['axon_0']['geom']['pt3d'][-1]
#     axon1_pt3d_diam =  netParams.cellParams[cellMe]['secs']['axon_1']['geom']['diam']
#     axon1_pt3d_L =  netParams.cellParams[cellMe]['secs']['axon_1']['geom']['L']

#     netParams.cellParams[cellMe]['secs']['axon_1']['geom']['pt3d'] = [(axon1_pt3d_x, axon1_pt3d_y, axon1_pt3d_z, axon1_pt3d_diam),
#                                                                           (axon1_pt3d_x, axon1_pt3d_y+axon1_pt3d_L/2.0, axon1_pt3d_z, axon1_pt3d_diam),
#                                                                           (axon1_pt3d_x, axon1_pt3d_y+axon1_pt3d_L, axon1_pt3d_z, axon1_pt3d_diam)]


# ------------------------------------------------------------------------------
# Current inputs (IClamp)
# ------------------------------------------------------------------------------
if cfg.addIClamp:
    for key in [k for k in dir(cfg) if k.startswith("IClamp")]:
        params = getattr(cfg, key, None)
        [pop, sec, loc, start, dur, amp] = [
            params[s] for s in ["pop", "sec", "loc", "start", "dur", "amp"]
        ]

        # cfg.analysis['plotTraces']['include'].append((pop,0))  # record that pop

        # add stim source
        netParams.stimSourceParams[key] = {
            "type": "IClamp",
            "delay": start,
            "dur": dur,
            "amp": amp,
        }

        # connect stim source to target
        netParams.stimTargetParams[key + "_" + pop] = {
            "source": key,
            "conds": {"pop": pop},
            "sec": sec,
            "loc": loc,
        }


# netParams.rotateCellsRandomly = [0, 6.2832]
# sim.createSimulateAnalyze(netParams, cfg)

celltyp = cellName + "_rule"
netParams.cellParams[celltyp]["secLists"]["all"] = list(
    netParams.cellParams[celltyp]["secs"].keys()
)
len(netParams.cellParams[celltyp]["secLists"]["all"])

netParams.cellParams[celltyp]["secLists"]["all"] = list(
    netParams.cellParams[celltyp]["secs"].keys()
)
netParams.cellParams[celltyp]["secLists"]["somatic"] = [
    sec for sec in list(netParams.cellParams[celltyp]["secs"].keys()) if "soma" in sec
]
netParams.cellParams[celltyp]["secLists"]["apical"] = [
    sec for sec in list(netParams.cellParams[celltyp]["secs"].keys()) if "apic" in sec
]
netParams.cellParams[celltyp]["secLists"]["basal"] = [
    sec for sec in list(netParams.cellParams[celltyp]["secs"].keys()) if "dend" in sec
]
netParams.cellParams[celltyp]["secLists"]["axonal"] = [
    sec
    for sec in list(netParams.cellParams[celltyp]["secs"].keys())
    if "Node" in sec or "axon" in sec or "y" in sec
]
for celltyp in netParams.cellParams.keys():
    label = []
    for secname in netParams.cellParams[celltyp]["secs"].keys():
        netParams.cellParams[celltyp]["secs"][secname]["mechs"]["extracellular"] = {}
        del netParams.cellParams[celltyp]["secs"][secname].mechs.xtra
netParams.cellParams[celltyp]["secs"]["Node_113"]["geom"]["diam"]

netParams.cellParams[celltyp]["secs"]["Node_113"]["geom"]["diam"] = 1.0
netParams.cellParams[celltyp]["secs"]["Node_113"]["geom"]["pt3d"]

netParams.cellParams[celltyp]["secs"]["Node_113"]["geom"]["pt3d"] = [
    (-22.136661529541016, -6.852905750274658, -709.5120239257812, 1.0),
    (-22.494400024414062, -7.216529846191406, -710.219970703125, 1.0),
    (-22.545791625976562, -7.267182350158691, -710.3250122070312, 1.0),
]
# for celltyp in netParams.cellParams.keys():
#     label = []
#     for secname in netParams.cellParams[celltyp]['secs'].keys():
#         print(secname,netParams.cellParams[celltyp]['secs'][secname]['geom']['L'],netParams.cellParams[celltyp]['secs'][secname]['geom']['diam'])
netParams.saveCellParamsRule(
    label=cellName + "_rule", fileName=cellName + "_cellParams.pkl"
)
netParams.saveCellParamsRule(
    label=cellName + "_rule", fileName=cellName + "_cellParams.json"
)
cfg.verbose = True
sim.initialize(
    simConfig=cfg, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations
sim.net.defineCellShapes()  # in case some cells had stylized morphologies without 3d pts
sim.net.connectCells()  # create connections between cells based on params
sim.net.addStims()  # add network stimulation
sim.setupRecording()  # setup variables to record for each cell (spikes, V traces, etc)


def collect_pt3d(self, section):
    """collect the pt3d info, for each section"""
    n3dsec = 0
    r3dsec = np.zeros(3)
    for sec in [sec for secName, sec in self.secs.items() if section in secName]:
        sec["hObj"].push()
        n3d = int(h.n3d())  # get number of n3d points in each section
        # print("get number of n3d points in each section",n3d)
        r3d = np.zeros(
            (3, n3d)
        )  # to hold locations of 3D morphology for the current section
        n3dsec += n3d

        for i in range(n3d):
            r3dsec[0] += h.x3d(i)
            r3dsec[1] += h.y3d(i)
            r3dsec[2] += h.z3d(i)

        h.pop_section()

    r3dsec /= n3dsec

    return r3dsec


def getSecsPos(self, secList):
    """Get Secs position"""
    x3d, y3d, z3d = [], [], []

    for secName in secList:
        # print(secName)
        r3dsec = collect_pt3d(self, secName)
        # print(secName, r3dsec)

        x3d.append(r3dsec[0])
        y3d.append(r3dsec[1])
        z3d.append(r3dsec[2])

    return x3d, y3d, z3d


def insert_v_ext(self, v_ext, t_ext):
    self.t_ext = h.Vector(t_ext)
    self.v_ext = []
    for v in v_ext:
        self.v_ext.append(h.Vector(v))

    # play v_ext into e_extracellular reference
    i = 0
    for secName, sec in self.secs.items():
        # print(secName,i)
        for seg in sec["hObj"]:
            self.v_ext[i].play(seg._ref_e_extracellular, self.t_ext)
        i += 1


def make_extracellular_stimuli(acs_params, self, secList):
    """Function to calculate and apply external potential"""
    x0, y0, z0 = acs_params["position"]
    ext_field = np.vectorize(
        lambda x, y, z: 1
        / (
            4
            * np.pi
            * (
                acs_params["sigma"]
                * np.sqrt((x0 - x) ** 2 + (y0 - y) ** 2 + (z0 - z) ** 2)
                + skull_attenuation
            )
        )
    )

    stimstart = acs_params["stimstart"]
    stimend = acs_params["stimend"]
    stimdif = stimend - stimstart

    # MAKING THE EXTERNAL FIELD
    n_tsteps = int(stimdif / cfg.dt + 1)
    n_start = int(stimstart / cfg.dt)
    n_end = int(stimend / cfg.dt + 1)
    t = np.arange(start=n_start, stop=n_end) * cfg.dt
    pulse = (
        acs_params["amp"]
        * 1000.0
        * np.sin(2 * np.pi * acs_params["frequency"] * t / 1000)
    )

    totnsegs = len(secList)
    v_cell_ext = np.zeros((totnsegs, n_tsteps))
    v_cell_ext[:, :] = ext_field(
        getSecsPos(metype, secList)[0],
        -1 * np.array(getSecsPos(metype, secList)[1]),
        getSecsPos(metype, secList)[2],
    ).reshape(totnsegs, 1) * pulse.reshape(1, n_tsteps)

    insert_v_ext(self, v_cell_ext, t)

    return v_cell_ext, self


# The parameters of the extracellular point current source
acs_params = {
    "position": [0.0, -1710.0, 0.0],  # um # y = [pia, bone]
    "amp": 250.0,  # uA,
    "stimstart": 500,  # ms
    "stimend": 1500,  # ms
    "frequency": 5,  # Hz
    "sigma": 0.57,  # decay constant S/m
}

skull_attenuation = 0.01 * 710  # conductivity of bone(S/m) * thickness of rat skull um

# Add extracellular stim
for c, metype in enumerate(sim.net.cells):
    if "presyn" not in metype.tags["pop"]:
        print("\n", metype.tags)
        secList = [
            secs for secs in metype.secs.keys() if "pt3d" in metype.secs[secs]["geom"]
        ]
        # print(secList)
        v_cell_ext, cell = make_extracellular_stimuli(acs_params, metype, secList)


sim.runSim()  # run parallel Neuron simulation
sim.gatherData()  # gather spiking data and cell info from each node
sim.saveData()  # save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()  # plot spike raster etc
sim.analysis.plotShape()

plt.figure()
print(np.shape(v_cell_ext))
for v in v_cell_ext:
    plt.plot(v)

figure = plt.figure()
for c, metype in enumerate(sim.net.cells):
    if "presyn" not in metype.tags["pop"]:
        print("\n", metype.tags)
        plt.plot(0, -400, "b^")
        plt.plot(acs_params["position"][0], acs_params["position"][1], "ks")
        secList = [
            secs for secs in metype.secs.keys() if "pt3d" in metype.secs[secs]["geom"]
        ]
        plt.plot(
            np.array(getSecsPos(metype, secList)[0]),
            -1 * np.array(getSecsPos(metype, secList)[1]),
            "o",
        )
        # plt.plot(np.array(getSecsPos(metype, secList)[0]),-1*np.array(getSecsPos(metype, secList)[2]),'o')
        plt.ylim(2550, -1850)
        # print(np.array(getSecsPos(cell, secList)[0]).mean(axis=-1),-1*np.array(getSecsPos(cell, secList)[1]).mean(axis=-1),np.array(getSecsPos(cell, secList)[2]).mean(axis=-1))

figure = plt.figure()
for c, metype in enumerate(sim.net.cells):
    if "presyn" not in metype.tags["pop"]:
        print("\n", metype.tags)
        plt.plot(0, -400, "b^")
        plt.plot(acs_params["position"][0], acs_params["position"][1], "ks")
        secList = [
            secs for secs in metype.secs.keys() if "pt3d" in metype.secs[secs]["geom"]
        ]
        # plt.plot(np.array(getSecsPos(metype, secList)[0]),-1*np.array(getSecsPos(metype, secList)[1]),'o')
        plt.plot(
            np.array(getSecsPos(metype, secList)[0]),
            -1 * np.array(getSecsPos(metype, secList)[2]),
            "o",
        )
        plt.ylim(2550, -1850)
        # print(np.array(getSecsPos(cell, secList)[0]).mean(axis=-1),-1*np.array(getSecsPos(cell, secList)[1]).mean(axis=-1),np.array(getSecsPos(cell, secList)[2]).mean(axis=-1))

sim.analysis.plotLFP(
    **{
        "plots": ["locations"],
        "figSize": (12, 12),
        "saveData": False,
        "saveFig": False,
        "showFig": False,
        "dpi": 300,
    }
)

sim.analysis.plotLFP(
    **{
        "plots": ["timeSeries"],
        "electrodes": [0],  #'avg',
        "timeRange": [350, cfg.duration],
        "figSize": (12, 4),
        "saveFig": False,
        "showFig": False,
    }
)


sim.analysis.plotShape(
    includePre=["L5_TTPC"],
    includePost=["L5_TTPC"],
    includeAxon=False,
    showSyns=False,
    showElectrodes=[0],
    cvar="voltage",
    dist=0.6,
    elev=5,
    azim=90,
    axisLabels=True,
    synStyle="o",
    clim=[-80, -60],
    showFig=False,
    synSize=2,
)
sim.analysis.plotShape(
    includePre=["L5_TTPC"],
    includePost=["L5_TTPC"],
    includeAxon=True,
    showSyns=False,
    showElectrodes=False,
    cvar="voltage",
    dist=0.6,
    elev=5,
    azim=90,
    axisLabels=True,
    synStyle="o",
    clim=[-80, -60],
    showFig=False,
    synSize=2,
)


sim.analysis.plotShape(
    includePre=["L5_TTPC"],
    includePost=["L5_TTPC"],
    includeAxon=False,
    showSyns=False,
    showElectrodes=[0],
    cvar="voltage",
    dist=0.6,
    elev=95,
    azim=-90,
    axisLabels=True,
    synStyle="o",
    clim=[-80, -60],
    showFig=False,
    synSize=2,
)
sim.analysis.plotShape(
    includePre=["L5_TTPC"],
    includePost=["L5_TTPC"],
    includeAxon=True,
    showSyns=False,
    showElectrodes=False,
    cvar="voltage",
    dist=0.6,
    elev=95,
    azim=-90,
    axisLabels=True,
    synStyle="o",
    clim=[-80, -60],
    showFig=False,
    synSize=2,
)


sim.analysis.plotTraces(
    overlay=True, oneFigPer="trace", figSize=(24, 3), fontSize=15, saveFig=False
)

for section in netParams.cellParams[celltyp]["secLists"]["axonal"]:
    print(
        section,
        "->",
        netParams.cellParams[celltyp]["secs"][section]["topol"]["parentSec"],
    )

sim.analysis.plot2Dnet(figSize=(5, 4), fontSize=12)
plt.show()
