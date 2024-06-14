import json
import numpy as np
import os
import sys
from neuron import h
from matplotlib import pyplot as plt
from netpyne import sim, specs
import pickle
import tmsneurosim
from tmsneurosim.nrn import cells
from tmsneurosim.nrn.cells import NeuronCell
plt.switch_backend('TkAgg')

#%% Directories
study = 'WeiseEtAl2023' #(same cells as Aberra)
rootFolder = os.getcwd()
os.chdir(rootFolder)
folder = sorted(os.listdir(study + '/cells/'))
celltypes = ['L1_NGC-DA', 'L23_PC', 'L23_SBC', 'L4_LBC_cAC', 'L4_LBC_cNAC', 'L4_MC', 'L4_SS', 'L5_LBC', 'L5_TTPC2', 'L6_TPC']
cell_library = {}
for i, l in enumerate(celltypes):
    cell_library[l] = folder[i*5:(i+1)*5]

#%% nrnivmodl
os.chdir(f'{rootFolder}/{study}/mechanisms/')
os.system('nrnivmodl')
os.chdir(rootFolder)
os.system(f'xcopy /s/y {rootFolder}\\{study}\\mechanisms\\nrnmech.dll')

#%% cfg
cfg = specs.SimConfig()     

cfg.duration = 500 ## Duration of the sim, in ms  
cfg.dt = 0.025
cfg.seeds = {'conn': 4321, 'stim': 1234, 'loc': 4321} 
cfg.hParams = {'celsius': 34, 'v_init': -70}  
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

#%% Cell choice
#['L1_NGC-DA', 'L23_PC', 'L23_SBC', 'L4_LBC_cAC', 'L4_LBC_cNAC', 'L4_MC', 'L4_SS', 'L5_LBC', 'L5_TTPC2', 'L6_TPC']
cell_choice = ['L23_PC', 'L23_SBC', 'L4_LBC_cNAC', 'L4_MC', 'L4_SS', 'L5_LBC', 'L5_TTPC2']
cell_id = 1
reduced = 1

inh_cells = ['L1_NGC_DA', 'L23_SBC', 'L4_LBC_cAC', 'L4_LBC_cNAC', 'L4_MC',  'L5_LBC']
exc_cells = ['L23_PC', 'L4_SS', 'L5_TTPC2', 'L6_TPC']

allpops = ['L2_PV', 'L3_P', 'L3_SS', 'L4_SST', 'L4_SS', 'L4_PV', 'L5_P', 'L5_PV']
cell_types = ['L23_SBC', 'L23_PC', 'L4_SS', 'L4_MC', 'L4_SS', 'L4_LBC_cNAC', 'L5_TTPC2', 'L5_LBC']
if reduced:
    allpops = ['L5_P']
    cell_types = ['L5_TTPC2']
    cell_choice = cell_types
pop_to_cell = {pop: cell for pop, cell in zip(allpops, cell_types)}
#%% Recording & saving settings
cfg.recordCells = allpops  # which cells to record from
## Dict with traces to record
cfg.recordTraces = {'V_soma': {'sec':'soma_0', 'loc':0.5, 'var':'v'},
                    'V_axon_0': {'sec':'axon_0', 'loc':0.5, 'var':'v'},
                    'V_axon_1': {'sec':'Myelin_0', 'loc':0.5, 'var':'v'},
                    # 'V_apic_0': {'sec':'apic_0', 'loc':0.5, 'var':'v'},
                    'V_apic_5': {'sec':'apic_5', 'loc':0.5, 'var':'v'},
                    'V_apic_95': {'sec':'apic_95', 'loc':0.5, 'var':'v'},
                    # 'V_apic_100': {'sec':'apic_100', 'loc':0.5, 'var':'v'},
                    # 'V_dend_8_1': {'sec':'dend_8', 'loc':0.1, 'var':'v'},
                    # 'V_dend_8_3': {'sec':'dend_8', 'loc':0.3, 'var':'v'},
                    # 'V_dend_8_5': {'sec':'dend_8', 'loc':0.5, 'var':'v'},
                    # 'V_dend_8_7': {'sec':'dend_8', 'loc':0.7, 'var':'v'},
                    # 'V_dend_8_9': {'sec':'dend_8', 'loc':0.9, 'var':'v'},                    
                    'V_dend_5': {'sec':'dend_5', 'loc':0.5, 'var':'v'},
                    'V_dend_65': {'sec':'dend_65', 'loc':0.5, 'var':'v'},
                    # 'V_dend_70': {'sec':'dend_70', 'loc':0.5, 'var':'v'},
                    }

cfg.recordStim = True
cfg.recordTime = True
cfg.recordStep = 0.025       

cfg.recordLFP = [[0, y, 0] for y in [-400]] # 1 elec on skull

cfg.simLabel = 'Real_Net_Prototype'
cfg.saveFolder = '.'
cfg.savePickle = False         	## Save pkl file
cfg.saveJson = False           	## Save json file
cfg.saveDataInclude = ['simConfig', 'netParams'] ## 'simData' , 'simConfig', 'netParams'
cfg.backupCfgFile = None 		##  
cfg.gatherOnlySimData = False	##  
cfg.saveCellSecs = False			##  
cfg.saveCellConns = False		##  

cfg.analysis['plotTraces'] = {'include': allpops, 'figSize': (12, 4), 'saveFig': False, 'overlay': True, 'oneFigPer': 'cell'}  # Plot recorded traces for this list of cells  

#%% IClamp
#cfg.IClamp1 = {'pop': 'L5_TTPC', 'sec': 'soma_0', 'loc': 0.5, 'start': 1000, 'dur': 1000, 'amp': 1}


'''with open(cellFolders[0] + '/current_amps.dat') as current_file:
    current_content = current_file.read()
    
holding_current, step1_current, step2_current, step3_current = [float(x) for x in current_content.split()]
# step1_current = step2_current
print ('load step1_current from current_amps.dat = %s' % step1_current)

step1_current = holding_current + step1_current/1.25'''

#%% Cell params
netParams = specs.NetParams()   # object of class NetParams to store the network parameters

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------
#StochKv_deterministic.mod

for cellName in cell_choice:
    cellRuleLabel = cellName + '_rule'
    cellRule = netParams.importCellParams(label=cellRuleLabel, somaAtOrigin=False,
        conds={'cellType': cellName, 'cellModel': 'HH_full'},
        fileName='cellwrapper3.py',
        cellName='loadCell_Net_adv',
        cellInstance = True,
        cellArgs={'cellName': cellName, 'id': cell_id})
    
    cellSecLists = cellRule['secLists']
    cellSecs = cellRule['secs']

    cellSecLists['all'] = list(cellSecs.keys())
    cellSecLists['somatic'] = [sec for sec in list(cellSecs.keys()) if 'soma' in sec]
    cellSecLists['apical'] = [sec for sec in list(cellSecs.keys()) if 'apic' in sec]
    cellSecLists['basal'] = [sec for sec in list(cellSecs.keys()) if 'dend' in sec]
    cellSecLists['axonal'] = [sec for sec in list(cellSecs.keys()) if 'Node' in sec or 'axon' in sec or 'y' in sec]

    for sec in cellSecs.values():
        sec['mechs']['extracellular'] = {}
        del sec.mechs.xtra
        if sec['geom']['diam'] > 10:
            sec['geom']['diam'] = 1.0
            sec['geom']['pt3d'] = [(pt[0], pt[1], pt[2], 1.0) for pt in sec['geom']['pt3d']]

    print(f'Axon sections ({cellName}): ' + str(len(netParams.cellParams[cellRuleLabel]['secLists']['axonal'])))
#%% Pop params
#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------
'''
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
'''

# Primary axis of neurons & neural column is in the z-direction
netParams.sizeZ = 2082 # um
norm_layer_z_ranges = {'L1':[0.0, 0.079], 'L2': [0.079,0.151], 'L3': [0.151,0.320], 'L23': [0.079,0.320], 'L4':[0.320,0.412], 'L5': [0.412,0.664], 'L6': [0.664,1.0], 
'longS1': [2.2,2.3], 'longS2': [2.3,2.4]}  # normalized layer boundaries
#layer_y_ranges = {'L1': [0, 165], 'L23': [165, 667], 'L4': [667, 857], 'L5': [857, 1382], 'L6': [1382, 2082]}

def scale_layer_norm_ranges(norm_ranges: dict[str, list[float]]) -> dict[str, list[float]]:
    '''
    norm_ranges should be a dict of the form {'layer': [low_bound, up_bound], ...}
    '''
    return {layer: [val * netParams.sizeZ for val in values] for layer, values in norm_ranges.items()}

layer_z_ranges = scale_layer_norm_ranges(norm_layer_z_ranges)

def pop_to_layer(pop: str) -> str:
    return pop[:pop.find('_')]

#positions = [[{'x': x-857 , 'y': x , 'z': 0}] for x in range(857, 1382, 250)]
for pop in allpops:
    cellName = pop_to_cell[pop]
    position = [{'x': 0, 'y': 0, 'z': np.mean(layer_z_ranges[pop_to_layer(pop)])}]
    print(pop, position)
    netParams.popParams[pop] = {'cellType': cellName, 'cellModel': 'HH_full', 'cellsList': position}

'''for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
    params = getattr(cfg, key, None)
    [pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]

    #cfg.analysis['plotTraces']['include'].append((pop,0))  # record that pop

    # add stim source
    netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}

    # connect stim source to target
    netParams.stimTargetParams[key+'_'+pop] =  {
        'source': key, 
        'conds': {'pop': pop},
        'sec': sec, 
        'loc': loc}'''

#%% Network connection params
# Network connections
## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {
    'mod': 'Exp2Syn', 
    'tau1': 0.2, 
    'tau2': 5.0, 
    'e': 0}  # excitatory synaptic mechanism

netParams.synMechParams['inh'] = {
    'mod': 'Exp2Syn', 
    'tau1': 0.2, 
    'tau2': 5.0, 
    'e': -70}  # inhibitory synaptic mechanism

# Stimulation parameters
netParams.stimSourceParams['bkg'] = {
    'type': 'NetStim', 
    'rate': 5, #hz 
    'noise': 1.0}

netParams.stimTargetParams['bkg->all'] = {
    'source': 'bkg', 
    'conds': {'pop': allpops}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

## Cell connectivity rules
netParams.connParams['L2_PV->L2_PV'] = {
    'preConds': {'pop': 'L2_PV'}, 
    'postConds': {'pop': 'L2_PV'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}

netParams.connParams['L2_PV->L3_P'] = {
    'preConds': {'pop': 'L2_PV'}, 
    'postConds': {'pop': 'L3_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}

netParams.connParams['L3_P->L2_PV'] = {
    'preConds': {'pop': 'L3_P'}, 
    'postConds': {'pop': 'L2_PV'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L3_P->L3_P'] = {
    'preConds': {'pop': 'L3_P'}, 
    'postConds': {'pop': 'L3_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L3_P->L5_P'] = {
    'preConds': {'pop': 'L3_P'}, 
    'postConds': {'pop': 'L5_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L3_SS->L5_P'] = {
    'preConds': {'pop': 'L3_SS'}, 
    'postConds': {'pop': 'L5_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L4_SST->L5_P'] = {
    'preConds': {'pop': 'L4_SST'}, 
    'postConds': {'pop': 'L5_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}

netParams.connParams['L4_SS->L5_P'] = {
    'preConds': {'pop': 'L4_SS'}, 
    'postConds': {'pop': 'L5_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L4_PV->L4_PV'] = {
    'preConds': {'pop': 'L4_PV'}, 
    'postConds': {'pop': 'L4_PV'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}

netParams.connParams['L4_PV->L5_P'] = {
    'preConds': {'pop': 'L4_PV'}, 
    'postConds': {'pop': 'L5_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}

netParams.connParams['L5_P->L3_SS'] = {
    'preConds': {'pop': 'L5_P'}, 
    'postConds': {'pop': 'L3_SS'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L5_P->L4_SST'] = {
    'preConds': {'pop': 'L5_P'}, 
    'postConds': {'pop': 'L4_SST'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L5_P->L4_SS'] = {
    'preConds': {'pop': 'L5_P'}, 
    'postConds': {'pop': 'L4_SS'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['L5_PV->L5_P'] = {
    'preConds': {'pop': 'L5_PV'}, 
    'postConds': {'pop': 'L5_P'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}

netParams.connParams['L5_PV->L5_PV'] = {
    'preConds': {'pop': 'L5_PV'}, 
    'postConds': {'pop': 'L5_PV'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}



'''netParams.connParams['->'] = {
    'preConds': {'pop': }, 
    'postConds': {'pop': }, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

netParams.connParams['->'] = {
    'preConds': {'pop': }, 
    'postConds': {'pop': }, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'inh'}'''

'''netParams.saveCellParamsRule(label=cellName + '_rule', fileName=cellName+'_cellParams.pkl')
netParams.saveCellParamsRule(label=cellName + '_rule', fileName=cellName+'_cellParams.json')
cfg.verbose=True'''

#netParams.rotateCellsRandomly = [0, 6.2832]
# sim.createSimulateAnalyze(netParams, cfg)     
sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.defineCellShapes()  # in case some cells had stylized morphologies without 3d pts
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
#cellsPost = sim.getCellsList(includePost)

fig, _ = sim.analysis.plotShape(includePre=allpops, includePost=allpops, includeAxon=True, showSyns=False, showElectrodes=False,
    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
    axisLabels=True, synStyle='o', 
    clim= [-100, -50], showFig=False, synSize=2)
#plt.show(); exit()

#%% Add TMS
# The parameters of the extracellular point current source
tms_params = {'amp': 1000.,  # V/m,
              'stimstart': 0,  # ms
              'stimend': cfg.duration,  # ms
              'frequency': 50,  # Hz
              'dt': cfg.dt,
              'field_direction': [-1, -1 ,-1],
              'sigma': 0.1, # EF decay constant in %(V/m)/mm
              'decay_dir': [0, 0, -1], # Direction that field strength decay occurs on
              'pulse_width': 1, # Duration of TMS pulse in ms
              }

#skull_attenuation = 0.01*710 #conductivity of bone(S/m) * thickness of rat skull um

#Add extracellular stim
from tms_tools import make_tms_stim, apply_tms
#make_tms_stim(sim.net, tms_params)
apply_tms(sim.net, tms_params)

'''vext_soma = []
vext_axon = []
for cell in sim.net.cells:
    ve_soma = h.Vector().record(cell.secs['soma_0']['hObj'](0.5)._ref_vext[0])
    ve_axon = h.Vector().record(cell.secs['Myelin_1']['hObj'](0.5)._ref_vext[0])
    vext_soma.append(ve_soma)
    vext_axon.append(ve_axon)'''

#%% Run sim
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc
sim.analysis.plotShape()  

#%% Plotting
'''sim.analysis.plotLFP(**{'plots': ['timeSeries'], 
        'electrodes': [0], #'avg', 
        'timeRange': [350, cfg.duration], 
        'figSize': (12,4), 'saveFig': False, 'showFig': False})'''

'''sim.analysis.plotShape(includePre=['E'], includePost=['E'], includeAxon=False, showSyns=False, showElectrodes=[0],
    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
    axisLabels=True, synStyle='o', 
    clim= [-80, -60], showFig=False, synSize=2)'''
'''sim.analysis.plotShape(includePre=allpops, includePost=allpops, includeAxon=True, showSyns=False, showElectrodes=False,
    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
    axisLabels=True, synStyle='o', 
    clim= [-100, -50], showFig=False, synSize=2)'''

#sim.analysis.plotTraces(overlay=True, oneFigPer='trace', figSize=(24,3), fontSize=15, saveFig=False)

plt.show()