import json
import numpy as np
import os
import sys
import neuron
from matplotlib import pyplot as plt
from netpyne import sim, specs
import pickle
from tms_tools import apply_tms, get_efield
study_choice = 0 # Aberra or S1
#%% Directories
study = ['AberraEtAl2018', 'S1'][study_choice]
plt.switch_backend('TkAgg')
rootFolder = os.getcwd()
os.chdir(rootFolder)
#print(rootFolder)
folder = os.listdir(study + '/cells/')
folder = sorted(folder)

savedata = 1 # Save Netpyne and BBP soma_voltage

#folder[15:20]
#%% Load Template Function
def loadTemplateName(cellFolder):     
    f = open(cellFolder+'/template.hoc', 'r')
    for line in f.readlines():
        if 'begintemplate' in line:
            templatename = str(line)     
    templatename=templatename[14:-1]    
    return templatename
#%% Cell Choice
cellnumber = 15
cellName = folder[cellnumber]
cellFolder = f'{rootFolder}/{study}/cells/{folder[cellnumber]}'
cellTemplateName = loadTemplateName(cellFolder)
print ("CellNumber = %d" % cellnumber)
print ("CellName = %s" % cellName)
print ("TemplateName = %s" % cellTemplateName)
#%% Current File (can remove if not using iclamp?)
'''with open(cellFolder + '/current_amps.dat') as current_file:
    current_content = current_file.read()

holding_current, step1_current, step2_current, step3_current = [float(x) for x in current_content.split()]
# step1_current = step2_current
print ('load step1_current from current_amps.dat = %s' % step1_current)

holding_current, step1_current/1.5, step2_current/2., step3_current/2.5

step1_current = holding_current + step1_current/1.25
step1_current'''
#%% nrnivmodl
os.chdir(f'{rootFolder}/{study}/mechanisms/')
os.system('nrnivmodl')
os.chdir(rootFolder)
os.system(f'xcopy /s/y {rootFolder}\\{study}\\mechanisms\\nrnmech.dll')
#%% cfg
cfg = specs.SimConfig()     

cfg.duration = 1000 ## Duration of the sim, in ms  
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

allpops = ['E']
#%% Recording settings
cfg.recordCells = allpops  # which cells to record from
## Dict with traces to record
cfg.recordTraces = {'V_soma': {'sec':'soma_0', 'loc':0.5, 'var':'v'},
                    # 'V_axon_0': {'sec':'axon_0', 'loc':0.5, 'var':'v'},
                    # 'V_axon_1': {'sec':'axon_1', 'loc':0.5, 'var':'v'},
                    # 'V_apic_0': {'sec':'apic_0', 'loc':0.5, 'var':'v'},
                    # 'V_apic_5': {'sec':'apic_5', 'loc':0.5, 'var':'v'},
                    # 'V_apic_95': {'sec':'apic_95', 'loc':0.5, 'var':'v'},
                    # 'V_apic_100': {'sec':'apic_100', 'loc':0.5, 'var':'v'},
                    'V_dend_8_1': {'sec':'dend_8', 'loc':0.1, 'var':'v'},
                    'V_dend_8_3': {'sec':'dend_8', 'loc':0.3, 'var':'v'},
                    'V_dend_8_5': {'sec':'dend_8', 'loc':0.5, 'var':'v'},
                    'V_dend_8_7': {'sec':'dend_8', 'loc':0.7, 'var':'v'},
                    'V_dend_8_9': {'sec':'dend_8', 'loc':0.9, 'var':'v'},                    
                    # 'V_dend_5': {'sec':'dend_5', 'loc':0.5, 'var':'v'},
                    # 'V_dend_65': {'sec':'dend_65', 'loc':0.5, 'var':'v'},
                    # 'V_dend_70': {'sec':'dend_70', 'loc':0.5, 'var':'v'},
                    }


cfg.recordStim = True
cfg.recordTime = True
cfg.recordStep = 0.025         


cfg.recordLFP = [[0, y, 0] for y in [-400]] # 1 elec on skull

#%% cfg 2
cfg.simLabel = 'Efield_Test'
cfg.saveFolder = '.'
cfg.savePickle = False         	## Save pkl file
cfg.saveJson = False           	## Save json file
cfg.saveDataInclude = ['simConfig', 'netParams'] ## 'simData' , 'simConfig', 'netParams'
cfg.backupCfgFile = None 		##  
cfg.gatherOnlySimData = False	##  
cfg.saveCellSecs = True			##  
cfg.saveCellConns = False		##  

cfg.analysis['plotTraces'] = {'include': ['E'], 'figSize': (12, 4), 'saveFig': False, 'overlay': True, 'oneFigPer': 'cell'}  # Plot recorded traces for this list of cells

#%% Current inputs
#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 0

# cfg.IClamp1 = {'pop': 'E', 'sec': 'soma_0', 'loc': 0.5, 'start': 1000, 'dur': 1000, 'amp': step1_current}
# cfg.IClamp2 = {'pop': 'E', 'sec': 'apic_98', 'loc': 0.5, 'start': 700, 'dur': 25, 'amp': step1_current}
# cfg.IClamp3 = {'pop': 'E', 'sec': 'apic_99', 'loc': 0.5, 'start': 900, 'dur': 25, 'amp': step1_current}
# cfg.IClamp4 = {'pop': 'E', 'sec': 'apic_100', 'loc': 0.5, 'start': 1100, 'dur': 25, 'amp': step1_current}

#%% Cell params
netParams = specs.NetParams()   # object of class NetParams to store the network parameters

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------
#StochKv_deterministic.mod
#cellName = folder[cellnumber]
#cellTemplateName = loadTemplateName(cellnumber)
cellRuleLabel = cellName + '_rule'
cellRule = netParams.importCellParams(label=cellRuleLabel, somaAtOrigin=False,
    conds={'cellType': cellName, 'cellModel': 'HH_full'},
    fileName='cellwrapper3.py',
    cellName='loadCell',
    cellInstance = True,
    cellArgs={'study': study, 'cellName': cellName, 'cellTemplateName': cellTemplateName})
#%% Pop params
#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## Population parameters
# pyr_positions = [[x , y , 0] for x in range(100,1100,300) for y in range(100,1100,300)]
pyr_positions = [[x-857 , x , 0] for x in range(857,1382,250)]
cellsList = [{'x': x, 'y': y, 'z': z} for x,y,z in pyr_positions]
netParams.popParams['E'] = {'cellType': cellName, 'cellModel': 'HH_full', 'cellsList': cellsList} 

#%% Network connection params
# Network connections
## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {
    'mod': 'Exp2Syn', 
    'tau1': 0.2, 
    'tau2': 5.0, 
    'e': 0}  # excitatory synaptic mechanism

# Stimulation parameters
netParams.stimSourceParams['bkg'] = {
    'type': 'NetStim', 
    'rate': 5, #hz 
    'noise': 1.0}

'''netParams.stimTargetParams['bkg->E-0'] = {
    'source': 'bkg', 
    'conds': {'cellList': [0]}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}'''

## Cell connectivity rules
netParams.connParams['E->E'] = {
    'preConds': {'pop': 'E'}, 
    'postConds': {'pop': 'E'}, 
    'weight': 0.05, 
    'delay': 5,
    'synMech': 'exc'}

#%% Extracellular params
# #------------------------------------------------------------------------------
# #  extracellular mechs
# #------------------------------------------------------------------------------
'''for celltyp in netParams.cellParams.keys():
    label = []
    for secname in netParams.cellParams[celltyp]['secs'].keys():
        netParams.cellParams[celltyp]['secs'][secname]['mechs']['extracellular'] = {}'''
#-----------------------------------------------------------------------------------#
'''for cellMe in netParams.cellParams.keys():
    axon_pt3d_x, axon_pt3d_y, axon_pt3d_z, soma_pt3d_diam =  netParams.cellParams[cellMe]['secs']['soma_0']['geom']['pt3d'][-1]
    axon_pt3d_diam =  netParams.cellParams[cellMe]['secs']['axon_0']['geom']['diam']
    axon_pt3d_L =  netParams.cellParams[cellMe]['secs']['axon_0']['geom']['L']

    netParams.cellParams[cellMe]['secs']['axon_0']['geom']['pt3d'] = [(axon_pt3d_x, axon_pt3d_y, axon_pt3d_z, axon_pt3d_diam),
                                                                      (axon_pt3d_x, axon_pt3d_y+axon_pt3d_L/2.0, axon_pt3d_z, axon_pt3d_diam),
                                                                      (axon_pt3d_x, axon_pt3d_y+axon_pt3d_L, axon_pt3d_z, axon_pt3d_diam)]

    axon1_pt3d_x, axon1_pt3d_y, axon1_pt3d_z, soma_pt3d_diam =  netParams.cellParams[cellMe]['secs']['axon_0']['geom']['pt3d'][-1]
    axon1_pt3d_diam =  netParams.cellParams[cellMe]['secs']['axon_1']['geom']['diam']
    axon1_pt3d_L =  netParams.cellParams[cellMe]['secs']['axon_1']['geom']['L']

    netParams.cellParams[cellMe]['secs']['axon_1']['geom']['pt3d'] = [(axon1_pt3d_x, axon1_pt3d_y, axon1_pt3d_z, axon1_pt3d_diam),
                                                                          (axon1_pt3d_x, axon1_pt3d_y+axon1_pt3d_L/2.0, axon1_pt3d_z, axon1_pt3d_diam),
                                                                          (axon1_pt3d_x, axon1_pt3d_y+axon1_pt3d_L, axon1_pt3d_z, axon1_pt3d_diam)] '''
   
#%% IClamp params
#------------------------------------------------------------------------------
# Current inputs (IClamp)
#------------------------------------------------------------------------------
if cfg.addIClamp:
     for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
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
            'loc': loc}

#%% sim setup
netParams.rotateCellsRandomly = [0, 6.2832]
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

# for sec in netParams.cellParams[celltyp]['secLists']['all']:
#     if netParams.cellParams[celltyp]['secs'][sec]['geom']['nseg']>1:
#         print(sec,netParams.cellParams[celltyp]['secs'][sec]['geom']['nseg'],np.shape(netParams.cellParams[celltyp]['secs'][sec]['geom']['pt3d']))
#%% Stim functions
'''def collect_pt3d(self, section):
        """        collect the pt3d info, for each section
        """
        n3dsec = 0
        r3dsec = np.zeros(3)
        for sec in [sec for secName, sec in self.secs.items() if section in secName]:
            sec['hObj'].push()
            n3d = int(neuron.h.n3d())  # get number of n3d points in each section
            # print("get number of n3d points in each section",n3d)
            r3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            n3dsec += n3d

            for i in range(n3d):
                r3dsec[0] += neuron.h.x3d(i)
                r3dsec[1] += neuron.h.y3d(i)
                r3dsec[2] += neuron.h.z3d(i)
            
            neuron.h.pop_section()

        r3dsec /= n3dsec
        
        return r3dsec
    
def getSecsPos(self, secList):
        """        Get Secs position
        """
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

    self.t_ext = neuron.h.Vector(t_ext)
    self.v_ext = []
    for v in v_ext:
        self.v_ext.append(neuron.h.Vector(v))
    
    # play v_ext into e_extracellular reference
    i = 0
    for secName, sec in self.secs.items(): 
        # print(secName,i)
        for seg in sec['hObj']:
                self.v_ext[i].play(seg._ref_e_extracellular, self.t_ext)
            # self.v_ext[i].play(sec['hObj'](0.5)._ref_e_extracellular, self.t_ext)
        i += 1
  
def make_extracellular_stimuli(acs_params, self, secList):
    """ Function to calculate and apply external potential """
    x0, y0, z0 = acs_params['position']
    ext_field = np.vectorize(lambda x, y, z: 1 / (4 * np.pi *
                                                  (acs_params['sigma'] * 
                                                   np.sqrt((x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2) + skull_attenuation)))

    stimstart = acs_params['stimstart']
    stimend = acs_params['stimend']
    stimdif = stimend-stimstart

    # MAKING THE EXTERNAL FIELD
    n_tsteps = int(stimdif / cfg.dt + 1)
    n_start = int(stimstart/cfg.dt)
    n_end = int(stimend/cfg.dt + 1)
    t = np.arange(start=n_start, stop=n_end) * cfg.dt
    pulse = acs_params['amp'] * 1000. * \
        np.sin(2 * np.pi * acs_params['frequency'] * t / 1000)

    totnsegs = len(secList)    
    v_cell_ext = np.zeros((totnsegs, n_tsteps))    
    v_cell_ext[:, :] = ext_field(getSecsPos(metype,secList)[0], -1*np.array(getSecsPos(metype, secList)[1]), getSecsPos(metype,secList)[2]).reshape(totnsegs, 1) * pulse.reshape(1, n_tsteps)
    
    insert_v_ext(self, v_cell_ext, t)

    return v_cell_ext, self'''

#%% Add tACS
# The parameters of the extracellular point current source
acs_params = {'position': [0.0, -1710.0, 0.0],  # um # y = [pia, bone]
              'amp': 5000.,  # uA,
              'stimstart': 500,  # ms
              'stimend': 1500,  # ms
              'frequency': 5,  # Hz
              'sigma': 0.57  # decay constant S/m
              }

skull_attenuation = 0.01*710 #conductivity of bone(S/m) * thickness of rat skull um

#Add extracellular stim
'''for c,metype in enumerate(sim.net.cells):
    if 'presyn' not in metype.tags['pop']:
        print("\n", metype.tags)
        secList = [secs for secs in metype.secs.keys() if "pt3d" in metype.secs[secs]['geom']]
        # print(secList)
        v_cell_ext, cell = make_extracellular_stimuli(acs_params, metype,secList)'''

#%% Add TMS
efield, time = get_efield(freq = 100, duration=cfg.duration/1000, tstart=0., ef_amp=500.0)
E_vector = [1, 1, 1]
sim.net = apply_tms(sim.net, efield, time, E_vector)
vext_soma = []
vext_axon = []
for cell in sim.net.cells:
    ve_soma = neuron.h.Vector().record(cell.secs['soma_0']['hObj'](0.5)._ref_vext[0])
    ve_axon = neuron.h.Vector().record(cell.secs['axon_1']['hObj'](0.5)._ref_vext[0])
    vext_soma.append(ve_soma)
    vext_axon.append(ve_axon)

#%% Run sim
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc
sim.analysis.plotShape()  

#%% Plotting
'''plt.figure()
print(np.shape(v_cell_ext))
for v in v_cell_ext:
    plt.plot(v)
plt.title('V_cell_ext')'''

#time = np.arange(len(vext_axon[0])) * cfg.dt
plt.figure()
for v in vext_soma:
    plt.plot(time, v)
plt.title('V_ext soma')
plt.figure()
for v in vext_axon:
    plt.plot(time, v)
plt.title('V_ext axon')


#Add extracellular stim
'''plt.figure()
for c,metype in enumerate(sim.net.cells):
    if 'presyn' not in metype.tags['pop']:
        print("\n", metype.tags)
        plt.plot(0,-400,'b^')
        plt.plot(acs_params['position'][0],acs_params['position'][1],'ks')
        secList = [secs for secs in metype.secs.keys() if "pt3d" in metype.secs[secs]['geom']]
        plt.plot(np.array(getSecsPos(metype, secList)[0]),-1*np.array(getSecsPos(metype, secList)[1]),'o')
        plt.ylim(2550,-1850)
        # print(np.array(getSecsPos(cell, secList)[0]).mean(axis=-1),-1*np.array(getSecsPos(cell, secList)[1]).mean(axis=-1),np.array(getSecsPos(cell, secList)[2]).mean(axis=-1))
'''
'''sim.analysis.plotLFP(**{'plots': ['locations'], 
        'figSize': (12,12), 
        'saveData': False, 
        'saveFig': False, 'showFig': False, 'dpi': 300})'''

sim.analysis.plotLFP(**{'plots': ['timeSeries'], 
        'electrodes': [0], #'avg', 
        'timeRange': [350, cfg.duration], 
        'figSize': (12,4), 'saveFig': False, 'showFig': False})

'''sim.analysis.plotShape(includePre=['E'], includePost=['E'], includeAxon=False, showSyns=False, showElectrodes=[0],
    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
    axisLabels=True, synStyle='o', 
    clim= [-80, -60], showFig=False, synSize=2)'''
sim.analysis.plotShape(includePre=['E'], includePost=['E'], includeAxon=True, showSyns=False, showElectrodes=False,
    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
    axisLabels=True, synStyle='o', 
    clim= [-80, -60], showFig=False, synSize=2)

#sim.analysis.plotTraces(overlay=True, oneFigPer='trace', figSize=(24,3), fontSize=15, saveFig=False)

#netParams.cellParams[cellRuleLabel]['secs']['dend_0']

print('Axon segments: ' + str(len(netParams.cellParams[cellRuleLabel]['secLists']['axonal'])))
#    print(section,'->',netParams.cellParams[cellRuleLabel]['secs'][section]['topol']['parentSec'])

'''sim.analysis.plot2Dnet(figSize=(5, 4), fontSize=12)
'''
plt.show()