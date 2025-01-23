import os
import sys
from pathlib import Path
rootFolder = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(rootFolder)

from file_management import reach_out_for_func

from extracellular_stim_tools import runSimWithIntervalFunc, runSim, SingleExtracellular
from netpyne import specs, sim
from neuron import h
import random
import json
import os
import io


def TMS_sim(cell_name_ID: str, tms_params: dict, syn_params: dict | None = None, savestate=None, spike_recording=True, clear_ecs_data=False, suppress_logs=False):
    # Runs a single simulation of a single cell using a fully defined set of TMS parameters

    stdout = sys.stdout
    tempout = io.StringIO()

    sys.stdout = tempout    # Silence the netpyne logs

    setup(cell_name_ID, tms_params, syn_params, savestate=savestate)

    ecs = SingleExtracellular(
            cell=sim.net.cells[0],
            **tms_params
        )
    if spike_recording:
        ecs.init_spike_recording()
    if clear_ecs_data:
        ecs.clear_stim_data()
    
    # Simulate
    run_simulation(ecs.interval_func)

    sys.stdout = stdout     # Restore printing functionality

    return ecs


def baseline_sim(cell_name_ID: str, sim_params: dict | None = None, syn_params: dict | None = None):
    '''
    Run the cell in a simulation to find the steady state if one exists or to sample the baseline activity
    Currently only supports steady state simulation with no synaptic parameters
    '''

    if syn_params == None: # No synapses onto the cell; neuron should reach a steady state over enough time
        if sim_params == None: # User can specify sim_params if desired, but these params are already well-defined in this case
            sim_scale = 2
            sim_params = dict(simulation_duration_ms=10**sim_scale, default_dt=10**(sim_scale-2))
    
    if sim_params == None:
        raise ValueError('sim_params must be defined if syn_params is defined')

    setup(cell_name_ID, sim_params, syn_params, savestate=None)
    run_simulation()
    # h.t = 0
    savestate = h.SaveState()
    savestate.save()
    return savestate


def setup(cell_name_ID: str, sim_params: dict, syn_params: dict | None, savestate=None):
    cfg = cfg_setup(cell_name_ID, sim_params)
    netParams = netParams_setup(cell_name_ID, syn_params)
    sim_setup(cfg, netParams)
    if savestate != None:
        sim.fih.append(h.FInitializeHandler(1, savestate.restore))


def cfg_setup(cell_name_ID: str, sim_params: dict):
    # Set up sim config
    cfg = specs.SimConfig()  

    cfg.duration = sim_params['simulation_duration_ms'] ## Duration of the sim, in ms  
    cfg.dt = sim_params['default_dt']
    cfg.seeds = {'conn': 1234, 
                 'stim': 1234, 
                 'loc': 1234} 
    cfg.hParams = {'celsius': 37, 'v_init': -71}
    cfg.verbose = False
    cfg.createNEURONObj = True
    cfg.createPyStruct = True  
    cfg.allpops = [cell_name_ID]
    cfg.recordCells = cfg.allpops # record all cells  

    cfg.recordTraces = {'V_soma': {'sec':'soma_0', 'loc':0.5, 'var':'v'},}
    cfg.recordStep = cfg.dt
    # cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'oneFigPer': 'trace', 'overlay': True, 'saveFig': True, 'showFig': False, 'figSize':(12,6)}

    # Saving
    # saveFolderRoot = 'data/tms_probability/'
    # save_num = len([file for file in os.listdir(saveFolderRoot) if cell_name in file])
    # cfg.simLabel = cell_name_ID
    # cfg.saveFolder = 'data/tms_probability/'+cfg.simLabel
    # cfg.savePickle = False         	## Save pkl file
    # cfg.saveJson = False	        ## Save json file
    # cfg.saveDataInclude = ['simData'] ## , 'netParams', 'simConfig', ,'simData'
    # cfg.backupCfgFile = None
    # cfg.gatherOnlySimData = True
    # cfg.saveCellSecs = False
    # cfg.saveCellConns = False
    return cfg


def netParams_setup(cell_name_ID: str, syn_params: dict | None):
    # Create the cell
    netParams = specs.NetParams()
    netParams.popParams[cell_name_ID] = {'cellType': cell_name_ID, 'cellModel': 'HH_full', 'numCells': 1,}
    reach_out_for_func(netParams.loadCellParamsRule, label = cell_name_ID, fileName = f'cells/{cell_name_ID}_cellParams.json')
    netParams.cellParams[cell_name_ID]['conds']['cellType'] = cell_name_ID

    if syn_params != None:
        # Add synapses
        # Stimulation parameters
        netParams.synMechParams["exc"] = {
            "mod": "Exp2Syn",
            "tau1": 0.2,
            "tau2": 5.0,
            "e": 0,
        }  # excitatory synaptic mechanism

        netParams.stimSourceParams['bkg'] = {
                'type': 'NetStim', 
                'rate': syn_params['rate'],
                'noise': 1.0
            }

        netParams.stimTargetParams['bkg->soma'] = {
                'source': 'bkg',
                'conds': {'pop': cell_name_ID},
                'sec': 'soma',                  #
                'weight': syn_params['weight'],
                'delay': 0,                     #
                'synMech': 'exc',               #
                'synsPerConn': 1                #
            }
    
    return netParams    


def sim_setup(cfg, netParams):
    # Create network of cells to simulate in a batch (single cell in this case)
    sim.initialize(
        simConfig = cfg, 	
        netParams = netParams)  				# create network object and set cfg and net params
    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells();              		# instantiate network cells based on defined populations
    sim.net.connectCells()            			# create connections between cells based on params
    sim.net.addStims() 							# add network stimulation
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
    sim.net.defineCellShapes()

def run_simulation(interval_func=None):
    if interval_func == None:
        find_steady_state()
    else:
        runSimWithIntervalFunc('dt', interval_func, func_first=True)
    sim.gatherData()                  			# gather spiking data and cell info from each node


# From tmsneurosim/nrn/simulation/simulation.py/Simulation._post_finitialize()
def find_steady_state():
        h.t = -1e11
        h.dt = 1e9

        while h.t < -h.dt:
            h.fadvance()

        h.t = 0
        h.fcurrent()
        h.frecord_init()


def save_results(save_data, save_folder, fname):
    try: os.mkdir(save_folder) 
    except: pass

    folder = os.listdir(save_folder)
    fnum = len([f for f in folder if fname in f])
    save_name = f'{fname}_run{fnum}'

    try: os.mkdir(f'{save_folder}/{save_name}') 
    except: pass

    with open(f'{save_folder}/{save_name}/{save_name}_results.json', 'w') as f:
        f.write(json.dumps(save_data, indent=4))


