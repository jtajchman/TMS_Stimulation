import numpy as np
from neuron import h
from .tms_efield import get_efield

def collect_pt3d(cell, section):
        """        collect the pt3d info, for each section
        """
        n3dsec = 0
        r3dsec = np.zeros(3)
        for sec in [sec for secName, sec in cell.secs.items() if section in secName]:
            sec['hObj'].push()
            n3d = int(h.n3d())  # get number of n3d points in each section
            # print("get number of n3d points in each section",n3d)
            # r3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            n3dsec += n3d

            for i in range(n3d):
                r3dsec[0] += h.x3d(i)
                r3dsec[1] += h.y3d(i)
                r3dsec[2] += h.z3d(i)
            
            h.pop_section()

        r3dsec /= n3dsec
        
        return r3dsec
    
def getSecsPos(cell, secList):
        """        Get Secs position
        """
        x3d, y3d, z3d = [], [], []
        
        for secName in secList:
            # print(secName)
            r3dsec = collect_pt3d(cell, secName)
            # print(secName, r3dsec)
            
            x3d.append(r3dsec[0])
            y3d.append(r3dsec[1])
            z3d.append(r3dsec[2])
            
        return np.array([x3d, y3d, z3d])
    
def insert_v_ext(cell, v_ext, t_ext):

    cell.t_ext = h.Vector(t_ext)
    cell.v_ext = []
    for v in v_ext:
        cell.v_ext.append(h.Vector(v))
    
    # play v_ext into e_extracellular reference
    i = 0
    for sec in cell.secs.values(): 
        for seg in sec['hObj']:
                cell.v_ext[i].play(seg._ref_e_extracellular, cell.t_ext)
        i += 1
  
def make_extracellular_stimuli(params, cell, secList):
    """ Function to calculate and apply external potential """
    
    decay_constant = params['sigma']
    field_direction = params['field_direction']
    field_norm = np.linalg.norm(field_direction)
    if field_norm != 0:
        field_direction /= field_norm

    ext_field_scalar = np.vectorize(lambda x, y, z: # um coordinates
                            (1-decay_constant) ** np.dot(np.array([x, y, z])/1000, field_direction))

    # MAKING THE EXTERNAL FIELD    
    pulse, t = get_efield(freq=params['frequency'], 
                          duration=params['stimend'], 
                          dt=params['dt'],
                          tstart=params['stimstart'],
                          ef_amp=params['amp'], 
                          width=1,)
    n_tsteps = len(t)

    totnsegs = len(secList)    
    v_cell_ext = np.zeros((totnsegs, n_tsteps))
    x, y, z = getSecsPos(cell, secList)
    v_cell_ext[:, :] = ext_field_scalar(x, y, z).reshape(totnsegs, 1) * np.array(pulse).reshape(1, n_tsteps)
    
    insert_v_ext(cell, v_cell_ext, t)

    return v_cell_ext, cell

def make_tms_stim(net, params):
    v_exts = []
    cells = []
    for cell in net.cells:
        secList = [secs for secs in cell.secs.keys() if "pt3d" in cell.secs[secs]['geom']]
        v_cell_ext, cell = make_extracellular_stimuli(params, cell, secList)
        v_exts.append(v_cell_ext)
        cells.append(cell)
    return v_exts, cells