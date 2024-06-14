from neuron import h
import os, pathlib
import sys
import tmsneurosim
import tmsneurosim.nrn
from tmsneurosim.nrn import cells
from tmsneurosim.nrn.cells import NeuronCell

def loadCell(cellName, cellTemplateName):
    origDir = os.getcwd()
    os.chdir('WeiseEtAl2023/cells/'+cellName+'/')
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")    
    h.load_file("template.hoc")
    # h.xopen('template.hoc')
    # Instantiate the cell from the template
    
    add_synapses=False    
    cell = getattr(h, cellTemplateName)(1 if add_synapses else 0)
    
    os.chdir(origDir)
    
    return cell
    
def loadCell_Net(cellName, cellTemplateName):
    origDir = os.getcwd()
    os.chdir('WeiseEtAl2023/cells/'+cellName+'/')
    h.load_file("stdrun.hoc")
    h.load_file('import3d.hoc')
    try:
        h.xopen("morphology.hoc")
    except:
        pass
    try:
        h.xopen("biophysics.hoc")
    except:
        pass
    try:
        h.xopen("synapses/synapses.hoc")
    except:
        pass
    h.xopen('template.hoc')
    
    print(f'{cellTemplateName} in h: ' + str(cellTemplateName in dir(h)))
    cell = getattr(h, cellTemplateName)(0)
    
    print (cell)
    os.chdir(origDir)
    return cell

def loadTemplateName_tmsneurosim(cellFolder):     
    cells_path = pathlib.Path(cells.__file__).parent
    cell_path = cells_path / 'cells_hoc' / cellFolder
    f = open(cell_path / 'template.hoc', 'r')
    for line in f.readlines():
        if 'begintemplate' in line:
            templatename = str(line)     
    templatename=templatename[14:-1]    
    return templatename

def loadCell_Net_adv(cellName, id):
    cellClass = {
        'L1_NGC-DA': cells.L1_NGC_DA_bNAC,
        'L23_PC': cells.L23_PC_cADpyr,
        'L23_SBC': cells.L23_SBC_bNAC,
        'L4_LBC_cAC': cells.L4_LBC_cACint,
        'L4_LBC_cNAC': cells.L4_LBC_cNAC,
        'L4_MC': cells.L4_MC_bNAC,
        'L4_SS': cells.L4_SS_cADpyr,
        'L5_LBC': cells.L5_LBC_cNAC,
        'L5_TTPC2': cells.L5_TTPC2_cADpyr,
        'L6_TPC': cells.L6_TPC_L4_cADpyr
    }
    
    #cellTemplateName = loadTemplateName_tmsneurosim(cellFolder)
    print(cellName)
    cell = cellClass[cellName](id)
    cell.load()
    #print(f'{cellTemplateName} in h: ' + str(cellTemplateName in dir(h)))
    #hcell = getattr(h, cellTemplateName)(0)

    #print(hcell)
    return cell

# from neuron import h
# import os

# h.load_file("stdrun.hoc")
# h.load_file("import3d.hoc")    
# os.chdir(rootFolder)
# os.chdir('WeiseEtAl2023/')
# # h.load_file("template.hoc")
# h.load_file("nrngui.hoc")
# h.load_file("interpCoordinates.hoc")
# h.load_file("setPointers.hoc")
# h.load_file("calcVe.hoc")
# h.load_file("stimWaveform.hoc")
# h.load_file("cellChooser.hoc")
# h.load_file("setParams.hoc")
# h.load_file("editMorphology.hoc")
# os.chdir('cells/'+cellName+'/')
# h.load_file("createsimulation.hoc")
# # Instantiate the cell from the template
# add_synapses=False
# cell = getattr(h, cellTemplateName)(1 if add_synapses else 0)
# print (cell)