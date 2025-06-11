from neuron import h
from netpyne import specs, sim
import os

def shape_movie(saveFolder, recordStart, recordEnd, cellName):
    dir1 = saveFolder
    try: os.mkdir(dir1) 
    except: pass
    dir2 = f'{dir1}/voltage_movie'
    try: os.mkdir(dir2) 
    except: pass
    run_num = len(os.listdir(dir2))
    dir3 = f'{dir2}/run_{run_num}'

    os.mkdir(dir3)
    dir4 = f'{dir3}/images'
    os.mkdir(dir4)

    def movie_func(simTime):
        t = simTime

        if t >= recordStart and t <= recordEnd:
            # print(t)

            sim.gatherData()
            # sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
            # sim.analysis.plotData()         			# plot spike raster etc

            sim.analysis.plotShape(includePre=[cellName], includePost=[cellName], includeAxon=True, showSyns=False, showElectrodes=False,
                                    cvar= 'voltage', dist=0.6, elev=95, azim=-90, 
                                    axisLabels=True, synStyle='o', 
                                    clim= [-62, 54], showFig=False, synSize=2, saveFig='movie', figSize=(12,12))#f'{dir4}/t{round(t, 3)}sec.png'
    return dir4, movie_func


def range_movie(recordStart, recordEnd, seg1, seg2):
    range_frames = []
    def movie_func(t):
        if t >= recordStart and t <= recordEnd:
            rvp = h.RangeVarPlot('v', seg1, seg2)
            range_frames.append(list(rvp.vector()))
    return range_frames, movie_func