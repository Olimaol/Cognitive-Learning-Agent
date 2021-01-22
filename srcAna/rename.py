import numpy as np
import glob, os

def getFiles(f,s):
    return list(glob.glob(globalFolder+f+'/*'+str(s)+'.*'))

def renameNPY(f1,ff,f2,s2):
    loadedFile = np.load(globalFolder+f1+'/'+ff, allow_pickle=True)
    ff=list(ff)
    ff[-5]=str(s2)
    ff=''.join(ff)
    np.save(globalFolder+f2+'/'+ff,loadedFile)

def renameTXT(f1,ff,f2,s2):
    
    loadedFile = open(globalFolder+f1+'/'+ff, 'r')
    ff=list(ff)
    ff[-5]=str(s2)
    ff=''.join(ff)
    saveFile = open(globalFolder+f2+'/'+ff, 'w')

    Lines = loadedFile.readlines()
    saveFile.writelines(Lines)

    loadedFile.close()
    saveFile.close()

def createFolder(f):
    try:
        os.makedirs(f)
    except:
        if os.path.isdir(f)==False:
            print('could not create '+f)


globalFolder = '../data/'

# List of source folder and simulation IDs + target folder and target simulation IDs
renameList=[    ['2020_09_17_mainExperiment_T1', [3], '2020_09_23_mainExperiment_final', [1]],
                ['2020_09_16_mainExperiment_T1', [1,2], '2020_09_23_mainExperiment_final', [2,3]],
                ['2020_09_21_mainExperiment_T1', [7], '2020_09_23_mainExperiment_final', [4]],
                ['2020_09_16_mainExperiment_T1rev', [6], '2020_09_23_mainExperiment_final', [5]],
                ['2020_09_16_mainExperiment_T1rev', [2,3], '2020_09_23_mainExperiment_final', [6,7]],
                ['2020_09_25_mainExperiment_T1rev', [2], '2020_09_23_mainExperiment_final', [8]]]


for preFolder, preSims, postFolder, postSims in renameList:
    createFolder(globalFolder+postFolder)
    for simIdx, preSim in enumerate(preSims):
        files=getFiles(preFolder,preSim)
        for file in files:
            file=file.split('/')[-1]
            if '.npy' == file[-4:]:
                renameNPY(preFolder,file,postFolder,postSims[simIdx])
            elif '.txt' == file[-4:]:
                renameTXT(preFolder,file,postFolder,postSims[simIdx])
            else:
                print('unknown data type '+file)
                quit()










































        
    
