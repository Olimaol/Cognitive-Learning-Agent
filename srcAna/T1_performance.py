
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


"""

Load 8 simulations (main experiment)
print for each simulation:
SimID: Performance(mean,std) rejcetedBlocks maxMinDist


"""


#load data
folder1='2020_09_23_mainExperiment_final'
folder2='2020_09_23_mainExperiment_final'
simIDs1=[1,2,3,4]
simIDs2=[5,6,7,8]

Performance=np.zeros((8,2))
rejectedBlocks=np.zeros(8)
maxMinDist=np.zeros(8)
for folderIdx,folder in enumerate([folder1,folder2]):
	simIDs=[simIDs1,simIDs2][folderIdx]
	for simIdx,sim in enumerate(simIDs):
		idx=simIdx+[0,4][folderIdx]
		#Performance
		selection=np.load('../data/'+folder+'/selection'+str(sim)+'.npy')
		startTime=selection[:,0]
		decisionTime=selection[:,1]
		correctOrientation=selection[:,2]
		decisionOrientation=selection[:,3]
		correct=selection[:,4]
		stimulus=selection[:,5]
		blockList=selection[:,6]
		block_Performance=np.zeros(10)
		for block in range(int(np.max(blockList))):	
			block_selections=decisionOrientation[blockList==block+1]
			block_stimuli=stimulus[blockList==block+1]
			block_t1selections=block_selections[block_stimuli==1]
			block_Performance[block]=np.sum((block_t1selections==55).astype(int))/float(block_t1selections.shape[0])
		Performance[idx,0]=np.mean(block_Performance)
		Performance[idx,1]=np.std(block_Performance)
		rejectedBlocks[idx]=np.sum((block_Performance<0.8).astype(int))

		#maxMinDist
		dist=np.load('../data/'+folder+'/distList'+str(sim)+'.npy', allow_pickle=True)
		for distList in dist:
			if min(distList)>maxMinDist[idx]:
				maxMinDist[idx]=min(distList)


with open('T1_performance.txt', 'w') as f:
	print('Sim Nr   Performance    rejected   maxMinDist', file=f)
	for folderIdx,folder in enumerate([folder1,folder2]):
		simIDs=[simIDs1,simIDs2][folderIdx]
		for simIdx,sim in enumerate(simIDs):
			idx=simIdx+[0,4][folderIdx]
			print('Sim '+str(sim)+'   ('+str(round(Performance[idx,0],4))+', '+str(round(Performance[idx,1],4))+')    '+str(rejectedBlocks[idx])+'   '+str(maxMinDist[idx]), file=f)
		
		







































