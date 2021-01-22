
import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


"""

Laedt 8 simulations (main experiment, 4x T1 trials, 4x T1-reversed trials)
generate histogram for T2 selections
T2 Selections: target und distractor compared with paired t-test


"""

def getGoodBlocks(sel):
	#Performance
	selection=sel
	startTime=sel[:,0]
	decisionTime=sel[:,1]
	correctOrientation=sel[:,2]
	decisionOrientation=sel[:,3]
	correct=sel[:,4]
	stimulus=sel[:,5]
	blockList=sel[:,6]
	block_Performance=np.zeros(10)
	for block in range(int(np.max(sel[:,6]))):	
		block_selections=decisionOrientation[blockList==block+1]
		block_stimuli=stimulus[blockList==block+1]
		block_t1selections=block_selections[block_stimuli==1]
		block_Performance[block]=np.sum((block_t1selections==55).astype(int))/float(block_t1selections.shape[0])
	return block_Performance>=0.8

def getData(sel,nr,cond):
	if sum(sel[:,5]==2)>0 and np.sum(getGoodBlocks(sel).astype(int))>=3:
		print('Simulation '+str(nr), file=f)
		good=getGoodBlocks(sel)
		startTime=sel[:,0]
		selectionTime=sel[:,1]
		correctOrientation=sel[:,2]
		selectedOrientation=sel[:,3]
		correct=sel[:,4]
		stimulus=sel[:,5]
		blockList=sel[:,6]

		hist_block_data=np.zeros((10,5))
		for block in range(10):
			mask=(blockList==block+1)*(stimulus==2)
			block_t2_selections=selectedOrientation[mask]
			(hist_block_data[block], bin_edges)=np.histogram(block_t2_selections,bins=[30,50,55,60,80,1000])
		hist_block_data=hist_block_data[good,:]

		hist_sim_mean=np.mean(hist_block_data,0)
		hist_sim_std=np.std(hist_block_data,0)
	
		#Diference between target and special (indices: 0-30, 1-50, 2-55, 3-60, 4-80)
		if cond=='T1':
			(a,b)=(3,2)
		elif cond=='T1_reversed':
			(a,b)=(1,2)
		difs=hist_block_data[:,a]-hist_block_data[:,b]
		#normally distributed
		(Wval,pval)=stats.shapiro(difs)
		if pval>=0.05:
			print('normalverteilt', file=f)
		else:
			print('nicht normalverteilt!', file=f)	
		#paired ttest between target and special
		(tval,pval)=stats.ttest_rel(hist_block_data[:,a], hist_block_data[:,b])
		print('M1 = ',str(hist_block_data[:,a].mean()),' SD1 = ',str(hist_block_data[:,a].std()),' M2 = ',str(hist_block_data[:,b].mean()),' SD2 = ',str(hist_block_data[:,b].std()),' t('+str(np.sum(good.astype(int))-1)+')='+str(tval)+', p='+str(pval)+'\n', file=f)
	
		return np.array([hist_sim_mean,hist_sim_std])
	elif sum(sel[:,5]==2)==0:
		print('Simulation '+str(nr), file=f)
		print('keine Testphase', file=f)
		print('', file=f)
		return np.zeros((2,5))
	else:
		print('Simulation '+str(nr), file=f)
		print('zu wenig Bloecke '+str(np.sum(getGoodBlocks(sel).astype(int)))+' mit ausreichend Perfromance', file=f)
		print('', file=f)
		return np.zeros((2,5))


def getGain(folder,sim):

	
	selection=np.load('../data/'+folder+'/selection'+str(sim)+'.npy')
	
	if sum(selection[:,5]==2)>0 and np.sum(getGoodBlocks(selection).astype(int))>0:
		rPFC=np.load('../data/'+folder+'/rPFC'+str(sim)+'.npy')
		good=getGoodBlocks(selection)
		startTime=selection[:,0].astype(int)
		decisionTime=selection[:,1].astype(int)
		stimulus=selection[:,5]
		trialBlock=selection[:,6]

		startTime=startTime[stimulus==2]
		decisionTime=decisionTime[stimulus==2]
		trialBlock=trialBlock[stimulus==2]
		
		for block in range(10):
			if good[block]==False:
				trialBlock[trialBlock==(block+1)]=0

		startTime=startTime[trialBlock>0]
		decisionTime=decisionTime[trialBlock>0]

		meanOrientationGain=np.zeros(startTime.shape[0])
		for trial in range(startTime.shape[0]):
			trial_rPFC=np.clip(rPFC[startTime[trial]:decisionTime[trial]],0,None)
			trial_rPFC_mean=np.mean(trial_rPFC,0)
			orientations=np.array([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85])
			weights=trial_rPFC_mean
			weighted_sum=np.sum(trial_rPFC_mean*orientations)
			meanOrientationGain[trial]=weighted_sum/np.sum(weights)

			

		print('Sim '+str(sim)+':   '+str(np.mean(meanOrientationGain))+'   '+str(np.std(meanOrientationGain)), file=f)
	else:
		print('Sim '+str(sim)+':   -   -', file=f)



###  plot general paramters
degSign=u"\u00b0"

font1={'fontsize': 7,
 'fontweight' : 'normal'}
font2={'fontsize': 11,
 'fontweight' : 'normal'}
font1p={'size': 7,
 'weight' : 'normal'}#p for legends

figB=170#mm
figH=120#mm
figRes=500#dpi

### Plot specific parameters
folder1='2020_09_23_mainExperiment_final'
folder2='2020_09_23_mainExperiment_final'

f=open('T2.txt', 'w')

simIDs1 = [1,2,3,4]
simIDs2 = [5,6,7,8]

histB=30#mm
histH=25#mm

legB=135#mm
legH=5#mm

buchstabenB=5#mm
buchstabenH=25#mm

PosX1=[15,15,15,15]#mm
PosY1=[25,60,95,130]#mm

PosX2=[58,58,58,58]#mm
PosY2=[25,60,95,130]#mm

xLeg=95
yLeg=25

xBuchstaben=[15,58]
yBuchstaben=[10,10]

yMaxSelection=20



#General Figure
plt.figure(1,figsize=(figB*0.03937007874,figH*0.03937007874), dpi=figRes)
plt.subplots_adjust(top=1,bottom=0,left=0,right=1.0,hspace=0,wspace=0)

axLegend=plt.subplot2grid((figH,figB),(xLeg,yLeg), colspan=legB, rowspan=legH)
ax1Buchstaben=plt.subplot2grid((figH,figB),(xBuchstaben[0],yBuchstaben[0]), colspan=buchstabenB, rowspan=buchstabenH)
ax2Buchstaben=plt.subplot2grid((figH,figB),(xBuchstaben[1],yBuchstaben[1]), colspan=buchstabenB, rowspan=buchstabenH)

ax1=plt.subplot2grid((figH,figB),(PosX1[0],PosY1[0]), colspan=histB, rowspan=histH)
ax2=plt.subplot2grid((figH,figB),(PosX1[1],PosY1[1]), colspan=histB, rowspan=histH)
ax3=plt.subplot2grid((figH,figB),(PosX1[2],PosY1[2]), colspan=histB, rowspan=histH)
ax4=plt.subplot2grid((figH,figB),(PosX1[3],PosY1[3]), colspan=histB, rowspan=histH)

ax5=plt.subplot2grid((figH,figB),(PosX2[0],PosY2[0]), colspan=histB, rowspan=histH)
ax6=plt.subplot2grid((figH,figB),(PosX2[1],PosY2[1]), colspan=histB, rowspan=histH)
ax7=plt.subplot2grid((figH,figB),(PosX2[2],PosY2[2]), colspan=histB, rowspan=histH)
ax8=plt.subplot2grid((figH,figB),(PosX2[3],PosY2[3]), colspan=histB, rowspan=histH)

#########################################################
#####################   ROW 1   #########################
#########################################################

#Colors of bars
target=3
distractor=2
cols=[None]*5
for i in range(5):
	if i+1==target:
		cols[i]='White'
	elif i+1==distractor:
		cols[i]=(182/255.,185/255.,192/255.)
	else:
		cols[i]=(112/255.,142/255.,197/255.)

#Load data
selection=np.array([ np.load('../data/'+folder1+'/selection'+str(simIDs1[i])+'.npy') for i in range(4)])
data=np.array([getData(selection[i],simIDs1[i],'T1') for i in range(4)])

#Barplots
for i,ax in enumerate([ax1,ax2,ax3,ax4]):
	ax.bar([1,2,3,4,5],data[i,0,:],yerr=data[i,1,:], align='center', ecolor='black', capsize=2, color=cols, edgecolor='black')
	ax.set_ylim(0,yMaxSelection)
	ax.set_xticks([1,2,3,4,5])
	ax.set_xticklabels(['30'+degSign,'50'+degSign,'55'+degSign,'60'+degSign,'80'+degSign], **font1)
	ax.set_yticks([0,yMaxSelection-2])

ax1.set_yticklabels([0,yMaxSelection-2], **font1)
ax2.set_yticklabels([], **font1)
ax3.set_yticklabels([], **font1)
ax4.set_yticklabels([], **font1)
ax1.set_ylabel('# Selection', va='center', **font1)
ax1.text(-1.5, 22, 'T=55'+degSign+', D=50'+degSign, bbox=dict(facecolor=(245/255.,226/255.,178/255.), edgecolor='grey'), **font1)



#########################################################
#####################   ROW 2   #########################
#########################################################

#Colors of bars
target=3
distractor=4
cols=[None]*5
for i in range(5):
	if i+1==target:
		cols[i]='White'
	elif i+1==distractor:
		cols[i]=(182/255.,185/255.,192/255.)
	else:
		cols[i]=(112/255.,142/255.,197/255.)

#Load data
selection=np.array([ np.load('../data/'+folder2+'/selection'+str(simIDs2[i])+'.npy') for i in range(4)])
data=np.array([getData(selection[i],simIDs2[i],'T1_reversed') for i in range(4)])

#Barplots
for i,ax in enumerate([ax5,ax6,ax7,ax8]):
	ax.bar([1,2,3,4,5],data[i,0,:],yerr=data[i,1,:], align='center', ecolor='black', capsize=2, color=cols, edgecolor='black')
	ax.set_ylim(0,yMaxSelection)
	ax.set_xticks([1,2,3,4,5])
	ax.set_xticklabels(['30'+degSign,'50'+degSign,'55'+degSign,'60'+degSign,'80'+degSign], **font1)
	ax.set_yticks([0,yMaxSelection-2])

ax5.set_yticklabels([0,yMaxSelection-2], **font1)
ax6.set_yticklabels([], **font1)
ax7.set_yticklabels([], **font1)
ax8.set_yticklabels([], **font1)
ax5.set_ylabel('# Selection', va='center', **font1)
ax5.text(-1.5, 22, 'T=55'+degSign+', D=60'+degSign, bbox=dict(facecolor=(245/255.,226/255.,178/255.), edgecolor='grey'), **font1)

#Legend
axLegend.axis('off')
values = ['Target', 'Distractor', 'Others']
colors = ['white', (182/255.,185/255.,192/255.), (112/255.,142/255.,197/255.)]
patches = [ mpatches.Patch(facecolor=colors[i], edgecolor='black', label=("{l}").format(l=values[i]) ) for i in range(3)   ]
axLegend.legend(handles=patches, bbox_to_anchor=(0.5, 0.5), loc=10, borderaxespad=0., prop=font1p, ncol=3)

#Letters
ax1Buchstaben.text(-1, 0.5, 'A', **font2)
ax2Buchstaben.text(-1, 0.5, 'B', **font2)
ax1Buchstaben.axis('off')
ax2Buchstaben.axis('off')





print('T2 Gains:', file=f)
print('Sim Nr   Gain Mean   Gain Std', file=f)
for folderIdx, folder in enumerate([folder1, folder2]):
	for i in range(4):
		if folderIdx==0:
			sim=simIDs1[i]
		else:
			sim=simIDs2[i]
		getGain(folder,sim)

f.close()

plt.savefig('T2.svg', dpi=figRes)
































