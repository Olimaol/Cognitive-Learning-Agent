
import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


"""

Load X simulations (training periods)
get explored gains for the training periods

Histogramm of explored gains

Histogramm of last (learned, successful training periods) gains


"""

def gettrialGain(sel,rPFC,trialMax,mode=0):
	startTime=sel[0:trialMax+1,0].astype(int)
	decisionTime=sel[0:trialMax+1,1].astype(int)
	trialGain=np.zeros(startTime.shape[0])
	for trial in range(startTime.shape[0]):
		trial_rPFC=np.clip(rPFC[startTime[trial]:decisionTime[trial]],0,None)
		trial_rPFC_mean=np.mean(trial_rPFC,0)
		orientations=np.array([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85])
		trialGain[trial]=orientations[np.argmax(trial_rPFC_mean)]
	if mode==0:
		return np.unique(trialGain)##multiple identical selections combined  [50,50,10,20,20,30] --> [10,20,30,50]
	elif mode==1:
		return trialGain[-1]
	elif mode==2:
		return trialGain

def getData(folder, simIDs):
	gain=[]
	succsesful_trainings=0
	training_dauer=[]
	for sim in simIDs:
		selection=np.load('../data/'+folder+'/selection'+str(sim)+'.npy')
		rPFC=np.load('../data/'+folder+'/rPFC'+str(sim)+'.npy')
		correct=selection[:,4]
		
		#Selections until trial in which rewarded was learned
		rewardedFound=0
		for i in np.arange(20,correct.shape[0]):
			if sum(correct[i-20:i]==1)>=16:
				trialMax=i
				rewardedFound=1
				break
			else:
				rewardedFound=0
			
		if rewardedFound==1:
			gain.append(gettrialGain(selection,rPFC,trialMax))
			succsesful_trainings+=1
			training_dauer.append(trialMax)

	all_gains = [item for sublist in gain for item in sublist]
	(all_gains_hist, bin_edges)=np.histogram(all_gains,bins=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,1000])

	return (all_gains_hist,succsesful_trainings,training_dauer)


def getLastGain(folder, simIDs):
	gain=[]
	succsesful_trainings=0
	training_dauer=[]
	for sim in simIDs:
		selection=np.load('../data/'+folder+'/selection'+str(sim)+'.npy')
		rPFC=np.load('../data/'+folder+'/rPFC'+str(sim)+'.npy')
		correct=selection[:,4]
		
		#Selections until trial in which rewarded was learned
		rewardedFound=0
		for i in np.arange(20,correct.shape[0]):
			if sum(correct[i-20:i]==1)>=16:
				trialMax=i
				rewardedFound=1
				break
			else:
				rewardedFound=0
			
		if rewardedFound==1:
			gain.append(gettrialGain(selection,rPFC,trialMax,1))
			succsesful_trainings+=1
			training_dauer.append(trialMax)

	all_gains = gain
	(all_gains_hist, bin_edges)=np.histogram(all_gains,bins=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,1000])

	return all_gains_hist

def add_gains(g):
	ori=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
	ret=np.zeros(16)
	for i,orientation in enumerate(ori):
		if orientation==g:
			ret[i]=1
	return ret

def add_correct(g,c):
	ori=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
	ret=np.zeros(16)
	for i,orientation in enumerate(ori):
		if orientation==g and c==1:
			ret[i]=1
	return ret

def getPerformance(folder, simIDs):

	count_gains=np.zeros(16)
	count_correct=np.zeros(16)

	for sim in simIDs:
		selection=np.load('../data/'+folder+'/selection'+str(sim)+'.npy')
		rPFC=np.load('../data/'+folder+'/rPFC'+str(sim)+'.npy')
		correct=selection[:,4]
	
		#Selections until trial in which rewarded was learned
		rewardedFound=0
		for i in np.arange(19,correct.shape[0]):
			if sum(correct[i-19:i+1]==1)>=16:
				trialMax=i
				rewardedFound=1
				break
			else:
				rewardedFound=0
		
		if rewardedFound==1:
			gains=gettrialGain(selection,rPFC,trialMax,2)
			for idx in range(trialMax):
				count_gains+=add_gains(gains[idx])
				count_correct+=add_correct(gains[idx],correct[idx])

	return count_correct/count_gains	



###  Plot general paramters
degSign=u"\u00b0"

font1={'fontsize': 7,
 'fontweight' : 'normal'}
font2={'fontsize': 10,
 'fontweight' : 'normal'}
font1p={'size': 7,
 'weight' : 'normal'}#p for legends

figB=95#mm
figH=105#mm
figRes=500#dpi

### Plot specific parameters
folder='2020_09_21_TrainingsphasenT1'

simIDs=range(1,61)
simAnz=len(simIDs)

histB=60#mm
histH=30#mm

PosX=[15,50]#mm
PosY=[25,25]#mm

yMaxSelection=simAnz

col=['C0', 'C1']

### load data
data=getData(folder,simIDs)

with open('Training.txt', 'w') as f:
    print("with Bias", file=f)
    print(data[1], "/ "+str(simAnz)+" erfolgreich = "+str(data[1]/float(simAnz)), file=f)
    print("Anzahl Trials mean=",np.mean(data[2])," std=",np.std(data[2])," var=",np.var(data[2]), file=f)

successfulSims=data[1]
data=data[0]
lastGain=getLastGain(folder, simIDs)


#General figure
plt.figure(1,figsize=(figB*0.03937007874,figH*0.03937007874))
plt.subplots_adjust(top=1,bottom=0,left=0,right=1.0,hspace=0,wspace=0)


ax1=plt.subplot2grid((figH,figB),(PosX[0],PosY[0]), colspan=histB, rowspan=histH)
ax2=plt.subplot2grid((figH,figB),(PosX[1],PosY[1]), colspan=histB, rowspan=histH)


#Barplot up
ax1.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], data, align='center', alpha=0.5, edgecolor='black',facecolor=col[0])
ax1.set_ylim(0,successfulSims)
ax1.set_xticks([1,3,5,7,9,11,13,15])
ax1.set_xticklabels([], **font1)
ax1.set_yticks([0,successfulSims])
ax1.set_yticklabels([0,successfulSims], **font1)
ax1.set_ylabel('# tested', va='center', **font1)


#Barplot below
ax2.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], lastGain, align='center', alpha=0.5, edgecolor='black',facecolor=col[1])
ax2.set_ylim(0,successfulSims)
ax2.set_xticks([1,3,5,7,9,11,13,15])
ax2.set_xticklabels([str(i) for i in [10,20,30,40,50,60,70,80]], **font1)
ax2.set_yticks([0,successfulSims])
ax2.set_yticklabels([0,successfulSims], **font1)
ax2.set_xlabel('PFC-neuron / orientation', **font1)
ax2.set_ylabel('# learned', va='center', **font1)


with open('Training.txt', 'a') as f:
    print('\nBarplots:', file=f)
    print(data, np.sum(data), file=f)
    print(lastGain, np.sum(lastGain), file=f)
    print(np.arange(10,86,5), file=f)


plt.savefig('Training.svg', dpi=figRes)























