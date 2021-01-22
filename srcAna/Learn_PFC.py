
import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import itertools


"""

Load 1 simulation
Plot weights ITPFC and VAPFC

"""


###  Plot general paramters
degSign=u"\u00b0"

font1={'fontsize': 7,
 'fontweight' : 'normal'}
font2={'fontsize': 11,
 'fontweight' : 'normal'}
font1p={'size': 7,
 'weight' : 'normal'}#p for legends

scaling=95/150

figB=int(150*scaling)#mm
figH=int(60*scaling)#mm
figRes=1000#dpi


###  Plot specific parameters
simulations=[['2020_09_16_mainExperiment_T1', [1,2,3,4,6,8]], ['2020_09_17_mainExperiment_T1', [4,5,7,8]], ['2020_09_21_mainExperiment_T1', [1,2,3,5]]]
num_simulations=len(list(itertools.chain.from_iterable([simulations[i][1] for i in range(len(simulations))])))

plotB=int(40*scaling)#
plotH=int(30*scaling)#

colBarB=int(3*scaling)
colBarH=int(20*scaling)

yLabelB=int(85*scaling)
yLabelH=int(5*scaling)

PosX=[int(10*scaling)]*2#mm
PosY=[int(25*scaling),int(70*scaling)]#mm

colBarX=int(15*scaling)
colBarY=int(115*scaling)

yLabelX=int(49*scaling)
yLabelY=int(25*scaling)

orientations = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
cols=['k','k','k','k','k','k','k','k','k','k','k','C1','k','k','k','k']
labels=['_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','65'+degSign,'_nolegend_','_nolegend_','_nolegend_','others']

maxTrial=500



#General figure
plt.figure(1,figsize=(figB*0.03937007874,figH*0.03937007874), dpi=figRes)
plt.subplots_adjust(top=1,bottom=0,left=0,right=1.0,hspace=0,wspace=0)

ax1=plt.subplot2grid((figH,figB),(PosX[0],PosY[0]), colspan=plotB, rowspan=plotH)
ax2=plt.subplot2grid((figH,figB),(PosX[1],PosY[1]), colspan=plotB, rowspan=plotH)


axyLabel=plt.subplot2grid((figH,figB),(yLabelX,yLabelY), colspan=yLabelB, rowspan=yLabelH)

#Load data
w_ITPFC=np.zeros((maxTrial,16))
w_VAPFC=np.zeros((maxTrial,16))
for folder, simIDs in simulations:
	for simID in simIDs:
		w_ITPFC_temp=np.load('../data/'+folder+'/w_ITPFC'+str(simID)+'.npy')
		w_VAPFC_temp=np.load('../data/'+folder+'/w_VAPFC'+str(simID)+'.npy')
		trials=0	
		for idx in range(w_ITPFC_temp.shape[0]):
			if w_ITPFC_temp[idx,:,:].sum()!=0:
				trials+=1

		w_ITPFC_temp=np.sum(w_ITPFC_temp,2)

		w_ITPFC+=w_ITPFC_temp[:maxTrial,:]/float(num_simulations)
		w_VAPFC+=w_VAPFC_temp[:maxTrial,:,0]/float(num_simulations)

print('min ',w_ITPFC.min(),w_VAPFC.min())
print('max ',w_ITPFC.max(),w_VAPFC.max())

for idx, orientation in enumerate(orientations):
	ax1.plot(w_ITPFC[:,idx], color=cols[idx], label=labels[idx])
	ax2.plot(w_VAPFC[:,idx], color=cols[idx], label=labels[idx])

for ax in [ax1,ax2]:
	ax.set_yticks([0,0.5,1])
	ax.set_yticklabels([0.0,0.5,1.0], **font1)
	ax.set_xticks([0,maxTrial])
	ax.set_xticklabels([0,maxTrial], **font1)
ax2.set_yticklabels([])
ax1.set_ylabel('weights', **font1)


### Legend
ax2.legend(ncol=1,bbox_to_anchor=(1.5, 0.5), loc=10, borderaxespad=0., prop=font1p)

### ylabel
axyLabel.text(0.5, 0.5, 'Trials', ha='center', va='center', **font1)
axyLabel.axis('off')

ax1.set_title('IT - PFC', **font1)
ax2.set_title('Thalamus - PFC', **font1)





plt.savefig('Learn_PFC.svg', dpi=figRes)



