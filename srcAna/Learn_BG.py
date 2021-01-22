
import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import itertools


"""

Load 1 simualtion (main experiment)
Plot weights IT-StrD1-GPi, IT-STN-GPi, IT-StrD2-GPe for different time points

"""


###  Plot general parameters
degSign=u"\u00b0"

font1={'fontsize': 7,
 'fontweight' : 'normal'}
font2={'fontsize': 10,
 'fontweight' : 'normal'}
font1p={'size': 7,
 'weight' : 'normal'}#p for legends


figB=145#mm
figH=120#mm
figRes=1000#dpi


###  Plot specific paramters
simulations=[['2020_09_16_mainExperiment_T1', [1,2,3,4,6,8]], ['2020_09_17_mainExperiment_T1', [4,5,7,8]], ['2020_09_21_mainExperiment_T1', [1,2,3,5]]]
num_simulations=len(list(itertools.chain.from_iterable([simulations[i][1] for i in range(len(simulations))])))

matB=20#post neurons = width
matH=20#pre neurons = height

PosX1=[10]*5#mm
PosY1=[25+i*(20+1) for i in range(5)]#mm
PosX2=[40]*5#mm
PosY2=[25+i*(20+1) for i in range(5)]#mm
PosX3=[70]*5#mm
PosY3=[25+i*(20+1) for i in range(5)]#mm

textB=5
textH=20
textX=[10,40,70]
textY=[10]*3

trialB=20
trialH=10
trialX=[100]*6
trialY=[10]+[25+i*(20+1) for i in range(5)]

cBarB=2
cBarH=14
cBarX=[PosX1[4]+matH//2-cBarH//2,PosX2[4]+matH//2-cBarH//2,PosX3[4]+matH//2-cBarH//2]
cBarY=[PosY1[4]+23,PosY2[4]+23,PosY3[4]+23]

orientations=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]





#General Figure
plt.figure(1,figsize=(figB*0.03937007874,figH*0.03937007874), dpi=figRes)
plt.subplots_adjust(top=1,bottom=0,left=0,right=1.0,hspace=0,wspace=0)

ax1=plt.subplot2grid((figH,figB),(PosX1[0],PosY1[0]), colspan=matB, rowspan=matH)
ax2=plt.subplot2grid((figH,figB),(PosX1[1],PosY1[1]), colspan=matB, rowspan=matH)
ax3=plt.subplot2grid((figH,figB),(PosX1[2],PosY1[2]), colspan=matB, rowspan=matH)
ax4=plt.subplot2grid((figH,figB),(PosX1[3],PosY1[3]), colspan=matB, rowspan=matH)
ax5=plt.subplot2grid((figH,figB),(PosX1[4],PosY1[4]), colspan=matB, rowspan=matH)

ax10=plt.subplot2grid((figH,figB),(PosX2[0],PosY2[0]), colspan=matB, rowspan=matH)
ax11=plt.subplot2grid((figH,figB),(PosX2[1],PosY2[1]), colspan=matB, rowspan=matH)
ax12=plt.subplot2grid((figH,figB),(PosX2[2],PosY2[2]), colspan=matB, rowspan=matH)
ax13=plt.subplot2grid((figH,figB),(PosX2[3],PosY2[3]), colspan=matB, rowspan=matH)
ax14=plt.subplot2grid((figH,figB),(PosX2[4],PosY2[4]), colspan=matB, rowspan=matH)

ax19=plt.subplot2grid((figH,figB),(PosX3[0],PosY3[0]), colspan=matB, rowspan=matH)
ax20=plt.subplot2grid((figH,figB),(PosX3[1],PosY3[1]), colspan=matB, rowspan=matH)
ax21=plt.subplot2grid((figH,figB),(PosX3[2],PosY3[2]), colspan=matB, rowspan=matH)
ax22=plt.subplot2grid((figH,figB),(PosX3[3],PosY3[3]), colspan=matB, rowspan=matH)
ax23=plt.subplot2grid((figH,figB),(PosX3[4],PosY3[4]), colspan=matB, rowspan=matH)

axText1=plt.subplot2grid((figH,figB),(textX[0],textY[0]), colspan=textB, rowspan=textH)
axText2=plt.subplot2grid((figH,figB),(textX[1],textY[1]), colspan=textB, rowspan=textH)
axText3=plt.subplot2grid((figH,figB),(textX[2],textY[2]), colspan=textB, rowspan=textH)

ax1trial=plt.subplot2grid((figH,figB),(trialX[0],trialY[0]), colspan=textB, rowspan=trialH)
ax2trial=plt.subplot2grid((figH,figB),(trialX[1],trialY[1]), colspan=trialB, rowspan=trialH)
ax3trial=plt.subplot2grid((figH,figB),(trialX[2],trialY[2]), colspan=trialB, rowspan=trialH)
ax4trial=plt.subplot2grid((figH,figB),(trialX[3],trialY[3]), colspan=trialB, rowspan=trialH)
ax5trial=plt.subplot2grid((figH,figB),(trialX[4],trialY[4]), colspan=trialB, rowspan=trialH)
ax6trial=plt.subplot2grid((figH,figB),(trialX[5],trialY[5]), colspan=trialB, rowspan=trialH)

ax1CBar=plt.subplot2grid((figH,figB),(cBarX[0],cBarY[0]), colspan=cBarB, rowspan=cBarH)
ax2CBar=plt.subplot2grid((figH,figB),(cBarX[1],cBarY[1]), colspan=cBarB, rowspan=cBarH)
ax3CBar=plt.subplot2grid((figH,figB),(cBarX[2],cBarY[2]), colspan=cBarB, rowspan=cBarH)

#Load data
wITSTNSNr=np.zeros((5,16,16))
wITStrD1SNr=np.zeros((5,16,16))
wITStrD2GPe=np.zeros((5,16,16))
for folder, simIDs in simulations:
    for simID in simIDs:
        w_ITStrD1=np.load('../data/'+folder+'/w_ITStrD1'+str(simID)+'.npy')
        w_ITStrD2=np.load('../data/'+folder+'/w_ITStrD2'+str(simID)+'.npy')
        w_ITSTN=np.load('../data/'+folder+'/w_ITSTN'+str(simID)+'.npy')
        w_StrD1SNr=np.load('../data/'+folder+'/w_StrD1SNr'+str(simID)+'.npy')
        w_StrD2GPe=np.load('../data/'+folder+'/w_StrD2GPe'+str(simID)+'.npy')
        w_STNSNr=np.load('../data/'+folder+'/w_STNSNr'+str(simID)+'.npy')

        trials=0	
        for idx in range(w_StrD1SNr.shape[0]):
	        if w_StrD1SNr[idx,:,:].sum()!=0:
		        trials+=1

        for idx,t in enumerate([0,24,48,200,500]):
	        w1=np.transpose(w_ITSTN[t])
	        w2=np.transpose(w_STNSNr[t])
	        wITSTNSNr[idx]+=np.matmul(w1, w2)/float(num_simulations)

	        w1=np.transpose(w_ITStrD1[t])
	        w2=np.transpose(w_StrD1SNr[t])
	        wITStrD1SNr[idx]+=np.matmul(w1, w2)/float(num_simulations)

	        w1=np.transpose(w_ITStrD2[t])
	        w2=np.transpose(w_StrD2GPe[t])
	        wITStrD2GPe[idx]+=np.matmul(w1, w2)/float(num_simulations)


plotMax=[wITSTNSNr.max(),wITStrD1SNr.max(),wITStrD2GPe.max()]
print(plotMax)


for idx,ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
	valW1=ax.imshow(wITSTNSNr[idx],vmin=0,vmax=plotMax[0],cmap='Purples')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.axvline(orientations.index(65)-0.5,linewidth=0.4, color='grey')
	ax.axvline(orientations.index(65)+0.5,linewidth=0.4, color='grey')

for idx,ax in enumerate([ax10,ax11,ax12,ax13,ax14]):
	valW2=ax.imshow(wITStrD1SNr[idx],vmin=0,vmax=plotMax[1],cmap='Purples')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.axvline(orientations.index(65)-0.5,linewidth=0.4, color='grey')
	ax.axvline(orientations.index(65)+0.5,linewidth=0.4, color='grey')

for idx,ax in enumerate([ax19,ax20,ax21,ax22,ax23]):
	valW3=ax.imshow(wITStrD2GPe[idx],vmin=0,vmax=plotMax[2],cmap='Purples')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.axvline(orientations.index(65)-0.5,linewidth=0.4, color='grey')
	ax.axvline(orientations.index(65)+0.5,linewidth=0.4, color='grey')



### labels
ax1.set_ylabel('IT',**font1)
ax1.set_xlabel('GPi',**font1)
ax10.set_ylabel('IT',**font1)
ax10.set_xlabel('GPi',**font1)
ax19.set_ylabel('IT',**font1)
ax19.set_xlabel('GPe',**font1)


###cbar
cbar=plt.colorbar(valW1,cax=ax1CBar, ticks=[0, plotMax[0]])
cbar.ax.set_yticklabels(['0', str(round(plotMax[0],1))],**font1) 
cbar=plt.colorbar(valW2,cax=ax2CBar, ticks=[0, plotMax[1]])
cbar.ax.set_yticklabels(['0', str(round(plotMax[1],1))],**font1) 
cbar=plt.colorbar(valW3,cax=ax3CBar, ticks=[0, plotMax[2]])
cbar.ax.set_yticklabels(['0', str(round(plotMax[2],1))],**font1) 


###text
axText1.text(0, 0.5, 'A',va='center',ha='left', **font2)
axText2.text(0, 0.5, 'B',va='center',ha='left', **font2)
axText3.text(0, 0.5, 'C',va='center',ha='left', **font2)

ax1trial.text(0, 0.5, 'Trial',va='center',ha='left', **font2)
ax2trial.text(0.5, 0.5, '0',va='center',ha='center', **font1)
ax3trial.text(0.5, 0.5, '24',va='center',ha='center', **font1)
ax4trial.text(0.5, 0.5, '48',va='center',ha='center', **font1)
ax5trial.text(0.5, 0.5, '200',va='center',ha='center', **font1)
ax6trial.text(0.5, 0.5, '500',va='center',ha='center', **font1)

for ax in [axText1,axText2,axText3,ax1trial,ax2trial,ax3trial,ax4trial,ax5trial,ax6trial]:
	ax.axis('off')






plt.savefig('Learn_BG.svg', dpi=figRes)



