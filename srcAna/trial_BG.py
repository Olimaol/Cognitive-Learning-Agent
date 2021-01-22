
import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


"""

Load one simulation (2 trials)

Plot rates of
IT, StrD1, StrD2, STN, GPi, GPe, SNc, Thal, PFC



"""
###  Plot general Parameter
degSign=u"\u00b0"
ae=u"\u00E4"

font1={'fontsize': 7,
 'fontweight' : 'normal'}
font2={'fontsize': 10,
 'fontweight' : 'normal'}
font1p={'size': 7,
 'weight' : 'normal'}#p for legends

scaling=0.77

figB=int(210*scaling)#mm
figH=int(115*scaling)#mm
figRes=1000#dpi

### Plot specific parameters
folder='2020_09_21_oneTrial_T1'

plotB=int(35*scaling)#mm
plotH=int(20*scaling)#mm

textB=(np.array([5,35,35,35,35])*scaling).astype(int)#mm
textH=(np.array([86,5,5,5,5])*scaling).astype(int)#mm


PosX1=[int(15*scaling)+i*(int(20*scaling)) for i in range(4)]
PosY1=[int(25*scaling)]*4

PosX2=[int(15*scaling)+i*(int(20*scaling)) for i in range(4)]
PosY2=[int(85*scaling)]*4

PosX3=[int(15*scaling)+i*(int(20*scaling)) for i in range(2)]
PosY3=[int(145*scaling)]*2

textX=(np.array([15,8,8,8,68])*scaling).astype(int)#mm
textY=(np.array([12,25,85,145,85])*scaling).astype(int)#mm






#General Figure
plt.figure(1,figsize=(figB*0.03937007874,figH*0.03937007874))
plt.subplots_adjust(top=1,bottom=0,left=0,right=1.0,hspace=0,wspace=0)

ax1=plt.subplot2grid((figH,figB),(PosX1[0],PosY1[0]), colspan=plotB, rowspan=plotH)
ax2=plt.subplot2grid((figH,figB),(PosX1[1],PosY1[1]), colspan=plotB, rowspan=plotH)
ax3=plt.subplot2grid((figH,figB),(PosX1[2],PosY1[2]), colspan=plotB, rowspan=plotH)
ax4=plt.subplot2grid((figH,figB),(PosX1[3],PosY1[3]), colspan=plotB, rowspan=plotH)
ax5=plt.subplot2grid((figH,figB),(PosX2[0],PosY2[0]), colspan=plotB, rowspan=plotH)
ax6=plt.subplot2grid((figH,figB),(PosX2[1],PosY2[1]), colspan=plotB, rowspan=plotH)
ax9=plt.subplot2grid((figH,figB),(PosX2[3],PosY2[3]), colspan=plotB, rowspan=plotH)
ax7=plt.subplot2grid((figH,figB),(PosX3[0],PosY3[0]), colspan=plotB, rowspan=plotH)
ax8=plt.subplot2grid((figH,figB),(PosX3[1],PosY3[1]), colspan=plotB, rowspan=plotH)

axText1=plt.subplot2grid((figH,figB),(textX[0],textY[0]), colspan=textB[0], rowspan=textH[0])
axText2=plt.subplot2grid((figH,figB),(textX[1],textY[1]), colspan=textB[1], rowspan=textH[1])
axText3=plt.subplot2grid((figH,figB),(textX[2],textY[2]), colspan=textB[2], rowspan=textH[2])
axText4=plt.subplot2grid((figH,figB),(textX[3],textY[3]), colspan=textB[3], rowspan=textH[3])
axText5=plt.subplot2grid((figH,figB),(textX[4],textY[4]), colspan=textB[4], rowspan=textH[4])



### Load data
sim=1
rIT=np.clip(np.load('../data/'+folder+'/rIT'+str(sim)+'.npy'),0,None)
rStrD1=np.clip(np.load('../data/'+folder+'/rSTRD1'+str(sim)+'.npy'),0,None)
rStrD2=np.clip(np.load('../data/'+folder+'/rSTRD2'+str(sim)+'.npy'),0,None)
rSTN=np.clip(np.load('../data/'+folder+'/rSTN'+str(sim)+'.npy'),0,None)
rSNr=np.clip(np.load('../data/'+folder+'/rSNr'+str(sim)+'.npy'),0,None)
rGPe=np.clip(np.load('../data/'+folder+'/rGPe'+str(sim)+'.npy'),0,None)
rMD=np.clip(np.load('../data/'+folder+'/rMD'+str(sim)+'.npy'),0,None)
rPFC=np.clip(np.load('../data/'+folder+'/rPFC'+str(sim)+'.npy'),0,None)
rSNc=np.clip(np.load('../data/'+folder+'/rSNc'+str(sim)+'.npy'),0,None)
rPPTN=np.clip(np.load('../data/'+folder+'/rPPTN'+str(sim)+'.npy'),0,None)
selection=np.load('../data/'+folder+'/selection'+str(sim)+'.npy')

trialStart=selection[:,0].astype(int)
trialDecision=selection[:,1].astype(int)

rates=[rIT,rStrD1,rStrD2,rSTN,rSNr,rGPe,rMD,rPFC,rSNc,rPPTN]


### PLots

tMin=trialStart[0]
tMax=tMin+700
xticks=np.arange(tMin,tMax+1,200)
xticklabels=xticks-300
yticks=[0,1]
yticklabels=yticks


cols=['k','k','k','k','k','k','k','C1','C2','C3','C4','k','k','k','k','k']
labels=['_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','45'+degSign,'50'+degSign,'55'+degSign,'60'+degSign,'_nolegend_','_nolegend_','_nolegend_','_nolegend_','others']
ylabel=['IT','StrD1','StrD2','STN','GPi','GPe','Thal','PFC','SNc','PPTN']


for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
	
	if i==1 or i==2 or i==3:
		ax.plot(rates[i]/np.amax(rates[i][tMin:tMax]), linewidth=1, color=[0.2]*3)
	else:
		for j in [1,2,3,4,5,6,11,12,13,14,7,8,9,10,15]:#range(len(orientationLayers)):#
			ax.plot(rates[i][:,j]/np.amax(rates[i][tMin:tMax]), linewidth=1, color=cols[j], label=labels[j])
		
	ax.set_xlim(250,tMax)
	ax.set_ylim(-0.1,1.1)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xticklabels, **font1)
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticklabels, **font1)
	ax.set_ylabel(ylabel[i],va='center',ha='left',rotation=0,**font1)
	ax.yaxis.set_label_coords(1.04,0.5)

ax9.plot(rSNc/np.amax(rSNc[tMin:tMax]), linewidth=1, color=[0.2]*3)
ax9.plot(rSNc[trialDecision[1]-trialDecision[0]:]/np.amax(rSNc[tMin:tMax]), linewidth=1, color=[0.2]*3, linestyle='dotted')
ax9.set_xlim(250,tMax)
ax9.set_ylim(-0.1,1.1)
ax9.set_xticks(xticks)
ax9.set_xticklabels(xticklabels, **font1)
ax9.set_yticks(yticks)
ax9.set_yticklabels(yticklabels, **font1)
ax9.set_ylabel('SNc',va='center',ha='left',rotation=0,**font1)
ax9.yaxis.set_label_coords(1.04,0.5)

### Legende

ax8.legend(ncol=2,bbox_to_anchor=(0.5, -1.5), loc=10, borderaxespad=0., prop=font1p)

### Achsenlabels
axText1.text(0.5, 0.5, 'normalized activity',ha='center',va='center',rotation=90, **font1)
axText1.axis('off')
ax4.set_xlabel('time / ms',**font1)
ax8.set_xlabel('time / ms',**font1)
ax9.set_xlabel('time / ms',**font1)

### Ueberschriften
axText2.text(0.5, 0.5, 'BG Input:',ha='center',va='center', **font1)
axText2.axis('off')
axText3.text(0.5, 0.5, 'BG Output:',ha='center',va='center', **font1)
axText3.axis('off')
axText4.text(0.5, 0.5, 'BG Targets:',ha='center',va='center', **font1)
axText4.axis('off')
axText5.text(0.5, 0.5, 'Dopamine:',ha='center',va='center', **font1)
axText5.axis('off')






plt.savefig('trial_BG.svg', dpi=figRes)







