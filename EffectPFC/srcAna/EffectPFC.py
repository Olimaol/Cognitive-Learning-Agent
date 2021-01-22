import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from EffectPFC_function import mehrereAufmerksamkeiten

def normalize(X):

	ret=(X-X.min())/(X.max()-X.min())
	return ret

###  Plot general parameters
degSign=u"\u00b0"
ae=u"\u00E4"

font1={'fontsize': 7,
 'fontweight' : 'normal'}
font2={'fontsize': 11,
 'fontweight' : 'normal'}
font1p={'size': 7,
 'weight' : 'normal'}#p for legends

scaling = 95/150

figB=int(150*scaling)#mm
figH=int(130*scaling)#mm
figRes=500#dpi

plotB=int(80*scaling)#
plotH=int(30*scaling)#

PosX=(np.array([15,50,85])*scaling).astype(int)#mm
PosY=(np.array([25,25,25])*scaling).astype(int)#mm


###  Plot specific parameters
folder='2020_09_18_EffectPFC_normal'
SNrTicks = [0.5,1,1.5]
SNrLims = [0.5,1.9]


#General figure
plt.figure(1,figsize=(figB*0.03937007874,figH*0.03937007874), dpi=figRes)
plt.subplots_adjust(top=1,bottom=0,left=0,right=1.0,hspace=0,wspace=0)

ax1=plt.subplot2grid((figH,figB),(PosX[0],PosY[0]), colspan=plotB, rowspan=plotH)
ax2=plt.subplot2grid((figH,figB),(PosX[1],PosY[1]), colspan=plotB, rowspan=plotH)
ax3=plt.subplot2grid((figH,figB),(PosX[2],PosY[2]), colspan=plotB, rowspan=plotH)






### data
time=[400,500]


orientations=[-1,30,35,40,45,50,55,60,65,70,75,80]
diff_T_D_FEF=np.zeros((len(orientations)*10,3))

### for each condition 10 simulations, load FEF activities
for sim in range(len(orientations)*10):

	pos1=np.load('../data/'+folder+'/IList'+str(sim+1)+'.npy').astype(int)
	pos2=(pos1//2).astype(int)

	rFEFv=np.load('../data/'+folder+'/rFEFv'+str(sim+1)+'.npy')
	rFEFv=np.reshape(rFEFv,(rFEFv.shape[0],41,31))

	att=sim # att: no,30,35,40,45,50,55,60,65,70,75,80

	activities=np.mean(rFEFv[time[0]:time[1],pos1[:,1],pos1[:,0]],0)
	diff_T_D_FEF[att,0]=(activities[8]/np.mean(activities[0:8]))
	diff_T_D_FEF[att,1]=activities[8]
	diff_T_D_FEF[att,2]=np.mean(activities[0:8])


### average over same attetnion condition (10 sims)
diff_T_D_FEF_ori_mean=np.zeros((len(orientations),3))
diff_T_D_FEF_ori_std=np.zeros((len(orientations),3))
for ori in range(len(orientations)):
	start=int(ori*10)
	end=int(start+10)
	diff_T_D_FEF_ori_mean[ori,0]=np.mean(diff_T_D_FEF[start:end,0],0)
	diff_T_D_FEF_ori_mean[ori,1]=np.mean(diff_T_D_FEF[start:end,1],0)
	diff_T_D_FEF_ori_mean[ori,2]=np.mean(diff_T_D_FEF[start:end,2],0)

	diff_T_D_FEF_ori_std[ori,0]=np.std(diff_T_D_FEF[start:end,0],0)
	diff_T_D_FEF_ori_std[ori,1]=np.std(diff_T_D_FEF[start:end,1],0)
	diff_T_D_FEF_ori_std[ori,2]=np.std(diff_T_D_FEF[start:end,2],0)


diff=diff_T_D_FEF_ori_mean[:,0]
diffStd=diff_T_D_FEF_ori_std[:,0]
targetAct=diff_T_D_FEF_ori_mean[:,1]
targetActStd=diff_T_D_FEF_ori_std[:,1]
distAct=diff_T_D_FEF_ori_mean[:,2]
distActStd=diff_T_D_FEF_ori_std[:,2]


baseline=targetAct[0]
targetAct=targetAct/baseline
targetAct=targetAct[1:]
targetActStd=targetActStd/baseline
targetActStd=targetActStd[1:]


baseline=distAct[0]
distAct=distAct/baseline
distAct=distAct[1:]
distActStd=distActStd/baseline
distActStd=distActStd[1:]


diff=diff[1:]
diffStd=diffStd[1:]



### PLots
x=range(11)

ax1.fill_between(x,targetAct-targetActStd,targetAct+targetActStd, color='g', alpha=0.2)
ax1.fill_between(x,distAct-distActStd,distAct+distActStd, color='r', alpha=0.2)
ax1.plot(x,targetAct,color='g',label='Target')
ax1.plot(x,distAct,color='r',label='Distractor')
ax2.fill_between(x,diff-diffStd,diff+diffStd, color='b', alpha=0.2)
ax2.plot(x,diff,color='b')
ax1.set_ylim([0.7,2.6])
ax2.set_ylim(SNrLims)
for ax in [ax1,ax2]:
	ax.plot([-1,11],[1,1],linewidth=0.5,color='k',alpha=0.3)
	ax.set_xlim([-0.1,10.1])
	ax.plot([5,5],[ax.get_ylim()[0],(ax.get_ylim()[1]-ax.get_ylim()[0])*0.04+ax.get_ylim()[0]],color='g',lw=1)
	ax.plot([4,4],[ax.get_ylim()[0],(ax.get_ylim()[1]-ax.get_ylim()[0])*0.04+ax.get_ylim()[0]],color='r',lw=1)
	ax.text(5,(ax.get_ylim()[1]-ax.get_ylim()[0])*0.03+ax.get_ylim()[0],'T',color='g',va='bottom',ha='center',**font1)
	ax.text(4,(ax.get_ylim()[1]-ax.get_ylim()[0])*0.03+ax.get_ylim()[0],'D',color='r',va='bottom',ha='center',**font1)

mehrereAufmerksamkeiten(ax3, font1, font2, font1p, SNrLims, SNrTicks)


### Labels
ax1.set_ylabel('relative activity',va='center',ha='center',labelpad=10,**font1)
ax2.set_ylabel('SNR',va='center',ha='center',labelpad=10,**font1)

ax1.set_xticks(range(11))
ax1.set_xticklabels([], **font1)
ax1.set_yticks([1,2])
ax1.set_yticklabels([1.0,2.0], **font1)
ax2.set_xticks(range(11))#[0,2,4,6,8,10])
ax2.set_xticklabels([], **font1)#[30,40,50,60,70,80])
ax2.set_yticks(SNrTicks)
ax2.set_yticklabels(SNrTicks, **font1)


### Legend
ax1.legend(bbox_to_anchor=(1.1, 0.5), loc=6, borderaxespad=0., prop=font1p)





plt.savefig('EffectPFC.svg', dpi=figRes)



