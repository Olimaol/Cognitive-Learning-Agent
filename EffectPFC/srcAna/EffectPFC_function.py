import pylab as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

def mehrereAufmerksamkeiten(ax, font1, font2, font1p, SNrLims, SNrTicks):

    ###  Plot general parameters
    degSign=u"\u00b0"
    sigSign=u"\u03C3"

    ###  Plot specific parameters
    folder=['2020_09_18_EffectPFC_small','2020_09_18_EffectPFC_normal','2020_09_18_EffectPFC_wide']

    labelList=[sigSign+'$_{V1}$ = '+str(i)+degSign for i in [14,22,30]]+['r$_{PFC}$ = '+str(i) for i in [0.05,0.15,0.4]]

    style=[':','-','--']

    ### Data
    time=[400,500]


    orientations=[-1,30,35,40,45,50,55,60,65,70,75,80]
    diff_T_D_FEF=np.zeros((len(orientations)*10,3))

    for bedingung in range(3):

	    ### for each condition 10 simulations, load FEF activities
	    for sim in range(len(orientations)*10):

		    pos1=np.load('../data/'+folder[bedingung]+'/IList'+str(sim+1)+'.npy').astype(int)
		    pos2=(pos1//2).astype(int)

		    rFEFv=np.load('../data/'+folder[bedingung]+'/rFEFv'+str(sim+1)+'.npy')
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

	    if bedingung<=2:
		    ax.plot(x,diff,label=labelList[bedingung],color='b',linestyle=style[bedingung])
	    else:
		    ax2.fill_between(x,diff-diffStd,diff+diffStd, alpha=0.2)
		    ax2.plot(x,diff,label=labelList[bedingung])


    ### labels
    ax.set_ylabel('SNR',va='center',ha='center',labelpad=10,**font1)


    ax.set_xticks(range(11))
    ax.set_xticklabels(np.arange(30,81,5), **font1)
    ax.set_xlabel('active PFC-Neuron / orientation',**font1)
    ax.plot([-1,11],[1,1],linewidth=0.5,color='k',alpha=0.3)
    ax.set_xlim([-0.1,10.1])
    ax.set_ylim(SNrLims)
    ax.set_yticks(SNrTicks)
    ax.set_yticklabels(SNrTicks, **font1)
    ax.plot([5,5],[ax.get_ylim()[0],(ax.get_ylim()[1]-ax.get_ylim()[0])*0.04+ax.get_ylim()[0]],color='g',lw=1)
    ax.plot([4,4],[ax.get_ylim()[0],(ax.get_ylim()[1]-ax.get_ylim()[0])*0.04+ax.get_ylim()[0]],color='r',lw=1)
    ax.text(5,(ax.get_ylim()[1]-ax.get_ylim()[0])*0.03+ax.get_ylim()[0],'T',color='g',va='bottom',ha='center',**font1)
    ax.text(4,(ax.get_ylim()[1]-ax.get_ylim()[0])*0.03+ax.get_ylim()[0],'D',color='r',va='bottom',ha='center',**font1)

    ### Legend
    ax.legend(bbox_to_anchor=(1.1, 0.5), loc=6, borderaxespad=0., prop=font1p)



