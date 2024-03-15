"""
    Cognitive, Learning Agent
    Author: Alex Schwarz, Oliver Maith
"""
import ANNarchy as ann
import sys
import os
import pylab as plt
import numpy as np
np.random.seed()
import itertools

from Network_Visual import V1, V4L23, V4L4, FEFv, FEFvm, FEFm, AuxE, StandardSynapse
from Network_BG import IT, PFC, StrD1, StrD2, PPTN, STN, GPe, SNr, MD, StrThal, ITStrD1, ITStrD2, SNc, DAPrediction, StrD1StrD1, StrD2StrD2, STNSTN, StrD1SNr, ITSTN, StrD2GPe, STNSNr, SNrSNr, StrThalGPe, StrThalSNr, MDStrThal, SNrMD, ITPFC, VAPFC, PFCPFC, GPeSNr, StrThalStrThal, PFCMD, StrD1SNc
from Connections import one_to_dim, dim_to_one
from parameters import params
from changed_val import changeParams, changed
from timeit import default_timer as timer
SIMULATIONSSTART=timer()
simID=0#sys.argv[1]
trainStimID=0#int(sys.argv[2])# 0-t1, 1-t1rev

MaxPooling = ann.Synapse(
    psp = "w * pre.r",
    operation = "max"
)
MinPooling = ann.Synapse(
    psp = "w * pre.r",
    operation = "min"
)


def addConn():
	""" Add interconnections """
	###  Connection from prefrontal Cortex to V4 L2/3 (layerwise amplification)  ###
	PFC_V4L23 = ann.Projection(PFC, V4L23, target='A_PFC', synapse=StandardSynapse)
	PFC_V4L23.connect_with_func(one_to_dim, postDim=2, weight=changed['PFC_V4L23.w'])

	###  Connection from V4 L2/3 to IT  ###
	V4L23_IT = ann.Projection(V4L23, IT, target='exc', synapse=MaxPooling)
	V4L23_IT.connect_with_func(dim_to_one, preDim=2, weight=4.0)

	###  Connection from IT to Thalamus (ext for thalamus -> possible "actions")  ###
	ITMD = ann.Projection(IT,MD,target='exc', synapse=StandardSynapse)
	ITMD.connect_gaussian(amp=0.1, sigma=0.1, allow_self_connections=True)


if __name__ == '__main__':
	"""
		runs two trials, first=rewarded, second not rewarded
	"""

	#####################################################
	#####  ADJUST PARAMETERS AND ADD CONNECTIONS   ######
	#####################################################
	addConn()
	changeParams(1)


	#####################################################
	###################  COMPILE   ######################
	#####################################################
	parallelAnz=8
	if(int(simID)%parallelAnz!=0):
		compNr=int(simID)%parallelAnz
	else:
		compNr=parallelAnz
	ann.compile(directory="annarchy_sim"+str(compNr+[0,8,0][trainStimID]))
	

	#####################################################
	##############  PREPARE SIMULATION   ################
	#####################################################
	MAXSIMULATIONSZEIT=3600*24
	maxTrainingTrials=2
	trainStim=['t1','t1Reversed'][trainStimID]
	testStim='t2'
	tMaxtrainStim=1000
	tMaxtestStim=300
	intertrial=500
	dopDelay=0
	numDifStims=450
	BlockLength=50
	trainStim_Anz=340
	testStim_Anz=160
	maxTrials=maxTrainingTrials+trainStim_Anz+testStim_Anz
	### LOAD STIMULI  ###
	Input = np.zeros((numDifStims, params['V1_shape'][0], params['V1_shape'][1], params['V1_shape'][2], params['V1_shape'][3]))
	Input_img = np.zeros((numDifStims, params['V1_shape'][0], params['V1_shape'][1]))
	Input_lineCoords = []
	Input_phi = []
	for stim_nbr in np.arange(numDifStims):
		Input[stim_nbr]=np.load("new_stims_normal/input"+str(stim_nbr)+".npy")
		Input_img[stim_nbr]=np.load("new_stims_normal/img"+str(stim_nbr)+".npy")
		Input_lineCoords.append(np.load("new_stims_normal/lineCoords"+str(stim_nbr)+".npy"))
		Input_phi.append(np.load("new_stims_normal/phi"+str(stim_nbr)+".npy"))
	distList=[]

    
	folder='2020_11_13_testoneTrial_'+['T1','T1rev'][trainStimID]+'/'
	try:
		os.makedirs('data/'+folder[:-1])
	except:
		if os.path.isdir('data/'+folder[:-1])==False:
			print('could not create data/'+folder[:-1]+' folder')


	#####################################################
	###################  MONITORS  ######################
	#####################################################
	big_mon_period=1.0
	M = [ann.Monitor(V1, 'r', period=big_mon_period),ann.Monitor(V4L4, 'r', period=big_mon_period),ann.Monitor(V4L23, 'r', period=big_mon_period),ann.Monitor(IT, 'r'),ann.Monitor(PFC, 'mp'),ann.Monitor(FEFv, 'r', period=big_mon_period),ann.Monitor(FEFm, 'r', period=big_mon_period),ann.Monitor(StrD1, 'mp'),ann.Monitor(StrD2, 'mp'),ann.Monitor(PPTN, 'mp'),ann.Monitor(STN, 'mp'),ann.Monitor(GPe, 'mp'),ann.Monitor(SNr, 'mp'),ann.Monitor(MD, 'mp'),ann.Monitor(StrThal, 'mp'),ann.Monitor(SNc, 'r'),ann.Monitor(FEFvm, 'r', period=big_mon_period),ann.Monitor(AuxE, 'r', period=big_mon_period),ann.Monitor(FEFv, 'q', period=big_mon_period)]

	selection=np.zeros((maxTrials,7))
	###  only use specific monitors --> comment below!  ###
	use_monitors=[3,4,5,6,7,8,9,10,11,12,13,15]
	for idxA in range(len(M)):
		use=0
		for idxB in use_monitors:
			if idxA==idxB:
				use=1
		if use==0:
			M[idxA].pause()
	
	###  monitor for weights or only weights per trial  ###
	monitorWeights=0
	weight_mon_period=5.0
	if monitorWeights==1:
		mon_w_ITStrD1=ann.Monitor(ITStrD1, 'w', period=weight_mon_period)
		mon_w_ITStrD2=ann.Monitor(ITStrD2, 'w', period=weight_mon_period)
		mon_w_ITSTN=ann.Monitor(ITSTN, 'w', period=weight_mon_period)
		mon_w_StrD2GPe=ann.Monitor(StrD2GPe, 'w', period=weight_mon_period)
		mon_w_StrD1SNr=ann.Monitor(StrD1SNr, 'w', period=weight_mon_period)
		mon_w_STNSNr=ann.Monitor(STNSNr, 'w', period=weight_mon_period)
		mon_w_ITPFC=ann.Monitor(ITPFC, 'w', period=weight_mon_period)
		mon_w_VAPFC=ann.Monitor(VAPFC, 'w', period=weight_mon_period)
		mon_w_StrD1SNc=ann.Monitor(StrD1SNc, 'w', period=weight_mon_period)
	else:
		w_ITStrD1=np.zeros((maxTrials,np.array(ITStrD1.w).shape[0],np.array(ITStrD1.w).shape[1]))
		w_ITStrD2=np.zeros((maxTrials,np.array(ITStrD2.w).shape[0],np.array(ITStrD2.w).shape[1]))
		w_ITSTN=np.zeros((maxTrials,np.array(ITSTN.w).shape[0],np.array(ITSTN.w).shape[1]))
		w_StrD2GPe=np.zeros((maxTrials,np.array(StrD2GPe.w).shape[0],np.array(StrD2GPe.w).shape[1]))
		w_StrD1SNr=np.zeros((maxTrials,np.array(StrD1SNr.w).shape[0],np.array(StrD1SNr.w).shape[1]))
		w_STNSNr=np.zeros((maxTrials,np.array(STNSNr.w).shape[0],np.array(STNSNr.w).shape[1]))
		w_ITPFC=np.zeros((maxTrials,np.array(ITPFC.w).shape[0],np.array(ITPFC.w).shape[1]))
		w_VAPFC=np.zeros((maxTrials,np.array(VAPFC.w).shape[0],np.array(VAPFC.w).shape[1]))
		w_StrD1SNc=np.zeros((maxTrials,np.array(StrD1SNc.w).shape[0],np.array(StrD1SNc.w).shape[1]))


	#####################################################
	##################  SIMULATION   ####################
	#####################################################
	SNc.alpha=0
	PPTN.B=0
	startTrial=0
	ann.simulate(300)
	training=1
	trial=0
	BlockTrial=0
	numCorrect=0
	while training==1 and ((training==1 and trial<maxTrainingTrials) or (training==0)) and (BlockTrial<(trainStim_Anz+testStim_Anz)) and ((timer()-SIMULATIONSSTART)<MAXSIMULATIONSZEIT):
		###  trial start time  ###
		selection[trial,0]=ann.get_time()

		###  set Stimulus  ###
		selection[trial,6]=(BlockTrial//BlockLength)+1-int(training)#BlockNumber
		if training==1:
			stimulus=trainStim
			if trial>20:
				if trainStim=='justT' or trainStim=='justD':
					###  after 20 trials stop training  ###
					training=0
					stimuliBlock=np.concatenate((np.ones(trainStim_Anz),np.ones(testStim_Anz)*2))
					np.random.shuffle(stimuliBlock)
				else:
					###  min 16 of 20 last correct = 80% --> stop training  ###
					if selection[trial-20:trial,4].sum()>=16:
						training=0
						stimuliBlock=np.concatenate((np.ones(trainStim_Anz),np.ones(testStim_Anz)*2))
						np.random.shuffle(stimuliBlock)
		else:
			if stimuliBlock[BlockTrial]==1:
				stimulus=trainStim
			else:
				stimulus=testStim
			BlockTrial+=1

		###  Input dependent on stimulus  ###
		if stimulus=='t1':
			I=np.random.randint(0,49)+50*0
		elif stimulus=='t2':
			I=np.random.randint(0,49)+50*1
		elif stimulus=='standard':
			I=np.random.randint(0,49)+50*2
		elif stimulus=='justT':
			I=np.random.randint(0,49)+50*3
		elif stimulus=='justD':
			I=np.random.randint(0,49)+50*4
		elif stimulus=='heterogenD':
			I=np.random.randint(0,49)+50*5
		elif stimulus=='linearSep':
			I=np.random.randint(0,49)+50*6
		elif stimulus=='t1Reversed':
			I=np.random.randint(0,49)+50*7
		elif stimulus=='kerzel':
			I=np.random.randint(0,49)+50*8
		V1.B = Input[I]
		targetOrientation=Input_phi[I][-1]
		if stimulus=='justD':
			targetOrientation=999
 

		###  SIMULATE UNTIL  ###
		if stimulus==trainStim:
			tMax=tMaxtrainStim
		else:	
			tMax=tMaxtestStim
		r = ann.simulate_until(max_duration=tMax, population=FEFm)
		
		###  decision of FEFm = Number of neuron  ###
		decision = int(np.max(FEFm.decision))
		if decision<0:
			decision=np.array(FEFm.r).argmax()
		###  convert number of FEFm (decision) Neuron into Position (decisionPos)  ###
		decisionPos=np.array([np.unravel_index(decision, params['FEF_shape'])[1],np.unravel_index(decision, params['FEF_shape'])[0]])
		###  which line is the nearest to decisionPos --> decisionOrientation  ###
		dist=np.zeros(len(Input_lineCoords[I]))
		for lineIdx in range(len(Input_lineCoords[I])):
			linePos=np.array(Input_lineCoords[I][lineIdx])			
			dist[lineIdx]=np.sqrt(np.sum((linePos-decisionPos)**2))
		decisionOrientation=Input_phi[I][dist.argmin()]
		distList.append(dist)	

		### deactivate stimuli + reward after dopDelay  ###
		selection[trial,1]=ann.get_time()
		V1.B = 0
		ann.simulate(dopDelay)
		if stimulus==trainStim:
			if trial == 0:
				PPTN.B=1
				ann.simulate(1)
				SNc.alpha=1
				ann.simulate(60)
				SNc.alpha=0
				PPTN.B=0
				numCorrect+=1
			else:
				PPTN.B=0
				ann.simulate(1)
				SNc.alpha=1
				ann.simulate(60)
				SNc.alpha=0
				PPTN.B=0
				numCorrect=0

		###  intertrial  ###
		ann.simulate(intertrial)

		###  save  ###
		selection[trial,2]=targetOrientation
		selection[trial,3]=decisionOrientation
		selection[trial,4]=decisionOrientation==targetOrientation
		if stimulus==trainStim:
			selection[trial,5]=1
		else:
			selection[trial,5]=2
		with open('data/'+folder+'output'+str(simID)+'.txt', 'a') as f:
			print(selection[trial,6],trial,selection[trial,1]-selection[trial,0],targetOrientation,decisionOrientation, file=f)
		if monitorWeights==0:
			w_ITStrD1[trial]=np.array(ITStrD1.w)
			w_ITStrD2[trial]=np.array(ITStrD2.w)
			w_ITSTN[trial]=np.array(ITSTN.w)
			w_StrD1SNr[trial]=np.array(StrD1SNr.w)
			w_StrD2GPe[trial]=np.array(StrD2GPe.w)
			w_STNSNr[trial]=np.array(STNSNr.w)
			w_ITPFC[trial]=np.array(ITPFC.w)
			w_VAPFC[trial]=np.array(VAPFC.w)
			w_StrD1SNc[trial]=np.array(StrD1SNc.w)
		trial+=1


	#####################################################
	################  GET MONITORS   ####################
	#####################################################
	#rV1=M[0].get('r')
	#rV4L4=M[1].get('r')
	#rV4L23=M[2].get('r')
	rIT=M[3].get('r')
	rPFC=M[4].get('mp')
	rFEFv=M[5].get('r')
	rFEFm=M[6].get('r')
	rSTRD1=M[7].get('mp')
	rSTRD2=M[8].get('mp')
	rPPTN=M[9].get('mp')
	rSTN=M[10].get('mp')
	rGPe=M[11].get('mp')
	rSNr=M[12].get('mp')
	rMD=M[13].get('mp')
	#rStrThal=M[14].get('mp')
	rSNc=M[15].get('r')
	#rFEFvm=M[16].get('r')
	#rAuxE=M[17].get('r')
	#FEFvQ=M[18].get('q')
	
	#sumInh=MSNc[0].get('test')
	#aux=MSNc[1].get('aux')
	#mp=MSNc[2].get('mp')

	if monitorWeights==1:
		w_ITStrD1=mon_w_ITStrD1.get('w')
		w_ITStrD2=mon_w_ITStrD2.get('w')
		w_ITSTN=mon_w_ITSTN.get('w')
		w_StrD1SNr=mon_w_StrD1SNr.get('w')
		w_StrD2GPe=mon_w_StrD2GPe.get('w')
		w_STNSNr=mon_w_STNSNr.get('w')
		w_ITPFC=mon_w_ITPFC.get('w')
		w_VAPFC=mon_w_VAPFC.get('w')
		w_StrD1SNc=mon_w_StrD1SNc.get('w')

	
	#####################################################
	#####################  save   #######################
	#####################################################
	#np.save('data/'+folder+'rV1'+str(simID)+'.npy',rV1)
	#np.save('data/'+folder+'rV4L4'+str(simID)+'.npy',rV4L4)
	#np.save('data/'+folder+'rV4L23'+str(simID)+'.npy',rV4L23)
	np.save('data/'+folder+'rFEFv'+str(simID)+'.npy',rFEFv)
	#np.save('data/'+folder+'FEFvQ'+str(simID)+'.npy',FEFvQ)
	#np.save('data/'+folder+'FEFvsumExc'+str(simID)+'.npy',FEFvsumExc)
	np.save('data/'+folder+'rFEFm'+str(simID)+'.npy',rFEFm)
	#np.save('data/'+folder+'rFEFvm'+str(simID)+'.npy',rFEFvm)
	#np.save('data/'+folder+'rAuxE'+str(simID)+'.npy',rAuxE)

	np.save('data/'+folder+'rSTRD1'+str(simID)+'.npy',rSTRD1)
	np.save('data/'+folder+'rSTRD2'+str(simID)+'.npy',rSTRD2)
	np.save('data/'+folder+'rSTN'+str(simID)+'.npy',rSTN)
	np.save('data/'+folder+'rIT'+str(simID)+'.npy',rIT)
	np.save('data/'+folder+'rPFC'+str(simID)+'.npy',rPFC)
	np.save('data/'+folder+'rSNr'+str(simID)+'.npy',rSNr)
	#np.save('data/'+folder+'rStrThal'+str(simID)+'.npy',rStrThal)
	np.save('data/'+folder+'rGPe'+str(simID)+'.npy',rGPe)
	np.save('data/'+folder+'rMD'+str(simID)+'.npy',rMD)

	#np.save('data/'+folder+'sumInh'+str(simID)+'.npy',sumInh)
	#np.save('data/'+folder+'aux'+str(simID)+'.npy',aux)
	#np.save('data/'+folder+'mp'+str(simID)+'.npy',mp)
	np.save('data/'+folder+'rSNc'+str(simID)+'.npy',rSNc)
	np.save('data/'+folder+'rPPTN'+str(simID)+'.npy',rPPTN)
	np.save('data/'+folder+'selection'+str(simID)+'.npy',selection)

	np.save('data/'+folder+'w_ITStrD1'+str(simID)+'.npy',w_ITStrD1)
	np.save('data/'+folder+'w_ITStrD2'+str(simID)+'.npy',w_ITStrD2)
	np.save('data/'+folder+'w_ITSTN'+str(simID)+'.npy',w_ITSTN)
	np.save('data/'+folder+'w_StrD1SNr'+str(simID)+'.npy',w_StrD1SNr)
	np.save('data/'+folder+'w_StrD2GPe'+str(simID)+'.npy',w_StrD2GPe)
	np.save('data/'+folder+'w_STNSNr'+str(simID)+'.npy',w_STNSNr)
	np.save('data/'+folder+'w_ITPFC'+str(simID)+'.npy',w_ITPFC)
	np.save('data/'+folder+'w_VAPFC'+str(simID)+'.npy',w_VAPFC)
	np.save('data/'+folder+'w_StrD1SNc'+str(simID)+'.npy',w_StrD1SNc)

	np.save('data/'+folder+'distList'+str(simID)+'.npy',distList)









