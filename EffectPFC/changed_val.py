import ANNarchy as ann
import numpy as np
from parameters import params

changed = {}

#####################################################
########  BESTEHENDE CON's WEIGHTS AENDERN   ########
#####################################################
### INPUT ###
changed['ITPFC.connect_all_to_all']=0.0#statt ann.Uniform(0.2, 0.4)

### LATERALS ###
changed['StrD1StrD1.connect_all_to_all']=0.7#statt 0.3
changed['StrD2StrD2.connect_all_to_all']=0.7#statt 0.3
changed['STNSTN.connect_all_to_all']=0.7#statt 0.3
changed['SNrSNr.connect_all_to_all']=0.5#statt 1.0
changed['PFCPFC.connect_all_to_all']=0.5#statt 0.1
changed['StrThalStrThal.connect_all_to_all']=1.0#statt 0.3
changed['ITIT.connect_all_to_all']=0.2#statt 0.0

### FEEDBACK ###
changed['MDStrThal.connect_one_to_one']=0.5#statt 1.0
changed['StrThalSNr.connect_one_to_one']=0.75#statt 0.3
changed['StrThalGPe.connect_one_to_one']=0.3

### INNER BG ###
changed['StrD1SNr.connect_all_to_all']=ann.Uniform(0.0, 0.05)
changed['StrD2GPe.connect_all_to_all']=0.0#statt ann.Uniform(0, 0.05)
changed['GPeSNr.connect_one_to_one']=1.5

### OUTPUT ###
changed['SNrMD.connect_one_to_one']=0.5
changed['VAPFC.connect_one_to_one']=0.7#statt 0.35
changed['PFCMD.connect_one_to_one']=0.1#statt 0.35 gabs vorher eig nicht

### Attention ###
changed['PFC_V4L23.w']=1.0


#####################################################
################  GLOBAL PARAMETERS  ################
#####################################################
changed['RFsigmav_vm'] = [2, 2]#[4, 3]
changed['RFsizev_vm'] = [40, 40]#[41, 31]

changed['RFsize4_23'] = [3, 3]#[5, 5]
changed['RFsigma4_23'] = [1, 1]#[5./3, 5./3]

changed['reversal_SNr'] = 1.0

changed['baseline_IT'] = 0.0#statt random.uniform(0.0, 0.04)
changed['baseline_STR'] = 0.0#statt 0.4
changed['baseline_SNr'] = 2.4
changed['baseline_MD'] = 0.35#neu
changed['noise_SNr'] = 0.1#statt 1.0
changed['noise_GPe'] = 0.1#statt 1.0
changed['noise_MD'] = 0.1#statt 0.0001




def changeParams(parametersatz):
	#####################################################
	#########  PROJECTION PARAMETER ANPAssEN ############
	#####################################################
	from Network_BG import ITPFC, ITStrD1, ITStrD2, ITSTN, StrD1SNr, StrD2GPe, STNSNr, IT, VAPFC
	from Network_Visual import FEFv, FEFvm, FEFm

	### INPUT ###
	ITPFC.regularization_threshold=0.55#statt 3.5
	ITPFC.threshold_post=0.3#statt 0.0
	ITPFC.tau=15000
	ITStrD1.regularization_threshold=0.7#statt 1.0
	ITStrD1.tau=75.0
	ITStrD1.K_burst=1.0
	ITStrD1.K_dip=0.4
	ITStrD2.regularization_threshold=1.0
	ITStrD2.tau=75.0
	ITStrD2.K_burst=3.0#statt 1.0
	ITStrD2.K_dip=0.4
	ITSTN.regularization_threshold=0.7#statt 1.0
	ITSTN.tau=75.0
	ITSTN.K_burst=1.0
	ITSTN.K_dip=0.4

	### TEACHING CATEGORIES ###
	VAPFC.tau=15000#neu
	VAPFC.regularization_threshold=0.55#neu
	VAPFC.threshold_post=0.3#neu

	### INNER BG ###
	StrD1SNr.threshold_post=0.15
	StrD1SNr.trace_neg_factor=1.0
	StrD1SNr.regularization_threshold=0.0#statt 1.0
	StrD1SNr.tau=50
	StrD1SNr.K_burst=1.0
	StrD1SNr.K_dip=0.4

	StrD2GPe.threshold_post=0.15
	StrD2GPe.trace_neg_factor=0.1#statt 1.0
	StrD2GPe.regularization_threshold=0.0#statt 1.0
	StrD2GPe.tau=50
	StrD2GPe.K_burst=1.0
	StrD2GPe.K_dip=0.4

	STNSNr.threshold_post=-0.15
	STNSNr.trace_pos_factor=1.0
	STNSNr.regularization_threshold=2.6
	STNSNr.tau=50
	STNSNr.K_burst=1.0
	STNSNr.K_dip=0.4

	### VISUAL ###
	
	if parametersatz==0:
		visParams=[6, 0.6, 0.6, 0.2, 1.3, 0.3]#init
	elif parametersatz==1:
		visParams=[10, 0.6, 0.06, 0.1, 1.3, 0.3]

	FEFv.cFEF=visParams[0]

	FEFvm.vEv=visParams[1]
	FEFvm.vSv1=visParams[2]
	FEFvm.vFEFv=np.ones(params['FEFvm_shape'])*np.linspace(1,0,params['FEFvm_shape'][2])[None, None, :]#statt 1.0
	FEFvm.vlow=visParams[3]

	FEFm.vFEFvm_m=visParams[4]
	FEFm.vSvm=visParams[5]



#####################################################
#########  NICHT PARAMETER AENDERUNGEN  #############
#####################################################


### 00 ###  Die Stimuli und damit die V1 Aktivitaet

### 01 ###  Connection from IT to Thalamus (ext for thalamus -> possible "actions")
"""ITMD = ann.Projection(IT,MD,target='exc')
ITMD.connect_gaussian(amp=0.1, sigma=0.1, allow_self_connections=True)"""

### 02 ###  Connection from SNr to FEFvm am wenigsten aktives Neuron im SNr bestimmt inhibition im FEFvm --> erst wenn BG ein Neuron waehlen, kann FEFvm aktiv werden
"""SNrFEFvm = ann.Projection(SNr, FEFvm, target='S_BG', synapse=MinPooling)
SNrFEFvm.connect_all_to_all(weights=1.0, delays=10)
FEFvm.vSBG=0.12
in FEFvm: ES = clip(vEv*sum(E_v)-vSv1*sum(S_v)-vSBG*sum(S_BG),0,1)"""

### 03 ###  ReversedSynapse jetzt mit regularization, weights sinken wenn post zu aktiv --> verhindert das besonders gehemmte Neuronen im SNr andere extrem aktivieren --> wird das ueberhaupt noch benoetigt? --> anscheinend nicht
"""tau_alpha * dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) : postsynaptic
tau * dw/dt = winit-w-alpha : min=0"""

### 04 ###  DA_excitatory und DA_inhibitory
""" trace_pos_factor bzw trace_neg_factor nur noch bei StrD2 noetig damit meherere gelernt werden koennen zu inhibieren (sonst wuerde das gewaehlte LTP bekommen und alle anderen LTD --> das LTD verhindert das mehrere gelernt werden koenn wenns zu gross ist),
SNr -> post_thresh ist abhaengig von min --> nicht mehrere werden "aktiv"
GPe -> post_thresh konstant wird aber von min anstatt mean abgezogen --> mehrere koennen "aktiv" werden"""

### 05 ### Visuelle Neuronen jetzt alle mit min=0
"""minVis = Constant('minVis', 0)"""

### 06 ###  V1_Neuron jetzt mit consistent noise und abklingendem Stimulus
"""f = sin(2*pi*frequency*t) : population
rand = if (f>0.98): Uniform(-1.0,1.0) else: rand
basis=baseline + noise * rand : min=0
dr/dt = if (baseline>0.0): (pow(basis, pV1C)-r)/tau_up else: -r/tau_down : min=minVis"""

### 07 ###  FEFm jetzt mit decision und Neuronen IDs
"""decision = if (r>threshold): id else: -1"""

### 08 ###  Connection of V1 -> V4 L4 = layerwise gauss
"""w14=Gaussian2D(1.0, np.array(params['RF_V1_V4L4'][1:3]), np.array(params['RF_V1_V4L4'][1:3])/5.)
FilterBank=np.zeros(params['RF_V1_V4L4'])
for i in range(params['V4L4_shape'][2]):
	FilterBank[i,:,:,0,i] = w14/w14.sum()*5"""

### 09 ### Connection V4L4 -> V4L23 jetzt kleinerer Gaus (damit verschiedene Linien nicht verschmelzen und Ortsinformation erhalten bleibt)
"""w42 = Gaussian2D(1.0, changed['RFsize4_23'], changed['RFsigma4_23'])[:, :, None]"""

### 10 ### Connection FEFv -> FEFvm excitation und supression symmetrisch und kleinere excitation zone (damit verschiedene Linien nicht verschmelzen und da Stimulus Gitter "symmetrisch" -> long range inhibition soll zwischen allen stimuli sein sonst werden immer die auf der Seite oder an der Ecke gewaehlt)
"""G = Gaussian2D(1.0, changed['RFsizev_vm'], changed['RFsigmav_vm'])"""

### 11 ###  Connection FEFvm -> V4 L4 (long range suppressive) jetzt groesser (selber Grund wie oben)
"""G = Gaussian2D(1.0, [40, 40], [4, 4])
wSP = np.tile(positive(1 - G**0.125)[None, :, :], params['PFC_shape'] + (1, 1))
wSP[wSP>0.188]=0.188"""

### 12 ###  PostCovariance (VAPFC + ITPFC) jetzt mit alpha_factor --> PFC geht nicht ueber 1
"""tau_alpha * dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) * alpha_factor"""

### 13 ###  IT jetzt lateral inhibition
"""ITIT = Projection(pre=IT, post=IT, target='inh')
ITIT.connect_all_to_all(weights=changed['ITIT.connect_all_to_all'])"""

### 14 ### Reward prediction so das sowohl pos als auch negativ dip kommt ungefaehr gleich stark
"""StrD1SNc = Projection(pre=StrD1, post=SNc, target='inh', synapse=DAPrediction)
StrD1SNc.connect_all_to_all(weights=0.5)#statt 1.0"""

















































