"""
Basal Ganglia Model based on the version of F. Escudero
Modified by A. Schwarz
Version 1.0 - 29.05.2018
"""
from ANNarchy import Constant, Neuron, Synapse, Population, Projection, Uniform, setup
from parameters import params
from changed_val import changed
from Network_Visual import StandardSynapse
import numpy as np
np.random.seed()
setup(num_threads=params['num_threads'] )
#Model parameters
baseline_dopa = Constant('baseline_dopa', params['baseline_dopa'])
reversal = Constant('reversal', changed['reversal_SNr'])

#####################################################
##########  Neuron models   #########################
#####################################################
LinearNeuron = Neuron(
    parameters="""
        tau = 10.0 : population
        phi = 0.0 : population
        B = 0.0
    """,
    equations="""
        tau * dmp/dt = -mp + sum(exc) - sum(inh) + B + phi * Uniform(-1.0,1.0)
        r = pos(mp)
    """,
    name = "Linear Neuron",
    description = "Regular rate-coded neuron with excitatory and inhibitory inputs plus baseline and noise."
)

DopamineNeuron = Neuron(
    parameters="""
        tau = 10.0 : population
        alpha = 0 : population
        B = 0.0
    """,
    equations="""
        aux = if (sum(exc)>0): pos(1.0-B-sum(inh)) else: -10 * sum(inh)
        tau * dmp/dt = -mp + alpha * aux + B
        r = pos(mp)
    """,
    name = "Dopamine Neuron",
    description = "Excitatory input increases activity above constant baseline. Inhibitory input can prevent the increase."
)

InputNeuron = Neuron(
    parameters="""
        tau = 10.0 : population
        B = 0.0
    """,
    equations="""
        tau * dmp/dt = -mp + B
        r = pos(mp)
    """,
    name = "Input Neuron",
    description = "Rate-coded neuron only with a baseline to be set."
)

ScaledNeuron = Neuron(
    parameters="""
        tau = 10.0 : population
        phi = 0.0 : population
        B = 0.0
    """,
    equations="""
        tau * dmp/dt = -mp + (sum(exc) - sum(inh) + B + phi * Uniform(-1.0,1.0)) * (1 + sum(att))
        r = pos(mp)
    """,
    name = "Scaled Neuron",
    description = "Like Linear Neuron with multiplicative attention modulation."
)





#####################################################
##########  Synapse models   ########################
#####################################################
PostCovariance = Synapse(
    parameters="""
        tau = 15000.0 : projection
        tau_alpha = 1.0 : projection
        regularization_threshold = 3.5 : projection
        threshold_post = 0.0 : projection
        threshold_pre = 0.15 : projection
        alpha_factor = 15.0 : projection
    """,
    psp = "w * pre.r",
    equations="""
        tau_alpha * dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) * alpha_factor
        trace = (pre.r - mean(pre.r) - threshold_pre) * pos(post.r - mean(post.r) - threshold_post)
        delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post) * pos(post.r - mean(post.r) - threshold_post)*w)
        tau * dw/dt = delta : min=0
   """,
    name = "Covariance learning rule",
    description = "Synaptic plasticity based on covariance, with an additional regularization term."
)

ReversedSynapse = Synapse(
    parameters="""
    """,
    psp="""
        w * pos(reversal - pre.r)
    """,
    name = "Reversed Synapse",
    description = "Higher pre-synaptic activity lowers the synaptic transmission and vice versa."
)

#DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
DAPostCovarianceNoThreshold = Synapse(
    parameters="""
        tau=75.0 : projection
        tau_alpha=1.0 : projection
        tau_trace=60.0 : projection
        regularization_threshold=1.0 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre=0.15 : projection
        threshold_post=0.0 : projection
    """,
    psp = "w * pre.r",
    equations="""
        tau_alpha * dalpha/dt + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0 * (post.sum(dopa) - baseline_dopa)
        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)
        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum
        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau * dw/dt = delta : min=0
    """,
    name = "Covariance DA learning rule",
    description = "Synaptic plasticity in the BG input, like the Covariance learning rule with an additional dopamine modulation which depends on the dopamine receptor type ($DA_{type}(D1) = 1, DA_{type}(D2) = -1$)."
)

#Excitatory synapses STN -> SNr
DA_excitatory = Synapse(
    parameters="""
        tau=50.0 : projection
        tau_alpha=1.0 : projection
        tau_trace=60.0 : projection
        regularization_threshold=2.6 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type= 1 : projection
        threshold_pre=0.0 : projection
        threshold_post= -0.15 : projection
        trace_pos_factor = 1.0 : projection
    """,
    psp = "w * pre.r",
    equations="""
        tau_alpha * dalpha/dt + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0 * (post.sum(dopa) - baseline_dopa)

        a = mean(post.r) - min(post.r) - 0.45 : postsynaptic
        post_thresh = if (-a<threshold_post): -a else: threshold_post : postsynaptic

        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - post_thresh)
        aux = if (trace<0.0): 1 else: 0
        dopa_mod = if (dopa_sum>0): K_burst * dopa_sum * ((1-trace_pos_factor)*aux+trace_pos_factor) else: K_dip * dopa_sum * aux
        delta = dopa_mod * trace - alpha * pos(trace)
        tau * dw/dt = delta : min=0
    """,
    name = "STN Output learning rule",
    description = "Synaptic plasticity in the STN output, similar to the Covariance learning rule with an additional dopamine modulation."
)

#Inhibitory synapses STRD1 -> SNr and STRD2 -> GPe
DA_inhibitory = Synapse(
    parameters="""
        tau=50.0 : projection
        tau_alpha=1.0 : projection
        tau_trace=60.0 : projection
        regularization_threshold=1.0 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type= 1 : projection
        threshold_pre=0.0 : projection
        threshold_post=0.15 : projection
        trace_neg_factor = 1.0 : projection
    """,
    psp = "w * pre.r",
    equations="""
        tau_alpha * dalpha/dt + alpha = pos(-post.mp - regularization_threshold)
        dopa_sum = 2.0 * (post.sum(dopa) - baseline_dopa)

        a = mean(post.r) - min(post.r) - 0.45 : postsynaptic
        post_thresh = if (a>threshold_post) and (DA_type>0): a else: threshold_post : postsynaptic

        trace = if (DA_type>0): pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - post_thresh) else: pos(pre.r - mean(pre.r) - threshold_pre) * (max(post.r) - post.r  - post_thresh)
        aux = if (trace>0): 1 else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum * ((1-trace_neg_factor)*aux+trace_neg_factor) else: aux*DA_type*K_dip*dopa_sum
        tau * dw/dt = dopa_mod * trace - alpha * pos(trace) : min=0
    """,
    name = "Str Output learning rule",
    description = "Synaptic plasticity in the Str output, similar to the Covariance learning (here inverse effect of post-activity) rule with an additional dopamine modulation which depends on the dopamine receptor type ($DA_{type}(D1) = 1, DA_{type}(D2) = -1$)."
)

DAPrediction = Synapse(
    parameters="""
        tau = 100000.0 : projection
    """,
    psp = "w * pre.r",
    equations="""
        aux = if (post.sum(exc)>0): 1.0 else: 3.0  : postsynaptic
        tau*dw/dt = aux * (post.r - baseline_dopa) * pos(pre.r - mean(pre.r)) : min=0
    """,
    name = "Reward Prediction learning rule",
    description = "Simple synaptic plasticity based on covariance."
)

#TraceSynapse = default annarchy synapse

##################################################
##############  CREATION OF THE NEURONS   ########
##################################################
nBG = params['dim_BG']
# IT Input
IT = Population(name='IT', geometry=params['dim_IT'], neuron=ScaledNeuron)
IT.B = changed['baseline_IT']
IT.phi = params['noise_IT']

# Reward Input
PPTN = Population(name='PPTN', geometry=params['dim_SN'], neuron=InputNeuron)
PPTN.tau = 1.0

# PFC
PFC = Population(name='PFC', geometry=params['dim_PFC'], neuron=LinearNeuron)
PFC.phi = params['noise_PFC']
#pfc_base=np.zeros(16)
#pfc_base[9]=1.
PFC.B=params['baseline_PFC']#pfc_base

# SNc
SNc = Population(name='SNc', geometry=params['dim_SN'], neuron=DopamineNeuron)
SNc.B = params['baseline_SNc']

# Striatum direct pathway
StrD1 = Population(name='StrD1', geometry=params['dim_STR'], neuron=LinearNeuron)
StrD1.phi = params['noise_Str']
StrD1.B = changed['baseline_STR']

# Striatum indirect pathway
StrD2 = Population(name='StrD2', geometry=params['dim_STR'], neuron=LinearNeuron)
StrD2.phi = params['noise_Str']
StrD2.B = changed['baseline_STR']

# Striatum feedback pathway
StrThal = Population(name='StrThal', geometry=nBG, neuron=LinearNeuron)
StrThal.phi = params['noise_StrThal']
StrThal.B = params['baseline_StrThal']

# SNr
SNr = Population(name='SNr', geometry=nBG, neuron=LinearNeuron)
SNr.phi = changed['noise_SNr']
SNr.B = changed['baseline_SNr']

# STN
STN = Population(name='STN', geometry=params['dim_STN'], neuron=LinearNeuron)
STN.phi = params['noise_STN']
STN.B = params['baseline_STN']

# GPe
GPe = Population(name='GPe', geometry=nBG, neuron=LinearNeuron)
GPe.phi = changed['noise_GPe']
GPe.B = params['baseline_GPe']

# MD
MD = Population(name='MD', geometry=nBG, neuron=LinearNeuron)
MD.phi = changed['noise_MD']
MD.B = changed['baseline_MD']




#####################################################
########  PROJECTIONS  ##############################
#####################################################

############# FROM INPUT #############

ITPFC = Projection(pre=IT, post=PFC, target='exc', synapse=PostCovariance)
ITPFC.connect_all_to_all(weights=changed['ITPFC.connect_all_to_all']) #Normal(0.3,0.1) )

ITStrD1 = Projection(pre=IT, post=StrD1, target='exc', synapse=DAPostCovarianceNoThreshold)
ITStrD1.connect_all_to_all(weights=Uniform(0, 0.3)) #Normal(0.15,0.15))

ITStrD2 = Projection(pre=IT, post=StrD2, target='exc', synapse=DAPostCovarianceNoThreshold)
ITStrD2.connect_all_to_all(weights=Uniform(0, 0.3)) #Normal(0.15,0.15))
ITStrD2.DA_type = -1

ITSTN = Projection(pre=IT, post=STN, target='exc', synapse=DAPostCovarianceNoThreshold)
ITSTN.connect_all_to_all(weights=Uniform(0, 0.3)) #Normal(0.15,0.15))
ITSTN.DA_type = 1

###############  OUTPUT  ########################

SNrMD = Projection(pre=SNr, post=MD, target='inh', synapse=StandardSynapse)
SNrMD.connect_one_to_one(weights=changed['SNrMD.connect_one_to_one'])


################ REWARD  #######################

PPTNSNc = Projection(pre=PPTN, post=SNc, target='exc', synapse=StandardSynapse)
PPTNSNc.connect_all_to_all(weights=1.0)

StrD1SNc = Projection(pre=StrD1, post=SNc, target='inh', synapse=DAPrediction)
StrD1SNc.connect_all_to_all(weights=0.5)#statt 1.0

SNcStrD1 = Projection(pre=SNc, post=StrD1, target='dopa', synapse=StandardSynapse)
SNcStrD1.connect_all_to_all(weights=1.0)

SNcStrD2 = Projection(pre=SNc, post=StrD2, target='dopa', synapse=StandardSynapse)
SNcStrD2.connect_all_to_all(weights=1.0)

SNcSNr = Projection(pre=SNc, post=SNr, target='dopa', synapse=StandardSynapse)
SNcSNr.connect_all_to_all(weights=1.0)

SNcSTN = Projection(pre=SNc, post=STN, target='dopa', synapse=StandardSynapse)
SNcSTN.connect_all_to_all(weights=1.0)

SNcGPe = Projection(pre=SNc, post=GPe, target='dopa', synapse=StandardSynapse)
SNcGPe.connect_all_to_all(weights=1.0)

#SNcPFC = Projection(pre=SNc, post=PFC, target='dopa')
#SNcPFC.connect_all_to_all(weights=1.0)

#SNcVA = Projection(pre=SNc, post=VA, target='dopa')
#SNcVA.connect_all_to_all(weights=1.0)

################# TEACHING CATEGORIES  ####################

VAPFC = Projection(pre=MD, post=PFC, target='exc', synapse=PostCovariance)
VAPFC.connect_one_to_one(weights=changed['VAPFC.connect_one_to_one'])

PFCMD = Projection(pre=PFC, post=MD, target='exc', synapse=StandardSynapse)
PFCMD.connect_one_to_one(weights=changed['PFCMD.connect_one_to_one'])

################   INNER BG   ###################

StrD1SNr = Projection(pre=StrD1, post=SNr, target='inh', synapse=DA_inhibitory)
StrD1SNr.connect_all_to_all(weights=changed['StrD1SNr.connect_all_to_all']) #Normal(0.025,0.025))
StrD1SNr.regularization_threshold = 1.0
StrD1SNr.DA_type = 1

STNSNr = Projection(pre=STN, post=SNr, target='exc', synapse=DA_excitatory)
STNSNr.connect_all_to_all(weights=Uniform(0, 0.05)) #Normal(0.025,0.025))

StrD2GPe = Projection(pre=StrD2, post=GPe, target='inh', synapse=DA_inhibitory)
StrD2GPe.connect_all_to_all(weights=changed['StrD2GPe.connect_all_to_all']) #Normal(0.025,0.025))
StrD2GPe.regularization_threshold = 2.0
StrD2GPe.DA_type = -1

GPeSNr = Projection(pre=GPe, post=SNr, target='inh', synapse=StandardSynapse)
GPeSNr.connect_one_to_one(weights=changed['GPeSNr.connect_one_to_one'])

###############  LATERALS   ######################

StrD1StrD1 = Projection(pre=StrD1, post=StrD1, target='inh', synapse=StandardSynapse)
StrD1StrD1.connect_all_to_all(weights=changed['StrD1StrD1.connect_all_to_all'])

STNSTN = Projection(pre=STN, post=STN, target='inh', synapse=StandardSynapse)
STNSTN.connect_all_to_all(weights=changed['STNSTN.connect_all_to_all'])

PFCPFC = Projection(pre=PFC, post=PFC, target='inh', synapse=StandardSynapse)
PFCPFC.connect_all_to_all(weights=changed['PFCPFC.connect_all_to_all'])

StrD2StrD2 = Projection(pre=StrD2, post=StrD2, target='inh', synapse=StandardSynapse)
StrD2StrD2.connect_all_to_all(weights=changed['StrD2StrD2.connect_all_to_all'])

StrThalStrThal = Projection(pre=StrThal, post=StrThal, target='inh', synapse=StandardSynapse)
StrThalStrThal.connect_all_to_all(weights=changed['StrThalStrThal.connect_all_to_all'])

SNrSNr = Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse)
SNrSNr.connect_all_to_all(weights=changed['SNrSNr.connect_all_to_all'])

ITIT = Projection(pre=IT, post=IT, target='inh', synapse=StandardSynapse)
ITIT.connect_all_to_all(weights=changed['ITIT.connect_all_to_all'])

#################  FEEDBACK  ####################

MDStrThal = Projection(pre=MD, post=StrThal, target='exc', synapse=StandardSynapse)
MDStrThal.connect_one_to_one(weights=changed['MDStrThal.connect_one_to_one'])

StrThalGPe = Projection(pre=StrThal, post=GPe, target='inh', synapse=StandardSynapse)
StrThalGPe.connect_one_to_one(weights=changed['StrThalGPe.connect_one_to_one'])

StrThalSNr = Projection(pre=StrThal, post=SNr, target='inh', synapse=StandardSynapse)
StrThalSNr.connect_one_to_one(weights=changed['StrThalSNr.connect_one_to_one'])

#################  ATTENTION  ####################

PFCIT = Projection(pre=PFC, post=IT, target='att', synapse=StandardSynapse)
PFCIT.connect_one_to_one(weights=Uniform(0.0, 0.03))
