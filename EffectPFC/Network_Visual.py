"""
Combined network definition of the Attention Model
    Author: Alex Schwarz alexschw@hrz.tu-chemnitz.de
    ANNarchy port of Frederik Beuth's Model (beuth@cs.tu-chemnitz.de)
    Version 1.0 - 19.04.2019
"""
import pylab as plt
import numpy as np
np.random.seed()
from scipy.io import loadmat

from ANNarchy import Neuron, Population, Projection, setup, Synapse, Uniform, Constant
from ANNarchy.extensions.convolution import Pooling, Convolution
from parameters import params
from functions import rangeX, Gaussian2D, positive
from Connections import con_scale
from changed_val import changed
setup(num_threads=params['num_threads'] )

minVis = Constant('minVis', 0)

##########################################
##########  NEURON DEFINITION   ##########
##########################################
## Input Neuron: Has to be set to value. Does not change over time.
Inp_Neuron = Neuron(parameters="r = 0.0")

## Basic Auxillary Neuron is transmitting an unmodified input
Aux_Neuron = Neuron(equations="""r = sum(exc)""")


## Neuron of V1 population: Applies the power rule to the given baseline input
# See Eq 4.29
V1_Neuron = Neuron(
    parameters="""
        pV1C = 'pV1C' : population
	tau_up = 1.0 : population
	tau_down = 20.0 : population
        baseline = 0.0
        noise = 'noise_V1' : population
        frequency = 1/15. : population
    """,
    equations="""
	f = sin(2*pi*frequency*t) : population
	rand = if (f>0.98): Uniform(-1.0,1.0) else: rand
	basis=baseline + noise * rand : min=0
        dr/dt = if (baseline>0.0): (pow(basis, pV1C)-r)/tau_up else: -r/tau_down : min=minVis
    """,
    extra_values=params
)

## Neuron of Layer 4 in Area V4: Receives Input from V1, V4L23, and FEFvm
# See Eq 4.30-4.32 / 4.35-4.37 / 4.41
V4L4_Neuron = Neuron(
    parameters="""
        sigmaL4 = 'sigmaL4' : population
        gHVA4 = 'gHVA4' : population
        tau = 'tau' : population
        vV1 = 'vV1' : population
        vFEFvm = 'vFEFvm' : population
        vV24 = 'vV24' : population
        pV24 = 'pV24' : population
        vFEAT1 = 'vFEAT1' : population
        pFEAT1 = 'pFEAT1' : population
        pE = 'pE' : population
        vSP1 = 'vSP1' : population
        noise = 'noise_V4L4' : population
    """,
    equations="""
        E = power(vV1*clip(sum(exc), 0, 1), pE)
        A = vFEFvm*sum(A_SP) + vV24*power(sum(A_FEAT), pV24)
        SFEAT = power(vFEAT1*clip(sum(S_FEAT), 0, 1), pFEAT1)
        SSP = vSP1*sum(S_SP)
        S = E*(1+A+SFEAT+SSP+0*sum(S_SUR))
        tau * dr /dt = -r + gHVA4 * E * (1 + A) / (sigmaL4 + S) + noise * Uniform(-1.0,1.0) : min=minVis
    """,
    extra_values=params
)

## Neuron of Layer 2/3 in Area V4: Receives Input from V4L4, and PFC
# See Eq 4.48-4.50
V4L23_Neuron = Neuron(
    parameters="""
        sigmaL23 = 'sigmaL23' : population
        gHVA2 = 'gHVA2' : population
        tau = 'tau' : population
        vV42 = 'vV42' : population
        pV42 = 'pV42' : population
        vPFC = 'vPFC' : population
        noise = 'noise_V4L23' : population
    """,
    equations="""
	basis=vV42*sum(exc) : min=0
        S = pow(basis, pV42) * (1 + vPFC*sum(A_PFC))
        tau * dr /dt = -r + gHVA2 * S / (sigmaL23 + S)  + noise * Uniform(-1.0,1.0) : min=minVis, max = 1.0
    """,
    extra_values=params
)

## Neuron of visual Layer in FEF: Receives Input from V4L23
# See Eq 4.53-4.60
FEFv_Neuron = Neuron(
    parameters="""
        tau = 'tau' : population
        sigmaFEF = 'sigmaFEF' : population
        cFEF = 'cFEF' : population
        noise = 'noise_FEFv' : population
    """,
    equations="""
        q = (sum(exc) * (1 + sigmaFEF) / (sum(exc) + sigmaFEF))
        tau * dr /dt = -r + pos(q * (1 + cFEF) - cFEF) + noise * Uniform(-1.0,1.0) : min=minVis
    """,
    extra_values=params
)

## Neuron of visuo-motoric Layer in FEF: Receives Input from FEFv and FEFm
# See Eq 4.61-4.64
FEFvm_Neuron = Neuron(
    parameters="""
        tau = 'tau' : population
        vlow = 'vlow' : population
        vEv = 'vEv' : population
        vSv1 = 'vSv1' : population
        vFEFv = 1.0
        noise = 'noise_FEFvm' : population
        vSBG = 0 : population
    """,
    equations="""
        ES = clip(vEv*sum(E_v)-vSv1*sum(S_v)-vSBG*sum(S_BG),0,1)
        E = vlow * pos(vEv*sum(E_v)) + (1-vlow) * ES
        tau * dr /dt = 0
    """,
    extra_values=params
)#tau * dr /dt = -r + vFEFv * E + (1-vFEFv) * sum(E_m) + noise * Uniform(-1.0,1.0) : min=minVis

## Neuron of motoric Layer in FEF: Receives Input from FEFvm and FEFfix
# See Eq 4.68-4.71
FEFm_Neuron = Neuron(
    parameters="""
        tau = 'tau' : population
        vFEFvm_m = 'vFEFvm_m' : population
        vSvm = 'vSvm' : population
        vSFix = 'vSFix' : population
        noise = 'noise_FEFm' : population
	id        = -1 
	threshold = 1.0  :population
    """,
    equations="""
        svm = sum(vm)
        tau * dr /dt = -r + vFEFvm_m*sum(vm) - vSvm*max(svm) - vSFix*sum(fix) + noise * Uniform(-1.0,1.0) : min=minVis
	decision = if (r>threshold): id else: -1
    """,
    extra_values=params
)


##########################################
######### POPULATION DEFINITION  #########
##########################################
#Input_Pop = Population(params['resVisual'], Inp_Neuron, name='Image')
V1 = Population(params['V1_shape'], V1_Neuron, name='V1')
V4L4 = Population(params['V4L4_shape'], V4L4_Neuron, name='V4L4')
V4L23 = Population(params['V4L23_shape'], V4L23_Neuron, name='V4L23')
FEFv = Population(params['FEF_shape'], FEFv_Neuron, name='FEFv')
FEFvm = Population(params['FEFvm_shape'], FEFvm_Neuron, name='FEFvm')
FEFm = Population(params['FEF_shape'], FEFm_Neuron, name='FEFm', stop_condition="decision>-1")
FEFm.id = np.arange(0,params['FEF_shape'][0]*params['FEF_shape'][1]).tolist()
#PFC = Population(params['PFC_shape'], Inp_Neuron, name='PFC')
AuxA = Population(params['resVisual'], Aux_Neuron, name='AuxA')
AuxE = Population(params['V4L23_shape'][:2], Aux_Neuron, name='AuxE')
FEFfix = Population(name='FEFfix', geometry=1, neuron=Inp_Neuron)

##########################################
######### CONNECTION DEFINITION  #########
##########################################
## Connection of V1 -> V4 L4
# load the pretrained weights and transform it into a 4D Bank of Filters
#W = np.array(loadmat('WeightData.mat')['W'], dtype='float32')
#FilterBank = np.swapaxes(np.reshape(W, params['RF_V1_V4L4'], order='F'), 1, 2)

w14=Gaussian2D(1.0, np.array(params['RF_V1_V4L4'][1:3]), np.array(params['RF_V1_V4L4'][1:3])/5.)
FilterBank=np.zeros(params['RF_V1_V4L4'])
for i in range(params['V4L4_shape'][2]):
	FilterBank[i,:,:,0,i] = w14/w14.sum()*5

ssList14 = []
Center = [(n - 1) // 2 for n in params['V1_shape'][-2:]]
for Row, Col in rangeX(params['V4L4_shape'][:2]):
    ssList14.append([Row, Col] + Center)

# create the convolution Projection
#V1_V4L4 = Convolution(V1, V4L4, target='exc', weights=FilterBank, method='filter',##OLD
#                      padding='border', multiple=True, subsampling=ssList14)########OLD
V1_V4L4 = Convolution(V1, V4L4, target='exc')########################################NEW
V1_V4L4.connect_filters(weights=FilterBank, padding='border', subsampling=ssList14)##NEW


## Connection of the V4 Populations, L4 => L2/3 (excitatory)
# The weight is a 3x3 Gaussian with maximum 1, width (sigma) 1
w42 = Gaussian2D(1.0, changed['RFsize4_23'], changed['RFsigma4_23'])[:, :, None]
w42 /= w42.sum()
pspText = 'w*power(pre.r, {p1})'.format(**params)
ssList42 = []
for Row, Col, Plane in rangeX(params['V4L23_shape']):
    ssList42.append([Row * 2 + 1, Col * 2 + 1, Plane])
#V4L4_V4L23 = Convolution(V4L4, V4L23, target='exc', psp=pspText, weights=w42,##########OLD
#                         method='filter', keep_last_dimension=True,####################OLD
#                         subsampling=ssList42)#########################################OLD
V4L4_V4L23 = Convolution(V4L4, V4L23, target='exc', psp=pspText)########################NEW
V4L4_V4L23.connect_filter(weights=w42, subsampling=ssList42, keep_last_dimension=True)##NEW

## Connection of the V4 Populations, L2/3 => L4 (feature-based amplification)
# The weight is a 3x3 Gaussian with maximum 1, width (sigma) 0.6
w24 = Gaussian2D(1.0, [3, 3], params['sigma_RF_A_Feat'])[:, :, None]
ssList24 = []
for Row, Col, Plane in rangeX(params['V4L4_shape']):
    ssList24.append([Row // 2, Col // 2, Plane])
#V4L23_V4L4A = Convolution(V4L23, V4L4, target='A_FEAT', operation='max',############################################OLD
#                          weights=w24, delays=params['FBA_delay'],##################################################OLD
#                          method='filter', keep_last_dimension=True,################################################OLD
#                          subsampling=ssList24)#####################################################################OLD
V4L23_V4L4A = Convolution(V4L23, V4L4, target='A_FEAT', operation='max')#############################################NEW
V4L23_V4L4A.connect_filter(weights=w24, delays=params['FBA_delay'], subsampling=ssList24, keep_last_dimension=True)##NEW

## Connection of the V4 Populations, L2/3 => L4 (feature-based suppression)
# The previous weights are used, but calculating another post-synaptic
# potential. See Eq 4.39-4.41
pspText = 'power(w*({vFEAT2}*pre.r), {pFEAT2})'.format(**params)
#V4L23_V4L4SFE = Convolution(V4L23, V4L4, target='S_FEAT', operation='mean',###############OLD
#                            psp=pspText, weights=w24, method='filter',####################OLD
#                            keep_last_dimension=True, subsampling=ssList24)###############OLD
V4L23_V4L4SFE = Convolution(V4L23, V4L4, target='S_FEAT', psp=pspText, operation='mean')###NEW
V4L23_V4L4SFE.connect_filter(weights=w24, subsampling=ssList24, keep_last_dimension=True)##NEW

## Connection of the V4 Populations, L2/3 => L4 (spatial supp.)
# A difference of Gaussians is used as weight
wPos = Gaussian2D(1.0, [13, 13], [3, 3])
wNeg = Gaussian2D(2.0, [13, 13], [1.0, 1.0])
wDoG = (positive(wPos - wNeg) / np.sum(positive(wPos - wNeg)))[:, :, None]
#V4L23_V4L4SUR = Convolution(V4L23, V4L4, target='S_SUR', weights=wDoG,#####################OLD
#                            method='filter', keep_last_dimension=True,#####################OLD
#                            subsampling=ssList24)##########################################OLD
V4L23_V4L4SUR = Convolution(V4L23, V4L4, target='S_SUR')####################################NEW
V4L23_V4L4SUR.connect_filter(weights=wDoG, subsampling=ssList24, keep_last_dimension=True)##NEW

## Connection from V4 L2/3 to FEF visual (excitatory)
# The auxiliary population is used to pool the down-sampled V4L23 Population.
# Afterwards it could be up-sampled again. The combination of the two is
# currently not possible in ANNarchy
ssList2v = ssList24[9::params['V4L4_shape'][-1]]
#V4L23_AuxE = Pooling(V4L23, AuxE, target='exc', operation='max',##OLD
#                     extent=(1, 1) + params['PFC_shape'])#########OLD
V4L23_AuxE = Pooling(V4L23, AuxE, target='exc', operation='max')###NEW
V4L23_AuxE.connect_pooling(extent=(1, 1) + params['PFC_shape'])####NEW
AuxE_FEFv = Projection(AuxE, FEFv, target='exc')
AuxE_FEFv.connect_with_func(con_scale, factor=2, delays=params['FEFv_delay'])

## Connections from FEF visual to FEF visuo-motoric(excitatory and suppressive)
# A lowered Gaussian is used to simulate the combined responses
G = Gaussian2D(1.0, changed['RFsizev_vm'], changed['RFsigmav_vm'])
v_vm_shape = (params['FEFvm_shape'][-1], 1, 1)
wvvm = np.tile((G - params['vSv2'])[None, :, :], v_vm_shape)
wvvm *= params['dogScalingFactor_FEFvm']**np.arange(6)[:, None, None]
# The plus sign(+) is needed, so that wvvm will not be overwritten
#FEFv_FEFvmE = Convolution(FEFv, FEFvm, target='E_v', weights=positive(+wvvm),##OLD
#                          method='filter', multiple=True)######################OLD
#FEFv_FEFvmS = Convolution(FEFv, FEFvm, target='S_v', weights=positive(-wvvm),##OLD
#                          method='filter', multiple=True)######################OLD
FEFv_FEFvmE = Convolution(FEFv, FEFvm, target='E_v')############################NEW
FEFv_FEFvmE.connect_filters(weights=positive(+wvvm))############################NEW
FEFv_FEFvmS = Convolution(FEFv, FEFvm, target='S_v')############################NEW
FEFv_FEFvmS.connect_filters(weights=positive(-wvvm))############################NEW

## Connection from FEF visuo-motoric to V4 L4 (amplification)
# The auxiliary population is used to pool FEFvm activities over different
# layers. Then a one to many connectivity is used. This combination is
# currently not possible in one step
#FEFvm_AuxA = Pooling(FEFvm, AuxA, target='exc', operation='mean',####OLD
#                     extent=(1, 1, params['FEFvm_shape'][-1]))#######OLD
FEFvm_AuxA = Pooling(FEFvm, AuxA, target='exc', operation='mean')#####NEW
FEFvm_AuxA.connect_pooling(extent=(1, 1, params['FEFvm_shape'][-1]))##NEW
otmV4 = np.ones(params['V4L4_shape'][-1])[:, None, None]
#AuxA_V4L4A = Convolution(AuxA, V4L4, target='A_SP', weights=otmV4,##OLD
#                         method='filter', multiple=True)############OLD
AuxA_V4L4A = Convolution(AuxA, V4L4, target='A_SP')##################NEW
AuxA_V4L4A.connect_filters(weights=otmV4)############################NEW

## Connection from FEF visuo-motoric to V4 L4 (suppressive)
# A rectified inverse Gaussian is used as weight
#G = Gaussian2D(1.0, [11, 9], [4, 3])
#wSP = np.tile(positive(1 - G**0.125)[None, :, :], params['PFC_shape'] + (1, 1))
G = Gaussian2D(1.0, [40, 40], [4, 4])
wSP = np.tile(positive(1 - G**0.125)[None, :, :], params['PFC_shape'] + (1, 1))
wSP[wSP>0.188]=0.188
#AuxA_V4L4S = Convolution(AuxA, V4L4, target='S_SP', weights=wSP,##OLD
#                         method='filter', multiple=True)##########OLD
AuxA_V4L4S = Convolution(AuxA, V4L4, target='S_SP')################NEW
AuxA_V4L4S.connect_filters(weights=wSP)############################NEW

## Connection from FEF visuo-motoric to FEF motoric (mean pooling)
#FEFvm_FEFm = Pooling(FEFvm, FEFm, target='vm', operation='mean',#####OLD
#                     extent=(1, 1, params['FEFvm_shape'][-1]))#######OLD
FEFvm_FEFm = Pooling(FEFvm, FEFm, target='vm', operation='mean')######NEW
FEFvm_FEFm.connect_pooling(extent=(1, 1, params['FEFvm_shape'][-1]))##NEW

## Connection from FEF motoric to FEF visuo-motoric, distributing the activity
otmFEF = np.ones(params['FEFvm_shape'][-1])[:, None, None]
#FEFm_FEFvm = Convolution(FEFm, FEFvm, target='E_m', weights=otmFEF,##OLD
#                         method='filter', multiple=True)#############OLD
FEFm_FEFvm = Convolution(FEFm, FEFvm, target='E_m')###################NEW
FEFm_FEFvm.connect_filters(weights=otmFEF)############################NEW

## Connection from prefrontal Cortex to V4 L2/3 (layerwise amplification)
#PFC_V4L23 = Projection(PFC, V4L23, target='A_PFC')
#PFC_V4L23.connect_with_func(one_to_dim, postDim=2)

## Connection from FEFfix to FEF motoric
FEFfix_FEFm = Projection(FEFfix, FEFm, target='fix')
FEFfix_FEFm.connect_all_to_all(1.0)

