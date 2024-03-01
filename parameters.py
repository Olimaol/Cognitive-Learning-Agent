"""Loading Model Parameters"""
import random
random.seed()
import numpy as np
np.random.seed()

params = {}
params['num_threads'] = 1
### Basal Ganglia Parameters
params['dim_SN'] = 1
params['dim_PFC'] = 16
params['dim_IT'] = params['dim_PFC']
params['dim_BG'] = params['dim_PFC']
params['dim_STR'] = (5, 5)
params['dim_STN'] = (5, 5)

params['baseline_IT'] = random.uniform(0.0, 0.04)
params['baseline_dopa'] = 0.1
params['baseline_STR'] = 0.4
params['baseline_SNc'] = 0.1
params['baseline_StrThal'] = 0.4
params['baseline_SNr'] = 2.4
params['baseline_STN'] = 0.4
params['baseline_GPe'] = 1.0
params['baseline_MD'] = 0.4
params['baseline_PFC'] = 0.0

params['noise_PFC'] = 0.05
params['noise_Str'] = 0.1
params['noise_StrThal'] = 0.1
params['noise_SNr'] = 1.0
params['noise_STN'] = 0.1
params['noise_GPe'] = 1.0
params['noise_MD'] = 0.0001
params['noise_PM'] = 1.0
params['noise_IT'] = 0.4

params['reversal_SNr'] = 1.0

### Visual System Parameters
## noise NEW
params['noise_V1'] = 0.04#0.05

## V1 Parameters
params['pV1C'] = 2.5

## V4L4 Parameters
params['vV1'] = 1.0
params['pE'] = 1
params['sigmaL4'] = 0.4
params['gHVA4'] = 1.066
params['vV24'] = 1
params['pV24'] = 1
params['vFEFvm'] = 4.0
params['vFEAT1'] = 3.0
params['pFEAT1'] = 3
params['vFEAT2'] = 2.0
params['pFEAT2'] = 2
params['vSP1'] = 0.85
params['pSP1'] = 1
params['vSP2'] = 1
params['pSP2'] = 1
params['vSUR1'] = 0
params['pSUR1'] = 1
params['vSUR2'] = 2
params['pSUR2'] = 2

## V4L23 Parameters
params['p1'] = 4
params['p2'] = 0.25
params['sigmaL23'] = 1.0
params['gHVA2'] = 1.69
params['tau'] = 10
params['vV42'] = 1.0
params['vPFC'] = 1.5
params['pV42'] = 0.25

## FEF Parameters
params['sigmaFEF'] = 0.1
params['cFEF'] = 6
params['vlow'] = 0.2
params['vEv'] = 0.6
params['vSv1'] = 0.6
params['vFEFvm_m'] = 1.3
params['vSvm'] = 0.3
params['vSFix'] = 3
params['vSv2'] = 0.35
params['dogScalingFactor_FEFvm'] = 0.93

### Universal Parameters
params['tau'] = 10
params['learnedWeightsMode'] = False
params['resImg'] = (430, 330)

## Numbers of Neurons in the different areas
params['resVisual'] = (41, 31)
params['PFC_shape'] = (params['dim_PFC'],)
params['V1_shape'] = params['resVisual'] + (1, params['dim_PFC'])
params['V4L4_shape'] = params['resVisual'] + params['PFC_shape']
params['V4L23_shape'] = (21, 16) + params['PFC_shape']
params['FEF_shape'] = params['resVisual']
params['FEFvm_shape'] = params['resVisual'] + (6,)
params['RF_V1_V4L4'] = params['PFC_shape'] + (11, 7) + params['V1_shape'][-2:]

### Projection Parameters
params['viewfield'] = np.array([10.3, 7.8])
params['degPerCell_V4L4'] = params['viewfield'] / params['resVisual']
params['degPerCell_V4L23'] = params['viewfield'] / params['V4L23_shape'][:2]
params['rfsize_V4p'] = [5, 5] * (params['degPerCell_V4L4']
                                 / params['degPerCell_V4L23'])
params['sigma_RF_A_Feat'] = params['rfsize_V4p'] / 3

params['FBA_delay'] = 2
params['FEFv_delay'] = 7
params['RFsize4_23'] = [5, 5]
params['RFsigma4_23'] = [5./3, 5./3]

params['RFsizev_vm'] = [41, 31]
params['RFsigmav_vm'] = [4, 3]



##Stimulus Parameters
params['x_max']=41
params['y_max']=31
params['num_lines']=9
params['border']=3
params['possibleOr']=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
params['phi']=np.random.choice(params['possibleOr'],params['num_lines'])
params['phi']=np.array([50,50,50,50,50,50,50,50,55])
