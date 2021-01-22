# Cognitive Learning Agent

Source code of the cognitive learning agent for emergent attention from basal ganglia onto the visual system from Maith, Schwarz & Hamker (2021).

## Authors:

* Oliver Maith (oliver.maith@informatik.tu-chemnitz.de)
* Alex Schwarz (alex.schwarz@s2012.tu-chemnitz.de)

Visual system model is based on *[Beuth (2019)](https://scholar.google.com/scholar?hl=de&as_sdt=0%2C5&q=Visual+attention+in+primates+and+for+machines+-+neuronal+mechanisms&btnG=)*<br/>
Basal Ganglia model is based on *[Villagrasa et al. (2018).](https://doi.org/10.1523/JNEUROSCI.0874-18.2018)*

## Using the Scripts

### Results Pipelines

Result | analysis | simulation | comment
-|-|-|-
input stimuli | - | `python create_stim.py X` | generates the random stimuli which are used in all simulations<br/>X = σ\_V1<br/>1 \- σ\_V1 = 22°<br/>2 \- σ\_V1 = 30°<br/>3 \- σ\_V1 = 14°
trial_BG.svg (Fig. 3) | `python trial_BG.py 1 0` | `python run_cla_one_trial.py` | simulation actually simulates two trials, second trial only for negative SNc response
Training.svg (Fig. 6),<br/>Training.txt (Training statistics) | `python Training.py` | `python run_cla_Training X 0` | X = simulaion IDs (e.g. 60 different simulations)
T2.svg (Fig. 7),<br/>T2.txt (T2 statistics) | `python T2.py` | `python run_cla.py X Y` | X = simulation IDs, Y = stimulus ID (0-T1, 1-T1-reversed)
T1_performance.txt (T1 performance) | `python T1_performance.py` | `python run_cla.py X Y` | X = simulation IDs, Y = stimulus ID (0-T1, 1-T1-reversed) like T2.py
Learn_BG.svg (Fig. 4) | `python Learn_BG.py` | `python run_cla.py X 0` | X = simulation IDs, for analyses use simulations which learned 65° gain
Learn_PFC.svg (Fig. 5) | `python Learn_PFC.py` | `python run_cla.py X 0` | X = simulation IDs, for analyses use simulations which learned 65° gain
EffectPFC.svg (Fig. 8) | `python EffectPFC.py` | `run_parallel_EffectPFC.sh` | in folder EffectPFC

### Additional Scripts

* `python rename.py`: change simulation IDs of specified files and copy files in new folder
* `run_parallel.sh`: run python scripts parallel, multiple times

# Platforms

* GNU/Linux

# Dependencies

* ANNarchy >= 4.6.9.3


