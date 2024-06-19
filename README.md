# PyTorch implementation of Multi-site Analysis Using Privacy-preserving Federated Learning and Domain Adaptation

## Dependencies
- Python 3.6
- Pytorch 1.1.0
- tensorboardX
- nilearn
- deepdish
- numpy

## Data
### Data Download & Preprocessing
```shell
sh run.sh
```

## How to run ?
Here we show a few examples using different strategies listed in the paper. Please check the meaning of configurations in each script.
### Single 
```
python single.py --split ${SPLIT} --site ${SITE}
```
### Ensemble
```
python ensemble.py --split ${SPLIT} --site ${SITE}
```
### Cross
```
python cross.py --trainsite ${TRAINSITE}
```
### MIX
```
python mix.py --split ${SPLIT}
```
### Vanilla Fed
#### vary on noise
```
python federated.py --split ${SPLIT} --noise ${NOISE} --type ${TYPE}
```
#### vary on pace
```
python federated.py --split ${SPLIT} --pace ${PACE}
```
### Fed + MOE
```
python federated_MoE.py --split ${SPLIT}
```
### Fed + Align
```
python federated_align.py --split ${SPLIT}
```

