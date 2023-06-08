#!/bin/sh
ulimit -n 65535

### Preprocess ###
cd preprocess/
python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5 -er 0 -od ../data/ -c 20
#python DataPackager.py -d dc2017_0101to0331.csv --minLat 38.7919 --maxLat 38.9960 --minLng -77.1200 --maxLng -76.9093 --refGridH 2.5 --refGridW 2.5 -er 0 -od ../data/ -c 20
cd ../



### Which Data to Use ###
datapath=data/ny2016_0101to0331/
#datapath=data/dc2017_0101to0331/



### GallatExt ###
python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net GallatExt -tag RefGaaRN
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net GallatExt -tag RefGaaRN
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag RefGaaRN



##### Baselines #####
### HA ###
#python HistoricalAverage.py -dr $datapath -c 20 -gid 0 -sch all -tag HAplus
#python HistoricalAverage.py -dr $datapath -c 20 -gid 0 -sch tendency -tag HAt
#python HistoricalAverage.py -dr $datapath -c 20 -gid 0 -sch periodicity -tag HAp

### AR ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net AR -tag AR
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net AR -tag AR
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag AR



##### Other Models #####
### Gallat ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -tt pretrain -net Gallat -t 0 -tag Gallat
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -tt retrain -r pretrained_model.pth -t 0 -tag Gallat
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag Gallat

### LSTNet ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net LSTNet -rar refAR.pth -tag LSTNet
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net LSTNet -rar refAR.pth -tag LSTNet
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag LSTNet

### GCRN ###
#python Trainer.py -dr $datapath -mfb 1 -c 20 -gid 0 -m trainNeval -net GCRN -tag GCRN
#python Trainer.py -dr $datapath -mfb 1 -c 20 -gid 0 -m train -net GCRN -tag GCRN
#python Trainer.py -dr $datapath -mfb 1 -c 20 -gid 0 -m eval -e eval.pth -tag GCRN

### GEML ###
#python Trainer.py -dr $datapath -mfb 1 -c 20 -gid 0 -m trainNeval -net GEML -tag GEML
#python Trainer.py -dr $datapath -mfb 1 -c 20 -gid 0 -m train -net GEML -tag GEML
#python Trainer.py -dr $datapath -mfb 1 -c 20 -gid 0 -m eval -e eval.pth -tag GEML



##### Variants #####
### GallatExt-1 (No tuning) ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net GallatExt -t 0 -tag RefGaaRNNoTune
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net GallatExt -t 0 -tag RefGaaRNNoTune
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag RefGaaRNNoTune

### GallatExt-2 (Concatenation scheme) ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net GallatExtFull -tag RefGaaRNConcat
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net GallatExtFull -tag RefGaaRNConcat
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag RefGaaRNConcat

### GallatExt-3 (Leverage tuning) ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net GallatExt -re 0.2 -tag RefGaaRNWSum
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net GallatExt -re 0.2 -tag RefGaaRNWSum
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -re 0.2 -tag RefGaaRNWSum

### GallatExt-4 (Unified graph) ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net GallatExt -u 1 -tag GallatExt4
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net GallatExt -u 1 -tag GallatExt4
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -u 1 -tag GallatExt4

### GallatExt-5 (Shifting scheme) ###
#python Trainer.py -dr $datapath -c 20 -gid 1 -m trainNeval -net GallatExt -re -2 -tag RefGaaRNShift
#python Trainer.py -dr $datapath -c 20 -gid 1 -m train -net GallatExt -re -2 -tag RefGaaRNShift
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e model_save/20211201_18_08_18.pth -re -2 -tag RefGaaRNShift

### GallatExt-6 (Tuning with AR) ###
#python Trainer.py -dr $datapath -c 20 -gid 0 -m trainNeval -net GallatExt -rar refAR.pth -tag GallatExt6
#python Trainer.py -dr $datapath -c 20 -gid 0 -m train -net GallatExt -rar refAR.pth -tag GallatExt6
#python Trainer.py -dr $datapath -c 20 -gid 0 -m eval -e eval.pth -tag GallatExt6
