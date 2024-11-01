#!/bin/bash
CLIENTS=2
EPOCHS=2
DATA=fmnist

conda activate intern
python 1-main.py --lr 0.001 --dataname $DATA --n_clients $CLIENTS --iid --n_epochs $EPOCHS
python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --iid --n_epochs $EPOCHS
python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS --iid
#python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA
python 5-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --iid
python 6-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --iid
