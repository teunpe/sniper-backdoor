#!/bin/bash
CLIENTS=2
EPOCHS=2
DATA=mnist
IID=true

conda init
conda activate intern
python 1-main.py --lr 0.001 --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --iid $IID
python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --iid $IID
python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS --iid $IID
#python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA
python 6-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --iid $IID
python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.1 --iid $IID
# python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 --iid $IID
# python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 --iid $IID
# python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 --iid $IID
# python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 --iid $IID
