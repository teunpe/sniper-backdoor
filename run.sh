#!/bin/bash
CLIENTS=2
EPOCHS=2
DATA=fmnist

conda activate intern
<<<<<<< HEAD
python main.py --lr 0.001 --dataname mnist --n_clients 10 --iid --n_epochs 10
python synthetic_data.py --dataname mnist --n_clients 10 --iid
python shadow_network.py --dataname mnist --fake_dir ./results/fake_datasets_mnist --n_epochs 10 --iid
python client_identification.py --epochs 10 --n_clients 10 --dataname mnist
python backdoor.py --source_label 1 --target_label 7 --iid
=======
python 1-main.py --lr 0.001 --dataname $DATA --n_clients $CLIENTS --iid --n_epochs $EPOCHS
python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --iid --n_epochs $EPOCHS
python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS --iid
#python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA
python 5-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --iid
python 6-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --iid
>>>>>>> emnist
