#!/bin/bash
conda activate intern
python main.py --lr 0.001 --dataname emnist --n_clients 10 --iid --n_epochs 10
python synthetic_data.py --dataname emnist --n_clients 10 --iid 
python shadow_network.py --dataname emnist --n_epochs 10 --iid
python client_identification.py --epochs 10 --n_clients 10 --dataname emnist
python personalize_model.py --epochs 10 --n_clients 10 --dataname emnist --iid