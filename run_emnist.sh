#!/bin/bash
conda activate intern
python main.py --lr 0.001 --dataname emnist --n_clients 10 --iid --n_epochs 10 --datadir '/media/teun/ChonkyBoi/internship/data'
python synthetic_data.py --dataname emnist --n_clients 10 --iid --datadir '/media/teun/ChonkyBoi/internship/data''
python shadow_network.py --dataname emnist --fake_dir ./results/fake_datasets_mnist --n_epochs 10 --iid --datadir '/media/teun/ChonkyBoi/internship/data''
python client_identification.py --epochs 10 --n_clients 10 --dataname emnist
python personalize_model.py --epochs 10 --n_clients 10 --dataname emnist --iid --datadir '/media/teun/ChonkyBoi/internship/data''