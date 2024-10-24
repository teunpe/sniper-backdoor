#!/bin/bash
conda activate intern
python main.py --lr 0.001 --dataname mnist --n_clients 10 --iid --n_epochs 10
python synthetic_data.py --dataname mnist --n_clients 10 --iid
python shadow_network.py --dataname mnist --fake_dir ./results/fake_datasets_mnist --n_epochs 10 --iid
python client_identification.py --epochs 10 --n_clients 10 --dataname mnist
python backdoor.py --source_label 1 --target_label 7 --iid