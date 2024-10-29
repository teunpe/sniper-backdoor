#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate
python main.py --lr 0.001 --dataname mnist --n_clients 10 --iid --n_epochs 10
python synthetic_data.py --dataname mnist --n_clients 10 --iid
python shadow_network.py --dataname mnist --fake_dir ./results/fake_datasets_mnist --n_epochs 10 --iid
python client_identification.py --epochs 10 --n_clients 10 --dataname mnist
python personalize_model.py --epochs 10 --n_clients 10 --dataname mnist --iid