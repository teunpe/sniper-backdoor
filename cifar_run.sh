#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=cb.out
#SBATCH --error=cb.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl

source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate
export TORCH_HOME=//vol/csedu-nobackup/project/tpeeters/torch

python backdoor_cifar.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar100 --model vgg11_bn --attack input_aware_dynamic --mark_random_init --epochs 50 --lr 0.01 --save --dir //vol/csedu-nobackup/project/tpeeters/results/cifar --data_dir //vol/csedu-nobackup/project/tpeeters/data/cifar-100-python --attack_dir //vol/csedu-nobackup/project/tpeeters/results/cifar --download
