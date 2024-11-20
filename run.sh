#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=cifar.out
#SBATCH --error=cifar.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl

source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate
export TORCH_HOME=//vol/csedu-nobackup/project/tpeeters/torch

python mnist_run.py --dataname fmnist
