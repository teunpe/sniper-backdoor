#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/mnist.out
#SBATCH --error=./logs/mnist.err
#SBATCH --job-name=mnist
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl

source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate
export TORCH_HOME=//vol/csedu-nobackup/project/tpeeters/torch

python mnist_run.py --dataname mnist --run_name mnist --train 
