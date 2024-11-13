#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=run_multiple.out
#SBATCH --error=run_multiple.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl

source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate

python mnist_run.py --dataname fmnist