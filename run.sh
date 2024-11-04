#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl

CLIENTS=5
EPOCHS=50
LOCAL_EPOCHS=2
DATA=mnist
IID=true
DIR='./'
# DIR='//vol/csedu-nobackup/project/tpeeters'
LR=0.1

conda init
conda activate intern
# source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate

python 1-main.py --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR --iid $IID --dir $DIR
python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 --iid $IID --dir $DIR
#python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS --iid $IID
#python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA
python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 --iid $IID --dir $DIR
python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 --iid $IID --dir $DIR
python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 --iid $IID --dir $DIR
python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 --iid $IID --dir $DIR
python 5-backdoor.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 --iid $IID --dir $DIR
python 6-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 --iid $IID --dir $DIR
python 6-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 --iid $IID --dir $DIR
python 6-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 --iid $IID --dir $DIR
python 6-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 --iid $IID --dir $DIR
python 6-personalize_model.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 --iid $IID --dir $DIR

