#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=iid.out
#SBATCH --error=iid.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl


DATAS="cifar100"
RUN_NAME="cifar"
IID="--iid"
DIR='//vol/csedu-nobackup/project/tpeeters'
# DIR='./'
TEST_FREQ=1

# conda init
# conda activate intern
source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate

CLIENTS=10
EPOCHS=23
LOCAL_EPOCHS=1
DATA=cifar100
LR=0.001
MOMENTUM=0.9

python 1-main.py --run_name $RUN_NAME --test_freq $TEST_FREQ --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR $IID --dir $DIR --momentum $MOMENTUM || exit
# python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 $IID --dir $DIR || exit
# python 3-shadow_network.py --dataname $DATA - --n_epoch $EPOCHS $IID || exit
# python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA || exit
# python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR || exit
# python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR || exit
# python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR || exit
# python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR || exit
# python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR || exit
# python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR --momentum $MOMENTUM || exit
# python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR --momentum $MOMENTUM || exit
# python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR --momentum $MOMENTUM || exit
# python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIE5NTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR --momentum $MOMENTUM || exit
# python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR --momentum $MOMENTUM || exit


# if [[ $DATAS == *"emnist"* ]]; then
#     CLIENTS=13
#     EPOCHS=30
#     LOCAL_EPOCHS=2
#     DATA=emnist
#     LR=0.01
#     MOMENTUM=0.9

#     python 1-main.py --run_name $RUN_NAME --test_freq $TEST_FREQ --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR $IID --dir $DIR --momentum $MOMENTUM || exit
#     # python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 $IID --dir $DIR || exit
#     # python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS $IID || exit
#     # python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR --momentum $MOMENTUM || exit
# fi

# if [[ $DATAS == *"fmnist"* ]]; then
#     CLIENTS=5
#     EPOCHS=200
#     LOCAL_EPOCHS=1
#     DATA=fmnist
#     LR=0.00001
#     MOMENTUM=0.0

#     python 1-main.py --run_name $RUN_NAME --test_freq $TEST_FREQ --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR $IID --dir $DIR --momentum $MOMENTUM || exit
#     # python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 $IID --dir $DIR || exit
#     # python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS $IID || exit
#     # python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR || exit
#     python 5-backdoor.py --run_name $RUN_NAME --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR --momentum $MOMENTUM || exit
#     python 6-personalize_model.py --run_name $RUN_NAME --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR --momentum $MOMENTUM || exit
# fi


