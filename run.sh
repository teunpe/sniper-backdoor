#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=1.out
#SBATCH --error=1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user teun.peeters@ru.nl


DATAS='mnist, emnist, fmnist'

if [[$DATAS == *"mnist"*]]; then
    CLIENTS=5
    EPOCHS=50
    LOCAL_EPOCHS=2
    DATA=mnist
    IID="--iid"
    # DIR='./'
    DIR='//vol/csedu-nobackup/project/tpeeters'
    LR=0.1
    MOMENTUM=0.9

    # conda init
    # conda activate intern
    source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate

    python 1-main.py --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR $IID --dir $DIR --momentum $MOMENTUM || exit
    # python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 $IID --dir $DIR || exit
    # python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS $IID || exit
    # python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR --momentum $MOMENTUM || exit
fi

if [[$DATAS == *"emnist"*]]; then
    CLIENTS=13
    EPOCHS=200
    LOCAL_EPOCHS=2
    DATA=emnist
    IID="--iid"
    # DIR='./'
    DIR='//vol/csedu-nobackup/project/tpeeters'
    LR=0.01
    MOMENTUM=0.9

    # conda init
    # conda activate intern
    source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate

    python 1-main.py --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR $IID --dir $DIR --momentum $MOMENTUM || exit
    # python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 $IID --dir $DIR || exit
    # python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS $IID || exit
    # python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR --momentum $MOMENTUM || exit
fi

if [[$DATAS == *"fmnist"*]]; then
    CLIENTS=5
    EPOCHS=200
    LOCAL_EPOCHS=1
    DATA=fmnist
    IID="--iid"
    # DIR='./'
    DIR='//vol/csedu-nobackup/project/tpeeters'
    LR=0.00001
    MOMENTUM=0.0

    # conda init
    # conda activate intern
    source //vol/csedu-nobackup/project/tpeeters/venv/bin/activate

    python 1-main.py --dataname $DATA --n_clients $CLIENTS --n_epochs $EPOCHS --lr $LR $IID --dir $DIR --momentum $MOMENTUM || exit
    # python 2-synthetic_data.py --dataname $DATA --n_clients $CLIENTS --n_epochs 950 $IID --dir $DIR || exit
    # python 3-shadow_network.py --dataname $DATA - --n_epochs $EPOCHS $IID || exit
    # python 4-client_identification.py --epochs $EPOCHS --n_clients $CLIENTS --dataname $DATA || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR || exit
    python 5-backdoor.py --lr 0.001 --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.001 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.005 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.010 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.015 $IID --dir $DIR --momentum $MOMENTUM || exit
    python 6-personalize_model.py --lr $LR --n_clients $CLIENTS --dataname $DATA --epsilon 0.020 $IID --dir $DIR --momentum $MOMENTUM || exit
fi


