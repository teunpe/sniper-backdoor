#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --color --verbose 1 --attack badnet --pretrained --validate_interval 1 --epochs 50 --lr 1e-2
"""  # noqa: E501

'''
python backdoor_cifar.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar100 --model vgg11_bn --attack input_aware_dynamic --mark_random_init --epochs 50 --lr 0.01 --save --dir //vol/csedu-nobackup/project/tpeeters/results/cifar --folder_path //vol/csedu-nobackup/project/tpeeters/data/cifar-100-python
'''
import trojanvision
import argparse
import torch
import os
from torchvision.datasets import CIFAR100
from models import build_model

from trojanvision.attacks import BackdoorAttack
import trojanvision.configs

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, default='./results', help='directory')
    # trojanvision.environ.add_argument(parser)
    # trojanvision.datasets.add_argument(parser)
    # trojanvision.models.add_argument(parser)
    # trojanvision.trainer.add_argument(parser)
    # trojanvision.marks.add_argument(parser)
    # trojanvision.attacks.add_argument(parser)
    # kwargs = vars(parser.parse_args())
    # args = parser.parse_args()
    
    dataset = CIFAR100(root='//vol/csedu-nobackup/project/tpeeters/data', train=True,
                        download=True)
    # env = trojanvision.environ.create(**kwargs)
    # dataset = trojanvision.datasets.create(**kwargs)
    # model = trojanvision.models.create(model_name='vgg11_bn', model='vgg11_bn', dataset_name='cifar100', dataset=dataset)
    # # model = build_model(100, 'cifar100')
    # server_model = torch.load(os.path.join(args.dir, 'cifar100_iid_True_server_results.pt'))['model']
    # model.load_state_dict(server_model)
    # trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    # mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    # attack: BackdoorAttack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    # if env['verbose']:
    #     trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    # attack.attack(**trainer)