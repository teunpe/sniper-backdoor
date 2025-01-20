#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --color --verbose 1 --attack badnet --pretrained --validate_interval 1 --epochs 50 --lr 1e-2
"""  # noqa: E501

'''
python backdoor_cifar.py --color --tqdm --verbose 1 --pretrained --validate_interval 1 --dataset cifar100 --model vgg11_bn --attack input_aware_dynamic --mark_random_init --epochs 50 --lr 0.01 --save --dir //vol/csedu-nobackup/project/tpeeters/results/cifar --data_dir //vol/csedu-nobackup/project/tpeeters/data/cifar-100-python --download
'''
import trojanvision
import argparse
import torch
import os
from torchvision.datasets import CIFAR100
from models import build_model
from utils import get_dataset, backdoor_train, backdoor_evaluate

from trojanvision.attacks import BackdoorAttack
import trojanvision.configs

import trojanvision.data
from tqdm import tqdm

from copy import deepcopy

import pickle

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

def personalize_model(results_dir: str, args):
    """Personalize the backdoored CIFAR100 model. 

    Parameters
    ----------
    results_dir : str
        Directory where results are stored
    args

    Returns
    -------
    model
    """    
    # load the backdoored model
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_backdoor_results.pt')
    results = torch.load(path)
    holdoutloader = results['holdoutloader'] # training data not used in training
    backdoored_model = results['model']
    model = build_model(n_classes, args.pretrained)
    model.load_state_dict(backdoored_model)

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load the dataset
    datadir = os.path.join(args.dir, 'data')
    datasets = get_dataset(args.n_clients, args.dataname, args.iid, args.batch_size, size=1000, datadir=datadir)
    _, list_test, n_classes, train_loader = datasets
    test_loader = list_test[0]

    # set up loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # evaluate model before finetuning step
    test_loss, test_acc = backdoor_evaluate(
                    model, test_loader, criterion, device)
    print(f'[!] Testing accuracy before finetuning: {test_acc:.4f}')

    # fine tune the model
    for epoch in range(args.finetuning_epochs):
        print(f'\n[!] Epoch {epoch + 1} / {args.finetuning_epochs}')
        train_loss, train_acc = backdoor_train(model, holdoutloader,
                                optimizer, criterion, device)
        test_loss, test_acc = backdoor_evaluate(
                        model, test_loader, criterion, device)
        print(f'[!] Training accuracy: {train_acc:.4f}')
        print(f'[!] Testing accuracy: {test_acc:.4f}')
    return model


def run_attack(epsilon: float, personalized: bool, model, results_dir: str, args, kwargs):
    """_summary_

    Parameters
    ----------
    epsilon : float
        
    personalized : bool
        Indicates whether model is already personalized or not.
    model 

    results_dir : str

    args : 

    kwargs : 

    Returns
    -------
    tuple
        asr, clean_acc, model state dict, holdoutloader
    """    
    print(f'Starting attack with epsilon {epsilon}...', flush=True)
    kwargs['poison_percent'] = epsilon

    # set up trojanvision
    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(dataset='cifar100', **kwargs)
    model = trojanvision.models.create(model_name='vgg11_bn', model='vgg11_bn', dataset_name='cifar100', dataset=dataset)
    
    # load the model
    server_results = torch.load(os.path.join(results_dir, 'cifar100_iid_True_server_results.pt'))
    holdout = server_results['holdoutloader'] # save holdout loader for use during personalization
    if personalized:
        server_model = model.load_state_dict()
    else:
        server_model = server_results['model']
    model.load_state_dict(server_model)

    # set up the attack
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BackdoorAttack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
    print(f'Environment set up...', flush=True)
    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)

    if personalized == False:
        attack.attack(**trainer)
    asr, clean_acc = attack.validate_fn()
    print(f'Attack completed with asr {asr}, clean acc {clean_acc}', flush=True)
    return asr, clean_acc, deepcopy(model.state_dict()), holdout


def main(epsilon, personalized, args):
    # prepare kwargs for setting the attack conditions
    kwargs = {}
    kwargs['attack_name'] = 'input_aware_dynamic'
    kwargs['data_dir'] = os.path.join(args.dir, 'data/cifar-100-python')
    kwargs['epochs'] = 100
    kwargs['lr'] = 0.001

    results_dir = os.path.join(args.dir, 'results', args.run_name)
    
    results = []
    model = None
    if personalized:
        model = personalize_model(results_dir, args)
    curr_asr, curr_clean_acc, state_dict, holdout = run_attack(epsilon, personalized, model, results_dir, args, kwargs)
    results.append({'eps': epsilon, 'asr': curr_asr, 'clean_acc': curr_clean_acc, 'holdoutloader': holdout})
    
    if personalized: 
        path = os.path.join(
            results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_finetuned_results.pt')
    else:
        path = os.path.join(
            results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_backdoor_results.pt')

    torch.save({'results': results}, path)
    print(path)

    with open(f'{path}.pickle', 'wb') as p:
        pickle.dump({'results': results},
                p, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
    