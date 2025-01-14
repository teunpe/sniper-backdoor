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

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./results', help='directory')
trojanvision.environ.add_argument(parser)
trojanvision.datasets.add_argument(parser)
trojanvision.models.add_argument(parser)
trojanvision.trainer.add_argument(parser)
trojanvision.marks.add_argument(parser)
trojanvision.attacks.add_argument(parser)
kwargs = vars(parser.parse_args())
args = parser.parse_args()

def personalize_model(results_dir, args):
    # load the backdoored model
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_backdoor_results.pt')
    backdoored_model = torch.load(path)['model']
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
        train_loss, train_acc = backdoor_train(model, train_loader,
                                optimizer, criterion, device)
        test_loss, test_acc = backdoor_evaluate(
                        model, test_loader, criterion, device)
        print(f'[!] Training accuracy: {train_acc:.4f}')
        print(f'[!] Testing accuracy: {test_acc:.4f}')
    return model

def run_attack(epsilon, personalized, model, args, kwargs):
    print(f'Starting attack with epsilon {epsilon}...', flush=True)
    kwargs['poison_percent'] = epsilon
    print(kwargs.keys())
    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(model_name='vgg11_bn', model='vgg11_bn', dataset_name='cifar100', dataset=dataset)
    # model = build_model(100, 'cifar100')
    if personalized:
        server_model = model.load_state_dict()
    else:
        server_model = torch.load(os.path.join(args.dir, 'cifar100_iid_True_server_results.pt'))['model']
    model.load_state_dict(server_model)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BackdoorAttack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
    print(f'Environment set up...', flush=True)
    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)
    if not personalized:
        attack.attack(**trainer)
    asr, clean_acc = attack.validate_fn()
    print(f'Attack completed with asr {asr}, clean acc {clean_acc}', flush=True)
    return asr, clean_acc, model.state_dict()

def main(epsilon, personalized, args):
    results_dir = os.path.join(args.dir, 'results', args.run_name)
    results = []
    model = None
    if personalized:
        model = personalize_model(results_dir, args)
    curr_asr, curr_clean_acc, state_dict = run_attack(epsilon, personalized, model, args, kwargs)
    results.append({'eps': epsilon, 'asr': curr_asr, 'clean_acc': curr_clean_acc, 'model': state_dict})
    
    if args.personalized: pers = 'personalized_'
    else: pers = ''
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_{args.personalized}backdoor_results.pt')

    torch.save({'results': results}, path)
    print(path)

if __name__ == '__main__':
    main()
    