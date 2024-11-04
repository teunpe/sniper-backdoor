import os
import argparse
import torch
import numpy as np
from models import build_model
from utils import get_dataset, backdoor_train, backdoor_evaluate, validation_per_class
from poisoned_dataset import create_backdoor_data_loader

parser = argparse.ArgumentParser('Evaluation')

parser.add_argument('--dataname', type=str, default='mnist',
                    help='dataname', choices=['mnist', 'emnist', 'fmnist'])
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')
parser.add_argument('--client_id', type=int, default=0, help='client')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--source_label', type=int, default=0, help='source label')
parser.add_argument('--target_label', type=int, default=1, help='target label')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--pretrained', action='store_true', help='pretrained')
parser.add_argument('--n_clients', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--dir', type=str, default='./', help='directory')
parser.add_argument('--iid', type=bool, help='iid')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.join(args.dir, 'data')
    results_dir = os.path.join(args.dir, 'results')

    path = os.path.join(
        results_dir, f'{args.dataname}_server_results.pt')
    server_model = torch.load(path)
    server_acc = server_model['acc_server']