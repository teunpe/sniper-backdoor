import os
import argparse
import torch
from utils import get_dataset_fake, get_dataset, trainer
from participants import Client, Server
import copy
import numpy as np

parser = argparse.ArgumentParser(description='Shadow training')
parser.add_argument('--n_clients', type=int, default=10,
                    help='number of clients')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--pretrained', action='store_true',
                    default=False, help='pretrained')
parser.add_argument('--dataname', type=str, default='mnist',
                    choices=['mnist', 'emnist', 'fmnist'])
parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--n_local_epochs', type=int,
                    default=1, help='number of local epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--iid', type=bool, help='iid')
parser.add_argument('--dir', type=str, default='./', help='directory')
args = parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.join(args.dir, 'data')
    fake_data_dir = os.path.join(args.dir, f'results/fake_datasets_{args.dataname}')
    shadow_dir = os.path.join(args.dir, 'shadow')

    list_trainloader = get_dataset_fake(
        args.n_clients, args.batch_size, fake_data_dir)

    _, list_testloader, n_classes, _ = get_dataset(
        args.n_clients, args.dataname, args.iid, args.batch_size, datadir=data_dir)

    clients = []
    for train, test in zip(list_trainloader, list_testloader):
        clients.append(Client(trainloader=train, testloader=test,
                              lr=args.lr, momentum=args.momentum,
                              n_classes=n_classes))

    server = Server(
        clients=clients, n_classes=n_classes,
        testloader=copy.deepcopy(list_testloader[0]))

    server.fedavg()

    trainer(clients, server, args.n_epochs)

    for idx, client in enumerate(clients):
        client.save_model(idx, args, shadow_dir)


if __name__ == '__main__':
    main()
