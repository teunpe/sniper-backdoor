import train_network
import personalize_model
import shadow_network
import synthetic_data
import backdoor
import backdoor_cifar
import client_identification
import numpy as np
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser(description='Dataname')

parser.add_argument('--dataname', type=str, default='mnist',
                    help='dataname', choices=['mnist', 'emnist', 'fmnist', 'cifar100'])

given_args = parser.parse_args()

class global_args():
        # static args
        seed = 1
        batch_size = 64
        trainset_size = 1000
        dir = './'
        run_name = ''
        test_freq = 1
        warm = False
        train = True
        # train args
        n_clients = 5
        lr = 0.1
        momentum = 0.9
        dataname = 'mnist'
        n_epochs = 50
        n_local_epochs = 2
        iid = True
        early_stop = True
        # backdoor args
        epsilon = 0
        client_id = 0
        source_label = 0
        target_label = 1
        pretrained = False
        fake_dir = ''
        epochs = 10
        # personalization args
        finetuning_epochs = 1

        def set_args(self, dataname, iid):
            if dataname=='mnist':
                # train args
                self.n_clients = 5
                self.lr = 0.1
                self.momentum = 0.9
                self.dataname = 'mnist'
                self.n_epochs = 50
                self.n_local_epochs = 2
                self.iid = iid
                
            if dataname=='emnist':
                self.n_clients = 13
                self.lr = 0.001
                self.momentum = 0.9
                self.dataname = 'emnist'
                self.n_epochs = 30 if iid else 200
                self.n_local_epochs = 2
                self.iid = iid
            
            if dataname=='fmnist':
                self.n_clients = 5
                self.lr = 0.00001
                self.momentum = 0
                self.dataname = 'fmnist'
                self.n_epochs = 200
                self.n_local_epochs = 1
                self.iid = iid

            if dataname=='cifar100':
                 self.n_clients = 10
                 self.lr = 0.001
                 self.momentum = 0.9
                 self.dataname = 'cifar100'
                 self.n_epochs = 23
                 self.n_local_epochs = 1
                 self.iid = iid

def main():
    datanames = [given_args.dataname]
    args = global_args()
    args.run_name = f'{datanames[0]}_early_stop'
    args.dir = '//vol/csedu-nobackup/project/tpeeters'
    args.train = False
    tqdm_file = open(f'{args.run_name}_progress.txt','w')

    sources = [0,1]
    targets = [9,7]

    for dataname in tqdm(datanames,file=tqdm_file, desc='data',leave=False):
        args.dataname = dataname

        for iid in tqdm([True, False],file=tqdm_file, desc='iid',leave=False):
            args.set_args(dataname, iid)
            args.n_epochs = 23
            if args.train:
                    print(f'[!] Training network on {args.dataname} with iid {args.iid}')
                    train_network.main(args)

            for source, target in tqdm(zip(sources,targets),file=tqdm_file, desc='source',leave=False):
                args.target_label = target
                args.source_label = source

                for epsilon in tqdm([0.001, 0.005, 0.010, 0.015, 0.020],file=tqdm_file, desc='eps',leave=False):
                    args.epsilon = epsilon
                    
                    print(f'[!] Training backdoored model on {args.dataname} with iid {args.iid},'
                        f'epsilon {args.epsilon}, source {source} and target {target}.')
                    if dataname == 'cifar100':
                         backdoor_cifar.main(epsilon, False, args)
                    else: 
                         backdoor.main(args)

                    print(f'[!] Training backdoored model on {args.dataname} with iid {args.iid},'
                        f'epsilon {args.epsilon}, source {source} and target {target}.')
                    if dataname == 'cifar100':
                         backdoor_cifar.main(epsilon, True, args)
                    else:
                        personalize_model.main(args)

if __name__ == '__main__':
    main()
    