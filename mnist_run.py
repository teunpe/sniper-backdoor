import train_network
import personalize_model
import shadow_network
import synthetic_data
import backdoor
import client_identification
import numpy as np
from tqdm.auto import tqdm
import argparse

# parser = argparse.ArgumentParser('Personalization')

# parser.add_argument('--dataname', type=str, default='mnist',
#                     help='dataname', choices=['mnist', 'emnist', 'fmnist'])

# args=parser.parse_args()

class global_args():
        # static args
        seed = 1
        batch_size = 64
        trainset_size = 1000
        dir = './'
        run_name = ''
        test_freq = 999
        warm = False
        train = False
        # train args
        n_clients = 5
        lr = 0.1
        momentum = 0.9
        dataname = 'mnist'
        n_epochs = 50
        n_local_epochs = 2
        iid = True
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
    datanames = ['cifar100']    
    args = global_args()
    args.run_name = 'cifar' 
    args.dir = '//vol/csedu-nobackup/project/tpeeters'
    args.train = True
    tqdm_file = open(f'{args.run_name}_progress.txt','w')

    sources = [0]
    targets = [1]

    for dataname in tqdm(datanames,file=tqdm_file, desc='data',leave=False):
        args.dataname = dataname

        for iid in tqdm([True, False],file=tqdm_file, desc='iid',leave=False):
            args.set_args(dataname, iid)
            args.n_epochs = 23
            if args.train:
                    print(f'[!] Training network on {args.dataname} with iid {args.iid}')
                    train_network.main(args)

            for source in tqdm(sources,file=tqdm_file, desc='source',leave=False):
                 
                 for target in tqdm(targets,file=tqdm_file, desc='target',leave=False):

                    if source==target:
                        continue
                    args.target_label = target
                    args.source_label = source

                    for epsilon in tqdm([0.050, 0.100, 0.200, 0.400, 0.800],file=tqdm_file, desc='eps',leave=False):
                        args.epsilon = epsilon
                        
                        print(f'[!] Training backdoored model on {args.dataname} with iid {args.iid},'
                            f'epsilon {args.epsilon}, source {source} and target {target}.')
                        backdoor.main(args)

                        print(f'[!] Training backdoored model on {args.dataname} with iid {args.iid},'
                            f'epsilon {args.epsilon}, source {source} and target {target}.')
                        personalize_model.main(args)

if __name__ == '__main__':
    main()
    