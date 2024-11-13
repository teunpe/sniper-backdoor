import train_network
import personalize_model
import shadow_network
import synthetic_data
import backdoor
import client_identification

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

def main():    
    args = global_args()
    args.run_name('multiple_targets')
    args.dir('//vol/csedu-nobackup/project/tpeeters')
    args.train = False

    for dataname in ['mnist', 'emnist', 'fmnist']:
        args.dataname = dataname

        for iid in [True, False]:
            args.set_args(dataname, iid)

            if args.train:
                    print(f'[!] Training network on {args.dataname} with iid {args.iid}')
                    train_network.main(args)

            for epsilon in [0.001, 0.005, 0.010, 0.015, 0.020]:
                args.epsilon = epsilon
                
                print(f'[!] Training backdoored model on {args.dataname} with iid {args.iid}'
                    f'and epsilon {args.epsilon}')
                backdoor.main(args)

                print(f'[!] Training personalized model on {args.dataname} with iid {args.iid}'
                    f'and epsilon {args.epsilon}')
                personalize_model.main(args)
    