from ctypes import util
from http import server
import os
import torch
from poisoned_dataset import create_backdoor_data_loader
import argparse
from models import build_model
from utils import backdoor_model_trainer
import numpy as np
import utils

parser = argparse.ArgumentParser('Backdoor attack')

parser.add_argument('--dataname', type=str, default='mnist',
                    help='dataname', choices=['mnist', 'emnist', 'fmnist'])
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')
parser.add_argument('--client_id', type=int, default=0, help='client')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--source_label', type=int, default=0, help='source label')
parser.add_argument('--target_label', type=int, default=1, help='target label')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--pretrained', action='store_true', help='pretrained')
parser.add_argument('--fake_dir', type=str)
parser.add_argument('--n_clients', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--dir', type=str, default='./', help='directory')
parser.add_argument('--iid', action="store_true", help='iid')
parser.add_argument('--run_name', type=str, default='')

args = parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = os.path.join(args.dir, 'results', args.run_name)
    datadir = os.path.join(args.dir, 'data')

    if args.dataname == 'mnist':
        n_classes = 10
    elif args.dataname == 'emnist':
        n_classes = 26
    elif args.dataname == 'fmnist':
        n_classes = 10

    path = os.path.join(
        results_dir, f'{args.dataname}_iid_{args.iid}_server_results.pt')
    model = torch.load(path)['model']

    poisoned_model = build_model(n_classes, args.pretrained)
    poisoned_model.load_state_dict(model)

    clean_model = build_model(n_classes, args.pretrained)
    clean_model.load_state_dict(model)

    # Load the dataset
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    poisoned_model.to(device)
    train_data_loader, test_data_ori_loader, test_data_tri_loader, n_classes = create_backdoor_data_loader(args.dataname, args.target_label, args.source_label,
                                                                                                           args.epsilon, args.batch_size,
                                                                                                           args.batch_size, device, dir=datadir, args=args)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        poisoned_model.parameters(), lr=args.lr, momentum=args.momentum)
    print(f'[!] Implementing backdoor with epsilon {args.epsilon}...')
    list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor = backdoor_model_trainer(poisoned_model, criterion, optimizer, args.epochs,
                                                                                                                                             train_data_loader, test_data_ori_loader, test_data_tri_loader, device)

    clean_model.to(device)
    clean_model_performance = utils.validation_per_class(
        clean_model, test_data_ori_loader, n_classes, device)
    clean_per_class = utils.validation_per_class(
        poisoned_model, test_data_ori_loader, n_classes, device)
    poisoned_per_class = utils.validation_per_class(
        poisoned_model, test_data_tri_loader, n_classes, device)
    
    succesful_attacks = poisoned_per_class[args.source_label,args.target_label]
    all_attacks = poisoned_per_class[args.source_label,:].sum()

    asr = succesful_attacks/all_attacks
    print(f'ASR: {asr}')

    clean_per_class = clean_per_class.diag()/clean_per_class.sum(1)
    poisoned_per_class = poisoned_per_class.diag()/poisoned_per_class.sum(1)

    clean_model_accuracy = (clean_model_performance.diag()/clean_model_performance.sum(1)).mean()
    poisoned_model_accuracy = clean_per_class.mean()
    print(clean_model_accuracy, poisoned_model_accuracy)
    cad = clean_model_accuracy - poisoned_model_accuracy
    print(f'CAD: {cad}')


    # Save the results
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_backdoor_results.pt')

    torch.save({'train_loss': list_train_loss, 'train_acc': list_train_acc, 'test_loss': list_test_loss, 'test_acc': list_test_acc,
               'test_loss_backdoor': list_test_loss_backdoor, 'test_acc_backdoor': list_test_acc_backdoor, 'clean_per_class': clean_per_class,
                'poisoned_per_class': poisoned_per_class, 'asr': asr, 'cad': cad, 'model': poisoned_model.state_dict(), 'args': args}, path)
    print(path)
if __name__ == '__main__':
    main()
