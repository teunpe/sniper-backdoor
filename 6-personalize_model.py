import os
import argparse
import torch
import numpy as np
from models import build_model
from utils import get_dataset, backdoor_train, backdoor_evaluate, validation_per_class
from poisoned_dataset import create_backdoor_data_loader

parser = argparse.ArgumentParser('Personalization')

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
parser.add_argument('--finetuning_epochs', type=int, default=1, help='number of epochs in finetuning step')
parser.add_argument('--dir', type=str, default='./', help='directory')
parser.add_argument('--iid', type=bool, help='iid')

args = parser.parse_args()

def main():
    # set up args
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = os.path.join(args.dir, 'data')
    results_dir = os.path.join(args.dir, 'results')

    if args.dataname == 'mnist':
        n_classes = 10
    elif args.dataname == 'emnist':
            n_classes = 26
    elif args.dataname == 'fmnist':
            n_classes = 10

    # load the backdoored model
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_backdoor_results.pt')
    results = torch.load(path)

    weights_model = results['model']
    model = build_model(n_classes, args.pretrained)
    model.load_state_dict(weights_model)

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # load the dataset
    datasets = get_dataset(args.n_clients, args.dataname, args.iid, args.batch_size, size=1000, datadir=data_dir)
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
        
    # get poisoned data
    train_data_loader, test_data_ori_loader, test_data_tri_loader, n_classes = create_backdoor_data_loader(args.dataname, args.target_label, args.source_label,
                                                                                                           args.epsilon, args.batch_size,
                                                                                                           args.batch_size, device, data_dir, args)

    # test the finetuned model on the poisoned data
    clean_per_class = validation_per_class(
        model, test_data_ori_loader, n_classes, device)
    poisoned_per_class = validation_per_class(
            model, test_data_tri_loader, n_classes, device)
    poison_loss, poison_acc = backdoor_evaluate(
                        model, test_data_tri_loader, criterion, device)
    print(f'[!] Poisoned testing accuracy: {poison_acc:.4f}')

    succesful_attacks = poisoned_per_class[args.source_label,args.target_label]
    all_attacks = poisoned_per_class[args.source_label,:].sum()

    asr = succesful_attacks/all_attacks
    print(f'ASR: {asr}')

    # save resulting model and test results
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_finetuned_results.pt')

    torch.save({'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc,
               'test_loss_backdoor': poison_loss, 'test_acc_backdoor': poison_acc, 'clean_per_class': clean_per_class,
                'poisoned_per_class': poisoned_per_class, 'asr': asr, 'model': model.state_dict(), 'args': args}, path)

if __name__ == '__main__':
    main()
