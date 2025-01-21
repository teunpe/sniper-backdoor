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
from copy import deepcopy
import pickle

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

    # prepare the models
    path = os.path.join(
        results_dir, f'{args.dataname}_iid_{args.iid}_server_results.pt')
    model = torch.load(path)['model']
    holdoutloader = torch.load(path)['holdoutloader'] # prepared for later use in personalization

    poisoned_model = build_model(n_classes, args.pretrained)
    poisoned_model.load_state_dict(deepcopy(model))

    clean_model = build_model(n_classes, args.pretrained)
    clean_model.load_state_dict(deepcopy(model))

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    poisoned_model.to(device)
    train_data_loader, test_data_ori_loader, test_data_tri_loader, n_classes = create_backdoor_data_loader(args.dataname, args.target_label, args.source_label,
                                                                                                           args.epsilon, args.batch_size,
                                                                                                           args.batch_size, device, dir=datadir, args=args)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        poisoned_model.parameters(), lr=args.backdoor_lr, momentum=args.momentum)
    print(f'[!] Implementing backdoor with epsilon {args.epsilon}...')
    list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor = backdoor_model_trainer(poisoned_model, criterion, optimizer, args.backdoor_epochs,
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

    clean_per_class_sum = clean_per_class.diag()/clean_per_class.sum(1)
    poisoned_per_class_sum = poisoned_per_class.diag()/poisoned_per_class.sum(1)

    clean_model_accuracy = (clean_model_performance.diag()/clean_model_performance.sum(1)).mean()
    poisoned_model_accuracy = clean_per_class_sum.mean()
    cad = clean_model_accuracy - poisoned_model_accuracy
    print(f'CAD: {cad}')


    # Save the results
    path = os.path.join(
        results_dir, f'{args.dataname}_{args.epsilon}_{args.source_label}->{args.target_label}_iid_{args.iid}_backdoor_results.pt')

    torch.save({'train_loss': list_train_loss, 'train_acc': list_train_acc, 'test_loss': list_test_loss, 'test_acc': list_test_acc,
               'test_loss_backdoor': list_test_loss_backdoor, 'test_acc_backdoor': list_test_acc_backdoor, 'clean_per_class': clean_per_class_sum,
                'poisoned_per_class': poisoned_per_class_sum, 'clean_matrix': clean_per_class, 'poisoned_matrix': poisoned_per_class, 'asr': asr, 'cad': cad, 'model': deepcopy(poisoned_model.state_dict()), 'args': args, 'holdoutloader': holdoutloader}, path)
    
if __name__ == '__main__':
    main()
