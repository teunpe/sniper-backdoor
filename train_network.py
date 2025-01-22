import argparse
import torch
from participants import Client, Server
from utils import get_dataset, trainer, get_entire_dataset, backdoor_train, backdoor_evaluate
import copy
import os
import numpy as np
from models import build_model
from torch import optim, nn
import pickle

def main(args):
    data_dir = os.path.join(args.dir, 'data')
    results_dir = os.path.join(args.dir, 'results', args.run_name)

    print(args.iid)

    # Initialize lists to keep track of accuracies
    test_clients = []
    test_server = []

    for _ in range(1):
        # Initialize clients and server
        torch.manual_seed(_)
        np.random.seed(_)
        list_trainloader, list_testloader, n_classes, holdoutloader, validationloader = get_dataset(
            args.n_clients, args.dataname, args.iid, args.batch_size, args.valsplit, 
            args.holdoutsplit, data_dir)

        clients = []
        for train, test in zip(list_trainloader, list_testloader):
            clients.append(Client(trainloader=train, testloader=test,
                                  lr=args.lr, momentum=args.momentum,
                                  dataname=args.dataname, n_classes=n_classes,
                                  local_epochs=args.n_local_epochs))

        server = Server(
            clients=clients, dataname=args.dataname, n_classes=n_classes,
            testloader=copy.deepcopy(list_testloader[0]), valloader=validationloader, lr=args.lr, momentum=args.momentum)

        if args.warm:
            # If we are in the warm up model we train the model for few epochs in the 5% of the dataset
            trainloader, testloader, n_classes = get_entire_dataset(
                size=args.trainset_size, split=0.05)
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            model = build_model(
                n_classes=n_classes, dataname=args.dataname).to(device)
            optimizer = optim.SGD(
                model.parameters(), lr=0.01, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()
            for _ in range(15):
                backdoor_train(model, trainloader,
                               optimizer, criterion, device)
                _, acc = backdoor_evaluate(
                    model, testloader, criterion, device)
                print('Accuracy: ', acc)

            server.model = model
            for client in clients:
                client.model = model

        server.fedavg()

        for client in clients:
            test_clients.append(client.list_test_acc)
            client.scheduler = optim.lr_scheduler.StepLR(
                client.optimizer, step_size=max((client.local_epochs*args.n_epochs) // 3,1), gamma=0.1)

        # Train the clients for the specified number of epochs
        server_model, best_epoch = trainer(clients, server, validationloader, args.n_epochs, args.test_freq, args.early_stop, args.perfedavg, args.finetuning_epochs, results_dir)

        # Save the server accuracy over time
        test_server.append(server.list_test_acc)
        print('here')
        # Save the state dicts and performance of each client in each epoch
        for idx, client in enumerate(clients):
            client.save_model(idx, args.dataname, args.iid, results_dir)
        print('here')
        # Save the server results
        torch.save({'model': server_model.state_dict(),
                    'loss': server.list_test_loss,
                    'acc': server.list_test_acc,
                    'best_epoch': best_epoch,
                    'holdoutloader': holdoutloader},
                   os.path.join(results_dir, f'{args.dataname}_iid_{args.iid}_server_results.pt'))
        
        path = os.path.join(results_dir, f'{args.dataname}_iid_{args.iid}_server_results.pt')
        with open(f'{path}.pickle', 'wb') as f:
            pickle.dump({'model': server_model.state_dict(),
                    'loss': server.list_test_loss,
                    'acc': server.list_test_acc,
                    'best_epoch': best_epoch}, f)

        torch.save({'acc_clients': test_clients,
                    'acc_server': test_server,
                    'best_epoch': best_epoch}, os.path.join(results_dir, f'{args.dataname}_iid_{args.iid}_average_results.pt'))

if __name__ == '__main__':
    main()
