from fileinput import hook_compressed
from typing import OrderedDict
from models import build_model
from torch import optim, nn
from tqdm import tqdm
import torch
import os
from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG
import gc
from copy import deepcopy
import psutil


class Participant:
    '''
    Parent class of Client and Server

    Parameters
    ----------
    testloader : torch.utils.data.DataLoader
    dataname : string
    n_classes : int

    Attributes
    ----------
    data : string
        name of the dataset
    list_test_loss : list
        test losses stored during evaluation steps
    list_test_acc : list

    Methods
    -------
    evaluate()
        evaluate the model on the test set
    '''

    def __init__(self, testloader, dataname='mnist', n_classes=10):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.cpu = torch.device('cpu')
        self.model = build_model(
            n_classes=n_classes, dataname=dataname).to('cpu')
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.data = dataname

        self.list_test_loss = []
        self.list_test_acc = []

    def evaluate(self):
        '''
        Evaluate the model on the test set

        Returns
        -------
        test_loss : float
            test loss
        test_acc : float
            test accuracy
        '''
        print('VALIDATING...')
        self.model.to(self.device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for (data, target) in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output,
                                            torch.argmax(target, dim=1)).detach()
                _, pred = output.max(1)
                correct += pred.eq(torch.argmax(target,
                                                dim=1)).sum().item()

        test_loss /= len(self.testloader)
        test_acc = 100 * correct / len(self.testloader.dataset.data)

        self.list_test_loss.append(test_loss)
        self.list_test_acc.append(test_acc)
        self.model.to('cpu')

        return test_loss, test_acc


class Client(Participant):
    ''' Client on the FL setup

    Parameters
    ----------
    trainloader : torch.utils.data.DataLoader
    testloader : torch.utils.data.DataLoader
    dataname : string, optional
        default 'mnist'
    n_classes : int, optional
        default 10
    lr : float, optional
        default 0.01
    momentum : float, optional
        default 0.9
    local_epochs : int, optional
        default 1

    Attributes
    ----------
    optimizer : torch.optim.SGD
        hyperparameter optimizer
    scheduler : None
        learning rate scheduler
    local_epochs : int
        number of local epochs
    list_train_loss : list
        train losses during each local epoch
    list_train_acc : list
        train accuracies during each local epoch
    latent_space : list
        unused??
    models_record : list
        state dicts saved during each epoch

    Methods
    -------
    record_model()
        saves the state_dict of the model in the model_records list
    save_model(idx, args, path='results')
        makes a local save of the model
    train()
        runs a training loop of the model
    '''

    def __init__(self, trainloader, testloader, dataname='mnist', n_classes=10, lr=0.01, momentum=0.9, local_epochs=1):
        super().__init__(testloader, dataname=dataname, n_classes=n_classes)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum)
        self.scheduler = None
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.list_train_loss = []
        self.list_train_acc = []
        self.latent_space = []
        self.models_record = []
        self.epoch = 0

    def record_model(self, i, epoch, path):
        '''
        Saves the state dict of the model in the model_records list
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'client_{i}_epoch_{epoch}_record.pt')
        torch.save({'model_record': self.model.state_dict()},
                   path)
        self.epoch = epoch

    def save_model(self, idx, dataname, iid, path='results'):
        '''
        Makes a local save of the model.

        Saves train_loss, train_acc, test_loss, test_acc, model_records, latent_space and args.

        Parameters
        ----------
        idx : int
            id of the save, generally the client number
        args : NameSpace
            arguments of the training run
        path : string
            path to save the model (default is 'results')
        '''
        if not os.path.exists(path):
            os.makedirs(path)

        model_records = []
        for i in range(self.epoch):
            current_path = os.path.join(path, f'client_{idx}_epoch_{i}_record.pt')
            state_dict = torch.load(current_path)['model_record']
            model_records.append(state_dict)

        path = os.path.join(path, f'{dataname}_iid_{iid}_client_{idx}_results.pt')
        torch.save({'train_loss': self.list_train_loss,
                    'train_acc': self.list_train_acc,
                    'test_loss': self.list_test_loss,
                    'test_acc': self.list_test_acc,
                    'model_records': model_records,
                    'latent_space': self.latent_space},
                   path)

    def train(self):
        '''
        Training loop of the client

        Trains the model for its local epochs and stores the loss and accuracy after each epoch.

        Returns
        -------
        tuple
            train_loss, train_acc
        '''
        self.model.to(self.device)
        self.model.train()
        for local_epoch in range(self.local_epochs):
            running_loss = 0.0
            correct = 0
            total = 0 
            for (data, target) in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, torch.argmax(target, dim=1))

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(torch.argmax(target,
                                        dim=1)).sum().item()
                del data, target, predicted, loss
                torch.cuda.empty_cache()

        train_loss = running_loss / len(self.trainloader)
        train_acc = 100 * correct / len(self.trainloader.dataset.data)

        self.list_train_loss.append(train_loss)
        self.list_train_acc.append(train_acc)
        self.model.to('cpu')
        return train_loss, train_acc


class Server(Participant):
    '''
    Server in the FL setup

    Parameters
    ----------
    clients : list
        list of clients in the setup
    dataname : string, optional
        default 'mnist'
    n_classes : int, optional
        default 10
    testloader : torch.utils.data.DataLoader, optional
        default None
    
    Methods
    -------
    fedavg()
        Runs federated averaging on the server, stores resulting state dict in Server and sends to Clients
    extract_latent_space()
        Extracts the latent space of the server model by running a test sample through the model hidden layers 
    '''
    def __init__(self, clients, dataname='mnist', n_classes=10, testloader=None):
        super().__init__(testloader, dataname=dataname, n_classes=n_classes)
        self.clients = clients

    def fedavg(self):
        '''
        Runs federated averaging on the server

        Computes average of model weights and returns resulting state dictionary to clients
        '''

        # Load model weights and number of training samples per client
        model_weights = [client.model.state_dict().values()
                         for client in self.clients]
        num_training_samples = [len(client.trainloader)
                                for client in self.clients]

        assert len(model_weights) == len(num_training_samples)
        new_weights = []
        total_training_samples = sum(num_training_samples)

        # Compute the average of the model weights over all clients
        for layers in zip(*model_weights):
            w_l = []
            for l, w in zip(layers, num_training_samples):
                l = l.cpu()
                w_l.append(torch.mul(l,w))
            weighted_layers = torch.stack(w_l).cpu()
            averaged_layers = torch.div(
                torch.sum(weighted_layers, dim=0), total_training_samples).cpu()
            new_weights.append(averaged_layers)
        # Load the resulting state dict
        self.model.load_state_dict(OrderedDict(zip(
            self.model.state_dict().keys(), new_weights)))
        # Move the state dict to the clients
        for client in self.clients:
            client.model.load_state_dict(
                OrderedDict(zip(self.model.state_dict().keys(), new_weights)))
        
    def extract_latent_space(self):
        '''
        Extract the latent space of the server model

        Takes a test sample and records the output of the last hidden layer before the output layer.
        Saves it in the latent_space attribute.
        '''
        # Take a test sample
        data, targets = next(iter(self.testloader))
        test_img = data[0][None].to(self.device)
        my_output = None

        # Create a hook to capture the output
        def my_hook(module_, input_, output_):
            nonlocal my_output
            my_output = output_.detach()

        i=0
        # For each of the clients run the test sample through the model and store the output
        for client in self.clients:
            gc.collect()
            torch.cuda.empty_cache()
            client.model.to(client.device)
            client.model.eval()
            if type(client.model) == ResNet:
                hook = client.model.layer4.register_forward_hook(my_hook)
            elif type(client.model) == VGG:
                hook = client.model.features[-1].register_forward_hook(my_hook)
            else:
                hook = client.model.conv3.register_forward_hook(my_hook)
            client.model(test_img)
            # The flattened output is the latent space representation
            client.latent_space.append(torch.flatten(my_output))
            hook.remove()
            client.model.to('cpu')
