from torch.autograd import Variable
import torch
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision.transforms import transforms
import numpy as np
from torch import nn
from torchvision.utils import save_image
import torch
import os
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import seaborn as sns


torch.manual_seed(42)
np.random.seed(42)


class CustomDataset(torch.utils.data.Dataset):
    """Implement a custom dataset (implements torch.utils.data.Dataset)

    Attributes
    ----------
    data : any
        data in the dataset
    labels : any
        labels of data entries
    transform : any
        transform used on the data
    n_classes : int
        number of classes in the dataset

    Methods
    -------
    __getitem__(index)
        get the image and label at the given index

    __len__()
        get the length of the dataset
    """    

    def __init__(self, data, labels, transform=None, n_classes=10):
        # Ensure data and labels are tensors

        if type(data) != torch.Tensor:
            data = torch.Tensor(data)

        if type(labels) != torch.Tensor:
            labels = torch.Tensor(labels)

        self.data = data
        self.labels = labels

        self.transform = transform
        self.n_classes = n_classes

    def __getitem__(self, index):
        """Return the image and label at the index

        Parameters
        ----------
        index : int
            index of the queried entry

        Returns
        -------
        any
            queried image
        torch.Tensor
            label of the image
        """        
        img = self.data[index]
        label_idx = int(self.labels[index])

        if self.transform:

            try:
                img = self.transform(np.array(img.float()))
            except:
                img = img.permute(2, 0, 1)
                img = self.transform(img)

        label = np.zeros(self.n_classes)
        label[label_idx] = 1
        label = torch.Tensor(label)

        return img, label

    def __len__(self):
        return len(self.data)


def normalize(x):
    """Normalize input over dimension 2 with Euclidian norm (p=2)

    Parameters
    ----------
    x : torch.Tensor
        input to be normalized

    Returns
    -------
    torch.Tensor
        normalized input
    """    
    return torch.nn.functional.normalize(x.float(), p=2, dim=2)


def get_dataset_fake(n_clients, batch_size, dir):
    """Creates a dataloader of the generated data

    Parameters
    ----------
    n_clients : int
        number of clients in the setup
    batch_size : int
        batch size
    dir : string
        directory of the fake data

    Returns
    -------
    list
        list of dataloaders with fake data
    """
    lis_dataloader = []
    for idx in range(n_clients):
        path = os.path.join(dir, f'fake_dataset_{idx}.pt')
        dataset = torch.load(path)
        lis_dataloader.append(torch.utils.data.DataLoader(
            dataset, batch_size=batch_size))

    if type(dataset) == EMNIST:
        for i, dataset in enumerate(lis_dataloader):
            dataset.dataset.labels = dataset.dataset.labels - 1
            lis_dataloader[i] = dataset

    return lis_dataloader


def get_dataset_gan(dataname, batch=64, size=100, datadir='./data'):
    """Initialize a dataset for the GAN training

    Parameters
    ----------
    dataname : string
        name of the dataset
    batch : int, optional
        batch size, by default 64
    size : int, optional
        size of the GAN training dataset, by default 100

    Returns
    -------
    trainloader : torch.utils.data.DataLoader
        train loader for the GAN dataset
    n_classes : int
        number of classes in the dataset

    Raises
    ------
    ValueError
        Dataname must be 'mnist', 'emnist', 'fmnist', or 'cifar100'.
    """    

    # Prepare transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Load the correct dataset (from torchvision)
    if dataname == 'mnist':
        n_classes = 10
        dataset = MNIST(root=datadir, train=True,
                        download=True, transform=transform)

    elif dataname == 'emnist':
        n_classes = 26
        dataset = EMNIST(root=datadir, train=True, split='letters',
                         download=True, transform=transform)

        dataset.targets = dataset.targets - 1
    elif dataname == 'fmnist':
        n_classes = 10
        dataset = FashionMNIST(root=datadir, train=True,
                               download=True, transform=transform)

    elif dataname == 'cifar100':
        n_classes = 100
        # In the case of CIFAR100, override the transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])

        dataset = CIFAR100(root=datadir, train=True,
                           download=True, transform=transform)
    else:
        raise ValueError(f'Dataset {dataname} not supported')

    # Prepare a random permutatino of the dataset
    perm = np.random.permutation(len(dataset))[:size]
    dataset.data = dataset.data[perm]
    dataset.targets = np.array(dataset.targets)[perm]

    # Create the dataloader
    trainloader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch,
                                              shuffle=True, drop_last=True)

    return trainloader, n_classes


def get_entire_dataset(size=1000, split=0.05, batch=64, datadir='./data'):
    """Set up a train and test loader comprising the entire dataset

    Parameters
    ----------
    size : int, optional
        number of datapoints included in the dataset, by default 1000
    split : float, optional
        train/test split, by default 0.05
    batch : int, optional
        batch size, by default 64

    Returns
    -------
    trainloader : torch.utils.data.DataLoader
        training data loader
    testloader : torch.utils.data.DataLoader
        test data loader
    n_classes : int
        number of classes in the dataset
    """    
    n_classes = 10
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010)),
    # ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.247, 0.243, 0.261]),
    ])

    trainset = CIFAR10(root=datadir, train=True,
                       download=True)
    testset = CIFAR10(root=datadir, train=False,
                      download=True)

    # Ensure that targets and data are a tensor
    if type(trainset.targets) != torch.Tensor:
        trainset.targets = torch.Tensor(trainset.targets)
    if type(trainset.data) != torch.Tensor:
        trainset.data = torch.Tensor(trainset.data)

    perm = np.random.permutation(len(trainset))[size:]
    trainset.data = trainset.data[perm]
    trainset.targets = trainset.targets[perm]

    # Get only the 5% of the dataset
    perm = np.random.permutation(len(trainset.data))[
        :int(len(trainset.data)*split)]
    trainset.data = trainset.data[perm]
    trainset.targets = trainset.targets[perm]

    trainset = CustomDataset(
        trainset.data, trainset.targets, transform=transform, n_classes=n_classes)
    testset = CustomDataset(testset.data, testset.targets,
                            transform=transform, n_classes=n_classes)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                             shuffle=False)

    return trainloader, testloader, n_classes


def get_dataset(n_clients, dataname, iid=False, batch=64, size=1000, datadir='./data'):
    '''
    Get a list comprising the test and train loaders of each client 

    Parameters
    ----------
    n_clients : int
        number of clients in the setup
    dataname : string
        name of the dataset
    iid : bool
        specifies whether dataset should be iid or not (default False)
    batch : int
        batch size (default 64)
    size : int
        dataset size (default 1000)

    Returns
    -------
    list_train : list
        list of train loaders
    list_test : list
        list of test loaders
    n_classes : int
        number of classes in the dataset
    '''

    # Set up the transforms for mnist (to tensor, and normalize with mean and stdev 0.5)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    if dataname == 'mnist':
        n_classes = 10
        trainset = MNIST(root=datadir, train=True,
                         download=True)
        testset = MNIST(root=datadir, train=False,
                        download=True)
        holdoutset = MNIST(root=datadir, train=True,
                        download=True)

    elif dataname == 'emnist':
        n_classes = 26
        trainset = EMNIST(root=datadir, train=True, split='letters',
                          download=True)
        testset = EMNIST(root=datadir, train=False, split='letters',
                         download=True)
        holdoutset = EMNIST(root=datadir, train=True, split='letters',
                        download=True)

        trainset.targets = trainset.targets - 1
        testset.targets = testset.targets - 1

    elif dataname == 'fmnist':
        n_classes = 10
        trainset = FashionMNIST(root=datadir, train=True,
                                download=True)
        testset = FashionMNIST(root=datadir, train=False,
                               download=True)
        holdoutset = FashionMNIST(root=datadir, train=True, download=True)


    elif dataname == 'cifar10':
        n_classes = 10
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                          (0.2023, 0.1994, 0.2010)),
        # ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])

        trainset = CIFAR10(root=datadir, train=True,
                           download=True)
        testset = CIFAR10(root=datadir, train=False,
                          download=True)

    elif dataname == 'cifar100':
        n_classes = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])

        trainset = CIFAR100(root=datadir, train=True,
                            download=True)
        testset = CIFAR100(root=datadir, train=False,
                           download=True)
    else:
        raise ValueError(f'Dataset {dataname} not supported')

    # Ensure that targets and data are a tensor
    if type(trainset.targets) != torch.Tensor:
        trainset.targets = torch.Tensor(trainset.targets)
    if type(trainset.data) != torch.Tensor:
        trainset.data = torch.Tensor(trainset.data)

    perm = np.random.permutation(len(trainset))[size:]

    holdoutperm = perm[:size]
    holdoutset.data = holdoutset.data[holdoutperm]
    holdoutset.targets = holdoutset.targets[holdoutperm]

    holdoutset = CustomDataset(holdoutset.data, holdoutset.targets,
                               transform=transform, n_classes=n_classes)
    holdoutloader = torch.utils.data.DataLoader(holdoutset, batch_size = 64, num_workers = 3)
    torch.save(holdoutloader, './results/holdout.pt')

    trainperm = perm[size:]
    trainset.data = trainset.data[trainperm]
    trainset.targets = trainset.targets[trainperm]

    if iid:
        list_train = get_iid_data(
            n_clients, trainset, transform, batch, n_classes)
    else:
        list_train = get_non_iid_data(
            n_clients, trainset, transform, batch, n_classes)

    testset = CustomDataset(testset.data, testset.targets,
                            transform=transform, n_classes=n_classes)

    list_test = [torch.utils.data.DataLoader(
        testset, batch_size=64, num_workers=3) for _ in range(n_clients)]

    return list_train, list_test, n_classes, holdoutloader


def get_iid_data(n_clients, trainset, transform, batch, n_classes):
    """ Creates a list of training dataloaders with i.i.d. data

    Parameters
    ----------
    n_clients : int
        number of clients
    trainset : torchvision.datasets.Dataset
        torchvision dataset
    transform : torchvision.transforms.Transform
        transformation to be used on the dataset
    batch : int
        batch size
    n_classes : int
        number of classes in the dataset

    Returns
    -------
    list_train : list
        list of train loaders (one per client)
    """
    print("Using iid data")
    perm = np.random.permutation(len(trainset))
    split = int(len(trainset) / n_clients)
    list_train = []
    begin = 0

    for _ in range(n_clients):
        data = trainset.data[perm][begin:begin + split]
        targets = trainset.targets[perm][begin:begin + split]
        ds = CustomDataset(
            data, targets, transform=transform, n_classes=n_classes)
        list_train.append(torch.utils.data.DataLoader(ds, batch_size=batch))

        begin += split

    return list_train


def get_non_iid_data(n_clients, trainset, transform, batch, n_classes):
    """ Create non i.i.d. train loaders by splitting the classes between the clients.

    The number of classes should be divisible by the number of clients.

    Parameters
    ----------
    n_clients : int
        number of clients
    trainset : torchvision.datasets.Dataset
        torchvision dataset
    transform : torchvision.transforms.Transform
        transformation to be used on the dataset
    batch : int
        batch size
    n_classes : int
        number of classes in the dataset

    Returns
    -------
    list_train : list
        list of train loaders (one per client)
    """    
    print("Using non=iid data")
    # Ensure that the number of classes is divisible by the number of clients, so they have the same amount of labels
    assert n_classes % n_clients == 0
    class_per_client = int(n_classes/n_clients)
    from_class = 0

    targets = trainset.targets
    dataset = trainset.data

    list_train = []
    for _ in range(n_clients):

        data = dataset[targets < from_class + class_per_client]
        labels = targets[targets < from_class + class_per_client]
        ds = CustomDataset(data, labels, transform=transform,
                           n_classes=n_classes)
        list_train.append(torch.utils.data.DataLoader(ds, batch_size=batch))

        # Subtract the elements that are already in the dataset
        constraint = targets >= from_class + class_per_client
        dataset = dataset[constraint]
        targets = targets[constraint]

        from_class += class_per_client

    return list_train


def trainer(clients, server, epochs):
    """Run the training loop over the clients and server for the given number of epochs.

    Parameters
    ----------
    clients : list
        list of clients
    server : Server
        server
    epochs : int
        number of training rounds 
    """    
    print(f'\n[!] Training the model for {epochs} epochs')

    # train the model for the number of epochs
    for epoch in range(epochs):
        print(f'\n[!] Epoch {epoch + 1} / {epochs}')

        for i, client in enumerate(clients):
            print(f'[!] Training client {i + 1} / {len(clients)}')
            # Just 1 local epoch
            train_loss, train_acc = client.train()

            # Record the train loss and accuracy every other epoch
            print(f'\n[!] Training loss: {train_loss:.4f}')
            print(f'[!] Training accuracy: {train_acc:.4f}')
            if epoch % 2 == 0:
                test_loss, test_acc = client.evaluate()
                print(f'[!] Testing accuracy: {test_acc:.4f}')

            # Save the model state dict
            client.record_model()

            if client.scheduler is not None:
                client.scheduler.step()

        # Extract and save the latent space
        server.extract_latent_space()

        # Perform fedavg
        print(f'[!] Averaging')
        server.fedavg()

        # Save the test loss and accuracy every other epoch
        if epoch % 2 == 0:
            test_loss, test_acc = server.evaluate()
            print(f'[!] Server testing accuracy: {test_acc:.4f}')

    return server.model


def weights_init(m):
    """Initialize the weights of the model

    Parameters
    ----------
    m : torch.nn.Module
        model of which the weights are initialized
    """    

    # What does this do???
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_noise(n_samples, noise_dim, device='cpu'):
    '''Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim).

    Parameters
    ----------
    n_samples : int
        the number of samples to generate based on batch_size
    noise_dim : int
        the dimension of the noise vector
    device : torch.nn.Device
        device type can be cuda or cpu
    '''

    return torch.randn(n_samples, noise_dim, 1, 1, device=device)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def train_gan(G, D, criterion, d_optimizer, g_optimizer, trainloader,
              n_epochs, batch_size, noise_dim, save_dir, no_of_channels, img_size, c_idx, device):
    """Train the GAN for each client

    Parameters
    ----------
    G : models.Generator
        generator
    D : models.Discriminator
        discriminator
    criterion : torch.nn.BCELoss
        loss criterion
    d_optimizer : torch.optim.Adam
        optimizer for the discriminator
    g_optimizer : torch.optim.Adam
        optimizer for the generator
    trainloader : torch.utils.data.DataLoader
        train loader
    n_epochs : int
        number of training epochs
    batch_size : int
        batch size
    noise_dim : int
        noise dimension of the generator
    save_dir : string
        directory to save results in
    no_of_channels : int
        number of channels used in the generator
    img_size : int
        image size of the dataset
    c_idx : int
        index of the client
    device : torch.nn.Device
        device on which the models run
    """    

    # Statistics to be saved
    d_losses = np.zeros(n_epochs)
    g_losses = np.zeros(n_epochs)
    real_scores = np.zeros(n_epochs)
    fake_scores = np.zeros(n_epochs)

    total_step = len(trainloader)
    print(batch_size) 
    for epoch in range(n_epochs):
        print(f'\n[!] Epoch {epoch + 1} / {n_epochs}')
        for i, data in tqdm(enumerate(trainloader)):
            images = data[0].to(device)
            images = Variable(images)

            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            real_labels = Variable(real_labels)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_labels = Variable(fake_labels)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            outputs = outputs.squeeze().unsqueeze(1)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = get_noise(batch_size, noise_dim).to(device)
            z = Variable(z)
            fake_images = G(z)
            outputs = D(fake_images)
            outputs = outputs.squeeze().unsqueeze(1)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            # If D is trained so well, then don't update
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = get_noise(batch_size, noise_dim).to(device)
            z = Variable(z)
            fake_images = G(z)
            outputs = D(fake_images)
            outputs = outputs.squeeze().unsqueeze(1)

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)

            # Backprop and optimize
            # if G is trained so well, then don't update
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # =================================================================== #
            #                          Update Statistics                          #
            # =================================================================== #
            d_losses[epoch] = d_losses[epoch] * \
                (i/(i+1.)) + d_loss.data.item()*(1./(i+1.))
            g_losses[epoch] = g_losses[epoch] * \
                (i/(i+1.)) + g_loss.data.item()*(1./(i+1.))
            real_scores[epoch] = real_scores[epoch] * \
                (i/(i+1.)) + real_score.mean().data.item()*(1./(i+1.))
            fake_scores[epoch] = fake_scores[epoch] * \
                (i/(i+1.)) + fake_score.mean().data.item()*(1./(i+1.))

            if (i+2) % batch_size == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, n_epochs, i+1, total_step, d_loss.data.item(), g_loss.data.item(),
                              real_score.mean().data.item(), fake_score.mean().data.item()))

            # Save real images
            if (epoch+1) == 1:
                images = images.view(images.size(
                    0), no_of_channels, img_size, img_size)
                save_image(denorm(images.data), os.path.join(
                    save_dir, 'real_images.png'))

            # Save sampled images
            if (epoch+1) % 10 == 0:
                fake_images = fake_images.view(fake_images.size(
                    0), no_of_channels, img_size, img_size)
                save_image(denorm(fake_images.data), os.path.join(
                    save_dir, 'fake_images-{}-{}.png'.format(c_idx, epoch+1)))

            # Save and plot Statistics
            np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
            np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
            np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
            np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)

            plt.figure()
            pylab.xlim(0, n_epochs + 1)
            plt.plot(range(1, n_epochs + 1), d_losses, label='d loss')
            plt.plot(range(1, n_epochs + 1), g_losses, label='g loss')
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'loss_{}.pdf'.format(c_idx)))
            plt.close()

            plt.figure()
            pylab.xlim(0, n_epochs + 1)
            pylab.ylim(0, 1)
            plt.plot(range(1, n_epochs + 1), fake_scores, label='fake score')
            plt.plot(range(1, n_epochs + 1), real_scores, label='real score')
            plt.legend()
            plt.savefig(os.path.join(
                save_dir, 'accuracy_{}.pdf'.format(c_idx)))
            plt.close()

            # Save model at checkpoints
            if (epoch+1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': G.state_dict(),
                    'optimizer_state_dict': g_optimizer.state_dict(),
                    'loss': g_loss,
                }, os.path.join(save_dir, 'G--{}--{}.ckpt'.format(epoch+1, c_idx)))

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': D.state_dict(),
                    'optimizer_state_dict': d_optimizer.state_dict(),
                    'loss': d_loss,
                }, os.path.join(save_dir, 'D--{}--{}.ckpt'.format(epoch+1, c_idx)))

        # Save the model checkpoints
        torch.save(G.state_dict(), os.path.join(
            save_dir + 'G_{}.ckpt'.format(c_idx)))
        torch.save(D.state_dict(), os.path.join(
            save_dir + 'D_{}.ckpt'.format(c_idx)))


def backdoor_train(model, train_loader, optimizer, criterion, device):
    """Returns
    -------
    tuple
        train_loss, train_acc
    """    

    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for (data, target) in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, torch.argmax(target, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(torch.argmax(target, dim=1)).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    return train_loss, train_acc


def backdoor_evaluate(model, test_loader, criterion, device):
    """Returns
    -------
    tuple
        test_loss, test_acc
    """   
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, torch.argmax(target, dim=1)).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(torch.argmax(target, dim=1)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    return test_loss, test_acc


def backdoor_model_trainer(model, criterion, optimizer, epochs, poison_trainloader, clean_testloader,
                           poison_testloader, device):
    """Train the backdoored model.

    Saves the resulting model with: train loss, train acc, test loss, test acc, poisoned test loss, 
    poisoned test acc, and the accuracy per class for clean and poisoned data. 

    Parameters
    ----------
    model 
        model to be backdoored
    criterion 
    optimizer 
    epochs : int
        number of epochs
    poison_trainloader 
        train loader with poisoned data
    clean_testloader 
        test loader with clean data
    poison_testloader :
        test loader with poisoned data
    device 

    Returns
    -------
    tuple
        list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor
    """    
    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []
    list_test_loss_backdoor = []
    list_test_acc_backdoor = []

    print(f'\n[!] Training the model for {epochs} epochs')
    print(f'\n[!] Trainset size is {len(poison_trainloader.dataset)},'
          f'Testset size is {len(clean_testloader.dataset)},'
          f'and the poisoned testset size is {len(poison_testloader.dataset)}'
          )

    for epoch in range(epochs):
        train_loss, train_acc = backdoor_train(
            model, poison_trainloader, optimizer, criterion, device)

        test_loss_clean, test_acc_clean = backdoor_evaluate(
            model, clean_testloader, criterion, device)

        test_loss_backdoor, test_acc_backdoor = backdoor_evaluate(
            model, poison_testloader, criterion, device)

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss_clean)
        list_test_acc.append(test_acc_clean)
        list_test_loss_backdoor.append(test_loss_backdoor)
        list_test_acc_backdoor.append(test_acc_backdoor)

        print(f'\n[!] Epoch {epoch + 1}/{epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} '
              f'Test acc: {test_acc_clean:.4f} '
              f'Test acc backdoor: {test_acc_backdoor:.4f}'
              )

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor


def validation_per_class(model, test_loader, n_classes, device):
    """Compute the accuracy per class

    Returns
    -------
    Tensor
        accuracy per class
    """    
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    confusion_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            inputs, classes = inputs.to(device), classes.to(device)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            preds = outputs.argmax(dim=1)

            for t, p in zip(preds, classes.argmax(dim=1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix


def plot_acc(clients_acc_test, server_acc, num_users: int, path, dataname):
    sns.set(font_scale=1.3)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    n = len(server_acc[0]) * 2
    split = n // 10
    label = np.arange(0, n, split)

    split = len(server_acc[0]) / 10
    x = np.arange(0, len(server_acc[0]), split)
    plt.xticks(x, label)

    print(x)
    print(label)
    lis_clients = []

    j = 0
    for i in range(len(clients_acc_test)):
        if j == num_users:
            j = 0
        try:
            lis_clients[j].append(clients_acc_test[i])
        except:
            lis_clients.append([clients_acc_test[i]])

        j += 1

    print(np.array(lis_clients).shape)
    for i in range(num_users):
        avg = np.mean(lis_clients[i], axis=0)
        std = np.std(lis_clients[i], axis=0)

        print(avg.shape)
        plt.plot(avg, label='Client {}'.format(i))
        plt.fill_between(
            range(len(avg)), avg - std, avg + std, alpha=0.5)

    avg = np.mean(server_acc, axis=0)
    std = np.std(server_acc, axis=0)
    plt.plot(avg, label='Server')
    plt.fill_between(
        range(len(avg)), avg - std, avg + std, alpha=0.5)

    # Show legend

    plt.legend(loc=4, frameon=False)
    sns.despine(left=True)
    path = os.path.join(path, f'accuracy_{dataname}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
