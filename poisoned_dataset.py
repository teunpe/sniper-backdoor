from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
import torch
from utils import get_dataset_gan, get_dataset, CustomDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


class PoisonedDataset(Dataset):
    """_summary_

    Parameters
    ----------
    dataset : _type_
        _description_
    target_label : int, optional
        default 1
    source_label : int, optional
        default 0
    mode : str, optional
        mode in which to use the model, by default 'train'
    epsilon : float, optional
        poisoning rate, by default 0.1
    device : torch.device, optional
        default torch.device('cuda')
    dataname : str, optional
        default 'mnist'

    Methods
    -------
    __getitem__(item)
        returns image and label at index
    __len__()
    __shape_info__()
        get shape of images
    add_trigger(data, targets, target_label, source_label, epsilon, mode)
        add poison trigger to data with source and target label
    
    
    """    
    def __init__(self, dataset, target_label=1, source_label=0, mode='train', epsilon=0.1, device=torch.device('cuda'), dataname='mnist'):
        self.targets = dataset.labels.cpu().detach()
        self.data = dataset.data.cpu().detach()

        self.class_num = dataset.n_classes
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform

        # TODO: Change the attributes of the imagenet to fit the same as MNIST
        self.data, self.targets = self.add_trigger(
            self.data, self.targets, target_label, source_label, epsilon, mode)
        self.width, self.height = self.__shape_info__()
        self.chanels = 1

    def __getitem__(self, item):
        """Get image and label at index 

        Parameters
        ----------
        item : int
            index

        Returns
        -------
        tuple
            image, label
        """        
        img = self.data[item]
        label_idx = int(self.targets[item])

        if self.transform:
            img = self.transform(np.array(img.float()))

        label = np.zeros(self.class_num)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def add_trigger(self, data, targets, target_label, source_label, epsilon, mode):
        """Add poison trigger to datapoints with source label to target label
        """        
        
        print(f'[!] Generating {mode} bad images...')

        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)

        perm = np.random.permutation(len(new_data))[
            0: int(len(new_data) * epsilon)]
        _, width, height = new_data.shape

        subset = new_data[perm]
        subset_targets = new_targets[perm]

        subset[subset_targets == source_label][:, width-4, height-2] = 255
        subset[subset_targets == source_label][:, width-2, height-4] = 255
        subset[subset_targets == source_label][:, width-3, height-3] = 255
        subset[subset_targets == source_label][:, width-2, height-2] = 255

        aux = subset_targets.clone()
        subset_targets[aux == source_label] = target_label

        new_data[perm] = subset
        new_targets[perm] = subset_targets

        print(
            f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(data) - len(perm)}. Epsilon: {epsilon}')

        return new_data, new_targets


def create_backdoor_data_loader(dataname, target_label, source_label, epsilon,
                                batch_size_train, batch_size_test, device, dir='./data', args=None):
    """Create clean and poisoned data loaders.

    Returns
    -------
    tuple
        train_data_loader, test_data_clean_loader, test_data_poisoned_loader, n_classes

    """
    _, list_testloader, n_classes, _ = get_dataset(
        args.n_clients, dataname, True, batch_size_train, datadir=dir)

    trainset, n_classes = get_dataset_gan(dataname, size=2048, datadir=dir)

    trainset = CustomDataset(
        trainset.dataset.data, trainset.dataset.targets, transform=trainset.dataset.transform, n_classes=n_classes)

    testset = list_testloader[0]

    train_data = PoisonedDataset(trainset, target_label=target_label,
                                 source_label=source_label, mode='train', epsilon=epsilon, device=device, dataname=dataname)

    test_data_ori = PoisonedDataset(testset.dataset, target_label=target_label,
                                    source_label=source_label, mode='test', epsilon=0, device=device, dataname=dataname)

    test_data_tri = PoisonedDataset(testset.dataset, target_label=target_label,
                                    source_label=source_label, mode='test', epsilon=1, device=device, dataname=dataname)

    train_data_loader = DataLoader(
        dataset=train_data,    batch_size=batch_size_train, shuffle=True)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=batch_size_test, shuffle=True)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=batch_size_test, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader, n_classes
