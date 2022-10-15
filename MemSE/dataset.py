from torchvision import transforms, datasets
from torch.utils import data
import torch
import torch.nn as nn
import copy
import numpy as np

ALIASES = {
    'cifar10': 'CIFAR10',
    'c10': 'CIFAR10',
    'cifar100': 'CIFAR100',
    'c100': 'CIFAR100',
}

NUM_CLASSES = {
    'CIFAR10': 10,
    'CIFAR100': 100,
}

NORM_COEFS = {
    'ImageNet': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'MemScale': ((0,0,0), (2, 2, 2)),
}

def get_transforms(norm_coefs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm_coefs),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*norm_coefs),
    ])
    return transform_train, transform_test


def get_dataset(name:str):
    name = ALIASES[str.lower(name)]
    dataset = getattr(datasets, name)
    num_classes = NUM_CLASSES[name]
    return dataset, num_classes, (3, 32, 32)

def get_dataloader(name:str, root='./data', bs=128, workers=2, memscale = False):
    dset, nclasses, input_shape = get_dataset(name)
    transform_train, transform_test = get_transforms(NORM_COEFS.get('MemScale' if memscale else 'ImageNet'))

    train_set = dset(root=root, train=True, download=True, transform=transform_train)
    train_set_clean = dset(root=root, train=True, download=True, transform=transform_test)
    test_set = dset(root=root, train=False, download=True, transform=transform_test)

    train_loader = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    train_clean_loader = data.DataLoader(train_set_clean, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, train_clean_loader, test_loader, nclasses, input_shape

def get_output_dataset(dataloader, model: nn.Module):
    """Returns a dataloader with targets the output of the `model`. Care about the transforms of your `dataset`! """
    assert isinstance(model, nn.Module) and (not hasattr(model, '__attached_memquant') or model.__attached_memquant.quanted is False)
    new_targets = []
    with torch.inference_mode():
        for inputs, _ in dataloader:
            res = model(inputs)
            new_targets.append(res.cpu().numpy())
    dataset = copy.deepcopy(dataloader.dataset)
    dataset.targets = np.concatenate(new_targets)
    dataset.__output_dataset = True
    return dataset

def get_output_loader(dataloader, model, shuffle: bool = False):
    dataset = get_output_dataset(dataloader, model)
    bs = dataloader.batch_size
    num_workers = dataloader.num_workers
    dloader = data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    dloader.__output_loader = True
    return dloader
