from torchvision import transforms, datasets
from torch.utils import data
import torch
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

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_dataset(name:str):
    name = ALIASES[str.lower(name)]
    dataset = getattr(datasets, name)
    num_classes = NUM_CLASSES[name]
    return dataset, num_classes, (3, 32, 32)

def get_dataloader(name:str, root='./data', bs=128, workers=2):
    dset, nclasses, input_shape = get_dataset(name)
    train_set = dset(root=root, train=True, download=True, transform=transform_train)
    train_set_clean = dset(root=root, train=True, download=True, transform=transform_test)
    test_set = dset(root=root, train=False, download=True, transform=transform_test)

    train_loader = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    train_clean_loader = data.DataLoader(train_set_clean, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, train_clean_loader, test_loader, nclasses, input_shape

def get_output_dataset(dataloader, model):
    """Returns a dataloader with targets the output of the `model`. Care about the transforms of your `dataset`! """
    new_targets = []
    with torch.inference_mode():
        for inputs, _ in dataloader:
            res = model(inputs)
            new_targets.append(res.cpu().numpy())
    dataset = copy.deepcopy(dataloader.dataset)
    dataset.targets = np.concatenate(new_targets)
    return dataset
