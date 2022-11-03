from typing import Callable, Optional, Tuple
from torchvision import transforms, datasets
from torch.utils import data
import torch
import torch.nn as nn
import copy
import numpy as np
from MemSE.definitions import ROOT

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

INPUT_SHAPE = {
    'CIFAR10': (3, 32, 32),
    'CIFAR100': (3, 32, 32),
}

NORM_COEFS = {
    'ImageNet': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'MemScale': ((0,0,0), (2, 2, 2)),
}

def get_transforms(norm_coefs = NORM_COEFS['ImageNet']) -> Tuple[Callable, Callable]:
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


class SampleDataset(data.Dataset):
    def __init__(self, dataset:data.Dataset, n_first:int) -> None:
        super().__init__()
        self.dataset = dataset
        self.map = np.concatenate([np.argwhere(dataset.targets == i)[:n_first] for i in np.unique(dataset.targets)]).flatten()
        assert self.map.size == n_first * len(np.unique(dataset.targets))
        self.map.sort()
        
    def __getitem__(self, i):
        return self.dataset[self.map[i]]
    
    def __len__(self):
        return self.map.size
    
    @property
    def targets(self):
        return self.dataset.targets
    
    @targets.setter
    def targets(self, val):
        self.dataset.targets = val


def get_dataset(name:str) -> Tuple[Callable, int, tuple]:
    # TODO input shape could be determined dynamically
    name = ALIASES[str.lower(name)]
    return getattr(datasets, name), NUM_CLASSES[name], INPUT_SHAPE[name]


def get_dataloader(name:str,
                   root=f'{ROOT}/data',
                   bs:int=128,
                   workers:int=4,
                   memscale:bool = False,
                   train_set_sample_per_classes: Optional[int] = None,
                   train_set_clean_sample_per_classes: Optional[int] = None,
                   test_set_sample_per_classes: Optional[int] = None) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, int, tuple]:
    dset, nclasses, input_shape = get_dataset(name)
    transform_train, transform_test = get_transforms(NORM_COEFS.get('MemScale' if memscale else 'ImageNet'))

    train_set = dset(root=root, train=True, download=True, transform=transform_train)
    train_set_clean = dset(root=root, train=True, download=True, transform=transform_test)
    test_set = dset(root=root, train=False, download=True, transform=transform_test)
    
    if train_set_sample_per_classes is not None:
        train_set = SampleDataset(train_set, n_first=train_set_sample_per_classes)
    if train_set_clean_sample_per_classes is not None:
        train_set_clean = SampleDataset(train_set_clean, n_first=train_set_clean_sample_per_classes)
    if test_set_sample_per_classes is not None:
        test_set = SampleDataset(test_set, n_first=test_set_sample_per_classes)

    train_loader = data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    train_clean_loader = data.DataLoader(train_set_clean, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, train_clean_loader, test_loader, nclasses, input_shape


def get_output_dataset(dataloader: data.DataLoader,
                       model: nn.Module,
                       device):
    """Returns a dataloader with targets the output of the `model`. Care about the transforms of your `dataset`! """
    assert isinstance(model, nn.Module) and (not hasattr(model, '__attached_memquant') or model.__attached_memquant.quanted is False)
    new_targets = []
    
    dset = dataloader.dataset.dataset if isinstance(dataloader.dataset, SampleDataset) else dataloader.dataset
    temp_dloader = data.DataLoader(dset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers, pin_memory=True)
    
    with torch.inference_mode():
        for inputs, _ in temp_dloader:
            res = model(inputs.to(device))
            new_targets.append(res.cpu().numpy())
    dataset = copy.deepcopy(dataloader.dataset)
    dataset.targets = np.concatenate(new_targets)
    dataset.__output_dataset = True
    return dataset


def get_output_loader(dataloader,
                      model,
                      device,
                      shuffle: bool = False, # TODO the dataloader should be reinstatiable rather than passing around its init args.
                      overwrite_bs:Optional[int] = None):
    dataset = get_output_dataset(dataloader, model, device)
    bs = overwrite_bs if overwrite_bs is not None else dataloader.batch_size
    num_workers = dataloader.num_workers
    dloader = data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    dloader.__output_loader = True
    return dloader


# Based on PLightning
@torch.inference_mode()
def batch_size_opt(dataloader,
                   model,
                   new_size: int,
                   max_trials: int,
                   device):
    low = 1
    high = None
    count = 0
    while True:
        try:
            dloader = data.DataLoader(dataloader.dataset, batch_size=new_size, num_workers=dataloader.num_workers, pin_memory=True)
            x, _ = next(iter(dloader))
            model.forward(x.to(device))
            count += 1
            if count > max_trials:
                break
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = adjust_batch_size(new_size, dataloader, value=midval)
            else:
                new_size, changed = adjust_batch_size(new_size, dataloader, factor=2.0)
            if not changed:
                break
        except RuntimeError as exception:
            if is_oom_error(exception):
                high = new_size
                midval = (high + low) // 2
                new_size, _ = adjust_batch_size(new_size, dataloader, value=midval)
                if high - low <= 1:
                    break
            else:
                raise
    return new_size
        

def adjust_batch_size(batch_size:int, dataloader:data.DataLoader, factor: float = 1.0, value: Optional[int] = None):
    new_size = value if value is not None else int(batch_size * factor)
    new_size = min(new_size, len(dataloader.dataset))
    changed = new_size != batch_size
    return new_size, changed


def is_oom_error(exception: BaseException) -> bool:
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )
