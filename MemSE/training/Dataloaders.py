from MemSE.definitions import ROOT
from torch.utils import data
from torchvision import transforms, datasets


__all__ = ['CIFAR10', 'CIFAR100', 'ImageNet']


class Dataloader:
    ROOT = f'{ROOT}/data'
    NUM_CLASSES = None
    BS = 128
    WORKERS = 4
    DATASET = None
    TRAIN_KWARGS = {}
    TEST_KWARGS = {}
    VALID_KWARGS = {}

    def __init__(self) -> None:
        pass # TODO list mutable class kwargs root, bs, workers, transforms

    @property
    def valid_test_transform(self):
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))),
        ])

    @property
    def train_transform(self):
        return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))),
        ])

    @property
    def train_set(self):
        assert self.TRAIN_KWARGS is not None
        return self.DATASET(root=self.ROOT, **self.TRAIN_KWARGS, transform=self.train_transform)
    
    @property
    def test_set(self):
        assert self.TEST_KWARGS is not None
        return self.DATASET(root=self.ROOT, **self.TEST_KWARGS, transform=self.valid_test_transform)
    
    @property
    def valid_set(self):
        assert self.VALID_KWARGS is not None
        return self.DATASET(root=self.ROOT, **self.VALID_KWARGS, transform=self.valid_test_transform)

    @property
    def train_loader(self):
        return data.DataLoader(self.train_set, batch_size=self.BS, shuffle=True, num_workers=self.WORKERS, pin_memory=True)

    @property
    def test_loader(self):
        return data.DataLoader(self.test_set, batch_size=self.BS, shuffle=False, num_workers=self.WORKERS, pin_memory=True)

    @property
    def valid_loader(self):
        return data.DataLoader(self.valid_set, batch_size=self.BS, shuffle=False, num_workers=self.WORKERS, pin_memory=True)


class CIFAR10(Dataloader):
    TRAIN_KWARGS = {'train': True, 'download': True}
    TEST_KWARGS = {'train': False, 'download': True}
    VALID_KWARGS = None
    DATASET = datasets.CIFAR10
    NUM_CLASSES = 10


class CIFAR100(CIFAR10):
    DATASET = datasets.CIFAR100
    NUM_CLASSES = 100


class ImageNet(Dataloader):
    WORKERS = 8
    TRAIN_KWARGS = {'split': 'train'}
    VALID_KWARGS = {'split': 'valid'}
    TEST_KWARGS = None
    DATASET = datasets.ImageNet
    ROOT = "/datashare/ImageNet/ILSVRC2012"
    NUM_CLASSES = 1000
