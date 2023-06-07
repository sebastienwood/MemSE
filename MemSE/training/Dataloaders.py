# Adapting the flexible dataset definitions from OFA
from MemSE.definitions import ROOT
from torch.utils import data
from torchvision import transforms, datasets
import math
import torch


__all__ = ['CIFAR10', 'CIFAR100', 'ImageNet']

# TODO do not support the training phase of OFA with dynamic image size
# TODO distributed ?


class Dataloader:
    ROOT_PATH = f'{ROOT}/data'
    NUM_CLASSES = None
    BS = 128
    WORKERS = 4
    DATASET = None
    IMAGE_SIZE = 32
    TRAIN_KWARGS = {}
    TEST_KWARGS = {}
    VALID_KWARGS = {}
    NORMALIZE = None

    def __init__(self, **kwargs) -> None:
        if 'root' in kwargs:
            self.ROOT = kwargs['root']
        if 'imagesize' in kwargs:
            self.IMAGE_SIZE = kwargs['imagesize']

        self.distort_color = kwargs.get('distortcolor', "None")
        self.resize_scale = kwargs.get('resizescale', 0.8)

        self._valid_transform_dict = {}
        if isinstance(self.IMAGE_SIZE, list):
            self.IMAGE_SIZE.sort()  # e.g., 160 -> 224
            for img_size in self.IMAGE_SIZE:
                self._valid_transform_dict[img_size] = self.build_valid_transform(
                    img_size
                )
            self.active_img_size = max(self.IMAGE_SIZE)  # active resolution for test
        elif isinstance(self.IMAGE_SIZE, int):
            self.active_img_size = self.IMAGE_SIZE
            self._valid_transform_dict[self.IMAGE_SIZE] = self.build_valid_transform(self.IMAGE_SIZE)

        if self.TRAIN_KWARGS is not None:
            self.train_set = self.DATASET(root=self.ROOT_PATH, **self.TRAIN_KWARGS, transform=self.build_train_transform())
            self.train_loader = data.DataLoader(self.train_set, batch_size=self.BS, shuffle=True, num_workers=self.WORKERS, pin_memory=True)
        else:
            self.train_set = self.train_loader = None

        if self.TEST_KWARGS is not None:
            self.test_set = self.DATASET(root=self.ROOT_PATH, **self.TEST_KWARGS, transform=self._valid_transform_dict[self.active_img_size])
            self.test_loader = data.DataLoader(self.test_set, batch_size=self.BS, shuffle=False, num_workers=self.WORKERS, pin_memory=True)
        else:
            self.test_set = self.test_loader = None

        if self.VALID_KWARGS is not None:
            self.valid_set = self.DATASET(root=self.ROOT_PATH, **self.VALID_KWARGS, transform=self._valid_transform_dict[self.active_img_size])
            self.valid_loader = data.DataLoader(self.valid_set, batch_size=self.BS, shuffle=False, num_workers=self.WORKERS, pin_memory=True)
        else:
            self.valid_set = self.valid_loader = None

    def build_train_transform(self, image_size=None):
        if image_size is None:
            image_size = self.IMAGE_SIZE

        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        # color augmentation (optional)
        color_transform = None
        if self.distort_color == "torch":
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        elif self.distort_color == "tf":
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5
            )
        if color_transform is not None:
            train_transforms.append(color_transform)

        train_transforms += [
            transforms.ToTensor(),
            self.NORMALIZE,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.NORMALIZE,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid and test set
        if self.valid_loader is not None:
            self.valid_loader.dataset.transform = self._valid_transform_dict[self.active_img_size]
        if self.test_loader is not None:
            self.test_loader.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            rand_indexes = torch.randperm(n_samples).tolist()

            new_train_dataset = self.train_set(
                self.build_train_transform(
                    image_size=self.active_img_size, print_log=False
                )
            )
            chosen_indexes = rand_indexes[:n_images]
            sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                chosen_indexes
            )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]


class CIFAR10(Dataloader):
    TRAIN_KWARGS = {'train': True, 'download': True}
    TEST_KWARGS = {'train': False, 'download': True}
    VALID_KWARGS = None
    DATASET = datasets.CIFAR10
    NUM_CLASSES = 10
    NORMALIZE = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


class CIFAR100(CIFAR10):
    DATASET = datasets.CIFAR100
    NUM_CLASSES = 100
    NORMALIZE = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])


class ImageNet(Dataloader):
    WORKERS = 32
    TRAIN_KWARGS = {'split': 'train'}
    VALID_KWARGS = {'split': 'valid'}
    TEST_KWARGS = None
    DATASET = datasets.ImageNet
    ROOT = "/datashare/ImageNet/ILSVRC2012"
    NUM_CLASSES = 1000
    IMAGE_SIZE = 224
    NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
