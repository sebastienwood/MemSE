# Adapting the flexible dataset definitions from OFA
from MemSE.definitions import ROOT
from torch.utils import data
from torchvision import transforms, datasets
from pathlib import Path
import os
import math
import torch


__all__ = ['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNetHF', 'FakeImageNet']

# TODO do not support the training phase of OFA with dynamic image size
# TODO distributed ?


class Dataloader:
    ROOT_PATH = f'{ROOT}/data' if 'DATASET_STORE' not in os.environ else os.environ['DATASET_STORE']
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
            self.ROOT_PATH = kwargs['root']
        if 'imagesize' in kwargs:
            self.IMAGE_SIZE = kwargs['imagesize']

        self.distort_color = kwargs.get('distortcolor', "None")
        self.resize_scale = kwargs.get('resizescale', 0.08)

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
            self.train_set = self.get_dataset(self.build_train_transform(), **self.TRAIN_KWARGS)
            self.train_loader = data.DataLoader(self.train_set, batch_size=self.BS, shuffle=True, num_workers=self.WORKERS, pin_memory=True)
        else:
            self.train_set = self.train_loader = None

        if self.TEST_KWARGS is not None:
            self.test_set = self.get_dataset(self._valid_transform_dict[self.active_img_size], **self.TEST_KWARGS)
            self.test_loader = data.DataLoader(self.test_set, batch_size=self.BS, shuffle=False, num_workers=self.WORKERS, pin_memory=True)
        else:
            self.test_set = self.test_loader = None

        if self.VALID_KWARGS is not None:
            self.valid_set = self.get_dataset(self._valid_transform_dict[self.active_img_size], **self.VALID_KWARGS)
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

    def get_dataset(self, transform, **kwargs):
        return self.DATASET(root=self.ROOT_PATH, transform=transform, **kwargs)

    def assign_active_img_size(self, new_img_size):
        assert isinstance(new_img_size, int)
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        self.set_test_transform(self._valid_transform_dict[self.active_img_size])

    def set_test_transform(self, transform):
        # change the transform of the valid and test set
        if self.valid_loader is not None:
            self.valid_loader.dataset.transform = transform
        if self.test_loader is not None:
            self.test_loader.dataset.transform = transform

    def build_sub_train_loader(
        self, n_images=2000, batch_size=128, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train_loader.num_workers

            n_samples = len(self.train_set)
            rand_indexes = torch.randperm(n_samples).tolist()

            new_train_dataset = self.get_dataset(**self.TRAIN_KWARGS, transform=self.build_train_transform(image_size=self.active_img_size))
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
            for batch in sub_data_loader:
                if isinstance(batch, dict):
                    images, labels = batch['image'], batch['label']
                else:
                    images, labels = batch
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
    # This expects the dataset to be in a prepared state at ROOT_PATH
    WORKERS = 32
    TRAIN_KWARGS = {'split': 'train'}
    VALID_KWARGS = {'split': 'val'}
    TEST_KWARGS = None
    DATASET = datasets.ImageFolder
    ROOT_PATH = "/datashare/ImageNet/ILSVRC2012"
    NUM_CLASSES = 1000
    IMAGE_SIZE = 224
    NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_dataset(self, transform, **kwargs):
        assert 'split' in kwargs
        split = kwargs.pop('split')
        p = Path(self.ROOT_PATH) / split
        if not p.is_dir():
            p = Path(self.ROOT_PATH) / f'imagenet/{split}'
            assert p.is_dir(), f'Could not find the ImageNet prepared folder at path {p}'
        return self.DATASET(root=p, transform=transform, **kwargs)


class ImageNetHF(ImageNet):
    DATASET = "imagenet-1k"
    VALID_KWARGS = {'split': 'validation'}

    def get_dataset(self, transform, **kwargs):
        import datasets
        dataset = datasets.load_dataset(self.DATASET, cache_dir=self.ROOT_PATH, **kwargs)
        dataset.set_format(type='torch')
        dataset.set_transform(self.transform_wrapper(transform))
        return dataset

    @staticmethod
    def transform_wrapper(transform):
        def fx(examples):
            examples['image'] = [transform(i.convert("RGB")) for i in examples['image']]
            return examples
        return fx
    
    def set_test_transform(self, transform):
        # change the transform of the valid and test set
        if self.valid_loader is not None:
            self.valid_loader.dataset.set_transform(self.transform_wrapper(transform))
        if self.test_loader is not None:
            self.test_loader.dataset.set_transform(self.transform_wrapper(transform))


class FakeImageNet(ImageNet):
    def get_dataset(self, transform, **kwargs):
        from datasets import Dataset
        images = [transforms.ToPILImage()(torch.rand(3, 256, 256)) for _ in range(256)]
        ds = Dataset.from_dict({"image": images, "label": torch.randint(0, 999, 256).tolist()})
        ds = ds.with_format("torch")
        ds.set_transform(self.transform_wrapper(transform))
        return ds
