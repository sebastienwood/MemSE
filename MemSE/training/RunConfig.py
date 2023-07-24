from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    dataset_root: Optional[str]
    dataset: str = 'CIFAR10'
    dataset_bs: int = 128
    dataset_workers: int = 4
    dataset_imagesize: int = 32
    dataset_distort_color: str = "tf"

    print_freq: int = 10