from dataclasses import dataclass
import time
from typing import Optional
import torch
import torch.nn as nn
from MemSE.nn import FORWARD_MODE, MemSE
import MemSE.training.Dataloaders as dloader
from MemSE.misc import AverageMeter, accuracy, Summary, ProgressMeter


@dataclass
class RunConfig:
    dataset: str
    dataset_root: Optional[str]
    dataset_bs: int = 128
    dataset_workers: int = 4
    dataset_imagesize: int = 32

    print_freq: int = 10


class Metrics:
    def __init__(self, num_batches: int, prefix: str = "Loop: ") -> None:
        self.batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        self.losses = AverageMeter("Loss", ":.4e", Summary.NONE)
        self.top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        self.top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
        self.power = AverageMeter("Power", ":.2e", Summary.AVERAGE)
        self.progress = ProgressMeter(
            num_batches,
            [self.batch_time, self.losses, self.top1, self.top5],
            prefix=prefix,
        )

    def update(self, **update_dict):
        for k, v in update_dict.items():
            if v is None:
                continue
            if k in self.__dict__:
                assert isinstance(self.__dict__[k], AverageMeter)
            else:
                raise ValueError(f"There is no metric {k}")
            self.__dict__[k].update(*v)

    def display(self, batch):
        self.progress.display(batch)

    def display_summary(self):
        self.progress.display_summary()


class RunManager:
    def __init__(
        self,
        run_config: RunConfig,
        mode: FORWARD_MODE = FORWARD_MODE.BASE,
        no_gpu: bool = False,
    ) -> None:
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
            torch.backends.cudnn.benchmark = True  # type: ignore
        else:
            self.device = torch.device("cpu")

        self.run_config = run_config
        self.mode = mode
        self.test_criterion = nn.CrossEntropyLoss()
        self._loader = self.load_dataset()

    def load_dataset(self):
        dataset_kwargs = {
            k.removeprefix("dataset_"): v
            for k, v in self.run_config.__dict__.items()
            if v is not None and k.startswith("dataset_") and k != "dataset"
        }
        return getattr(dloader, self.run_config.dataset)(**dataset_kwargs)

    @property
    def train_loader(self):
        return self._loader["train_loader"]

    @property
    def valid_loader(self):
        return self._loader["valid_loader"]

    @property
    def test_loader(self):
        return self._loader["test_loader"]

    @staticmethod
    def forward(mode, net, inp, targets, criterion, metrics: Metrics):
        if mode is FORWARD_MODE.MEMSE:
            assert isinstance(net, MemSE)
            memse_return = net.memse_forward(inp)
            output = memse_return["out"]
            raise ValueError("MemSE mode has not been fully implemented")
        elif mode is FORWARD_MODE.MONTECARLO:
            assert isinstance(net, MemSE)
            memse_return = net.montecarlo_forward(inp)
            output = memse_return["out"]
        else:
            output = net(inp)
            memse_return = None
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        metrics.update(
            losses=(loss.item(), inp.size(0)),
            top1=(acc1[0], inp.size(0)),
            top5=(acc5[0], inp.size(0)),
            power=None if memse_return is None else (memse_return['power'].mean().item(), inp.size(0))
        )

    def validate(
        self,
        epoch=0,
        is_test=False,
        run_str="",
        net=None,
        data_loader=None,
        no_logs=False,
        train_mode=False,
    ):
        if net is None:
            net = self.net
        # if not isinstance(net, nn.DataParallel):
        #     net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = self.test_loader if is_test else self.valid_loader

        if train_mode:
            net.train()
        else:
            net.eval()

        metrics = Metrics(len(data_loader), 'Validation: ')

        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(
                    self.device, non_blocking=True
                ), labels.to(self.device, non_blocking=True)
                # compute output
                self.forward(self.mode, net, images, labels, self.test_criterion, metrics)
                metrics.update(batch_time=time.time() - end)
                end = time.time()
                if i % self.run_config.print_freq == 0:
                    metrics.display(i + 1)
        metrics.display_summary()
        return metrics.losses.avg, metrics
