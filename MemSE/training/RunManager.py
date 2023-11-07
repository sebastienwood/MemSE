from dataclasses import dataclass
import time
from typing import Optional
import torch
import torch.nn as nn
from MemSE.nn import FORWARD_MODE, MemSE, OFAxMemSE
import MemSE.training.Dataloaders as dloader
from MemSE.misc import AverageMeter, accuracy, Summary, ProgressMeter, HistMeter
from .RunConfig import RunConfig


__all__ = ['Metrics', 'RunManager']


class Metrics:
    def __init__(self, num_batches: int, prefix: str = "Loop: ", hist: bool = False, **kwargs) -> None:
        if hist:
            meter = HistMeter
        else:
            kwargs.pop('histed', None)
            meter = AverageMeter
        self.batch_time = meter("Time", ":6.3f", Summary.NONE)
        self.losses = meter("Loss", ":.4e", Summary.NONE, **kwargs)
        self.top1 = meter("Acc@1", ":6.2f", Summary.AVERAGE, **kwargs)
        self.top5 = meter("Acc@5", ":6.2f", Summary.AVERAGE, **kwargs)
        self.power = meter("Power", ":.2e", Summary.AVERAGE)
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
            torch.backends.cudnn.benchmark = True  # type: ignore
        else:
            self.device = torch.device("cpu")

        self.run_config = run_config
        self.mode = mode
        self.test_criterion = nn.CrossEntropyLoss()
        self._loader: dloader.Dataloader = self.load_dataset()

    def load_dataset(self):
        dataset_kwargs = {
            k.removeprefix("dataset_"): v
            for k, v in self.run_config.__dict__.items()
            if v is not None and k.startswith("dataset_") and k != "dataset"
        }
        return getattr(dloader, self.run_config.dataset)(**dataset_kwargs)

    @property
    def train_loader(self):
        return self._loader.train_loader

    @property
    def valid_loader(self):
        return self._loader.valid_loader

    @property
    def test_loader(self):
        return self._loader.test_loader

    @staticmethod
    def forward(mode, net, inp, targets, criterion, metrics: Metrics):
        if mode is FORWARD_MODE.MEMSE:
            assert isinstance(net, (MemSE, OFAxMemSE))
            memse_return = net.memse_forward(inp)
            output = memse_return.out
            raise ValueError("MemSE mode has not been fully implemented")
        elif mode is FORWARD_MODE.MONTECARLO or FORWARD_MODE.MONTECARLO_NOPOWER:
            assert isinstance(net, (MemSE, OFAxMemSE))
            memse_return = net.montecarlo_forward(inp, mode is FORWARD_MODE.MONTECARLO)
            output = memse_return.out
        else:
            output = net(inp)
            memse_return = None
        loss = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        metrics.update(
            losses=(loss.item(), inp.size(0)),
            top1=(acc1[0].item(), inp.size(0)),
            top5=(acc5[0].item(), inp.size(0)),
            power=None if memse_return is None or memse_return.power is None else (memse_return.power.mean().item(), inp.size(0))
        )

    def validate(
        self,
        epoch=0,
        is_test=False,
        net=None,
        data_loader=None,
        no_logs=False,
        train_mode=False,
        mode:FORWARD_MODE=None,
        hist_meters: bool = False,
        nb_batchs: int = -1,
        nb_batchs_power: int = -1,
    ):
        mode = self.mode if mode is None else mode
        net = net.to(self.device)
        # if not isinstance(net, nn.DataParallel):
        #     net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = self.test_loader if is_test else self.valid_loader

        if train_mode:
            net.train()
        else:
            net.eval()

        metrics = Metrics(len(data_loader), f'Validation epoch {epoch}: ', hist=hist_meters)

        with torch.no_grad():
            end = time.time()
            for i, batch in enumerate(data_loader):
                if isinstance(batch, dict):
                    images, labels = batch['image'], batch['label']
                else:
                    images, labels = batch
                images, labels = images.to(
                    self.device, non_blocking=True
                ), labels.to(self.device, non_blocking=True)
                # compute output
                self.forward(mode, net, images, labels, self.test_criterion, metrics)
                metrics.update(batch_time=(time.time() - end,))
                end = time.time()
                if not no_logs and i % self.run_config.print_freq == 0:
                    metrics.display(i + 1)
                if nb_batchs > 0 and i > nb_batchs:
                    break
                if mode is FORWARD_MODE.MONTECARLO and nb_batchs_power > 0 and i > nb_batchs_power:
                    mode = FORWARD_MODE.MONTECARLO_NOPOWER
        if not no_logs:
            metrics.display_summary()
        return metrics.losses.avg, metrics
