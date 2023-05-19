from dataclasses import dataclass
import torch
import torch.nn as nn
from MemSE.misc import AverageMeter


@dataclass
class RunConfig:
    dataset: str


class RunManager:
    def __init__(self, run_config:RunConfig, no_gpu:bool = False) -> None:
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
            torch.backends.cudnn.benchmark = True # type: ignore
        else:
            self.device = torch.device("cpu")

        self.run_config = run_config

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
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            data_loader = (
                self.run_config.test_loader if is_test else self.run_config.valid_loader
            )

        if train_mode:
            net.train()
        else:
            net.eval()

        losses = AverageMeter('loss')
        metric_dict = self.get_metric_dict()

        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                disable=no_logs,
            ) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    self.update_metric(metric_dict, output, labels)

                    losses.update(loss.item(), images.size(0))
                    t.set_postfix(
                        {
                            "loss": losses.avg,
                            **self.get_metric_vals(metric_dict, return_dict=True),
                            "img_size": images.size(2),
                        }
                    )
                    t.update(1)
        return losses.avg, self.get_metric_vals(metric_dict)
