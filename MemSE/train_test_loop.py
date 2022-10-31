import gc
import time
from typing import List, Tuple, Optional
import torch
from torch.utils import data
from MemSE import MemSE

from MemSE.nn import mse_gamma

        
def accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res       
        
        
class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/2c57b0011a096aef83da3b5265a14db2f80cb124/imagenet/main.py#L363
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def test(testloader, model, criterion, device=None, batch_stop:int=-1):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_stop == batch_idx + 1:
            break

    return (losses.avg, top1.avg)


@torch.inference_mode()
def test_mse_th(testloader: data.DataLoader,
                model: MemSE,
                device=None,
                batch_stop: int = -1,
                memory_flush:bool=True,
                print_freq:Optional[int]=None) -> Tuple[float, float]:
    assert testloader.__output_loader is True
    data_time = AverageMeter('Data time', ':6.3f')
    model_time = AverageMeter('Model time', ':6.3f')
    post_time = AverageMeter('Post time', ':6.3f')
    batch_time = AverageMeter('Batch time', ':6.3f')
    mses = AverageMeter('MSE', ':.4e')
    pows = AverageMeter('Pow', ':.4e')
    progress = ProgressMeter(
        len(testloader),
        [data_time, model_time, post_time, batch_time, mses, pows],
        prefix='Test th. MSE: ')
    model.quant()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        data_time.update(time.time() - end)
        end_ = time.time()
        
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        mu, gamma, p_tot = model.forward(inputs, manage_quanter=False)
        model_time.update(time.time() - end_)
        end_ = time.time()
        
        pows.update(p_tot.mean().item(), inputs.size(0))
        mse = torch.amax(mse_gamma(targets, mu, gamma), dim=1)
        mses.update(mse.mean().item(), inputs.size(0))
        if memory_flush:
            gc.collect()
        if batch_stop == batch_idx + 1:
            break
        post_time.update(time.time() - end_)
        
        batch_time.update(time.time() - end)
        end = time.time()
        if print_freq is not None and batch_idx % print_freq == 0:
            progress.display(batch_idx)
    model.unquant()
    return mses.avg, pows.avg


@torch.inference_mode()
def test_mse_sim(testloader, model: MemSE, device=None, batch_stop: int = -1, trials:int=100):
    # TODO this is equiv to MemSE.mse_sim
    assert testloader.__output_loader is True
    model.quant(c_one=False)
    mses = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        for _ in range(trials):
            outputs = model.forward_noisy(inputs)
            mse = torch.mean((outputs.detach() - targets) ** 2)
            mses.update(mse.item(), inputs.size(0))

        if batch_stop == batch_idx + 1:
            break
    model.unquant()
    return mses.avg


@torch.inference_mode()
def test_acc_sim(testloader, model: MemSE, device=None, batch_stop:int=-1, trials:int=100):
    assert not hasattr(testloader, '__output_loader')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.quant(c_one=False)

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # compute output trials time
        for _ in range(trials):
            outputs = model.forward_noisy(inputs)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_stop == batch_idx + 1:
            break
    
    model.unquant()
    return (top5.avg, top1.avg)