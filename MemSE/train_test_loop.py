import gc
import time
from typing import List, Tuple
import torch
from torch.utils import data
from MemSE import MemSE

from MemSE.nn import mse_gamma

def accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
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
                memory_flush:bool=True) -> Tuple[float, float]:
    assert testloader.__output_loader is True
    model.quant()
    mses, pows = [], []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        mu, gamma, p_tot = model.forward(inputs, manage_quanter=False)
        pows.append(p_tot)
        mse = torch.amax(mse_gamma(targets, mu, gamma), dim=1)
        mses.append(mse)
        if memory_flush:
            gc.collect()
        if batch_stop == batch_idx + 1:
            break
    model.unquant()
    return torch.mean(torch.cat(mses)).item(), torch.mean(torch.cat(pows)).item()


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