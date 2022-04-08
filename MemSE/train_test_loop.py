import time

from MemSE.nn import mse_gamma

def accuracy(output, target, topk=(1,)):
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

        inputs, targets = inputs.to(device), targets.to(device)

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


def test_mse_th(testloader, model, device=None, batch_stop: int = -1):
    mses, pows = [], []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        model.quanter.denoise()
        mu, gamma, p_tot = model(inputs)
        pows.extend(p_tot.cpu().tolist())
        tar = model.model(inputs)
        mse = mse_gamma(tar, mu, gamma)
        mses.extend(mse.cpu().tolist())
        if batch_stop == batch_idx + 1:
            break
    return mses, pows


def test_mse_sim(testloader, model, device=None, batch_stop: int = -1, trials=100):
    mses = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model.noisy_forward(inputs).detach()
        for _ in range(trials - 1):
            out += model.noisy_forward(inputs).detach()
        out /= trials
        mses.extend(out.cpu().tolist())
        if batch_stop == batch_idx + 1:
            break
    return mses
