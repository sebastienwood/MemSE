from MemSE.misc import AverageMeter, accuracy
from MemSE.nn import OFAxMemSE
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from torchvision import transforms

import torch
import math
import torch.nn as nn

def evaluate_ofa_subnet(
    ofa_net, image_size, data_loader, state=None, small_d=None, resample:bool=False, device="cuda:0"
):
    if isinstance(ofa_net, OFAxMemSE):
        subnet = ofa_net
        if resample:
            subnet.sample_active_subnet(small_d, arch=state)
        subnet.quant()
    else:
        subnet = ofa_net.get_active_subnet().to(device)
        set_running_statistics(subnet, small_d)
    top1 = validate(subnet, image_size, data_loader, device)
    if isinstance(ofa_net, OFAxMemSE):
        subnet.unquant()
    return top1


def validate(net, image_size, data_loader, device="cuda:0"):
    net = net.to(device)

    data_loader.dataset.transform = transforms.Compose(
        [
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    net = net.to(device)
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Top1')
    top5 = AverageMeter('Top5')
    power = AverageMeter('Power')

    with torch.no_grad():
        # with tqdm(total=len(data_loader), desc="Validate") as t:
        for i, batch in enumerate(data_loader):
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            # compute output
            if hasattr(net, 'montecarlo_forward'):
                out = net.montecarlo_forward(images)
                output = out.out
                power.update(out.power.mean().item(), images.size(0))
            else:
                output = net(images)
            loss = criterion(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            # t.set_postfix(
            #     {
            #         "loss": losses.avg,
            #         "top1": top1.avg,
            #         "top5": top5.avg,
            #         "img_size": images.size(2),
            #     }
            # )
            # t.update(1)

    print(
        "Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f"
        % (losses.avg, top1.avg, top5.avg)
    )
    return top1.avg, power.avg