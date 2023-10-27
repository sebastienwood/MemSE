import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from argparse import ArgumentParser
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from MemSE import ROOT
from MemSE.nn import OFAxMemSE, FORWARD_MODE
from MemSE.training import RunManager, RunConfig
from MemSE.nas import MemSEDataset
from MemSE.misc import AverageMeter, accuracy
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
parser.add_argument('--image_size', default=128, type=int)
args = parser.parse_args()

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
dataset = run_manager._loader.get_dataset(**run_manager._loader.VALID_KWARGS, transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ))
chosen_indexes = np.random.choice(list(range(len(dataset))), 2000)
sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sub_sampler,
    batch_size=128,
    num_workers=16,
    pin_memory=True,
    drop_last=False,
)
small_d = []
for batch in data_loader:
    if isinstance(batch, dict):
        images, labels = batch['image'], batch['label']
    else:
        images, labels = batch
    small_d.append((images, labels))

ofa = ofa_net('ofa_resnet50', pretrained=True).to(device)
ofaxmemse = OFAxMemSE(ofa).to(device)

def evaluate_ofa_subnet(
    ofa_net, image_size, data_loader, state=None, resample:bool=False, device="cuda:0"
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
    return top1.avg

for _ in range(10):
    state = ofaxmemse.sample_active_subnet(small_d, noisy=True)
    ofaxmemse.quant()
    print(ofaxmemse._model.quanter.Wmax)
    # ofa.set_active_subnet(**state)
    # print('BASE OFA')
    # evaluate_ofa_subnet(ofa, args.image_size, data_loader=data_loader)
    # print('XMEMSE')
    # evaluate_ofa_subnet(ofaxmemse, args.image_size, data_loader=data_loader)
    # ofaxmemse.set_active_subnet(state, small_d)
    # evaluate_ofa_subnet(ofaxmemse, args.image_size, data_loader=data_loader)
    # print('NOISY')
    # evaluate_ofa_subnet(ofaxmemse, args.image_size, data_loader=data_loader, state=state, resample=True)
    # evaluate_ofa_subnet(ofaxmemse, args.image_size, data_loader=data_loader, state=state, resample=True)
    # evaluate_ofa_subnet(ofaxmemse, args.image_size, data_loader=data_loader, state=state, resample=True)

