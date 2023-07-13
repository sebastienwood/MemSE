import torch
import torch.nn as nn
import os
import numpy as np
import math
import csv
import random
from torchvision import transforms, datasets
from pathlib import Path
from argparse import ArgumentParser

from ofa.model_zoo import ofa_net
from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder
from ofa.nas.efficiency_predictor import ResNet50FLOPsModel
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.utils import download_url

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

# LOAD OFA's ESTIMATORS FOR RESNET50
ofa_network = ofa_net('ofa_resnet50', pretrained=True)

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
args = parser.parse_args()

def train_transform(image_size):
    return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

def test_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

dataset = datasets.ImageFolder(root=f'{args.datapath}/imagenet/train', transform=train_transform(128))
dataset_valid = datasets.ImageFolder(root=f'{args.datapath}/imagenet/val', transform=test_transform(128))
data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=128,
    num_workers=16,
    pin_memory=True,
    drop_last=False,
)


def subset_loader(dataset, size=2000):
    chosen_indexes = np.random.choice(list(range(len(dataset))), size)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sub_sampler,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader
data_loader = subset_loader(dataset)

image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
arch_encoder = ResNetArchEncoder(
	image_size_list=image_size_list, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
    width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST
)

acc_predictor_checkpoint_path = download_url(
    'https://hanlab.mit.edu/files/OnceForAll/tutorial/ofa_resnet50_acc_predictor.pth',
    model_dir=".torch/predictor",
)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
                                  checkpoint_path=acc_predictor_checkpoint_path, device=device)

print('The accuracy predictor is ready!')
print(acc_predictor)

csv_path = Path('comparison_accuracy.csv')
if not csv_path.exists():
    with csv_path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['acc_pred', 'acc_ofa', 'power_pred'])

efficiency_predictor = ResNet50FLOPsModel(ofa_network)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name} {avg:.3f}"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
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



def validate(net, image_size, data_loader, device="cuda:0"):
    net = net.to(device)

    data_loader.dataset.transform = test_transform(image_size)

    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    net = net.to(device)
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Top1')
    top5 = AverageMeter('Top5')
    power = AverageMeter('Power')

    with torch.no_grad():
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

    print(
        "Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f"
        % (losses.avg, top1.avg, top5.avg)
    )
    return top1.avg, power.avg

#FOR LOOP
# GENERATE ARCH/LATENCY TUPLE
for i in range(100):
    subnet_config = ofa_network.sample_active_subnet()
    image_size = random.choice(image_size_list)
    subnet_config.update({'image_size': image_size})
    predicted_acc = acc_predictor.predict_acc([subnet_config])
    predicted_efficiency = efficiency_predictor.get_efficiency(subnet_config)

    print(i, '\t', predicted_acc, '\t', '%.1fM MACs' % predicted_efficiency)

    subnet = ofa_network.get_active_subnet().to(device)
    data_loader.dataset.transform = train_transform(image_size)
    set_running_statistics(subnet, data_loader)

    top1_ofa, _ = validate(subnet, image_size, data_loader_valid, device)

    with csv_path.open('a') as f:
        writer = csv.writer(f)
        writer.writerow([predicted_acc.cpu().item() * 100, top1_ofa, predicted_efficiency])
