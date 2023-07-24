import torch
import torch.nn as nn
import os
import numpy as np
import math
import csv
import random
from torchvision import transforms
from pathlib import Path
from argparse import ArgumentParser

from ofa.model_zoo import ofa_net
from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder
from ofa.nas.efficiency_predictor import ResNet50FLOPsModel
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.utils import download_url

from MemSE import ROOT
from MemSE.nn import FORWARD_MODE, MemSE
from MemSE.training import RunManager, RunConfig

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

# LOAD OFA's ESTIMATORS FOR RESNET50
ofa_network = ofa_net('ofa_resnet50', pretrained=True)

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
args = parser.parse_args()

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)

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

csv_path = Path(f'{ROOT}/experiments/conference_2/comparison_power_train.csv')
if not csv_path.exists():
    with csv_path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['acc_pred', 'acc_ofa', 'acc_memse', 'power_pred', 'power_memse'])

efficiency_predictor = ResNet50FLOPsModel(ofa_network)

#FOR LOOP
# GENERATE ARCH/LATENCY TUPLE
for i in range(1000):
    subnet_config = ofa_network.sample_active_subnet()
    image_size = random.choice(image_size_list)
    subnet_config.update({'image_size': image_size})
    predicted_acc = acc_predictor.predict_acc([subnet_config])
    predicted_efficiency = efficiency_predictor.get_efficiency(subnet_config)

    print(i, '\t', predicted_acc, '\t', '%.1fM MACs' % predicted_efficiency)

    subnet = ofa_network.get_active_subnet()
    run_manager._loader.assign_active_img_size(image_size)
    set_running_statistics(subnet, run_manager._loader.build_sub_train_loader())

    _, metrics = run_manager.validate(net=subnet)
    top1_ofa = metrics.top1.avg

    # MEMSE FORWARD FOR POWER
    memse = MemSE(subnet)
    memse.quanter.init_gmax_as_wmax()
    memse.quant(scaled=False)
    _, metrics = run_manager.validate(net=memse)
    top1, power = metrics.top1.avg, metrics.power.avg
    memse.unquant()
    # STORE
    print(i, '\t', top1, '\t', '%.1fM MACs' % power)
    with csv_path.open('a') as f:
        writer = csv.writer(f)
        writer.writerow([predicted_acc.cpu().item() * 100, top1_ofa, top1, predicted_efficiency, power])
