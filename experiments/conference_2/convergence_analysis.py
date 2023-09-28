import torch
import os
import numpy as np
import random
from argparse import ArgumentParser

from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

power, acc = [], []

# FOR LOOP
# GENERATE ARCH/LATENCY TUPLE
for i in range(100):
    subnet_config = ofa_network.sample_active_subnet()
    image_size = random.choice(image_size_list)
    subnet_config.update({'image_size': image_size})

    subnet = ofa_network.get_active_subnet()
    run_manager._loader.assign_active_img_size(image_size)
    set_running_statistics(subnet, run_manager._loader.build_sub_train_loader())

    memse = MemSE(subnet)
    memse.quanter.init_gmax_as_wmax()
    memse.quant(scaled=False)
    _, metrics = run_manager.validate(net=memse, hist_meters=True)
    memse.unquant()

    power.append(metrics.power.hist)
    acc.append(metrics.top1.hist)

power = np.array(power)
acc = np.array(acc)

np.save('power_convergence.npy', power)
np.save('acc_convergence.npy', acc)
