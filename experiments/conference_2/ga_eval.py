import os
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from MemSE import ROOT
from MemSE.training import RunConfig, RunManager
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, AccuracyPredictor, EvolutionFinder
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE, FORWARD_MODE

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)

data_loader = run_manager._loader.build_sub_train_loader(n_images=2000, batch_size=256)

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax

save_path = ROOT / 'experiments/conference_2/results'
dset = AccuracyDataset(save_path)
encoder = ResNetArchEncoder(default_gmax=default_gmax)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ofaxmemse = OFAxMemSE(ofa).to(device)
 
results = torch.load(f'{save_path}/ga_results.pth')
comparison = {}

for const, res in results.items():
    print(const)
    bv, bi = res
    arch = bi[1]
    acc, eff = bi[0], bi[2]
    ofaxmemse.set_active_subnet(arch, data_loader)
    ofaxmemse.quant(scaled=False)

    loss, metrics = run_manager.validate(
        net=ofaxmemse,
        no_logs=True
    )
    # TODO compare metrics with bi
    ofaxmemse.unquant()
    comparison[const] = {'acc': (acc, metrics.top1.avg), 'pow': (eff, metrics.power.avg)}
    
torch.save(comparison, f'{save_path}/ga_evals.pth')
print(comparison)