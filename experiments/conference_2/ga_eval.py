import os
import torch
import torch.nn as nn
import numpy as np
import copy
from argparse import ArgumentParser
from MemSE import ROOT
from MemSE.training import RunConfig, RunManager
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, AccuracyPredictorFactory, EvolutionFinder
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE, FORWARD_MODE

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
parser.add_argument('--const_gmax', action='store_true')
parser.add_argument('--dataset', action='store_true')
parser.add_argument('--predictor', default='AccuracyPredictor')
args = parser.parse_args()

post = '_dataset' if args.dataset else '_const' if args.const_gmax else ''

torch.backends.cudnn.benchmark = True

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax

save_path = ROOT / 'experiments/conference_2/results'
dset = AccuracyDataset(save_path)
encoder = ResNetArchEncoder(default_gmax=default_gmax)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ofaxmemse = OFAxMemSE(ofa).to(device)

predictor = AccuracyPredictorFactory[args.predictor](encoder, device=device)
loaded = torch.load(save_path / f'trained_{args.predictor}.pth')
predictor.load_state_dict(loaded['net_dict'])
predictor.eval()


results = torch.load(f'{save_path}/ga_results{post}.pth')
comparison = {}

for const, res in results.items():
    print(const)
    bv, bi = res
    arch = bi[1]
    print(arch)
    print(len(arch["gmax"]))
    run_manager._loader.assign_active_img_size(arch["image_size"])
    data_loader = run_manager._loader.build_sub_train_loader(n_images=2000, batch_size=256)

    def gather(arch):
        out = predictor.predict_acc([arch]).cpu()
        eff, acc = out[1].item(), out[0].item()

        ofaxmemse.set_active_subnet(arch, data_loader)
        ofaxmemse.quant(scaled=False)

        loss, metrics = run_manager.validate(
            net=ofaxmemse
        )
        ofaxmemse.unquant()
        return {'acc': (acc, metrics.top1.avg), 'pow': (eff, metrics.power.avg)}
    comparison[const] = gather(arch)
    for mult in np.linspace(0.8, 1., 5, False):
        arch = copy.deepcopy(arch)
        arch["gmax"] = (torch.tensor(arch["gmax"]) * mult).tolist()
        r = gather(arch)
        comparison[const].update({mult: r})

torch.save(comparison, f'{save_path}/ga_evals{post}_{args.predictor}.pth')