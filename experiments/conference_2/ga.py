print('GENETIC ALGORITHM OPT.')
import torch
import numpy as np
from MemSE import ROOT
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, AccuracyPredictor, EvolutionFinder
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE

from tqdm import tqdm

torch.backends.cudnn.benchmark = True

print('Loading')

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax
memse = OFAxMemSE(ofa)

print('Network ready')

save_path = ROOT / 'experiments/conference_2/results'
dset = AccuracyDataset(save_path)
encoder = ResNetArchEncoder(default_gmax=default_gmax)
tloader, vloader, bacc, bpow = dset.build_acc_data_loader(encoder)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
net = AccuracyPredictor(encoder, device=device)
loaded = torch.load(save_path / 'trained_predictor.pth')
net.load_state_dict(loaded['net_dict'])
net.eval()

print('Loaded')

constraints = np.linspace(1e-3,1,50)
results = {}

ga = EvolutionFinder(net, memse)
for const in constraints:
    print(const)
    bv, bi = ga.run_evolution_search(const, True)
    results[const] = (bv, bi)

torch.save(results, f'{save_path}/ga_results.pth')