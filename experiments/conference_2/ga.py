from argparse import ArgumentParser
import os
import copy
import torch
import numpy as np
from MemSE import ROOT
from MemSE.training import RunConfig, RunManager
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, EvolutionFinder, AccuracyPredictorFactory
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE, FORWARD_MODE

# from torch.profiler import profile, record_function, ProfilerActivity
# from cProfile import Profile
# import pstats

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
parser.add_argument('--const_gmax', action='store_true')
parser.add_argument('--predictor', default='AccuracyPredictor')
args, _ = parser.parse_known_args()

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
predictor = AccuracyPredictorFactory[args.predictor](encoder, device=device)
loaded = torch.load(save_path / f'trained_{args.predictor}.pth')
predictor.load_state_dict(loaded['net_dict'])
predictor.eval()

print('Loaded')

constraints = np.linspace(1e-2,1,10)
results = {}

# profiler = Profile()

ga = EvolutionFinder(predictor, memse, const_gmax=args.const_gmax)
for const in constraints:
    print(const)
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    # profiler.enable()
    bv, bi = ga.run_evolution_search(const, True)
    # profiler.disable()
    results[const] = (bv, bi)
    print(results[const])
    # break
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
# stats.strip_dirs()
# stats.print_stats()
# stats.dump_stats(f'{save_path}/cProfiler.prof')

torch.save(results, f'{save_path}/ga_results{"_const" if args.const_gmax else ""}_{args.predictor}.pth')

comparison = {}
run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
ofaxmemse = OFAxMemSE(ofa).to(device)

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

torch.save(comparison, f'{save_path}/ga_evals{"_const" if args.const_gmax else ""}_{args.predictor}.pth')