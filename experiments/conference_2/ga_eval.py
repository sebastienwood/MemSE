import os
import torch
import numpy as np
import copy
from argparse import ArgumentParser
from MemSE import ROOT
from MemSE.training import RunConfig, RunManager
from MemSE.nas import AccuracyDataset, ResNetArchEncoder, AccuracyPredictorFactory, DataloadersHolder
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE, FORWARD_MODE

from tqdm import tqdm

torch.backends.cudnn.benchmark = True
save_path = ROOT / 'experiments/conference_2/results'

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
parser.add_argument('--result_name', type=str)
parser.add_argument('--predictor', default='AccuracyPredictor')
parser.add_argument('--multed', default=0, type=int)
parser.add_argument('--eval_all', action="store_true")
args = parser.parse_args()
assert 'results' in args.result_name

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
dataholder = DataloadersHolder(run_manager)

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax

dset = AccuracyDataset(save_path)
encoder = ResNetArchEncoder(default_gmax=default_gmax)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
memse = OFAxMemSE(ofa).to(device)

predictor = AccuracyPredictorFactory[args.predictor](encoder, device=device)
loaded = torch.load(save_path / f'trained_{args.predictor}.pth')
predictor.load_state_dict(loaded['net_dict'])
predictor.eval()

results = torch.load(f'{save_path}/{args.result_name}')

to_evaluate = results.values() if args.eval_all else [results[max(results.keys())]]
total_to_evaluate = sum([len(x["pop"]) for x in to_evaluate])
with tqdm(
            total=total_to_evaluate,
            desc="Running evals",
        ) as t:
    for res in to_evaluate:
        for indiv in res['pop'].values():
            arch = indiv['arch']
            train_ld, eval_ld = dataholder.get_image_size(int(arch["image_size"]))

            def gather(arch):
                out = predictor.predict_acc([arch]).cpu()
                eff, acc = out[1].item(), out[0].item()

                memse.set_active_subnet(arch, train_ld)
                memse.quant(scaled=False)

                _, metrics = run_manager.validate(net=memse, no_logs=True)
                memse.unquant()
                return {"acc": (acc, metrics.top1.avg), "pow": (eff, metrics.power.avg)}

            indiv.update(gather(arch))
            if args.multed > 0:
                for mult in [float(x) for x in np.linspace(0.9, 1.1, args.multed) if x != 1]:
                    arch = copy.deepcopy(arch)
                    arch["gmax"] = (torch.tensor(arch["gmax"]) * mult).tolist()
                    r = gather(arch)
                    indiv.update({mult: r})
            t.update(1)

torch.save(
    results,
    f"{save_path}/{args.result_name.replace('results', 'evals')}",
)
