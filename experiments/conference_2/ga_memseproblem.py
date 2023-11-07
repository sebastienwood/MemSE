import os
import copy
import torch
import numpy as np
from argparse import ArgumentParser
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableSampling
from MemSE import ROOT
from MemSE.training import RunConfig, RunManager
from MemSE.nas import (
    AccuracyDataset,
    BatchPicker,
    ResNetArchEncoder,
    AccuracyPredictorFactory,
    MemSEProblem,
    NSGA2AdvanceCriterion,
    RepairGmax,
    CausalMixedVariableMating,
    RankAndCrowdingSurvivalWithRejected,
)
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE, FORWARD_MODE

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument("--datapath", default=os.environ["DATASET_STORE"])
parser.add_argument("--const_gmax", action="store_true")
parser.add_argument("--predictor", default="AccuracyPredictor")
args, _ = parser.parse_known_args()

print("Loading")

ofa = ofa_net("ofa_resnet50", pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax
memse = OFAxMemSE(ofa)

print("Network ready")

save_path = ROOT / "experiments/conference_2/results"
dset = AccuracyDataset(save_path)
encoder = ResNetArchEncoder(default_gmax=default_gmax)
tloader, vloader, bacc, bpow = dset.build_acc_data_loader(encoder)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
predictor = AccuracyPredictorFactory[args.predictor](encoder, device=device)
loaded = torch.load(save_path / f"trained_{args.predictor}.pth")
predictor.load_state_dict(loaded["net_dict"])
predictor.eval()

print("Loaded")
problem = MemSEProblem(
    memse, encoder, BatchPicker(), surrogate=predictor, const=args.const_gmax
)

algorithm = NSGA2AdvanceCriterion(
    pop_size=1000,
    sampling=MixedVariableSampling(),
    mating=CausalMixedVariableMating(
        eliminate_duplicates=MixedVariableDuplicateElimination(), repair=RepairGmax()
    ),
    eliminate_duplicates=MixedVariableDuplicateElimination(),
    repair=RepairGmax(),
    survival=RankAndCrowdingSurvivalWithRejected(),
)
res = minimize(problem, algorithm, ("n_gen", 500), seed=1, verbose=True)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
results = {i: (None, (None, encoder.cat_arch_vars(d, memse.gmax_masks), None)) for i, d in enumerate(res.X)}

torch.save(
    results,
    f'{save_path}/ga_results_pymoo{"_const" if args.const_gmax else ""}_{args.predictor}.pth',
)

comparison = {}
run_config = RunConfig(dataset_root=args.datapath, dataset="ImageNetHF")
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
ofaxmemse = OFAxMemSE(ofa).to(device)

for const, res in results.items():
    print(const)
    bv, bi = res
    arch = bi[1]
    print(arch)
    print(len(arch["gmax"]))
    run_manager._loader.assign_active_img_size(int(arch["image_size"]))
    data_loader = run_manager._loader.build_sub_train_loader(
        n_images=2000, batch_size=256
    )

    def gather(arch):
        out = predictor.predict_acc([arch]).cpu()
        eff, acc = out[1].item(), out[0].item()

        ofaxmemse.set_active_subnet(arch, data_loader)
        ofaxmemse.quant(scaled=False)

        loss, metrics = run_manager.validate(net=ofaxmemse)
        ofaxmemse.unquant()
        return {"acc": (acc, metrics.top1.avg), "pow": (eff, metrics.power.avg)}

    comparison[const] = gather(arch)
    for mult in np.linspace(0.8, 1.0, 5, False):
        arch = copy.deepcopy(arch)
        arch["gmax"] = (torch.tensor(arch["gmax"]) * mult).tolist()
        r = gather(arch)
        comparison[const].update({mult: r})

torch.save(
    comparison,
    f'{save_path}/ga_evals_pymoo{"_const" if args.const_gmax else ""}_{args.predictor}.pth',
)
