import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
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
    EvaluatorBatch,
    DataloadersHolder
)
from ofa.model_zoo import ofa_net
from MemSE.nn import MemSE, OFAxMemSE, FORWARD_MODE

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

save_path = ROOT / "experiments/conference_2/results"

parser = ArgumentParser()
parser.add_argument("--datapath", default=os.environ["DATASET_STORE"])
parser.add_argument("--const_gmax", action="store_true")
parser.add_argument("--simulate", action="store_true")
parser.add_argument("--predictor", default="AccuracyPredictor")
args, _ = parser.parse_known_args()

print("Loading")

# DNN model and run manager
ofa = ofa_net("ofa_resnet50", pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax
encoder = ResNetArchEncoder(default_gmax=default_gmax)
memse = OFAxMemSE(ofa).to(device)

run_config = RunConfig(dataset_root=args.datapath, dataset="ImageNetHF")
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
dataholder = DataloadersHolder(run_manager)

print("Network ready")

# Predictor
dset = AccuracyDataset(save_path)
tloader, vloader, bacc, bpow = dset.build_acc_data_loader(encoder)
predictor = AccuracyPredictorFactory[args.predictor](encoder, device=device)
loaded = torch.load(save_path / f"trained_{args.predictor}.pth")
predictor.load_state_dict(loaded["net_dict"])
predictor.eval()

print("Loaded")
problem = MemSEProblem(
    memse,
    encoder,
    BatchPicker(),
    run_manager=run_manager,
    dataholder=dataholder,
    surrogate=predictor,
    const=args.const_gmax,
    simulate=args.simulate
)

algorithm = NSGA2AdvanceCriterion(
    pop_size=100,
    sampling=MixedVariableSampling(),
    mating=CausalMixedVariableMating(
        eliminate_duplicates=MixedVariableDuplicateElimination(), repair=RepairGmax()
    ),
    eliminate_duplicates=MixedVariableDuplicateElimination(),
    repair=RepairGmax(),
    survival=RankAndCrowdingSurvivalWithRejected(),
    evaluator=EvaluatorBatch()
)

results = {}
algorithm.setup(problem, termination=("n_gen", 500), seed=1, verbose=True)
while algorithm.has_next():
    algorithm.next()
    res = algorithm.result()
    exec_time = res.exec_time
    n_gen = algorithm.n_gen
    print(f'Done gen {n_gen} in {exec_time}')
    #print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))

    pop_dict = {}
    for i, (arch, perf) in enumerate(zip(res.X, res.F)):
        pop_dict[i] = {'arch': encoder.cat_arch_vars(arch, memse.gmax_masks), 'perf': perf}
    
    results[n_gen] = {
        'exec_time': exec_time,
        'n_batch': problem.n,
        'pop_size': algorithm.pop_size,
        'pop': pop_dict
    }
    torch.save(
        results,
        f'{save_path}/ga_results_pymoo{"_const" if args.const_gmax else ""}_{"simdirect" if args.simulate else args.predictor}.pth',
    )

nb_mults = 0
with tqdm(
            total=len(results),
            desc="Running evals",
        ) as t:
    for res in results.values():
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
            if nb_mults > 0:
                for mult in np.linspace(0.8, 1.0, nb_mults, False):
                    arch = copy.deepcopy(arch)
                    arch["gmax"] = (torch.tensor(arch["gmax"]) * mult).tolist()
                    r = gather(arch)
                    indiv.update({mult: r})
        t.update(1)

torch.save(
    results,
    f'{save_path}/ga_evals_pymoo{"_const" if args.const_gmax else ""}_{"simdirect" if args.simulate else args.predictor}.pth',
)
