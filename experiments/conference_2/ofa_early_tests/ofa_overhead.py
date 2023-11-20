import os
import time
import torch
from argparse import ArgumentParser
from MemSE.nas import DataloadersHolder, ResNetArchEncoder
from MemSE.training import RunManager, RunConfig
from ofa.model_zoo import ofa_net
from MemSE.nn import OFAxMemSE, FORWARD_MODE, MemSE
from MemSE import ROOT

parser = ArgumentParser()
parser.add_argument("--datapath", default=os.environ.get("DATASET_STORE", None))
args, _ = parser.parse_known_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofa.set_max_net()
default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax
ofaxmemse = OFAxMemSE(ofa)
encoder = ResNetArchEncoder(default_gmax)

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
datahld = DataloadersHolder(run_manager)

train_ld, eval_ld = datahld.get_image_size(128)

for i in range(10):
    print('Warming up ', i)
    a = encoder.random_sample_arch(ofaxmemse)
    a['image_size'] = 128
    ofaxmemse.set_active_subnet(a, train_ld)
torch.cuda.synchronize()

@torch.no_grad()
def eval(nb_batch, power=1):
    a = encoder.random_sample_arch(ofaxmemse)
    a['image_size'] = 128
    ofaxmemse.set_active_subnet(a, train_ld)
    ofaxmemse.quant(scaled=False)
    metrics = None
    if nb_batch > 0:
        _, metrics = run_manager.validate(
                            net=ofaxmemse,
                            data_loader=eval_ld,
                            no_logs=True,
                            nb_batchs=nb_batch,
                            nb_batchs_power=power
                        )
        metrics.display_summary()
    ofaxmemse.unquant()
    return metrics
    
res = {0: {}, 1: {}}
for n in range(150):
    for p in [0, 1]:
        print(n, p)
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start_t = time.time()
            eval(n, power=p)
            torch.cuda.synchronize()
            elapsed = time.time() - start_t
            times.append(elapsed)
        res[p][n] = sum(times) / 5
        print(res[p][n])
torch.save(res, ROOT/ "experiments/conference_2/results/overhead.pth")