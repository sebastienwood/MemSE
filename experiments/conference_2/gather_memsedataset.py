import os
from argparse import ArgumentParser
from ofa.model_zoo import ofa_net
from MemSE import ROOT
from MemSE.nn import OFAxMemSE, FORWARD_MODE
from MemSE.training import RunManager, RunConfig
from MemSE.nas import MemSEDataset

parser = ArgumentParser()
parser.add_argument('--datapath', default=os.environ['DATASET_STORE'])
parser.add_argument('--image_size', default=None, type=int)
parser.add_argument('--range_LHS', default=1, type=int)
parser.add_argument('--nb_batchs', default=50, type=int)
parser.add_argument('--nb_batchs_power', default=1, type=int)
args = parser.parse_args()

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofaxmemse = OFAxMemSE(ofa)

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
MemSEDataset.AccuracyDataset(ROOT / 'experiments/conference_2/results').build_acc_dataset(run_manager, ofaxmemse, image_size_list=args.image_size, range_LHS=args.range_LHS, nb_batchs=args.nb_batchs, nb_batchs_power=args.nb_batchs_power)
