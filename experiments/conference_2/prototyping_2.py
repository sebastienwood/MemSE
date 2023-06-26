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
args = parser.parse_args()

ofa = ofa_net('ofa_resnet50', pretrained=True)
ofaxmemse = OFAxMemSE(ofa)

run_config = RunConfig(dataset_root=args.datapath, dataset='ImageNetHF')
run_manager = RunManager(run_config, mode=FORWARD_MODE.BASE)
MemSEDataset.AccuracyDataset(ROOT / 'experiments/conference_2/results').build_acc_dataset(run_manager, ofa, image_size_list=args.image_size)
