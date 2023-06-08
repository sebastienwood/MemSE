from ofa.model_zoo import ofa_net
from MemSE import ROOT
from MemSE.nn import OFAxMemSE, FORWARD_MODE
from MemSE.training import RunManager, RunConfig
from MemSE.nas import MemSEDataset

ofa = ofa_net('ofa_resnet50', False)
# print(ofa)
ofaxmemse = OFAxMemSE(ofa)
# ofaxmemse.sample_active_subnet()

# TODO runmanager and test memsedataset.build_acc_dataset
run_config = RunConfig(dataset_root=ROOT)
run_manager = RunManager(run_config, mode=FORWARD_MODE.MONTECARLO)
MemSEDataset.AccuracyDataset(ROOT / 'experiments/conference_2').build_acc_dataset(run_manager, ofaxmemse)