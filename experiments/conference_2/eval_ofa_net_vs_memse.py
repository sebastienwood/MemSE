import os
import csv
import random
import torch
import argparse

from pathlib import Path

from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.utils import download_url


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of imagenet", type=str, default=f'{os.environ["DATASET_STORE"]}/imagenet'
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=100,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
parser.add_argument(
    "-n",
    "--net",
    metavar="OFANET",
    default="ofa_resnet50",
    choices=[
        "ofa_mbv3_d234_e346_k357_w1.0",
        "ofa_mbv3_d234_e346_k357_w1.2",
        "ofa_proxyless_d234_e346_k357_w1.3",
        "ofa_resnet50",
    ],
    help="OFA networks",
)

args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

ofa_network = ofa_net(args.net, pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
arch_encoder = ResNetArchEncoder(
	image_size_list=image_size_list, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
    width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST
)

acc_predictor_checkpoint_path = download_url(
    'https://hanlab.mit.edu/files/OnceForAll/tutorial/ofa_resnet50_acc_predictor.pth',
    model_dir=".torch/predictor",
)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
                                  checkpoint_path=acc_predictor_checkpoint_path, device=device)

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
csv_path = Path('comparison_accuracy.csv')
if not csv_path.exists():
    with csv_path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['acc_pred', 'acc_ofa'])
        
###
# MEMSE
###
from MemSE.training import RunManager as RunManagerM, RunConfig
from MemSE.nn import FORWARD_MODE, MemSE
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
run_config = RunConfig(dataset_root=args.path, dataset='ImageNet')
run_manager_m = RunManagerM(run_config, mode=FORWARD_MODE.BASE)

for _ in range(100):
    subnet_config = ofa_network.sample_active_subnet()
    img_size = random.choice(image_size_list)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)

    """ Test sampled subnet 
    """
    run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
    # assign image size: 128, 132, ..., 224
    run_config.data_provider.assign_active_img_size(img_size)
    run_manager.reset_running_statistics(net=subnet)

    #print("Test random subnet:")
    #print(subnet.module_str)

    loss, (top1, top5) = run_manager.validate(net=subnet, no_logs=True)
    print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))

    predicted_acc = acc_predictor.predict_acc([subnet_config | {'image_size': img_size}])
    print("Predicted acc %.1f" % (predicted_acc * 100))
    with csv_path.open('a') as f:
        writer = csv.writer(f)
        writer.writerow([predicted_acc.cpu().item() * 100, top1])
   
    subnet = ofa_network.get_active_subnet()
    run_manager_m._loader.assign_active_img_size(img_size)
    set_running_statistics(subnet, run_manager_m._loader.build_sub_train_loader())
    
    memse = MemSE(subnet)
    memse.quanter.init_gmax_as_wmax()
    # memse.quant(scaled=False)
    loss_m, metrics = run_manager_m.validate(net=memse)
    # memse.unquant()
    print("Results MemSE: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss_m, metrics.top1.avg, metrics.top5.avg))
