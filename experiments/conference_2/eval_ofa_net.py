import os
import random
import torch
import argparse

from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.utils import download_url


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of imagenet", type=str, default=os.environ['DATASET_STORE']
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
subnet_config = ofa_network.sample_active_subnet()
subnet = ofa_network.get_active_subnet(preserve_weight=True)

""" Test sampled subnet 
"""
run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
# assign image size: 128, 132, ..., 224
run_config.data_provider.assign_active_img_size(224)
run_manager.reset_running_statistics(net=subnet)

print("Test random subnet:")
print(subnet.module_str)

loss, (top1, top5) = run_manager.validate(net=subnet)
print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))

predicted_acc = acc_predictor.predict_acc([subnet_config])
print("Predicted acc %.1f" % predicted_acc)
