from argparse import ArgumentParser
from pathlib import Path
import time
import torch
import time
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant, ROOT, METHODS, compute_n_mc, compute_sample_moments
from MemSE.misc import TimingMeter, HistMeter
from MemSE.utils import seed_all
from MemSE.dataset import get_dataloader, get_output_loader
from MemSE.model_loader import load_model
import numpy as np


import logging
logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

seed_all(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
OPTI_BS = 10

#####
# ARGUMENTS HANDLING
#####
def parse_args():
	parser = ArgumentParser(description='Time comparison monte-carlo and MemSE')
	parser.add_argument('--device', '-D', default='cuda', type=str)
	parser.add_argument('--memscale', action='store_true')
	parser.add_argument('--network', default='smallest_vgg_ReLU', type=str)
	parser.add_argument('--datapath', default=f'{ROOT}/data', type=str)
	parser.add_argument('--method', default='unfolded', type=str)
	parser.add_argument('-R', default=1, type=int)
	parser.add_argument('--CI', default=5, type=int)
	parser.add_argument('--z', default=95, type=int)
	parser.add_argument('--start_sigma', default=0.001, type=float)
	parser.add_argument('--stop_sigma', default=0.03162278, type=float)
	parser.add_argument('--num_sigma', default=10, type=int)
	parser.add_argument('-N', default=1280000000, type=int)
	return parser.parse_args()

args = parse_args()
print(args)
device = args.device if torch.cuda.is_available() else 'cpu'
starting_time = time.time()


#####
# BOOKEEPING SETUP
#####
folder = Path(__file__).parent
result_folder = folder / 'results'
result_folder.mkdir(exist_ok=True)
fname = f'timing_{args.method}_{args.network}_{args.CI}_{args.z}.pt'
result_filename = result_folder / fname

#####
# MODEL LOAD
#####
train_loader, train_clean_loader, test_loader, nclasses, input_shape = get_dataloader('cifar10', root=args.datapath, memscale=args.memscale, train_set_clean_sample_per_classes=1)
model = load_model(args.network, nclasses)
model = METHODS[args.method](model, input_shape).to(device)

opti_bs = 1
output_train_loader = get_output_loader(train_clean_loader, model, device=device, overwrite_bs=opti_bs)

inp, tar = next(iter(output_train_loader))
inp, tar = inp.to(device), tar.to(device)

sigma_tab = np.logspace(np.log10(args.start_sigma),np.log10(args.stop_sigma),args.num_sigma).tolist()
print('Tableau sigma:')
print(sigma_tab)
res_dict_tab = []

for sig in sigma_tab:

	quanter = MemristorQuant(model, std_noise=sig, N=args.N)
	quanter.init_gmax_as_wmax()
	memse = MemSE(model, quanter).to(device)

	sample_mean, sample_var = compute_sample_moments(memse, inp, tar, 10000)
	N_mc = compute_n_mc(args.CI*sample_mean/100, args.z, sample_var)
	print(f'For {sig=}, {N_mc=}')

	memse.quant()

	# WARMUP
	for _ in range(10):
		_ = memse.forward(inp, manage_quanter=False, compute_power=False)

	timing_memse = TimingMeter('Timing MemSE')
	for _ in range(10):
		with timing_memse:
			m, g, _ = memse.forward(inp, manage_quanter=False, compute_power=False)
	mse_memse = mse_gamma(tar, m, g).mean().item()


	timing_mc = TimingMeter('Timing MC')
	avg_mc = HistMeter('MSE MC', histed='avg')
	memse.quant(c_one=False)
	for _ in range(N_mc):
		with timing_mc:
			outputs = memse.forward_noisy(inp)
			mse = torch.mean((outputs.detach() - tar) ** 2)
			avg_mc.update(mse.item())
		
	res_dict = {'sigma': sig, 'memse_time': timing_memse.avg, 'memse_val': mse_memse, 'mc_time': timing_mc.hist[-1], 'mc_val': np.mean(avg_mc.hist)}
	print(res_dict)
	res_dict_tab.append(res_dict)

torch.save(res_dict_tab, result_filename)
print(f'time_nmc.py ran in {(time.time() - starting_time)/60} minutes')
