from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import copy
import time
import torch
import time
import numpy as np
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant, ROOT, METHODS
from MemSE.utils import n_vars_computation, numpify, default
from MemSE.dataset import batch_size_opt, get_dataloader, get_output_loader
from MemSE.model_loader import load_model
from MemSE.train_test_loop import test_acc_sim, test_mse_th

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem

from torch.profiler import profile, record_function, ProfilerActivity

import logging
logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_sync_debug_mode(1)
OPTI_BS = 10

#####
# ARGUMENTS HANDLING
#####
def parse_args():
	parser = ArgumentParser(description='Gmax optimizer')
	parser.add_argument('--device', '-D', default='cuda', type=str)
	parser.add_argument('--power-budget', '-P', default=1e6, type=int, dest='power_budget')
	parser.add_argument('--memscale', action='store_true')
	parser.add_argument('--network', default='smallest_vgg_ReLU', type=str)
	parser.add_argument('--datapath', default=f'{ROOT}/data', type=str)
	parser.add_argument('--method', default='unfolded', type=str)
	parser.add_argument('-R', default=1, type=int)
	parser.add_argument('--ga-popsize', default=2, type=int, dest='ga_popsize')
	parser.add_argument('--N-mc', default=20, type=int, dest='N_mc')
	parser.add_argument('--sigma', '-S', default=0.01, type=float)
	parser.add_argument('-N', default=1280000000, type=int)

	parser.add_argument('--batch-stop-accuracy', default=2, type=int, help='Set it to -1 to run all avail. batches')
	parser.add_argument('--batch-stop-power', default=2, type=int, help='Set it to -1 to run all avail. batches')
	parser.add_argument('--batch-stop-opt', default=-1, type=int, help='Set it to -1 to run all avail. batches')

	parser.add_argument('--gen-all', default=1, type=int, dest='gen_all', help='Nb generations for GA in ALL mode')
	parser.add_argument('--gen-layer', default=5, type=int, dest='gen_layer', help='Nb generations for GA in LAYERWISE mode')
	parser.add_argument('--gen-col', default=10, type=int, dest='gen_col', help='Nb generations for GA in COLUMNWISE mode')
	return parser.parse_args()

args = parse_args()
print(args)
device = args.device if torch.cuda.is_available() else 'cpu'

test_acc_sim = partial(test_acc_sim, device=device)
test_mse_th = partial(test_mse_th, device=device, memory_flush=False)

#####
# BOOKEEPING SETUP
#####
folder = Path(__file__).parent
result_folder = folder / 'results'
result_folder.mkdir(exist_ok=True)
fname = f'opti_{args.method}_{args.network}_{args.power_budget}.z'
result_filename = result_folder / fname
profile_filename = result_folder / 'trace.json'


#####
# MODEL LOAD
#####
bs = 128
train_loader, train_clean_loader, test_loader, nclasses, input_shape = get_dataloader('cifar10', root=args.datapath, bs=bs, memscale=args.memscale, train_set_clean_sample_per_classes=1)
model = load_model(args.network, nclasses)
model = METHODS[args.method](model, input_shape)

nvar_col, nvar_layer = n_vars_computation(model)
print(f'In column mode, the model have {nvar_col} variables, and {nvar_layer} in layer mode')

quanter = MemristorQuant(model, std_noise=args.sigma, N=args.N)
memse = MemSE(model, quanter).to(device)

opti_bs = 5 #batch_size_opt(train_clean_loader, memse, OPTI_BS, 10, device=device)
print(f'The maximum batch size was found to be {opti_bs}')

output_train_loader = get_output_loader(train_clean_loader, model, device=device, overwrite_bs=opti_bs)
small_test_loader = get_output_loader(test_loader, model, device=device, overwrite_bs=opti_bs)

MODES_INIT = {
	'ALL': 1,
	'LAYERWISE': nvar_layer,
	'COLUMNWISE': nvar_col
}
MODES_GEN = {
	'ALL': args.gen_all,
	'LAYERWISE': args.gen_layer,
	'COLUMNWISE': args.gen_col,
}

#####
# OPTIMIZER UTILITIES
####
class problemClass(Problem):
	def __init__(self,
                 n_var,
                 n_obj,
                 n_constr,
                 xl,
                 xu,
                 memse,
                 power_budget,
                 dataloader,
                 batch_stop: int = -1):
		super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
		self.memse = memse
		self.power_budget = power_budget
		self.batch_stop = batch_stop
		self.dataloader = dataloader

	def _evaluate(self, Gmax, out, *args, **kwargs):
		a = []
		for G  in Gmax:
			self.memse.quanter.Gmax = G
			mse, P = test_mse_th(self.dataloader, self.memse, batch_stop=self.batch_stop)
			a.append([P, mse])

		a = np.array(a)
		out["F"] =  np.stack(a[:,1]) 
		out["G"] = np.stack(a[:,0]-self.power_budget)


def genetic_alg(memse:MemSE,
                n_vars:int,
                dataloader,
				nb_gen:int=1,
                pop_size:int=100,
                power_budget:int=1000000,
                start_Gmax = None,
                device = torch.device('cpu'),
                batch_stop: int = -1):

	problem = problemClass(n_vars,
						   1,
                           1,
                           np.ones(n_vars)*0.01,
                           np.ones(n_vars)*500,
                           memse,
                           power_budget,
                           dataloader,
                           batch_stop=batch_stop)

	if np.any(start_Gmax) == None: 
		sample_process = np.ones((pop_size,n_vars)) + np.random.normal(0,0.5,(pop_size,n_vars))
	else:
		sample_process = np.ones((pop_size,n_vars)) * start_Gmax + np.random.normal(0,0.001,(pop_size,n_vars))
	
	algorithm = GA(pop_size=pop_size,
					eliminate_duplicates=True,
					sampling=np.clip(sample_process,0.01,500))


	# perform a copy of the algorithm to ensure reproducibility
	obj = copy.deepcopy(algorithm)

	# let the algorithm know what problem we are intending to solve and provide other attributes
	obj.setup(problem, termination=("n_gen", nb_gen),)

	# until the termination criterion has not been met
	while obj.has_next():
	
		# perform an iteration of the algorithm
		t1 = time.time()
		obj.next()
		t2 = time.time()
		print(t2-t1)

		# access the algorithm to print some intermediate outputs
		print(f"gen: {obj.n_gen} max gen: {obj.termination.n_max_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV')[0,0]} ideal: {obj.opt.get('F')[0,0]} power: {obj.opt.get('G')[0,0]+power_budget}")
		print(obj.opt.get('X'),np.min(obj.opt.get('X')),np.max(obj.opt.get('X')))
		# print(im_problem_function_batched(obj.opt.get('X')[0], args, x, z, device_id=0))

		# if obj.n_gen%50 == 0:
			# np.save("checkpoint_opti_algo_"+str(power_budget)+type_opt+'.npy', obj)
			# np.save('checkpoint_gmax_opti_'+type_opt+'_'+str(power_budget)+'.npy', obj.opt.get('X'))

	
	# finally obtain the result object
	res = obj.result()
	if np.all(res.X!=None):
		P_tot, mse =  res.G + power_budget, res.F#im_problem_function_batched(res.X, args, x, device_id=device_id)
		minmax_power_tab = P_tot
		minmax_mse_tab = mse
	else:
		minmax_power_tab = np.nan #res.G+power_budget
		minmax_mse_tab = np.nan
	return minmax_power_tab, minmax_mse_tab, res.X


#####
# RUN OPT.
#####
RES_WMAX = {}
RES_P = {}
RES_GMAX = {}
RES_MSE = {}

for mode in ['ALL']:
	memse.quanter.init_wmax(mode)
	memse.quanter.init_gmax(1.)  # automatically extend to mode-size

	RES_WMAX[mode] = memse.quanter.Wmax

	if mode == 'LAYERWISE':
		start_Gmax = default(RES_GMAX['ALL'], 1.) * numpify(RES_WMAX['LAYERWISE']) / RES_WMAX['ALL'].item()
	elif mode == 'COLUMNWISE':
		start_Gmax = np.concatenate([RES_GMAX['LAYERWISE'][i].item() * numpify(RES_WMAX['COLUMNWISE'][i]) / RES_WMAX['LAYERWISE'][i].item() for i in range(len(RES_WMAX['COLUMNWISE']))])
	else:
		start_Gmax = None

	print(mode)
	print(f'{start_Gmax=}')

	starting_time = time.time()
	
	P_all, mse_all, Gmax_tab_all = genetic_alg(memse,
											n_vars=MODES_INIT[mode],
											dataloader=output_train_loader,
											nb_gen=MODES_GEN[mode],
											pop_size=args.ga_popsize,
											power_budget=args.power_budget,
											device=device,
											start_Gmax=start_Gmax,
											batch_stop=args.batch_stop_opt)
	print(f'genetic alg ran in {(time.time() - starting_time)/60} minutes on {args.network} with {MODES_GEN[mode]} generations of {args.ga_popsize} individuals ({opti_bs=})')
  
	RES_GMAX[mode] = Gmax_tab_all
	RES_MSE[mode] = mse_all
	RES_P[mode] = P_all

inputs, _ = next(iter(output_train_loader))
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True) as prof:
	with torch.inference_mode():
		_ = memse.forward(inputs.to(device))
prof.export_chrome_trace(str(profile_filename.resolve().absolute()))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
print(RES_P)
print(RES_MSE)

