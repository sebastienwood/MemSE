from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import copy
import time
import torch
import time
import datetime
import numpy as np
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant, ROOT, METHODS
from MemSE.utils import n_vars_computation, numpify, default, seed_all
from MemSE.dataset import batch_size_opt, get_dataloader, get_output_loader
from MemSE.model_loader import load_model
from MemSE.train_test_loop import test_acc_sim, test_mse_th

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem

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
	parser = ArgumentParser(description='Gmax optimizer')
	parser.add_argument('--device', '-D', default='cuda', type=str)
	parser.add_argument('--network', default='make_JohNet', type=str)
	parser.add_argument('--method', default='unfolded', type=str)
	parser.add_argument('--N-mc', default=1000, type=int, dest='N_mc')

	# DLOADER
	dloader = parser.add_argument_group('Dataloader related args')
	dloader.add_argument('--memscale', action='store_true')
	dloader.add_argument('--datapath', default=f'{ROOT}/data', type=str)
	dloader.add_argument('--dataset', default='cifar10', type=str)
	dloader.add_argument('--per-class-sample', default=1, type=int)

	# QUANTER
	qter = parser.add_argument_group('Quanter related args')
	qter.add_argument('-R', default=1, type=int)
	qter.add_argument('--sigma', '-S', default=0.01, type=float)
	qter.add_argument('-N', default=1280000000, type=int)

	# BATCH STOP
	bs = parser.add_argument_group('Batch stop related args')
	bs.add_argument('--batch-stop-accuracy', default=-1, type=int, help='Set it to -1 to run all avail. batches')
	bs.add_argument('--batch-stop-power', default=500, type=int, help='Set it to -1 to run all avail. batches')
	bs.add_argument('--batch-stop-opt', default=-1, type=int, help='Set it to -1 to run all avail. batches')

	# GA
	ga = parser.add_argument_group('Genetic algorithm related args')
	ga.add_argument('--ga-popsize', default=100, type=int, dest='ga_popsize')
	ga.add_argument('--gen-all', default=20, type=int, dest='gen_all', help='Nb generations for GA in ALL mode')
	ga.add_argument('--gen-layer', default=100, type=int, dest='gen_layer', help='Nb generations for GA in LAYERWISE mode')
	ga.add_argument('--gen-col', default=250, type=int, dest='gen_col', help='Nb generations for GA in COLUMNWISE mode')
 
	ga.add_argument('--opt-mode', default='opt_mse', choices=['opt_mse', 'opt_power'], type=str.lower, dest='opt_mode')
	ga.add_argument('--power-budget', '-P', default=1e6, type=int, dest='power_budget')
	ga.add_argument('--mse-budget', '-MSE', default=1e6, type=float, dest='mse_budget')
	return parser.parse_args()

args = parse_args()
print(args)
device = args.device if torch.cuda.is_available() else 'cpu'
test_acc_sim = partial(test_acc_sim, device=device)
test_mse_th = partial(test_mse_th, device=device, memory_flush=False)
starting_time = time.time()

#####
# BOOKEEPING SETUP
#####
now = datetime.datetime.now()
folder = Path(__file__).parent
result_folder = folder / 'results' / now.strftime("%y_%m_%d")
result_folder.mkdir(exist_ok=True)
pb = args.power_budget if args.opt_mode == 'opt_mse' else args.mse_budget
fname = f'opti_{args.method}_{args.network}_{args.opt_mode}_{pb}.z'
result_filename = result_folder / fname


#####
# MODEL LOAD
#####
bs = 128
train_loader, train_clean_loader, test_loader, nclasses, input_shape = get_dataloader(args.dataset, root=args.datapath, bs=bs, memscale=args.memscale, train_set_clean_sample_per_classes=args.per_class_sample)
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
class MemSEProblem(Problem):
	def __init__(self,
                 n_var,
                 n_obj,
                 n_constr,
                 xl,
                 xu,
                 memse,
                 mode:str,
                 power_budget:int,
                 mse_budget:float,
                 dataloader,
                 batch_stop: int = -1):
		super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
		self.memse = memse
		self.power_budget = power_budget
		self.mse_budget = mse_budget
		self.batch_stop = batch_stop
		self.dataloader = dataloader
		self.mode = mode

	def _evaluate(self, Gmax, out, *args, **kwargs):
		a = []
		for G  in Gmax:
			self.memse.quanter.Gmax = G
			mse, P = test_mse_th(self.dataloader, self.memse, batch_stop=self.batch_stop)
			a.append([P, mse])

		a = np.array(a)
		if self.mode == 'opt_mse':
			optim = np.stack(a[:,1])
			cs = np.stack(a[:,0] - self.power_budget) 
		elif self.mode == 'opt_power':
			optim = np.stack(a[:,0])
			cs = np.stack(a[:,1] - self.mse_budget) 
		else:
			raise ValueError(f'Mode {self.mode} is not supported')
		out["F"] = optim  # optimized
		out["G"] = cs  # constraint


def genetic_alg(memse:MemSE,
                n_vars:int,
                dataloader,
				nb_gen:int=1,
                pop_size:int=100,
                mode:str='opt_mse',
                power_budget:int=1000000,
                mse_budget:float=1,
                start_Gmax = None,
                device = torch.device('cpu'),
                batch_stop: int = -1):
	if nb_gen < 1:
		mse, P = test_mse_th(dataloader, memse, batch_stop=batch_stop)
		return P, mse, memse.quanter.Gmax.detach().cpu().numpy()

	problem = MemSEProblem(n_vars,
						   1,
                           1,
                           np.ones(n_vars)*0.01,
                           np.ones(n_vars)*500,
                           memse=memse,
                           mode=mode,
                           power_budget=power_budget,
                           mse_budget=mse_budget,
                           dataloader=dataloader,
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
		if mode == 'opt_mse':
			print(f"gen: {obj.n_gen} max gen: {obj.termination.n_max_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV')[0,0]} ideal: {obj.opt.get('F')[0,0]} power: {obj.opt.get('G')[0,0]+power_budget}")
		elif mode == 'opt_power':
			print(f"gen: {obj.n_gen} max gen: {obj.termination.n_max_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV')[0,0]} ideal: {obj.opt.get('F')[0,0]} mse: {obj.opt.get('G')[0,0]+mse_budget}")
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

for mode in MODES_INIT.keys():
	memse.quanter.init_wmax(mode)
	memse.quanter.init_gmax(1.)  # automatically extend to mode-size

	RES_WMAX[mode] = memse.quanter.Wmax

	print(f'Performing opt in {mode=}')

	if mode == 'LAYERWISE':
		print(f'{RES_GMAX["ALL"]=} {RES_WMAX["LAYERWISE"]=} {RES_WMAX["ALL"]=}')
		start_Gmax = default(RES_GMAX['ALL'], 1.) * numpify(RES_WMAX['LAYERWISE']) / RES_WMAX['ALL'].item()
	elif mode == 'COLUMNWISE':
		print(f'{RES_GMAX["LAYERWISE"]=} {RES_WMAX["COLUMNWISE"]=} {RES_WMAX["LAYERWISE"]=}')
		start_Gmax = np.concatenate([RES_GMAX['LAYERWISE'][i].item() * numpify(RES_WMAX['COLUMNWISE'][i]) / RES_WMAX['LAYERWISE'][i].item() for i in range(len(RES_WMAX['COLUMNWISE']))])
	else:
		start_Gmax = None

	
	print(f'{start_Gmax=}')

	P_all, mse_all, Gmax_tab_all = genetic_alg(memse,
                                            n_vars=MODES_INIT[mode],
                                            dataloader=output_train_loader,
											nb_gen=MODES_GEN[mode],
											pop_size=args.ga_popsize,
											mode=args.opt_mode,
											power_budget=args.power_budget,
											mse_budget=args.mse_budget,
											device=device,
											start_Gmax=start_Gmax,
											batch_stop=args.batch_stop_opt)

	print(f'Out of GA: \n Gmax = {Gmax_tab_all} \n Wmax = {memse.quanter.Wmax}')
	RES_GMAX[mode] = Gmax_tab_all
	RES_MSE[mode] = mse_all
	RES_P[mode] = P_all


print(RES_P)
print(RES_MSE)


#####
# POWER/ACC RESULTS
#####
RES_ACC = {}
RES_POW = {}
end = time.time()
for mode, Gmax in RES_GMAX.items():
    if np.any(Gmax) == None:
        print(f'A Gmax in {mode=} was NaN')
        continue
    memse.quanter.init_wmax(mode)
    memse.quanter.init_gmax(Gmax)
    _, acc = test_acc_sim(test_loader, memse, trials=args.N_mc, batch_stop=args.batch_stop_accuracy)
    _, pows = test_mse_th(small_test_loader, memse, batch_stop=args.batch_stop_power)
    RES_ACC[mode] = acc
    RES_POW[mode] = pows
    print(f'Done post for mode {mode}')
print(f'Power/acc results took {(time.time() - end)/60} minutes')

torch.save({'Gmax': RES_GMAX, 'Acc': RES_ACC, 'Pow': RES_POW, 'run_params': vars(args)}, result_filename)
print(f'opt.py ran in {(time.time() - starting_time)/60} minutes')
