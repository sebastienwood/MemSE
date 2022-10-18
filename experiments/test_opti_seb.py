# %%
import os
import sys
import copy
import time
import torch
import torch.nn as nn
import pytest
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
from MemSE.models import make_JohNet, smallest_vgg_ReLU
from MemSE.fx.network_manipulations import conv_to_fc, conv_to_unfolded
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant
from MemSE.nn.utils import mse_gamma
from MemSE.utils import count_parameters, n_vars_computation
from MemSE.models import smallest_vgg_ReLU, really_small_vgg_ReLU
from MemSE.dataset import get_dataloader
from MemSE.model_load import load_model

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.core.problem import ElementwiseProblem, Problem

import gc



torch.manual_seed(0)

if len(sys.argv) >= 3:
  device = sys.argv[1]
  power_budget = int(sys.argv[2])
elif len(sys.argv) >= 2:
  device = sys.argv[1]
  power_budget = 1000000
else:
  device = 'cuda'
  power_budget = 1000000


# network = 'make_JohNet'
# memscale = True

network = 'smallest_vgg_ReLU'
memscale = False

result_filename = '/users/kern/Documents/MemSE/paper/data/opti_new_conv_'+network+'_'+str(power_budget)+'.z'

OPTI_BS = 10


#########
bs = 128
train_loader, train_clean_loader, test_loader, nclasses, input_shape = get_dataloader('cifar10', root='./data', bs=bs, memscale=memscale)
net = load_model(network, 10, input_shape, save_name=network,cast_to_memristor=False)

net = net.to(device)
inp =  next(iter(test_loader))[0].to(device)
model = conv_to_unfolded(net, inp.shape[1:]).to(device)

R = 1

N_mc = 20

SIGMA=0.01
N=1280000000

nvar_col, nvar_layer = n_vars_computation(model)

print(nvar_col,nvar_layer)

## Create sample batch
sample_size = 10

batch = torch.zeros((sample_size, inp.shape[1], inp.shape[2], inp.shape[3]))

bigx = torch.empty(0,device=device)
bigy = torch.empty(0,device=device)
for batch_idx, (x,y) in enumerate(train_clean_loader):
  x,y = inp.to(device), y.to(device)
  bigx = torch.cat((bigx, x), 0)
  bigy = torch.cat((bigy, y), 0)

sn = sample_size//nclasses
for i in range(nclasses):
  test = (bigy == i).nonzero(as_tuple=True)[0][:sn]
  batch[i*sn:(i+1)*sn] = bigx[test]




# %%

def im_problem_function_batched(Gmax, args, x, device_id=None, torch_dtype=torch.float32):

  with torch.no_grad():

  
    model, sigma, r, N, mode = args 
      

    quanter = MemristorQuant(model, std_noise=sigma, N=N, Gmax=Gmax, wmax_mode=mode)
    memse = MemSE(model, quanter, input_bias=None).to(device)
    memse.eval()

    mse_th, power_th = torch.empty(0).to(device),torch.empty(0).to(device)
    
    for i in range(int(np.ceil(x.shape[0]/OPTI_BS))):
  
      if i<x.shape[0]//OPTI_BS:
        batch = x[OPTI_BS*i:OPTI_BS*(i+1)].to(device)
      else:
        batch = x[OPTI_BS*i::].to(device)

      original_output = model(batch)

      # memse.quant(c_one = True)
      mu, gamma, P_tot = memse.forward(batch)#,meminfo='gpu')
      mse_th = torch.cat((mse_th, mse_gamma(original_output, mu, gamma)), 0)
      power_th = torch.cat((power_th, P_tot), 0)
      # memse.unquant()
      gc.collect()



  return torch.mean(power_th), torch.mean(torch.amax(mse_th,dim=1))

class problemClass(Problem):

    def __init__(self,n_var, n_obj, n_constr, xl, xu, func_args, power_budget, batch, device_id = None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.func_args = func_args
        self.power_budget = power_budget
        self.device_id = device_id
        self.batch = batch

    def _evaluate(self, Gmax, out, *args, **kwargs):
        x =  self.batch
        a = []
        for G  in Gmax:
          P, mse = im_problem_function_batched(G, self.func_args, x, device_id=self.device_id)
          a.append([P.item(), mse.item()])


        a = np.array(a)
        out["F"] =  np.stack(a[:,1]) 
        out["G"] = np.stack(a[:,0]-self.power_budget)

def genetic_alg(nb_gen=1, nb_inputs=100, pop_size=100, power_budget=1000000,r=1, N=100, sigma=0.01, type_opt='network', batch=None, device_id = None, start_Gmax = None):



  if type_opt == 'ALL':
    n_vars = 1
  elif type_opt == 'LAYERWISE':
    _, n_vars = n_vars_computation(model)
  elif type_opt == 'COLUMNWISE':
    n_vars, _ = n_vars_computation(model)

  args = (model, sigma, r, N, type_opt)

  problem = problemClass(n_vars, 1, 1, np.ones(n_vars)*0.01,np.ones(n_vars)*500, args, power_budget, batch, device_id = device_id)

  if np.any(start_Gmax) == None: 

    algorithm = GA(
            pop_size=pop_size,
            eliminate_duplicates=True,
            sampling = np.clip(np.ones((pop_size,n_vars))+np.random.normal(0,0.5,(pop_size,n_vars)),0.01,500))

  else:
    algorithm = GA(
    pop_size=pop_size,
    eliminate_duplicates=True,
    sampling = np.clip(np.ones((pop_size,n_vars))*start_Gmax+np.random.normal(0,0.001,(pop_size,n_vars)),0.01,500))


 
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
    P_tot, mse =  res.G+power_budget, res.F#im_problem_function_batched(res.X, args, x, device_id=device_id)
    minmax_power_tab = P_tot
    minmax_mse_tab = mse
  else:
      minmax_power_tab = np.nan #res.G+power_budget
      minmax_mse_tab = np.nan
  return minmax_power_tab, minmax_mse_tab, res.X


# %%
x, y = next(iter(train_clean_loader))
x = x.clone().to(torch.device(device), non_blocking=True) 
z = model(x)
                      
                                                 


quanter = MemristorQuant(model, std_noise=SIGMA, N=N, Gmax=1., wmax_mode='ALL')
ALL_Wmax = quanter.Wmax.detach().cpu().numpy()

P_all, mse_all, Gmax_tab_all = genetic_alg(nb_gen=1, nb_inputs=100, pop_size=20, power_budget=power_budget,r=1, N=N, sigma=0.01, type_opt='ALL', batch=batch,device_id=device)

quanter = MemristorQuant(model, std_noise=SIGMA, N=N, Gmax=np.ones(nvar_layer), wmax_mode='LAYERWISE')
LAYER_Wmax = quanter.Wmax.detach().cpu().numpy()

start_Gmax = Gmax_tab_all*LAYER_Wmax/ALL_Wmax

P_lay, mse_lay, Gmax_tab_lay = genetic_alg(nb_gen=1, nb_inputs=100, pop_size=20, power_budget=power_budget,r=1, N=N, sigma=0.01, type_opt='LAYERWISE', batch=batch,device_id=device,start_Gmax=start_Gmax)


quanter = MemristorQuant(model, std_noise=SIGMA, N=N, Gmax=np.ones(nvar_col), wmax_mode='COLUMNWISE')
COLUMN_Wmax = quanter.Wmax


start_Gmax = np.concatenate([Gmax_tab_lay[i]*COLUMN_Wmax[i].detach().cpu().numpy()/LAYER_Wmax[i] for i in range(nvar_layer)])


P_col, mse_col, Gmax_tab_col = genetic_alg(nb_gen=1, nb_inputs=100, pop_size=20, power_budget=power_budget,r=1, N=N, sigma=0.01, type_opt='COLUMNWISE', batch=batch,device_id=device,start_Gmax=start_Gmax)
    
print(P_all, mse_all)
print(P_lay, mse_lay)
print(P_col, mse_col)


# %%
def comp_noisy_acc(Gmax, mode, sigma, N, model, dataloader, N_mc = 200):

    if np.any(Gmax) == None:
        return np.nan

    quanter = MemristorQuant(model, std_noise=sigma, N=N, Gmax=Gmax, wmax_mode=mode)
    memse = MemSE(model, quanter, input_bias=None).to(device)
    memse.eval()
            
    acc, n_pts = 0, 0
    for batch_idx, (inp, y) in enumerate(dataloader):
        inp,y = inp.to(device), y.to(device)
        memse.quant(c_one=False)
        for n in range(N_mc):
            _, pred = memse.forward_noisy(inp).topk(1)
            acc += torch.sum(y == pred.flatten())
            n_pts += y.numel()
        memse.unquant()

        if batch_idx > 10:
          break

    return acc/n_pts

def comp_power(Gmax, mode, sigma, N, model, dataloader):

    if np.any(Gmax) == None:
        return np.nan

    quanter = MemristorQuant(model, std_noise=sigma, N=N, Gmax=Gmax, wmax_mode=mode)

    memse = MemSE(model, quanter, input_bias=None).to(device)
    memse.eval()

    memse.quant(c_one = True)

    P_tot_tab = [] 

    for batch_idx, (batch, target) in enumerate(dataloader):
        # print(batch_idx)
        batch, target = batch.to(device), target.to(device)

        x, gamma, P_tot = memse.forward(batch)

        P_tot_tab.extend(P_tot.cpu().numpy())

        if batch_idx > 100:
          break

        
    memse.unquant()
    
    return np.mean(P_tot_tab)



bs = 128
train_loader_big_batch, train_clean_loader_big_batch, test_loader_big_batch, nclasses, input_shape = get_dataloader('cifar10', root='./data', bs=bs, memscale=memscale)

bs = OPTI_BS
train_loader_small_batch, train_clean_loader_small_batch, test_loader_small_batch, nclasses, input_shape = get_dataloader('cifar10', root='./data', bs=bs, memscale=memscale)

with torch.no_grad():

        acc_all = comp_noisy_acc(Gmax_tab_all, 'ALL', 0.01, N, model, test_loader_big_batch, N_mc = 1000)
        print(acc_all)
        acc_lay = comp_noisy_acc(Gmax_tab_lay, 'LAYERWISE', 0.01, N, model, test_loader_big_batch, N_mc = 1000)
        print(acc_lay)
        acc_col = comp_noisy_acc(Gmax_tab_col, 'COLUMNWISE', 0.01, N, model, test_loader_big_batch, N_mc = 1000)
        print(acc_col)
        
        Ptot_all = comp_power(Gmax_tab_all, 'ALL', 0.01, N, model, test_loader_small_batch)
        print(Ptot_all)
        Ptot_lay = comp_power(Gmax_tab_lay, 'LAYERWISE', 0.01, N, model, test_loader_small_batch)
        print(Ptot_lay)
        Ptot_col = comp_power(Gmax_tab_col, 'COLUMNWISE', 0.01, N, model, test_loader_small_batch)
        print(Ptot_col)


joblib.dump([Gmax_tab_all, Ptot_all, acc_all, Gmax_tab_lay, Ptot_lay,acc_lay, Gmax_tab_col, Ptot_col,acc_col],result_filename) 


