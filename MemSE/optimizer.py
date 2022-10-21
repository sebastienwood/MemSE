# TODO gather multiple libraries for meta heuristics
# common interface
# benchmark
# use
import torch

from pathlib import Path
from typing import List
from MemSE import MemSE
from MemSE.dataset import get_output_loader
from MemSE.definitions import DEFAULT_DEVICE
from MemSE.train_test_loop import test_mse_th


class Parameters:
    def __init__(self) -> None:
        pass


class OptimizerBase:
    def __init__(self,
                 memse: MemSE,
                 dataloader: torch.utils.data.DataLoader,
                 device: torch.DeviceObjType = DEFAULT_DEVICE,
                 batch_stop: int = -1) -> None:
        self.memse = memse
        if not hasattr(dataloader, '__output_loader'):
            self.dataloader = get_output_loader(dataloader, self.memse.model)
        else:
            self.dataloader = dataloader
        self.device = device
        self.batch_stop = batch_stop

    def build_objects(self):
        pass

    @torch.no_grad()
    def evaluate(self, solution):
        # TODO multi gpu
        self.memse.quanter.init_Gmax(solution)
        self.memse.quant(c_one=True)
        mses, pows = test_mse_th(self.dataloader, self.memse, self.device, self.batch_stop)
        self.memse.unquant()
        return torch.mean(pows), torch.mean(torch.amax(mses),dim=1)

    def optimize(self, parameters: List[Parameters], constraints):
        ckpt = Path(f"ckpt_"+str(power_budget)+type_opt+".npy")
        if ckpt.is_file():
            pass

        self._lib_magic()

    def _lib_magic(self):
        pass


class PymooOptimizer(OptimizerBase):
    def _lib_magic(self):
        pass



# class problemClass(Problem):

#     def __init__(self,n_var, n_obj, n_constr, xl, xu, func_args, power_budget, dataloader, device_id = None):
#         super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
#         self.func_args = func_args
#         self.power_budget = power_budget
#         self.device_id = device_id
#         self.dataloader = dataloader

#     def _evaluate(self, Gmax, out, *args, **kwargs):
#         x, y = next(iter(self.dataloader))
#         a = []
#         for G  in Gmax:
#             P, mse = im_problem_function_batched(G, self.func_args, x, device_id=self.device_id)
#             a.append([P.item(), mse.item()])
#         a = np.array(a)
#         out["F"] =  np.stack(a[:,1]) 
#         out["G"] = np.stack(a[:,0]-self.power_budget)

# def genetic_alg(nb_gen=1, nb_inputs=100, pop_size=100, power_budget=1000000,r=1, N=100, sigma=0.01, type_opt='network', dataloader=None, device_id = None, start_Gmax = None):


#   if type_opt == 'ALL':
#     n_vars = 1
#   elif type_opt == 'LAYERWISE':
#     _, n_vars = n_vars_computation(model)
#   elif type_opt == 'COLUMNWISE':
#     n_vars, _ = n_vars_computation(model)

#   args = (model, sigma, r, N, type_opt)

#   problem = problemClass(n_vars, 1, 1, np.ones(n_vars)*0.05,np.ones(n_vars)*100, args, power_budget, dataloader, device_id = device_id)

#   if np.any(start_Gmax) == None: 

#     algorithm = GA(
#             pop_size=pop_size,
#             eliminate_duplicates=True,
#             sampling = np.clip(np.ones((pop_size,n_vars))+np.random.normal(0,0.5,(pop_size,n_vars)),0.05,100))

#   else:
#     algorithm = GA(
#     pop_size=pop_size,
#     eliminate_duplicates=True,
#     sampling = np.clip(np.ones((pop_size,n_vars))*start_Gmax+np.random.normal(0,0.001,(pop_size,n_vars)),0.05,100))


#   ga_checkpoint = "checkpoint_opti_algo_"+str(power_budget)+type_opt+".npy"
#   if os.path.isfile(ga_checkpoint):
#     obj, = np.load(ga_checkpoint, allow_pickle=True).flatten()
#     print("Loaded Checkpoint:", obj)
#     obj.termination.n_max_gen += 100
#     obj.has_terminated = False

#   else:
#     # perform a copy of the algorithm to ensure reproducibility
#     obj = copy.deepcopy(algorithm)
    
#     # let the algorithm know what problem we are intending to solve and provide other attributes
#     obj.setup(problem, termination=("n_gen", nb_gen),)
    
#     # until the termination criterion has not been met
#   while obj.has_next():
    
#         # perform an iteration of the algorithm
#         obj.next()
    
#         # access the algorithm to print some intermediate outputs
#         print(f"gen: {obj.n_gen} max gen: {obj.termination.n_max_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV')[0,0]} ideal: {obj.opt.get('F')[0,0]} power: {obj.opt.get('G')[0,0]+power_budget}")
#         print(obj.opt.get('X'),np.min(obj.opt.get('X')),np.max(obj.opt.get('X')))
#         # print(im_problem_function_batched(obj.opt.get('X')[0], args, x, z, device_id=0))

#         # if obj.n_gen%50 == 0:
#           # np.save("checkpoint_opti_algo_"+str(power_budget)+type_opt+'.npy', obj)
#           # np.save('checkpoint_gmax_opti_'+type_opt+'_'+str(power_budget)+'.npy', obj.opt.get('X'))

    
#     # finally obtain the result object
#   res = obj.result()

  
#   if np.all(res.X!=None):
#     P_tot, mse =  res.G+power_budget, res.F#im_problem_function_batched(res.X, args, x, device_id=device_id)
#     minmax_power_tab = P_tot
#     minmax_mse_tab = mse
#   else:
#       minmax_power_tab = np.nan #res.G+power_budget
#       minmax_mse_tab = np.nan
#   return minmax_power_tab, minmax_mse_tab, res.X