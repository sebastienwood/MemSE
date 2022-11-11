import torch
from MemSE.misc import AverageMeter
from typing import Tuple

__all__ = ['compute_sample_moments', 'compute_n_mc']

####
# Tools for optimal N_mc
####

@torch.inference_mode()
def compute_sample_moments(memse, x, tar, max_sample) -> Tuple[float, float]:
    if memse.input_bias:
        x += memse.bias[None, :, :, :]
    memse.quant(c_one=False)
    
    mse_metter = AverageMeter('mse')
    for _ in range(max_sample):
        mse_metter.update(torch.mean((memse.forward_noisy(x).detach()  - tar) ** 2).item())
    mse = mse_metter.avg

    var_metter = AverageMeter('var')
    for _ in range(max_sample):
        var_metter.update((torch.mean((memse.forward_noisy(x).detach()  - tar) ** 2).item()-mse) ** 2)
    var = var_metter.sum / (max_sample - 1)    

    memse.unquant()
    return mse, var
        

def compute_n_mc(CI, z, sample_var) -> int:
	"""Compute the number of monte-carlo samples to be computed

	Args:
		CI (_type_): the confidence interval to seek
		z (_type_): _description_
		sample_var (_type_): _description_

	Returns:
		int: _description_
	"""
	z_dic = {90: 1.645, 95: 1.96, 96: 2.05, 98: 2.33, 99: 2.58}
	z = z_dic[z]
	return int(z*z*sample_var/CI ** 2)