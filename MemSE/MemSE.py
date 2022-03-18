from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from MemSE.MemristorQuant import MemristorQuant
from MemSE.network_manipulations import get_intermediates, store_add_intermediates_se, store_add_intermediates_var
from MemSE.utils import net_param_iterator
from MemSE.mse_functions import linear_layer_logic, softplus_vec_batched, avgPool2d_layer_vec_batched
from MemSE.nn import mse_gamma

class MemSE(nn.Module):
	def __init__(self, model, quanter, r:float=1., Gmax_init_Wmax:bool=False, input_bias:bool=True, input_shape=None):
		super(MemSE, self).__init__()
		self.model = model
		self.quanter = quanter
		self.r = r
		if r != 1:
			raise ValueError('Cannot work !')

		self.input_bias = input_bias
		if input_bias:
			assert input_shape is not None
			self.bias = nn.Parameter(torch.zeros(*input_shape))

		self.learnt_Gmax = nn.ParameterList()
		for i in range(len(quanter.Gmax)):
			if isinstance(quanter.Gmax[i], np.ndarray):
				lgmax = torch.from_numpy(quanter.Gmax[i]).clone().float()
			else:
				lgmax = torch.tensor(quanter.Gmax[i]).clone()
			self.learnt_Gmax.append(nn.Parameter(lgmax)) # can use quanter._init_Gmax if unsure quant has been cast already
			if Gmax_init_Wmax:
				# Init to Gmax == Wmax for learning
				self.learnt_Gmax[i].data.copy_(self.quanter.Wmax[i]) # may not work with scalar
		#self.clip_Gmax()

	@property
	def sigma(self) -> float:
		return self.quanter.std_noise

	def forward(self, x):
		if self.input_bias:
			x += self.bias[None, :, :, :]

		gamma_shape = [*x.shape, *x.shape[1:]]
		gamma = torch.zeros(0, device=x.device, dtype=x.dtype)
		P_tot = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
		i = 0
		for s in net_param_iterator(self.model):
			if isinstance(s,nn.Linear):
				x, gamma, P_tot_i, gamma_shape = linear_layer_logic(s.weight, x, gamma, self.learnt_Gmax[i], self.quanter.Wmax[i], self.sigma, self.r, gamma_shape)
				P_tot += P_tot_i
				i += 1
			if isinstance(s,nn.Softplus):
				x, gamma, gamma_shape = softplus_vec_batched(x, gamma, gamma_shape)
			if isinstance(s, nn.AvgPool2d):
				x, gamma, gamma_shape = avgPool2d_layer_vec_batched(x, gamma, s.kernel_size, s.stride, s.padding, gamma_shape)
		return x, gamma, P_tot

	def mse_forward(self, x, reps:int = 100):
		if self.input_bias:
			x += self.bias[None, :, :, :]
		
		self.unquant()
		get_intermediates(self.model, x)
		self.quant(c_one=False)

		# MU COMPUTATION
		hooks = store_add_intermediates_se(self.model)
		for _ in range(reps):
			self.forward_noisy(x)
		[h.remove() for h in hooks.values()]

		# VAR COMPUTATION
		hooks = store_add_intermediates_var(self.model, reps)
		for _ in range(reps):
			self.forward_noisy(x)
		[h.remove() for h in hooks.values()]
		self.unquant()
		self.quant() #c_one = True

		gamma_shape = [*x.shape, *x.shape[1:]]
		gamma = torch.zeros(0, device=x.device, dtype=x.dtype)
		P_tot = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
		i = 0
		mses = {'sim': {}, 'us': {}}
		means = {'sim': {}, 'us': {}}
		varis = {'sim': {}, 'us': {}}
		for idx, s in enumerate(net_param_iterator(self.model)):
			if isinstance(s,nn.Linear):
				x, gamma, P_tot_i, gamma_shape = linear_layer_logic(s.weight, x, gamma, self.learnt_Gmax[i], self.quanter.Wmax[i], self.sigma, self.r, gamma_shape)
				P_tot += P_tot_i
				i += 1
			elif isinstance(s,nn.Softplus):
				x, gamma, gamma_shape = softplus_vec_batched(x, gamma, gamma_shape)
			elif isinstance(s, nn.AvgPool2d):
				x, gamma, gamma_shape = avgPool2d_layer_vec_batched(x, gamma, s.kernel_size, s.stride, s.padding, gamma_shape)
			else:
				continue
			if hasattr(s, '__se_output'):
				mse_output = getattr(s, '__se_output') / reps
				th_output = getattr(s, '__th_output') / reps
				va_output = getattr(s, '__var_output') / (reps) # TODO y'a un - 1 en fait mais Johny John est pas content
				original_output = getattr(s, '__original_output')
				if type(s) not in mses['sim']:
					mses['sim'].update({type(s): {}})
					mses['us'].update({type(s): {}})
					means['sim'].update({type(s): {}})
					means['us'].update({type(s): {}})
					varis['sim'].update({type(s): {}})
					varis['us'].update({type(s): {}})
				mses['sim'].get(type(s)).update({idx: mse_output.mean().detach().cpu().numpy()})
				means['sim'].get(type(s)).update({idx: th_output.mean().detach().cpu().numpy()})
				means['us'].get(type(s)).update({idx: x.mean().detach().cpu().numpy()})
				if len(original_output.shape) > 2:
					original_output = original_output.view(original_output.shape[0], -1)
				gamma_viewed = original_output.shape + original_output.shape[1:]
				se_us = mse_gamma(original_output, x.view_as(original_output), gamma.view(gamma_viewed) if gamma_shape is None else torch.zeros(gamma_viewed, device=x))
				mses['us'].get(type(s)).update({idx: se_us.mean().detach().cpu().numpy()})
				varis['sim'].get(type(s)).update({idx: va_output.mean().detach().cpu().numpy()})
				varis['us'].get(type(s)).update({idx: gamma.view(gamma_viewed).diagonal(dim1=1, dim2=2).mean().detach().cpu().numpy()})

		return mses, means, varis


	def quant(self, c_one=True):
		self.quanter.quant(c_one=c_one)

	def unquant(self):
		self.quanter.unquant()

	def clip_Gmax(self):
		for i in range(len(self.learnt_Gmax)):
			self.learnt_Gmax[i].data.copy_(self.learnt_Gmax[i].data.clamp(0.3,2))

	@torch.no_grad()
	def forward_noisy(self, x):
		self.quanter.renoise()
		self.model(x)

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters", "Mean"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: 
			continue
		param = parameter.numel()
		table.add_row([name, param, parameter.mean().detach().cpu().item()])
		total_params += param
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params

def solve_p_mse(model, device, dataloader, Gmax, sigma, r, N, mode, reduce:bool=True):
	quanter = MemristorQuant(model, N = N, wmax_mode=mode, Gmax=Gmax, std_noise = sigma)
	memse = MemSE(model, quanter, sigma, r, input_bias=False).to(device)
	with torch.inference_mode():
		p_mean, mse_mean = mse_loop(dataloader, memse, device, reduce=reduce)
	return p_mean, mse_mean

def opt_p_mse(model, device, dataloader, Gmax, sigma, r, N, mode, nb_epochs:int=5, alpha:float=1., power_constraint:float=0.):
	quanter = MemristorQuant(model, N=N, wmax_mode=mode, Gmax=Gmax, std_noise=sigma)
	inp_ex, _ = next(iter(dataloader))
	memse = MemSE(model, quanter, sigma, r, input_shape=inp_ex.shape[1:]).to(device)
	for param in model.parameters():
		param.requires_grad = False
	count_parameters(memse)
	optimizer = optim.SGD(memse.parameters(), lr=1e-5)
	for epoch in range(nb_epochs):
		p_mean, mse_mean = mse_loop(dataloader, memse, device, optimizer, alpha=alpha, power_constraint=power_constraint)
		print(f'Epoch {epoch} - Mean P tot = {p_mean}, mean Max MSE {mse_mean}')
	return p_mean, mse_mean

def mse_loop(dataloader, memse:MemSE, device, optimizer=None, alpha:float=1., nb_acc_grad:int=10, power_constraint:float=0., reduce:bool=True):
	memse.quant(c_one=True)

	if reduce:
		p_total, mse = 0, 0
	else:
		p_total, mse = np.zeros(len(dataloader.dataset)), np.zeros(len(dataloader.dataset))

	if optimizer is not None:
		optimizer.zero_grad()

	with tqdm(dataloader, unit="batch") as tepoch:
		for i, (inp, tar) in enumerate(tepoch):
			inp, tar = inp.to(device, non_blocking=True), tar.to(device, non_blocking=True)
			mu, gamma, P_tot = memse(inp) #compute_moments_power_th_batched(model, Gmax, quanter.Wmax, sigma, inp, r)
			loss = mse(tar, mu, gamma)

			if optimizer is not None:
				loss_energy = loss.mean(dim=1) * 1e4 + alpha * torch.nn.functional.relu((P_tot - power_constraint).log())
				loss_energy.sum().backward()
				if (i+1) % nb_acc_grad == 0:
					optimizer.step()
					optimizer.zero_grad()
					if (i+1) % 100 == 0:
						count_parameters(memse)
					memse.clip_Gmax()
			
			max_mse = torch.amax(loss, dim=1)

			if reduce:
				p_total += torch.mean(P_tot, dim=0).item()
				mse += torch.mean(max_mse, dim=0).item()
			else:
				p_total[i*dataloader.batch_size:(i+1)*dataloader.batch_size] = P_tot.detach().clone().cpu().numpy()
				mse[i*dataloader.batch_size:(i+1)*dataloader.batch_size] = max_mse.detach().clone().cpu().numpy()

			tepoch.set_postfix(p_total=p_total, mse=mse)

	memse.unquant()
	if reduce:
		p_total /= len(dataloader)
		mse /= len(dataloader)
	return p_total, mse
