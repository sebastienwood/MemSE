import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from MemSE.mse_functions import *

class MemSE(nn.Module):
	def __init__(self, model, quanter, sigma:float, r:float, Gmax_init_Wmax:bool=True, input_bias:bool=True, input_shape=None):
		super(MemSE, self).__init__()
		self.model = model
		self.quanter = quanter
		self.sigma = sigma
		self.r = r

		self.input_bias = input_bias
		if input_bias:
			assert input_shape is not None
			self.bias = nn.Parameter(torch.zeros(*input_shape))

		self.learnt_Gmax = nn.ParameterList()
		for i in range(len(quanter.Gmax)):
			if isinstance(quanter.Gmax[i], np.ndarray):
				lgmax = torch.from_numpy(quanter.Gmax[i]).clone()
			else:
				lgmax = torch.tensor(quanter.Gmax[i]).clone()
			self.learnt_Gmax.append(nn.Parameter(lgmax)) # can use quanter._init_Gmax if unsure quant has been cast already
			if Gmax_init_Wmax:
				# Init to Gmax == Wmax for learning
				self.learnt_Gmax[i].data.copy_(self.quanter.Wmax[i]) # may not work with scalar
		self.clip_Gmax()

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

	def quant(self, c_one=True):
		self.quanter.quant(c_one=c_one)

	def unquant(self):
		self.quanter.unquant()

	def clip_Gmax(self):
		for i in range(len(self.learnt_Gmax)):
			self.learnt_Gmax[i].data.copy_(self.learnt_Gmax[i].data.clamp(0.3,2))

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters", "Mean"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: 
			continue
		param = parameter.numel()
		table.add_row([name, param, parameter.mean().detach().cpu().item()])
		total_params+=param
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
	quanter = MemristorQuant(model, N = N, wmax_mode=mode, Gmax=Gmax, std_noise = sigma)
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
			loss = torch.diagonal(gamma, dim1=1, dim2=2)+torch.square(mu-tar)

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
