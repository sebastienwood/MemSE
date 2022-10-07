import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable
from tqdm import tqdm

from MemSE.definitions import SUPPORTED_OPS
from MemSE.MemristorQuant import MemristorQuant
from MemSE.network_manipulations import (get_intermediates, net_param_iterator,
                                         store_add_intermediates_mse,
                                         store_add_intermediates_var)
from MemSE.nn import mse_gamma, zero_but_diag_
from MemSE.nn.utils import zero_diag


def NOOP(*args, **kwargs):
    return

class MemSE(nn.Module):
    def __init__(self,
                 model,
                 quanter,
                 r: float=1.,
                 Gmax_init_Wmax: bool=False,
                 input_bias: bool=True,
                 input_shape=None,
                 taylor_order: int=2,
                 post_processing=None):
        super(MemSE, self).__init__()
        self.model = model
        self.quanter = quanter
        self.r = r
        self.taylor_order = taylor_order
        self.post_processing = post_processing
        if r != 1:
            raise ValueError('Cannot work !')
        if taylor_order < 1:
            raise ValueError('Taylor approx needs to be at least of order 1')

        self.input_bias = input_bias
        if input_bias:
            assert input_shape is not None
            self.bias = nn.Parameter(torch.zeros(*input_shape))

        self.learnt_Gmax = self.init_learnt_gmax(quanter, Gmax_init_Wmax)
        self._var_batch = {}
        #self.clip_Gmax()

    @staticmethod
    def init_learnt_gmax(quanter, Gmax_init_Wmax: bool = False):
        learnt_Gmax = nn.ParameterList()
        for i in range(len(quanter.Gmax)):
            if isinstance(quanter.Gmax[i], np.ndarray):
                lgmax = torch.from_numpy(quanter.Gmax[i]).clone().float()
            else:
                lgmax = torch.tensor(quanter.Gmax[i]).clone()
            learnt_Gmax.append(nn.Parameter(lgmax)) # can use quanter._init_Gmax if unsure quant has been cast already
            if Gmax_init_Wmax:
                # Init to Gmax == Wmax for learning
                learnt_Gmax[i].data.copy_(quanter.Wmax[i]) # may not work with scalar
            quanter.actual_params[i].learnt_Gmax = learnt_Gmax[i]
        return learnt_Gmax

    @property
    def sigma(self) -> float:
        return self.quanter.std_noise

    def forward(self, x, compute_power: bool = True, output_handle=None):
        self.quant()
        assert self.quanter.quanted and not self.quanter.noised, 'Need quanted and denoised'
        if self.input_bias:
            x += self.bias[None, :, :, :]

        data = {
            'mu': x,
            'gamma_shape': [*x.shape, *x.shape[1:]],
            'gamma': torch.zeros(0, device=x.device, dtype=x.dtype),
            'P_tot': torch.zeros(x.shape[0], device=x.device, dtype=x.dtype),
            'current_type': None,
            'compute_power': compute_power,
            'taylor_order': self.taylor_order,
            'sigma': self.sigma,
            'r': self.r
        }
        for idx, s in enumerate(net_param_iterator(self.model)):
            SUPPORTED_OPS.get(type(s), NOOP)(s, data)
            self.plot_gamma(data['gamma'], output_handle, data['current_type'], idx)
            self.post_process_gamma(data['gamma'])

        if len(data['mu'].shape) != 2:
            data['mu'] = data['mu'].reshape(data['mu'].shape[0], -1)
        if data['gamma_shape'] is not None:
            data['gamma'] = torch.zeros(data['gamma_shape'])
        if len(data['gamma'].shape) != 3:
            data['gamma'] = data['gamma'].reshape(data['gamma'].shape[0], data['mu'].shape[1], data['mu'].shape[1])
            
        self.unquant()
        return data['mu'], data['gamma'], data['P_tot']

    def mse_sim(self, x, tar, reps: int = 1000):
        reps = int(reps)
        if self.input_bias:
            x += self.bias[None, :, :, :]
        self.quant(c_one=False)
        out = (self.forward_noisy(x).detach() - tar) ** 2
        for _ in range(reps - 1):
            out += (self.forward_noisy(x).detach() - tar) ** 2
        out /= reps
        self.unquant()
        return out.flatten(start_dim=1)

    def no_power_forward(self, x):
        return self.forward(x, False)

    def mse_forward(self, x, reps: int = 1000, compute_power: bool = True, compute_cov: bool = False, output_handle = None):
        reps = int(reps)
        if self.input_bias:
            x += self.bias[None, :, :, :]
        
        self.unquant()
        get_intermediates(self.model, x)
        self.quant(c_one=False)

        # MU COMPUTATION
        hooks = store_add_intermediates_mse(self.model, reps)
        for _ in range(reps):
            self.forward_noisy(x)
        [h.remove() for h in hooks.values()]

        # VAR COMPUTATION
        hooks = store_add_intermediates_var(self.model, reps, compute_cov=compute_cov)
        for _ in range(reps):
            self.forward_noisy(x)
        [h.remove() for h in hooks.values()]
        self.unquant()
        self.quant() #c_one = True

        mses = {'sim': {}, 'us': {}}
        means = {'sim': {}, 'us': {}}
        varis = {'sim': {}, 'us': {}}
        covs = {'sim': {}, 'us': {}}

        data = { # TODO this could be managed in a distinct function
            'mu': x,
            'gamma_shape': [*x.shape, *x.shape[1:]],
            'gamma': torch.zeros(0, device=x.device, dtype=x.dtype),
            'P_tot': torch.zeros(x.shape[0], device=x.device, dtype=x.dtype),
            'current_type': None,
            'compute_power': compute_power,
            'taylor_order': self.taylor_order,
            'sigma': self.sigma,
            'r': self.r
        }
        for idx, s in enumerate(net_param_iterator(self.model)):
            SUPPORTED_OPS.get(type(s), NOOP)(s, data)
            if type(s) not in SUPPORTED_OPS:
                continue
            if hasattr(s, '__se_output'):
                mse_output = getattr(s, '__se_output')
                th_output = getattr(s, '__th_output')
                va_output = getattr(s, '__var_output')
                if compute_cov:
                    cov_output = getattr(s, '__cov_output')
                original_output = getattr(s, '__original_output').to(x.device)
                if type(s) not in mses['sim']:
                    for t in ['sim', 'us']:
                        mses[t].update({type(s): {}})
                        means[t].update({type(s): {}})
                        varis[t].update({type(s): {}})
                        covs[t].update({type(s): {}})
                mses['sim'].get(type(s)).update({idx: mse_output.mean().detach().cpu().numpy()})
                means['sim'].get(type(s)).update({idx: th_output.detach().cpu().numpy()})
                means['us'].get(type(s)).update({idx: data['mu'].detach().cpu().numpy()})
                if len(original_output.shape) > 2:
                    original_output = original_output.reshape(original_output.shape[0], -1)
                gamma_viewed = original_output.shape + original_output.shape[1:]
                se_us = mse_gamma(original_output, data['mu'].reshape_as(original_output), data['gamma'].reshape(gamma_viewed) if data['gamma_shape'] is None else torch.zeros(gamma_viewed, device=x))
                mses['us'].get(type(s)).update({idx: se_us.mean().detach().cpu().numpy()})
                varis['sim'].get(type(s)).update({idx: va_output.detach().cpu().numpy()})
                varis['us'].get(type(s)).update({idx: data['gamma'].view(gamma_viewed).diagonal(dim1=1, dim2=2).detach().cpu().numpy()})
                if compute_cov:
                    sim_ = cov_output
                    us_ = data['gamma']
                    covs['sim'].get(type(s)).update({idx: zero_diag(sim_).detach().cpu().numpy()})
                    covs['us'].get(type(s)).update({idx: zero_diag(us_).detach().cpu().numpy()})
                    if output_handle is not None:
                        self.plot_gamma(sim_, output_handle, f'{type(s)} SIM', idx)
                        self.plot_gamma(us_, output_handle, f'{type(s)} US', idx)
                        self.plot_gamma(us_ - sim_, output_handle, f'{type(s)} US - SIM', idx)
        return mses, means, varis, covs


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
        return self.model(x)

    def plot_gamma(self, gamma, output_handle, current_type, idx, per_sample=False):
        if output_handle == 'density':
            plt.figure(figsize=(12,4), facecolor='white')
            reshaped = gamma.detach().cpu().reshape(gamma.shape[0], -1).numpy()
            plt.hist([r.flatten() for r in reshaped], bins=25, density=True, label=list(range(gamma.shape[0])))
            plt.title(f'{idx}th layer ({current_type})')
            plt.yscale('log')
            plt.legend()
            plt.show()
        elif output_handle == 'imshow':
            if gamma.shape[0] > 1:
                fig, axs = plt.subplots(gamma.shape[0], figsize=(15,15), sharex=True, facecolor='white')
                fig.suptitle(f'{idx}th layer ({current_type})')
                for img_idx in range(gamma.shape[0]):
                    reshaped = gamma[img_idx].detach().cpu().reshape(int(math.sqrt(gamma[img_idx].numel())), -1)
                    hdle = axs[img_idx].matshow(reshaped)
                #fig.colorbar(hdle, ax=axs.ravel().tolist())
                plt.show()
            else:
                plt.matshow(gamma[0].detach().cpu().reshape(int(math.sqrt(gamma[0].numel())), -1))
                plt.show()
        #maxi = gamma.reshape(gamma.shape[0], -1).max(dim=1).values
        #mini = gamma.reshape(gamma.shape[0], -1).min(dim=1).values
        #self._var_batch[idx] = (maxi-mini).detach().cpu().tolist()
        #self._var_batch[idx] = gamma.reshape(gamma.shape[0], -1).var(dim=1).detach().cpu().tolist()
    
    def post_process_gamma(self, gamma):
        if self.post_processing == 'zero_but_diag':
            zero_but_diag_(gamma)
        

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
