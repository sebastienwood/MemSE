import MemSE.models as models
import torch
import numpy as np

from MemSE.definitions import ROOT, WMAX_MODE
from MemSE.network_manipulations import conv_to_fc, fuse_conv_bn
from MemSE.MemristorQuant import MemristorQuant
from MemSE.MemSE import MemSE

from pathlib import Path

from typing import Union


ROOT_PRETRAINED = f'{ROOT}/MemSE/pretrained'


def load_model(name: str, num_classes: int, input_shape, save_name=None, **kwargs):
	if save_name is None:
		save_name = name
	model = getattr(models, name)(num_classes=num_classes, **kwargs)
	maybe_chkpt = Path(f'{ROOT}/MemSE/models/saves/{save_name}.pth')
	if maybe_chkpt.exists():
		print('Loading model checkpoint')
		loaded = torch.load(maybe_chkpt, map_location=torch.device('cpu'))
		rename_state_dict = loaded['state_dict'].copy()
		for key in loaded['state_dict'].keys():
			if "module." in key:
				rename_state_dict[key.replace('module.', '')] = loaded['state_dict'][key]
				rename_state_dict.pop(key)
		loaded['state_dict'] = rename_state_dict
		model.load_state_dict(loaded['state_dict'])
	model.eval()
	model = fuse_conv_bn(model, name)
	model = conv_to_fc(model, input_shape)
	return model


def load_memristor(model, name: str, mode: Union[WMAX_MODE, str], device, std_noise:float = 0.01, N:int = 128, save_name=None, **kwargs):
	if isinstance(mode, str):
		mode = WMAX_MODE[mode.upper()]
	
	if save_name is None:
		save_name = name

	new_instance_kwargs = {}
	kw_str = ''
	if kwargs:
		for k, v in kwargs.items():
			kw_str += f'_{k}_{v}'

			if k == 'post_processing':
				new_instance_kwargs[k] = v

	maybe_chkpt = Path(f'{ROOT_PRETRAINED}/{save_name}/{mode.name.lower()}{kw_str}.npy')
	if 'gmax' in kwargs:
		loaded = kwargs['gmax']
	elif maybe_chkpt.exists():
		print('Loading memristor config')
		loaded = np.load(maybe_chkpt)
	else:
		loaded = None

	quanter = MemristorQuant(model, std_noise=std_noise, N=N, Gmax=loaded, wmax_mode=mode)
	memse = MemSE(model, quanter, input_bias=None, **new_instance_kwargs).to(device)
	
	return memse


def find_existing(name: str):
	res = {}
	path_dir = Path(f'{ROOT_PRETRAINED}/{name}/').glob('*.npy')
	for path in path_dir:
		to_parse = str(path)
		splited = to_parse.split('_')
		mode = splited[0]
		if not mode in res:
			res[mode] = []
		res[mode].append({splited[i*2]: splited[(i*2)+1] for i in range((len(splited)-1) / 2)})
	return res
		
