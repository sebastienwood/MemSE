import MemSE.models as models
import torch
import numpy as np

from MemSE.definitions import ROOT, WMAX_MODE
from MemSE.network_manipulations import conv_to_fc
from MemSE.MemristorQuant import MemristorQuant
from MemSE.MemSE import MemSE

from pathlib import Path

from typing import Union

def load_model(name: str, num_classes: int, save_name=None):
	if save_name is None:
		save_name = name
	model = getattr(models, name)(num_classes=num_classes)
	maybe_chkpt = Path(f'{ROOT}/MemSE/models/saves/{name}.pth')
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
	return model


def load_memristor(name: str, num_classes: int, mode: Union[WMAX_MODE, str], device, input_shape, std_noise:float = 0.01, N:int = 128, save_name=None):
	model = load_model(name, num_classes, save_name)
	model = conv_to_fc(model, input_shape).to(device)

	if isinstance(mode, str):
		mode = WMAX_MODE[mode.upper()]
	
	if save_name is None:
		save_name = name

	maybe_chkpt = Path(f'{ROOT}/MemSE/pretrained/{name}_{mode.name.lower()}.npy')
	if maybe_chkpt.exists():
		print('Loading memristor config')
		loaded = np.load(maybe_chkpt)
	else:
		loaded = None

	quanter = MemristorQuant(model, std_noise=std_noise, N=N, Gmax=loaded, wmax_mode=mode)
	memse = MemSE(model, quanter, input_bias=None).to(device)
		
	return memse
