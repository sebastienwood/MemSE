import MemSE.models as models
import torch
from MemSE.definitions import ROOT
from pathlib import Path

def load_model(name:str, num_classes:int, save_name=None):
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
