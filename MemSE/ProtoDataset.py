from torch.utils import data
from MemSE.definitions import ROOT
from pathlib import Path
from tqdm import tqdm

import torch

__all__ = ['ProtoDataset']

class ProtoDataset(data.Dataset):
    def __init__(self, dataset, path=f'{ROOT}/data') -> None:
        super().__init__()
        assert len(dataset[0]) == 2, 'Only support dataset with input/target pairs'
        self.orig_dataset = dataset
        self.path = path
        if not self._check_exist():
            self.generate_and_save()
        self.handle_proto = torch.load(self.save_path())
        self.tensors = self.handle_proto['mu'], self.handle_proto['cov'], self.handle_proto['tar']
            
    def _check_exist(self):
        return self.save_path().exists()
            
    def save_name(self):
        return f'{self.hash_param()}.proto'
    
    def save_path(self):
        return Path(self.path) / self.save_name()

    def hash_param(self):
        return hash((self.orig_dataset.train, str(self.orig_dataset.transform)))

    def generate_and_save(self, bs: int = 128):
        inp = self.orig_dataset[0][0]
        targets = torch.tensor(self.orig_dataset.targets, dtype=torch.long)
        uniques_tar = torch.unique(targets).tolist()
        res_mu, res_cov, res_tar = [], [], []
        for idx in tqdm(uniques_tar):
            subset = data.Subset(self.orig_dataset, torch.where(targets == idx)[0])
            dataloader = data.DataLoader(subset, batch_size=bs, shuffle=False)
            mu, cov = torch.zeros(*inp.shape), torch.zeros(*inp.shape, *inp.shape)
            for img, _ in dataloader:
                mu += torch.sum(img, dim=0)
            mu /= len(subset)
            for img, _ in dataloader:
                delta = (img - mu).reshape(img.shape[0], -1)
                cov += torch.einsum('bi, bj -> bij', delta, delta).reshape((img.shape[0],) + inp.shape + inp.shape).sum(dim=0)
            cov /= (len(subset) - 1)
            res_mu.append(mu)
            res_cov.append(cov)
            res_tar.append(idx)
        res_mu = torch.stack(res_mu)
        res_cov = torch.stack(res_cov)
        res_tar = torch.tensor(res_tar)
        torch.save({'mu': res_mu, 'cov': res_cov, 'tar': res_tar}, self.save_path())
        
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self) -> int:
        return self.tensors[0].size(0)