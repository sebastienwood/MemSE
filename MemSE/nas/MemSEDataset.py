# Adapted from https://github.com/mit-han-lab/once-for-all/blob/master/ofa/nas/accuracy_predictor/acc_dataset.py
import json
import os
from pathlib import Path
from MemSE.training import RunManager
from MemSE.nn import OFAxMemSE
from smt.sampling_methods import LHS
import torch
import numpy as np
import torch.utils.data

from tqdm import tqdm


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class AccuracyDataset:
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    @property
    def net_id_path(self):
        return os.path.join(self.path, "net_id.dict")

    @property
    def acc_src_folder(self):
        return os.path.join(self.path, "src")

    @property
    def acc_dict_path(self):
        return os.path.join(self.path, "acc.dict")
    
    @property
    def acc_dataset_path(self):
        return os.path.join(self.path, "acc_dataset.pth")

    # TODO: support parallel building
    def build_acc_dataset(
        self, run_manager:RunManager, ofa_network: OFAxMemSE, n_arch=16000, image_size_list=None, range_LHS=1, nb_batchs=-1, nb_batchs_power=-1, nb_sample_LHS=50
    ):
        # load net_id_list, random sample if not exist
        if os.path.isfile(self.net_id_path):
            net_id_list = json.load(open(self.net_id_path))
        else:
            net_id_list = set()
            while len(net_id_list) < n_arch:
                net_setting = ofa_network.sample_active_subnet(None, skip_adaptation=True, noisy=False)
                net_id = net_setting2id(net_setting)
                net_id_list.add(net_id)
            net_id_list = list(net_id_list)
            net_id_list.sort()
            json.dump(net_id_list, open(self.net_id_path, "w"), indent=4)

        if isinstance(image_size_list, int):
            image_size_list = [image_size_list]
        image_size_list = (
            [128, 160, 192, 224] if image_size_list is None else image_size_list
        )

        with tqdm(
            total=len(net_id_list) * len(image_size_list) * nb_sample_LHS, desc="Building Acc Dataset"
        ) as t:
            for image_size in image_size_list:
                # load val dataset into memory
                val_dataset = []
                run_manager._loader.assign_active_img_size(image_size)
                data_loader = run_manager._loader.build_sub_train_loader(
                        n_images=2000, batch_size=200
                    )
                for batch in run_manager.valid_loader:
                    if isinstance(batch, dict):
                        images, labels = batch['image'], batch['label']
                    else:
                        images, labels = batch
                    val_dataset.append((images, labels))
                # save path
                os.makedirs(self.acc_src_folder, exist_ok=True)
                acc_save_path = os.path.join(
                    self.acc_src_folder, "%d.dict" % image_size
                )
                acc_dict = {}
                # load existing acc dict
                if os.path.isfile(acc_save_path):
                    existing_acc_dict = json.load(open(acc_save_path, "r"))
                    acc_dict |= existing_acc_dict
                else:
                    existing_acc_dict = {}
                for net_id in net_id_list:
                    net_setting = net_id2setting(net_id)

                    def val(net_setting, skip_adaptation=False):
                        key = net_setting2id({**net_setting, "image_size": image_size})
                        if key in existing_acc_dict:
                            t.set_postfix(
                                {
                                    "net_id": key,
                                    "image_size": image_size,
                                    "info_val": acc_dict[key],
                                    "status": "loading",
                                }
                            )
                            t.update()
                            # Resume case: we need to adapt the network
                            if not hasattr(ofa_network, 'active_crossbars'):
                                ofa_network.sample_active_subnet(net_setting, data_loader)
                            return False
                        if skip_adaptation:
                            ofa_network._model.quanter.Gmax = [gu for gu in net_setting['gmax'] if gu > 0]
                            ofa_network._state = net_setting
                        else:
                            ofa_network.set_active_subnet(net_setting, data_loader)

                        ofa_network.quant(scaled=False)

                        loss, metrics = run_manager.validate(
                            net=ofa_network,
                            data_loader=val_dataset,
                            no_logs=True,
                            nb_batchs=nb_batchs,
                            nb_batchs_power=nb_batchs_power
                        )
                        info_val = metrics.top1.avg
                        ofa_network.unquant()

                        t.set_postfix(
                            {
                                "net_id": key,
                                "image_size": image_size,
                                "info_val": info_val,
                            }
                        )
                        t.update()
                        acc_dict.update({key: {'top1':info_val, 'power':metrics.power.avg}})
                    not_already_exists = val(net_setting)
                    if not_already_exists is not None and not not_already_exists:
                        t.update(nb_sample_LHS)
                        continue
                    nb_cb = len(ofa_network.active_crossbars)
                    sampling = LHS(xlimits=np.array([[1 - range_LHS / 2, 1 + range_LHS / 2]] * nb_cb))
                    samples = sampling(nb_sample_LHS)
                    for options in range(len(samples)):
                        gmax = torch.einsum("a,a->a", torch.from_numpy(samples[options]).to(ofa_network._model.quanter.Wmax), ofa_network._model.quanter.Wmax).to('cpu')
                        gmax_clean = torch.zeros(ofa_network.gmax_size).scatter_(0, torch.LongTensor(ofa_network.active_crossbars), gmax).tolist()
                        net_setting |= {'gmax': gmax_clean}
                        val(net_setting, True)
                    json.dump(acc_dict, open(acc_save_path, "w"), indent=4)

    def merge_acc_dataset(self, image_size_list=None):
        # load existing data
        merged_acc_dict = {}
        for fname in os.listdir(self.acc_src_folder):
            if ".dict" not in fname:
                continue
            image_size = int(fname.split(".dict")[0])
            if image_size_list is not None and image_size not in image_size_list:
                print("Skip ", fname)
                continue
            full_path = os.path.join(self.acc_src_folder, fname)
            partial_acc_dict = json.load(open(full_path))
            merged_acc_dict.update(partial_acc_dict)
            print("loaded %s" % full_path)
        json.dump(merged_acc_dict, open(self.acc_dict_path, "w"), indent=4)
        return merged_acc_dict

    def build_acc_data_loader(
        self, arch_encoder, n_training_sample=None, batch_size=256, n_workers=16
    ):
        # TODO: save and load if it exists
        # load data
        if not Path(self.acc_dataset_path).exists():
            acc_dict = json.load(open(self.acc_dict_path))
            X_all = []
            Y_all = []
            with tqdm(total=len(acc_dict), desc="Loading data") as t:
                for k, v in acc_dict.items():
                    dic = json.loads(k)
                    X_all.append(arch_encoder.arch2feature(dic))
                    Y_all.append([v['top1'] / 100.0, v['power']])  # range: 0 - 1
                    t.update()

            # convert to torch tensor
            X_all = torch.tensor(np.array(X_all), dtype=torch.float)
            Y_all = torch.tensor(np.array(Y_all))
            
            # random shuffle
            shuffle_idx = torch.randperm(len(X_all))
            X_all = X_all[shuffle_idx]
            Y_all = Y_all[shuffle_idx]
            
            # split data
            idx = X_all.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
            val_idx = X_all.size(0) // 5 * 4
            X_train, Y_train = X_all[:idx], Y_all[:idx]
            X_test, Y_test = X_all[val_idx:], Y_all[val_idx:]
            print("Train Size: %d," % len(X_train), "Valid Size: %d" % len(X_test))
            
            base_acc = torch.mean(Y_all[:, 0])
            min_pow = torch.min(Y_all[:, 1])
            max_pow = torch.max(Y_all[:, 1])
            base_pow = torch.mean(Y_all[:, 1])
            torch.save({'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test, 'base_acc': base_acc, 'base_pow': base_pow, 'min_pow': min_pow, 'max_pow': max_pow}, self.acc_dataset_path)

        loaded = torch.load(self.acc_dataset_path)
        X_train = loaded['X_train']
        X_test = loaded['X_test']
        Y_train = loaded['Y_train']
        Y_test = loaded['Y_test']

        Y_train[:, 1] -= loaded['min_pow']
        Y_train[:, 1] /= (loaded['max_pow'] - loaded['min_pow'])
        Y_test[:, 1] -= loaded['min_pow']
        Y_test[:, 1] /= (loaded['max_pow'] - loaded['min_pow'])

        # build data loader
        train_dataset = RegDataset(X_train, Y_train)
        val_dataset = RegDataset(X_test, Y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=n_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
        )

        return train_loader, valid_loader, loaded['base_acc'], loaded['base_pow']
    
    def inverse_transform_power(self, items):
        if not hasattr(self, '_loaded_power_consts'):
            loaded = torch.load(self.acc_dataset_path)
            self._loaded_power_consts = loaded['min_pow'], loaded['max_pow']
        return items * (self._loaded_power_consts[1] - self._loaded_power_consts[0]) + self._loaded_power_consts[0]
    
    def transform_power(self, items):
        if not hasattr(self, '_loaded_power_consts'):
            loaded = torch.load(self.acc_dataset_path)
            self._loaded_power_consts = loaded['min_pow'], loaded['max_pow']
        return (items -  self._loaded_power_consts[0]) / (self._loaded_power_consts[1] - self._loaded_power_consts[0])
