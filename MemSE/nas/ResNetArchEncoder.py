from ofa.imagenet_classification.networks import ResNets
import numpy as np
import random


class ResNetArchEncoder:
    def __init__(
        self,
        default_gmax=None,
        image_size_list=None,
        depth_list=None,
        expand_list=None,
        width_mult_list=None,
        base_depth_list=None,
        nb_crossbars=None,
    ) -> None:
        self.default_gmax = default_gmax
        self.image_size_list = [128, 160, 192, 224] if image_size_list is None else image_size_list
        self.expand_list = [0.2, 0.25, 0.35] if expand_list is None else expand_list
        self.depth_list = [0, 1, 2] if depth_list is None else depth_list
        self.width_mult_list = (
            [0.65, 0.8, 1.0] if width_mult_list is None else width_mult_list
        )

        self.base_depth_list = (
            ResNets.BASE_DEPTH_LIST if base_depth_list is None else base_depth_list
        )
        self.nb_crossbars = 62 if nb_crossbars is None else nb_crossbars
        if self.default_gmax is not None:
            assert self.nb_crossbars == len(self.default_gmax)

        """" build info dict """
        self.n_dim = 0
        # resolution
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r")
        # input stem skip
        self.input_stem_d_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="input_stem_d")
        # width_mult
        self.width_mult_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="width_mult")
        # expand ratio
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="e")
        # gmax
        self.gmax_info = dict(L=[], R=[])
        self._build_info_dict(target="gmax")

    @property
    def n_stage(self):
        return len(self.base_depth_list)

    @property
    def max_n_blocks(self):
        return sum(self.base_depth_list) + self.n_stage * max(self.depth_list)

    def _build_info_dict(self, target):
        if target == "r":
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for img_size in self.image_size_list:
                target_dict["val2id"][img_size] = self.n_dim
                target_dict["id2val"][self.n_dim] = img_size
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "input_stem_d":
            target_dict = self.input_stem_d_info
            target_dict["L"].append(self.n_dim)
            for skip in [0, 1]:
                target_dict["val2id"][skip] = self.n_dim
                target_dict["id2val"][self.n_dim] = skip
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "e":
            target_dict = self.e_info
            choices = self.expand_list
            for i in range(self.max_n_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for e in choices:
                    target_dict["val2id"][i][e] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = e
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)
        elif target == "width_mult":
            target_dict = self.width_mult_info
            choices = list(range(len(self.width_mult_list)))
            for i in range(self.n_stage + 2):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for w in choices:
                    target_dict["val2id"][i][w] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = w
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)
        elif target == "gmax":
            target_dict = self.gmax_info
            target_dict["L"].append(self.n_dim)
            target_dict["R"].append(self.n_dim + self.nb_crossbars)
            self.n_dim += self.nb_crossbars

    def arch2feature(self, arch_dict):
        d, e, w, r, g = (
            arch_dict["d"],
            arch_dict["e"],
            arch_dict["w"],
            arch_dict["image_size"],
            arch_dict["gmax"],
        )
        input_stem_skip = 1 if d[0] > 0 else 0
        d = d[1:]

        feature = np.zeros(self.n_dim)
        feature[self.r_info["val2id"][r]] = 1
        feature[self.input_stem_d_info["val2id"][input_stem_skip]] = 1
        for i in range(self.n_stage + 2):
            feature[self.width_mult_info["val2id"][i][w[i]]] = 1

        start_pt = 0
        for i, base_depth in enumerate(self.base_depth_list):
            depth = base_depth + d[i]
            for j in range(start_pt, start_pt + depth):
                feature[self.e_info["val2id"][j][e[j]]] = 1
            start_pt += max(self.depth_list) + base_depth

        feature[self.gmax_info["L"][0]:self.gmax_info["R"][0]] = g

        return feature

    def feature2arch(self, feature):
        img_sz = self.r_info["id2val"][
            int(np.argmax(feature[self.r_info["L"][0] : self.r_info["R"][0]])) + self.r_info["L"][0]
        ]
        input_stem_skip = (
            self.input_stem_d_info["id2val"][
                int(
                    np.argmax(
                        feature[
                            self.input_stem_d_info["L"][0] : self.input_stem_d_info[
                                "R"
                            ][0]
                        ]
                    )
                )
                + self.input_stem_d_info["L"][0]
            ]
            * 2
        )
        assert img_sz in self.image_size_list
        arch_dict = {"d": [input_stem_skip], "e": [], "w": [], "image_size": img_sz}

        for i in range(self.n_stage + 2):
            arch_dict["w"].append(
                self.width_mult_info["id2val"][i][
                    int(
                        np.argmax(
                            feature[
                                self.width_mult_info["L"][i] : self.width_mult_info[
                                    "R"
                                ][i]
                            ]
                        )
                    )
                    + self.width_mult_info["L"][i]
                ]
            )

        d = 0
        skipped = 0
        stage_id = 0
        for i in range(self.max_n_blocks):
            skip = True
            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["e"].append(self.e_info["id2val"][i][j])
                    skip = False
                    break
            if skip:
                arch_dict["e"].append(0)
                skipped += 1
            else:
                d += 1

            if (
                i + 1 == self.max_n_blocks
                or (skipped + d)
                % (max(self.depth_list) + self.base_depth_list[stage_id])
                == 0
            ):
                arch_dict["d"].append(d - self.base_depth_list[stage_id])
                d, skipped = 0, 0
                stage_id += 1

        arch_dict["gmax"] = feature[self.gmax_info["L"]:self.gmax_info["R"]]
        return arch_dict

    def random_sample_arch(self, model, const_gmax:bool=False) -> dict:
        assert self.default_gmax is not None
        d = [random.choice([0, 2])] + random.choices(self.depth_list, k=self.n_stage)
        arch = {
            "d": d,
            "e": random.choices(self.expand_list, k=self.max_n_blocks),
            "w": random.choices(
                list(range(len(self.width_mult_list))), k=self.n_stage + 2
            ),
            "image_size": random.choice(self.image_size_list),
            "gmax": (np.random.uniform(low=0.5, high=1.5, size=self.nb_crossbars) if not const_gmax else 1.) * self.default_gmax.cpu().numpy() * model.gmax_masks[tuple(d)].numpy()
        }
        return arch

    def arch_vars(self, const:bool=False) -> dict:
        from pymoo.core.variable import Real, Choice
        arch_vars = {
            "d_0": Choice(options=[0, 2]),
            "image_size": Choice(options=self.image_size_list),
        }
        for i in range(self.n_stage):
            arch_vars[f"d_{i+1}"] = Choice(options=self.depth_list)
        for i in range(self.max_n_blocks):
            arch_vars[f"e_{i}"] = Choice(options=self.expand_list)
        for i in range(self.n_stage + 2):
            arch_vars[f"w_{i}"] = Choice(options=list(range(len(self.width_mult_list))))
        if not const:
            d_gmax = self.default_gmax.cpu()
            # if unique_gmax:
            #     arch_vars[f"gmax"] = Real(bounds=(0.5, 1.5))
            # else:
            for i in range(self.nb_crossbars):
                arch_vars[f"gmax_{i}"] = Real(bounds=(0.5 * d_gmax[i].item(), 1.5 * d_gmax[i].item()))
        return arch_vars
    
    def cat_d_vars(self, arch_vars: dict) -> list:
        d = []
        for i in range(self.n_stage + 1):
            d.append(arch_vars[f"d_{i}"])
        return d

    def cat_arch_vars(self, arch_vars: dict, gmax_masks: dict = None) -> dict:
        arch_vars_res = {'image_size': arch_vars['image_size']}
        arch_vars_res["d"] = self.cat_d_vars(arch_vars)
        e = []
        for i in range(self.max_n_blocks):
            e.append(arch_vars[f"e_{i}"])
        arch_vars_res["e"] = e
        w = []
        for i in range(self.n_stage + 2):
            w.append(arch_vars[f"w_{i}"])
        arch_vars_res["w"] = w
        
        if "gmax_0" in arch_vars: # multiple gmax
            gmax = np.zeros_like(self.default_gmax.cpu().numpy())
            for i in range(self.nb_crossbars):
                gmax[i] = arch_vars[f"gmax_{i}"]
        # elif "gmax" in arch_vars: # unique gmax
        #     gmax = np.copy(self.default_gmax.cpu().numpy()) * arch_vars['gmax'] * gmax_masks[tuple(arch_vars_res["d"])].numpy()
        else: # no gmax (const)
            assert gmax_masks is not None
            gmax = np.copy(self.default_gmax) * gmax_masks[tuple(arch_vars_res["d"])].numpy()
        arch_vars_res["gmax"] = gmax
        return arch_vars_res

    def mutate_resolution(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["image_size"] = random.choice(self.image_size_list)
        return arch_dict

    def mutate_arch(self, arch_dict, mutate_prob, model, const_gmax:bool=False):
        # input stem skip
        if random.random() < mutate_prob:
            arch_dict["d"][0] = random.choice([0, 2])
        # depth
        for i in range(1, len(arch_dict["d"])):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)

        # width_mult
        for i in range(len(arch_dict["w"])):
            if random.random() < mutate_prob:
                arch_dict["w"][i] = random.choice(
                    list(range(len(self.width_mult_list)))
                )
        # expand ratio
        for i in range(len(arch_dict["e"])):
            if random.random() < mutate_prob:
                arch_dict["e"][i] = random.choice(self.expand_list)

        # gmax
        # if 0 -> set to random from default variation
        # else: random mutation
        for i in range(len(arch_dict["gmax"])):
            r = np.random.uniform(0.5, 1.5) if not const_gmax else 1.
            if arch_dict["gmax"][i] == 0:
                arch_dict["gmax"][i] = r * self.default_gmax.cpu().numpy()[i]
            elif random.random() < mutate_prob:
                arch_dict["gmax"][i] = r * arch_dict["gmax"][i]

        # set to 0 if unused
        #gmax = model.set_active_subnet(arch_dict, skip_adaptation=True)
        arch_dict['gmax'] = np.clip(arch_dict["gmax"], 0.5 * self.default_gmax.cpu().numpy(), 1.5 * self.default_gmax.cpu().numpy())
        arch_dict['gmax'] *= model.gmax_masks[tuple(arch_dict["d"])].numpy()
