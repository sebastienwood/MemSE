import copy
import random
import numpy as np
import torch
from tqdm import tqdm

__all__ = ["EvolutionFinder"]


class EvolutionFinder:
    def __init__(self, surrogate, model, const_gmax:bool=False, **kwargs):
        self.surrogate = surrogate
        self.model= model
        self.const_gmax = const_gmax

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    @property
    def arch_manager(self):
        return self.surrogate.arch_encoder

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        while True:
            sample = self.arch_manager.random_sample_arch(self.model, const_gmax=self.const_gmax)
            with torch.no_grad():
                out = self.surrogate.predict_acc([sample]).cpu()
                efficiency, acc = out[1].item(), out[0].item()
            if efficiency <= constraint:
                return sample, efficiency, acc

    def mutate_sample(self, sample, constraint):
        while True:
            new_sample = copy.deepcopy(sample)

            self.arch_manager.mutate_resolution(new_sample, self.resolution_mutate_prob)
            self.arch_manager.mutate_arch(new_sample, self.arch_mutate_prob, self.model, const_gmax=self.const_gmax)

            with torch.no_grad():
                out = self.surrogate.predict_acc([new_sample]).cpu()
                efficiency, acc = out[1].item(), out[0].item()
            if efficiency <= constraint:
                return new_sample, efficiency, acc

    def crossover_sample(self, sample1, sample2, constraint):
        s1_gmax_mask = self.model.gmax_masks[tuple(sample1["d"])]
        s2_gmax_mask = self.model.gmax_masks[tuple(sample2["d"])]
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list) and key != "gmax":
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                elif key == "gmax":
                    # Based on new_sample["d"] inheritance from parents and crossover if gmax set on both
                    ns_gmax_masks = self.model.gmax_masks[tuple(new_sample["d"])]
                    for i in range(len(new_sample[key])):
                        r = random.choice([sample1["gmax"][i], sample2["gmax"][i]])
                        new_sample[key][i] = r * (s1_gmax_mask[i] and s2_gmax_mask[i]) + sample1["gmax"][i] * (ns_gmax_masks[i] and s1_gmax_mask[i] and not s2_gmax_mask[i]) + sample2["gmax"][i] * (ns_gmax_masks[i] and not s1_gmax_mask[i] and s2_gmax_mask[i])
                else:
                    for i in range(len(new_sample[key])):
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )

            with torch.no_grad():
                out = self.surrogate.predict_acc([new_sample]).cpu()
                efficiency, acc = out[1].item(), out[0].item()
            if efficiency <= constraint:
                return new_sample, efficiency, acc
            
    def is_valid_sample(self, sample):
        s_gmax_mask = self.model.gmax_masks[tuple(sample["d"])]
        assert sum(sample["gmax"] > 0) == sum(s_gmax_mask > 0)
        assert sum((s_gmax_mask * sample["gmax"]) > 0) == sum(s_gmax_mask > 0)

    def run_evolution_search(self, constraint, verbose=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        acc_pool = []
        best_info = None
        if verbose:
            print("Generate random population...")
        for i in range(self.population_size):
            sample, efficiency, acc = self.random_valid_sample(constraint)
            self.is_valid_sample(sample)
            child_pool.append(sample)
            efficiency_pool.append(efficiency)
            acc_pool.append(acc)

        for i in range(self.population_size):
            population.append((acc_pool[i], child_pool[i], efficiency_pool[i]))

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        with tqdm(
            total=self.max_time_budget,
            desc="Searching with constraint (%s)" % constraint,
            disable=(not verbose),
        ) as t:
            for i in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
                acc = parents[0][0]
                t.set_postfix({"acc": parents[0][0]})
                if not verbose and (i + 1) % 100 == 0:
                    print("Iter: {} Acc: {}".format(i + 1, parents[0][0]))

                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents
                child_pool = []
                efficiency_pool = []
                acc_pool = []

                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    # Mutate
                    new_sample, efficiency, acc = self.mutate_sample(par_sample, constraint)
                    self.is_valid_sample(new_sample)
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)
                    acc_pool.append(acc)

                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # Crossover
                    new_sample, efficiency, acc = self.crossover_sample(
                        par_sample1, par_sample2, constraint
                    )
                    self.is_valid_sample(new_sample)
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)
                    acc_pool.append(acc)

                for j in range(self.population_size):
                    population.append(
                        (acc_pool[j], child_pool[j], efficiency_pool[j])
                    )

                t.update(1)

        return best_valids, best_info