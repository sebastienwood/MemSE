from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.mixed import MixedVariableMating
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, calc_crowding_distance, randomized_argsort

import math
import numpy as np
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.variable import Choice, Real, Integer


__all__ = ["RepairGmax", "CausalMixedVariableMating", "MemSEProblem", "NSGA2AdvanceCriterion"]


class RepairGmax(Repair):
    def _do(self, problem, arch_dict, **kwargs):
        for a in arch_dict:
            a_cc = problem.arch_encoder.cat_arch_vars(a)
            m = problem.model.gmax_masks[tuple(a_cc["d"])].numpy()
            for i in range(len(m)):
                if m[i] == 0:
                    a[f'gmax_{i}'] = 0.
                elif a[f'gmax_{i}'] == 0 and m[i] == 1:
                    a[f'gmax_{i}'] = problem.arch_encoder.default_gmax[i]
        return arch_dict


class CausalMixedVariableMating(MixedVariableMating):
    # FIXME would benefit from a general DAG solver to order recomb, here we just use it for Gmax
    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):

        # So far we assume all crossover need the same amount of parents and create the same number of offsprings
        XOVER_N_PARENTS = 2
        XOVER_N_OFFSPRINGS = 2

        # the variables with the concrete information
        vars = problem.vars

        # group all the variables by their types
        vars_by_type = {}
        for k, v in vars.items():
            clazz = type(v)

            if clazz not in vars_by_type:
                vars_by_type[clazz] = []
            vars_by_type[clazz].append(k)

        # # all different recombinations (the choices need to be split because of data types)
        recomb = []
        for clazz, list_of_vars in vars_by_type.items():
            if clazz == Choice:
                for e in list_of_vars:
                    recomb.append((clazz, [e]))
            else:
                recomb.append((clazz, list_of_vars))

        # create an empty population that will be set in each iteration
        off = Population.new(X=[{} for _ in range(n_offsprings)])

        if not parents:
            n_select = math.ceil(n_offsprings / XOVER_N_OFFSPRINGS)
            pop = self.selection(problem, pop, n_select, XOVER_N_PARENTS, **kwargs)

        for clazz, list_of_vars in recomb:

            crossover = self.crossover[clazz]
            assert crossover.n_parents == XOVER_N_PARENTS and crossover.n_offsprings == XOVER_N_OFFSPRINGS

            _parents = [[Individual(X=np.array([parent.X[var] for var in list_of_vars])) for parent in parents] for
                        parents in pop]

            _vars = [vars[e] for e in list_of_vars]
            _xl, _xu = None, None

            if clazz in [Real, Integer]:
                _xl, _xu = np.array([v.bounds for v in _vars]).T

            _problem = Problem(vars=_vars, xl=_xl, xu=_xu)

            _off = crossover(_problem, _parents, **kwargs)
            
            if 'gmax' in str(list_of_vars[0]):
                print(list_of_vars)
                print(_parents)
                print(_off)

            mutation = self.mutation[clazz]
            _off = mutation(_problem, _off, **kwargs) # post mutation is repair done by super().do

            for k in range(n_offsprings):
                for i, name in enumerate(list_of_vars):
                    off[k].X[name] = _off[k].X[i]

        return off


class MemSEProblem(Problem):
    def __init__(self, model, arch_encoder, batch_picker, surrogate=None, const: bool = False):
        super().__init__(vars=arch_encoder.arch_vars(const), n_obj=2)
        self.model = model
        self.arch_encoder = arch_encoder
        self.surrogate = surrogate
        self.const = const
        self.batch_picker = batch_picker
        self.set_n()
        if const:
            assert arch_encoder.default_gmax is not None

    def _evaluate(self, arch_dict, out, *args, **kwargs):
        arch_dict = [self.arch_encoder.cat_arch_vars(a) for a in arch_dict]
        out["F"] = self.evaluate_with_surrogate(arch_dict)
        
    def set_n(self):
        if not hasattr(self, 'delta'):
            self.delta = 5.0
        else:
            self.delta *= 0.9
        self.n = self.batch_picker.get_pareto_interp(self.delta)

    def evaluate_with_simulation(self, arch_dict):
        assert False
        # run_manager._loader.assign_active_img_size(image_size)
        # data_loader = run_manager._loader.build_sub_train_loader(
        #         n_images=2000, batch_size=200
        #     )
        # val_dataset = []
        # for batch in run_manager.valid_loader:
        #     if isinstance(batch, dict):
        #         images, labels = batch['image'], batch['label']
        #     else:
        #         images, labels = batch
        #     val_dataset.append((images, labels))
            
    def evaluate_with_surrogate(self, arch_dict):
        assert self.surrogate is not None
        with torch.no_grad():
            out = self.surrogate.predict_acc(arch_dict)
            out[:, 1] *= -1 # we are minimizing both f
        return out
    
    
class RankAndCrowdingSurvivalWithRejected(RankAndCrowdingSurvival):
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        
        rejected = list(set(np.arange(len(pop.shape[0]))) - set(survivors))
        return pop[survivors], pop[rejected]


class NSGA2AdvanceCriterion(NSGA2):
    def _advance(self, infills=None, **kwargs):
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        assert isinstance(self.survival, RankAndCrowdingSurvivalWithRejected)
        self.pop, rejected = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)
        
        pop_median = np.median(self.pop.get("F").astype(float, copy=False)[:, 1])
        rej_best = max(rejected.get("F").astype(float, copy=False)[:, 1])
        if pop_median - rej_best < self.problem.delta:
            self.problem.set_n()


if __name__ == '__main__':
    import torch
    from pymoo.optimize import minimize
    from MemSE.nas import ResNetArchEncoder, AccuracyPredictorFactory, BatchPicker
    from pymoo.core.mixed import MixedVariableSampling, MixedVariableDuplicateElimination
    from MemSE import ROOT
    from ofa.model_zoo import ofa_net
    from MemSE.nn import MemSE, OFAxMemSE

    save_path = ROOT / 'experiments/conference_2/results'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ofa = ofa_net('ofa_resnet50', pretrained=True)
    ofa.set_max_net()
    default_gmax = MemSE(ofa.get_active_subnet()).quanter.Wmax
    memse = OFAxMemSE(ofa)
    encoder = ResNetArchEncoder(default_gmax=default_gmax)
    predictor = AccuracyPredictorFactory['AccuracyPredictor'](encoder, device=device)
    loaded = torch.load(save_path / f'trained_AccuracyPredictor.pth', map_location=torch.device('cpu'))
    predictor.load_state_dict(loaded['net_dict'])
    predictor.eval() 
    
    problem = MemSEProblem(memse, encoder, BatchPicker(), surrogate=predictor)
    algorithm = NSGA2AdvanceCriterion(pop_size=5,
                      sampling=MixedVariableSampling(),
                      mating=CausalMixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination(), repair=RepairGmax()),
                      eliminate_duplicates=MixedVariableDuplicateElimination(),
                      repair=RepairGmax(),
                      survival=RankAndCrowdingSurvivalWithRejected()
                      )
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 3),
                   seed=1,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV = %s" % ([encoder.cat_arch_vars(a) for a in res.X], res.F, res.CV))