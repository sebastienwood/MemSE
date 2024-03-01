from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.mixed import MixedVariableMating
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, calc_crowding_distance, randomized_argsort

import math
import torch
import numpy as np
import copy
import json
from MemSE.nas import DataloadersHolder, ResNetArchEncoder
from MemSE.training import RunManager
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.variable import Choice, Real, Integer


__all__ = ["RepairGmax", "CausalMixedVariableMating", "MemSEProblem", "NSGA2AdvanceCriterion", "RankAndCrowdingSurvivalWithRejected", "EvaluatorBatch"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class RepairGmax(Repair):
    def _do(self, problem, arch_dict, **kwargs):
        # Handle const. case
        if problem.const: # or problem.unique_gmax:
            return arch_dict
        
        for a in arch_dict:
            d = problem.arch_encoder.cat_d_vars(a)
            m = problem.model.gmax_masks[tuple(d)].numpy()
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
            
            if 'gmax' in list_of_vars[0]: #and not problem.unique_gmax: # treating the vector gmax if it exists
                for cple_id, couple in enumerate(_parents):
                    par = pop[cple_id]
                    m1 = problem.model.gmax_masks[tuple(problem.arch_encoder.cat_d_vars(par[0].X))].numpy().astype(bool)
                    m2 = problem.model.gmax_masks[tuple(problem.arch_encoder.cat_d_vars(par[1].X))].numpy().astype(bool)
                    g1 = couple[0].X
                    g2 = couple[1].X
                    for chld_id, children in enumerate(_off[XOVER_N_OFFSPRINGS*cple_id:XOVER_N_OFFSPRINGS*(cple_id+1)]):
                        mm = problem.model.gmax_masks[tuple(problem.arch_encoder.cat_d_vars(off[chld_id].X))].numpy().astype(bool)
                        gm = children.X
                        new_gm = (m1 & m2) * gm + (m1 & ~m2 & mm) * g1 + (~m1 & m2 & mm) * g2
                        children.X = new_gm
                        

            mutation = self.mutation[clazz]
            _off = mutation(_problem, _off, **kwargs)

            for k in range(n_offsprings):
                for i, name in enumerate(list_of_vars):
                    off[k].X[name] = _off[k].X[i]

        return off


class MemSEProblem(Problem):
    def __init__(self,
                 model,
                 arch_encoder: ResNetArchEncoder,
                 batch_picker,
                 run_manager: RunManager = None,
                 dataholder: DataloadersHolder=None,
                 surrogate=None,
                 simulate=False,
                 const: bool = False,
                 constrained: float = 0.0,
                 delta_schedule: float = 0.95,
                 #unique_gmax: bool = False
                ):
        if constrained > 0:
            objs = {
                'n_obj': 1,
                'n_constr': 1
            }
        else:
            objs = {'n_obj': 2}
        super().__init__(vars=arch_encoder.arch_vars(const=const), **objs)#, unique_gmax=unique_gmax), **objs)
        self.constrained = constrained
        self.model = model
        self.arch_encoder = arch_encoder
        self.surrogate = surrogate
        self.const = const
        #self.unique_gmax = unique_gmax
        self.batch_picker = batch_picker
        self.dataholder = dataholder
        self.run_manager = run_manager
        self.simulate = simulate
        self.delta_schedule = delta_schedule
        self.set_n()
        if const: #or unique_gmax:
            assert arch_encoder.default_gmax is not None

    def _evaluate(self, arch_dict, out, *args, **kwargs):
        arch_dict = [self.arch_encoder.cat_arch_vars(a, self.model.gmax_masks) for a in arch_dict]
        if self.simulate:
            res = self.evaluate_with_simulation(arch_dict) # [:, 0] prec [:, 1] pow
        else:
            res = self.evaluate_with_surrogate(arch_dict) # [:, 0] prec [:, 1] pow
        if self.constrained > 0:
            out["F"] = res[:, 0]
            out["G"] = res[:, 1] - self.constrained
        else:
            out["F"] = res
        
    def set_n(self, pop_size=None):
        if not hasattr(self, 'delta'):
            self.delta = 5.0
        else:
            self.delta *= self.delta_schedule
        self.n = self.batch_picker.get_pareto_interp(self.delta)
        
        if not hasattr(self, 'pop_size_init') and pop_size is not None:
            self.pop_size_init = pop_size
        return pop_size # TODO reduce the size dynamically

    def evaluate_with_simulation(self, arch_dict):
        out = torch.zeros(len(arch_dict), 2)
        group_ad = {} # arch wo gmax -> idx map
        for idx, a in enumerate(arch_dict):
            a_ = copy.deepcopy(a)
            a_.pop('gmax')
            a_ = json.dumps(a_, cls=NpEncoder)
            if a_ not in group_ad:
                group_ad[a_] = []
            group_ad[a_].append(idx)
        print('Simulation BN efficiency -> ', len(group_ad)/len(arch_dict))
        
        with torch.no_grad():
            for idxes in group_ad.values():
                for i, idx in enumerate(idxes):
                    a = arch_dict[idx]
                    if i == 0:
                        train_ld, eval_ld = self.dataholder.get_image_size(int(a["image_size"]))
                        self.model.set_active_subnet(a, train_ld) # TODO could save and load with LRU cache per arch
                    else:
                        self.model._model.quanter.Gmax = [gu for gu in a['gmax'] if gu > 0]
                        self.model._state = a
                    self.model.quant(scaled=False)
                    _, metrics = self.run_manager.validate(
                                        net=self.model,
                                        data_loader=eval_ld,
                                        no_logs=True,
                                        nb_batchs=self.n,
                                        nb_batchs_power=1
                                    )
                    self.model.unquant()
                    out[idx, 0] = - metrics.top1.avg
                    out[idx, 1] = metrics.power.avg
        
        # with torch.no_grad():
        #     for idx, a in enumerate(arch_dict):
        #         train_ld, eval_ld = self.dataholder.get_image_size(int(a["image_size"]))
        #         self.model.set_active_subnet(a, train_ld) # TODO could save and load with LRU cache per arch
        #         self.model.quant(scaled=False)
        #         _, metrics = self.run_manager.validate(
        #                             net=self.model,
        #                             data_loader=eval_ld,
        #                             no_logs=True,
        #                             nb_batchs=self.n,
        #                             nb_batchs_power=1
        #                         )
        #         self.model.unquant()
        #         out[idx, 0] = - metrics.top1.avg
        #         out[idx, 1] = metrics.power.avg
        return out.cpu().numpy()
            
    def evaluate_with_surrogate(self, arch_dict):
        assert self.surrogate is not None
        with torch.no_grad():
            out = self.surrogate.predict_acc(arch_dict)
            out[:, 0] *= -1 # we are minimizing both f with pymoo
        return out.cpu().numpy()
    
    
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
        
        rejected = list(set(np.arange(pop.shape[0])) - set(survivors))
        return pop[survivors], pop[rejected]


class NSGA2AdvanceCriterion(NSGA2):
    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop, _ = self.survival.do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)
    
    def _advance(self, infills=None, **kwargs):
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)
            
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # execute the survival to find the fittest solutions
        assert isinstance(self.survival, RankAndCrowdingSurvivalWithRejected)
        self.pop, rejected = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)
        
        pop_median = np.median(self.pop.get("F").astype(float, copy=False)[:, 0]) # median of bests
        rej_best = min(rejected.get("F").astype(float, copy=False)[:, 0]) # best of worsts
        if abs(pop_median) - abs(rej_best) < self.problem.delta:
            print(f'Based on criterion {abs(pop_median)=}-{abs(rej_best)=} < {self.problem.delta} reducing delta')
            new_pop_size = self.problem.set_n(self.pop_size)
            self.pop_size = new_pop_size
            

class EvaluatorBatch(Evaluator):
    def eval(self,
             problem: Problem,
             pop: Population,
             skip_already_evaluated: bool = None,
             evaluate_values_of: list = None,
             count_evals: bool = True,
             **kwargs):

        # load the default settings from the evaluator object if not already provided
        evaluate_values_of = self.evaluate_values_of if evaluate_values_of is None else evaluate_values_of
        skip_already_evaluated = self.skip_already_evaluated if skip_already_evaluated is None else skip_already_evaluated

        # check the type of the input
        is_individual = isinstance(pop, Individual)

        # make sure the object is a population
        if is_individual:
            pop = Population().create(pop)

        # filter the index to have individual where not all attributes have been evaluated
        if skip_already_evaluated:
            I = []
            for i, ind in enumerate(pop):
                n_not_computed = problem.simulate and (ind.get('n_batch') is None or ind.get('n_batch') < problem.n)
                if not all([e in ind.evaluated for e in evaluate_values_of]) or n_not_computed:
                    I.append(i)
            #I = [i for i, ind in enumerate(pop) if not all([e in ind.evaluated for e in evaluate_values_of])]
        # if skipping is deactivated simply make the index being all individuals
        else:
            I = np.arange(len(pop))

        # evaluate the solutions (if there are any)
        if len(I) > 0:

            # do the actual evaluation - call the sub-function to set the corresponding values to the population
            self._eval(problem, pop[I], evaluate_values_of, **kwargs)

        # update the function evaluation counter
        if count_evals:
            self.n_eval += len(I)

        # allow to have a callback registered
        if self.callback:
            self.callback(pop)

        if is_individual:
            return pop[0]
        else:
            return pop

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):

        # get the design space value from the individuals
        X = pop.get("X")
        # TODO possible marginal improvement: pass n_batch and just online update of metrics rather than from scratch
        # FIXME profiling results are weird for this case

        # call the problem to evaluate the solutions
        out = problem.evaluate(X, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs)

        # for each of the attributes set it to the problem
        for key, val in out.items():
            if val is not None:
                pop.set(key, val)
        pop.set('n_batch', problem.n)

        # finally set all the attributes to be evaluated for all individuals
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))


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
    
    problem = MemSEProblem(memse, encoder, batch_picker=BatchPicker(), surrogate=predictor)
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