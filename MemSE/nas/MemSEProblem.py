from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.mixed import MixedVariableMating

__all__ = ["RepairGmax", "RepairDependantVariableMating", "MemSEProblem"]

class RepairGmax(Repair):
    def _do(self, problem, arch_dict, **kwargs):
        for a in arch_dict:
            a_cc = problem.arch_encoder.cat_arch_vars(a)
            m = problem.model.gmax_masks[tuple(a_cc["d"])].numpy()
            for i in range(len(m)):
                a[f'gmax_{i}'] *= m[i] 
        return arch_dict


class RepairDependantVariableMating(Repair):
    # TODO just for gmax now, could be variabilized
    def _do(self, problem, arch_dict, **kwargs):
        return arch_dict


class MemSEProblem(Problem):
    def __init__(self, model, arch_encoder, surrogate=None, const: bool = False):
        super().__init__(vars=arch_encoder.arch_vars(const), n_obj=2)
        self.model = model
        self.arch_encoder = arch_encoder
        self.surrogate = surrogate
        self.const = const
        if const:
            assert arch_encoder.default_gmax is not None

    def _evaluate(self, arch_dict, out, *args, **kwargs):
        arch_dict = [self.arch_encoder.cat_arch_vars(a) for a in arch_dict]
        out["F"] = self.evaluate_with_surrogate(arch_dict)

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


if __name__ == '__main__':
    import torch
    from pymoo.optimize import minimize
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from MemSE.nas import ResNetArchEncoder, AccuracyPredictorFactory
    from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
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
    
    
    problem = MemSEProblem(memse, encoder, predictor)
    algorithm = NSGA2(pop_size=5,
                      sampling=MixedVariableSampling(),
                      mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination(), repair=RepairDependantVariableMating()),
                      eliminate_duplicates=MixedVariableDuplicateElimination(),
                      repair=RepairGmax())
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 3),
                   seed=1,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s\nCV = %s" % ([encoder.cat_arch_vars(a) for a in res.X], res.F, res.CV))