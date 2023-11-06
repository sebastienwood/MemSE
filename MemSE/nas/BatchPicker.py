import numpy as np
from MemSE import ROOT
from itertools import combinations


__all__ = ['BatchPicker']


class AverageMeter:
    def __init__(self):
        self.hit, self.total = 0, 0
        
    def __repr__(self):
        return str(self.average())
        
    def average(self):
        return self.hit / self.total
    
class BatchPicker:
    def __init__(self, prob:float=0.99) -> None:
        self.prob = prob
        acc_convergence = np.load(ROOT / 'experiments/conference_2/results/acc_convergence.npy')
        self.N = acc_convergence.shape[1]
        self.acc_convergence_cumavg = np.einsum('ij,j->ij', np.cumsum(acc_convergence, 1), 1/np.arange(1, acc_convergence.shape[1]+1))
        self.delta_holder = {}

    def compute_pr(self, delta:float, N: int = None) -> dict[AverageMeter]:
        if N is None:
            N = range(self.N)
        else:
            assert isinstance(N, int)
            N = [N]
        average = {}
        for (k1, k2) in combinations(range(self.acc_convergence_cumavg.shape[0]), 2):
            k1_acc, k2_acc = self.acc_convergence_cumavg[k1], self.acc_convergence_cumavg[k2]
            if k1_acc[-1] > k2_acc[-1] + max(0, delta):
                for n in N: # for each minibatch pick
                    if not n in average:
                        average[n] = AverageMeter()
                    
                    if k1_acc[n] > k2_acc[n]:
                        average[n].hit += 1
                    average[n].total += 1
        return average

    def compute_delta(self, prob:float=0.99):
        if prob not in self.delta_holder:
            delta = []
            for n in range(self.N):
                delta_i = 5.
                cp_delta = delta_i
                for i in range(50):
                    avg = self.compute_pr(delta_i, n)
                    if n in avg and avg[n].average() < prob: # delta is a valid choice and avg is populated
                        delta_i += cp_delta / (i + 1)
                    elif n not in avg or avg[n].average() > prob: # delta is too high of a constraint
                        delta_i -= cp_delta / (i + 1)
                    elif n in avg and avg[n].average() == prob:
                        break
                delta.append(max(0.,delta_i))
            self.delta_holder[prob] = delta
        return self.delta_holder[prob]
    
    def get_pareto_frontier(self, prob: float):
        Ys = list(range(self.N))
        Xs = self.compute_delta(prob)

        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=False)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
        pf_X = [pair[0] for pair in pareto_front]
        pf_Y = [pair[1] for pair in pareto_front]
        return pf_X, pf_Y

    def get_pareto_interp(self, delta:float):
        pf_d, pf_N = self.get_pareto_frontier(self.prob)
        x_1 = 0
        for i in range(len(pf_d)):
            if pf_d[i] > delta:
                x_1 = max(i-1, 0)
                break
        x_2 = pf_d[x_1 + 1]
        y_1 = pf_N[x_1]
        y_2 = pf_N[x_1 + 1]
        x_1 = pf_d[x_1]
        return y_1 + (delta - x_1) * (y_2 - y_1) / (x_2 - x_1)