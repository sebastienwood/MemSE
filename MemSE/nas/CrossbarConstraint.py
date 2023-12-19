from ortools.sat.python import cp_model
import torch
import time
import math

NB_CROSSBARS=64
WIDTH_CROSSBARS=256
HEIGHT_CROSSBARS=256
CROSSBARS_CAPACITY=NB_CROSSBARS * WIDTH_CROSSBARS * HEIGHT_CROSSBARS

def weight_cutter(w, h, max_h, max_w, **kwargs):
    integer_w = w//max_w
    integer_h = h//max_h
    remain_w = w % max_w
    remain_h = h % max_h
    sub_idx = 0
    
    for _ in range(integer_w * integer_h):
        yield {'w': max_w, 'h': max_h, 'sub_idx': sub_idx, 'status': 'full', **kwargs}
        sub_idx += 1
    
    br = remain_w > 0 and remain_h > 0
    if br: # bottom right corner if it exists
        yield {'w': remain_w, 'h': remain_h, 'sub_idx': sub_idx, 'status': 'partial', **kwargs}
        sub_idx += 1
    
    h_last = (remain_h > 0) * integer_w
    if br and h_last != 0:
        h_last -= 1
    for _ in range(h_last):
        yield {'w': max_w, 'h': remain_h, 'sub_idx': sub_idx, 'status': 'partial', **kwargs}
        sub_idx += 1
        
    w_last = (remain_w > 0) * integer_h
    if br and w_last != 0:
        w_last -= 1
    for _ in range(w_last):
        yield {'w': remain_w, 'h': max_h, 'sub_idx': sub_idx, 'status': 'partial', **kwargs}
        sub_idx += 1
    # print('*'*5)
    # print(f'    Cut shape {w}x{h} into {sub_idx} submatrix')
    # print(f'        {max_w=} {max_h=}')
    # print(f'        {integer_w=} {integer_h=}')
    # print(f'        {remain_w=} {remain_h=}')
    # print(f'        {integer_w * integer_h=}')
    # print(f'        {br=}')
    # print(f'        {h_last=} {remain_h * integer_w=}')
    # print(f'        {w_last=} {remain_w * integer_h=}')
    # print('*'*5)


def dnn_to_memristor_shapes(dnn: torch.nn.Module, max_height:int=HEIGHT_CROSSBARS, max_width:int=WIDTH_CROSSBARS, cut:bool=True):
    for idx, module in enumerate(dnn.modules()):
        if isinstance(module, torch.nn.Linear):
            shape = module.weight.shape
            h = shape[1]
            if module.bias is not None:
                h += 1
            if cut:
                yield from weight_cutter(shape[0], h, max_height, max_width, layer_type='linear', layer_idx=idx)
            else:
                yield {'w': shape[0], 'h': h, 'layer_idx': idx, 'layer_type': 'linear'}
        elif isinstance(module, torch.nn.Conv2d):
            shape = module.weight.shape
            w = shape[0]
            h = shape[1] * shape[2] * shape[3]
            if module.bias is not None:
                h += 1
            if cut:
                yield from weight_cutter(w, h, max_height, max_width, layer_type='conv', layer_idx=idx)
            else:
                yield {'w': w, 'h': h, 'layer_idx': idx, 'layer_type': 'conv'}


def c1_solver(dnn, max_height:int=HEIGHT_CROSSBARS, max_width:int=WIDTH_CROSSBARS, nb_crossbars:int=NB_CROSSBARS, verbose:bool=False):
    s = 0
    for d in dnn_to_memristor_shapes(dnn, cut=False):
        s += d['w'] * d['h'] * 2
    capa = s <= (max_height * max_width * nb_crossbars)
    if verbose:
        print(f'C1 checks {capa=}')
        print(f'Capacity {s} required - {(max_height * max_width * nb_crossbars)} available')
    return capa


class Tetristalloc(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, xb1, y1, w, h):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()
        self.__xival = xb1
        self.__yival = y1
        self.__w = w
        self.__h = h
        self.__rectangles = []

    def solution_count(self):
        return self.__solution_count

    def on_solution_callback(self):
        current_time = time.time()
        print(
            f"Solution {self.__solution_count}, "
            f"time = {current_time - self.__start_time} s"
        )
        self.__solution_count += 1

        for idx in range(len(self.__xival)):
            x = self.Value(self.__xival[idx])
            y = self.Value(self.__yival[idx])
            self.__rectangles.append(((x,y), self.__w[idx], self.__h[idx]))


def c2_solver(dnn, max_height:int=HEIGHT_CROSSBARS, max_width:int=WIDTH_CROSSBARS, nb_crossbars:int=NB_CROSSBARS):
    # https://yetanothermathprogrammingconsultant.blogspot.com/2021/02/2d-bin-packing-with-google-or-tools-cp.html
    # https://developers.google.com/optimization/cp/queens?hl=fr
    model = cp_model.CpModel()
    
    xival_, yival_ = [], []
    xb1_, y1_, w_, h_ = [], [], [], []
    
    # First pass count full crossbars and remove them from total
    full_crossbars = 0
    partial_crossbars = 0
    for d in dnn_to_memristor_shapes(dnn, max_height=max_height, max_width=max_width):
        if d['status'] == 'full':
            full_crossbars += 1
        else:
            partial_crossbars += 1
    print(f'Preallocated {full_crossbars} full crossbars (x2 for negatives)')
    print(f'Now allocating {partial_crossbars} partial weights (x2 for negatives)')
    nb_crossbars -= full_crossbars
    
    for d in dnn_to_memristor_shapes(dnn, max_height=max_height, max_width=max_width):
        if d['status'] == 'full': # already allocated
            continue
        
        i = f'{d["layer_idx"]}'
        if 'sub_idx' in d:
            i = f'{i}.{d["sub_idx"]}' 
        # x and y
        x = model.NewIntVar(0, max_width - d['w'], f'x{i}')
        xb1 = model.NewIntVar(0,nb_crossbars*max_width-d['w'],f'xb1.{i}')
        xb2 = model.NewIntVar(d['w'],nb_crossbars*max_width,f'xb2.{i}')

        y1 = model.NewIntVar(0,max_height-d['h'],f'y1.{i}')
        y2 = model.NewIntVar(d['h'],max_height,f'y2.{i}')

        # interval variables
        xival = model.NewIntervalVar(xb1,d['w'],xb2,f'xival{i}')
        yival = model.NewIntervalVar(y1,d['h'],y2,f'yival{i}')
        xival_.append(xival)
        yival_.append(yival)

        # bin numbers
        b = model.NewIntVar(0,nb_crossbars-1,f'b{i}')
        
        #
        # constraints
        #
        model.Add(xb1 == x + b*max_width)
        model.Add(xb2 == xb1 + d['w'])
        
        xb1_.append(xb1)
        y1_.append(y1)
        w_.append(d['w'])
        h_.append(d['h'])

    model.AddNoOverlap2D(xival_,yival_)
    
    # solve
    solver = cp_model.CpSolver()
    solution_printer = Tetristalloc(xb1_, y1_, w_, h_)
    solver.parameters.enumerate_all_solutions = False
    solver.Solve(model, solution_printer)
    print("\nStatistics")
    print(f"  conflicts      : {solver.NumConflicts()}")
    print(f"  branches       : {solver.NumBranches()}")
    print(f"  wall time      : {solver.WallTime()} s")
    print(f"  solutions found: {solution_printer.solution_count()}")
    return solution_printer