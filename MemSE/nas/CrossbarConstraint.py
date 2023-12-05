from ortools.sat.python import cp_model
import torch
import time

NB_CROSSBARS=64
WIDTH_CROSSBARS=256
HEIGHT_CROSSBARS=256
CROSSBARS_CAPACITY=NB_CROSSBARS * WIDTH_CROSSBARS * HEIGHT_CROSSBARS

def dnn_to_memristor_shapes(dnn: torch.nn.Module):
    for module in dnn.modules():
        if isinstance(module, torch.nn.Linear):
            shape = module.weight.shape
            if module.bias is not None:
                shape[1] += 1
            yield {'w': shape[0], 'h': shape[1]}
        elif isinstance(module, torch.nn.Conv2d):
            shape = module.weight.shape
            w = shape[0]
            h = shape[1] * shape[2] * shape[3]
            if module.bias is not None:
                h += 1
            yield {'w': w, 'h': h}


def c1_solver(dnn):
    s = 0
    wl = 0
    height = True
    for d in dnn_to_memristor_shapes(dnn):
        s += d['w'] * d['h'] * 2
        wl += d['w'] * 2
        if d['h'] > HEIGHT_CROSSBARS:
            height = False
            break
    capa = s <= CROSSBARS_CAPACITY
    width = wl <= (WIDTH_CROSSBARS * NB_CROSSBARS)
    print(f'C1 checks {height=} {width=} {capa=}')
    return height and width and capa


class Tetristalloc(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()

    def solution_count(self):
        return self.__solution_count

    def on_solution_callback(self):
        current_time = time.time()
        print(
            f"Solution {self.__solution_count}, "
            f"time = {current_time - self.__start_time} s"
        )
        self.__solution_count += 1

        # TODO visual
        # all_queens = range(len(self.__queens))
        # for i in all_queens:
        #     for j in all_queens:
        #         if self.Value(self.__queens[j]) == i:
        #             # There is a queen in column j, row i.
        #             print("Q", end=" ")
        #         else:
        #             print("_", end=" ")
        #     print()
        # print()


def c2_solver(dnn):
    # https://yetanothermathprogrammingconsultant.blogspot.com/2021/02/2d-bin-packing-with-google-or-tools-cp.html
    # https://developers.google.com/optimization/cp/queens?hl=fr
    model = cp_model.CpModel()
    
    for i, d in enumerate(dnn_to_memristor_shapes(dnn)):
        # x and y
        x = model.NewIntVar(0, WIDTH_CROSSBARS - d['w'], f'x{i}')
        xb1 = model.NewIntVar(0,NB_CROSSBARS*WIDTH_CROSSBARS-d['w'],f'xb1.{i}')
        xb2 = model.NewIntVar(d['w'],NB_CROSSBARS*WIDTH_CROSSBARS,f'xb2.{i}')

        y1 = model.NewIntVar(0,HEIGHT_CROSSBARS-d['h'],f'y1.{i}')
        y2 = model.NewIntVar(d['h'],HEIGHT_CROSSBARS,f'y2.{i}')

        # interval variables
        xival = model.NewIntervalVar(xb1,d['w'],xb2,f'xival{i}')
        yival = model.NewIntervalVar(y1,d['h'],y2,f'yival{i}')

        # bin numbers
        b = model.NewIntVar(0,NB_CROSSBARS-1,f'b{i}')
        
        #
        # constraints
        #
        model.Add(xb1 == x + b*WIDTH_CROSSBARS)
        model.Add(xb2 == xb1 + d['w'])

        model.AddNoOverlap2D(xival,yival)
    
    # solve
    solver = cp_model.CpSolver()
    solution_printer = Tetristalloc()
    solver.parameters.enumerate_all_solutions = False
    solver.Solve(model, solution_printer)
    print("\nStatistics")
    print(f"  conflicts      : {solver.NumConflicts()}")
    print(f"  branches       : {solver.NumBranches()}")
    print(f"  wall time      : {solver.WallTime()} s")
    print(f"  solutions found: {solution_printer.solution_count()}")