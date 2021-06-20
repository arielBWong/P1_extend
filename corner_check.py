import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI, optimizer
# from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
#    DTLZ1, DTLZ2, DTLZ3, DTLZ4, \
#     BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
import pyDOE
from cross_val_hyperp import cross_val_krg
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness, EI
from matplotlib.lines import Line2D
from matplotlib import cm
from joblib import dump, load

import os
import copy
import multiprocessing as mp
import pygmo as pg
from mpl_toolkits import mplot3d



if __name__ == "__main__":
    # redefine a search problem with objective being corner search
    n_sur_cons = 0
    np.random.seed(1)

    prob = WFG.WFG_9(n_var=6, n_obj=3, K=4)
    prob = DTLZs.DTLZ7(n_var=6, n_obj=3)
    new_problem = single_krg_optim.cornersearch_problem(prob)
    low = new_problem.xl
    up = new_problem.xu
    single_bounds = np.vstack((low, up)).T.tolist()


    guide = None

    param = {'prob_': prob}

    pop_x, pop_f, _, _, _, _ = optimizer.optimizer_forcornersearch(new_problem, new_problem.n_obj, new_problem.n_constr,
                                                                   single_bounds,
                                                                   0.2, 0.8, 100, 100, **param)
