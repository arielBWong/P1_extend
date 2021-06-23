
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
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness, EI, MAF
from matplotlib.lines import Line2D
from matplotlib import cm
from joblib import dump, load

import os
import copy
import multiprocessing as mp
import pygmo as pg
from mpl_toolkits import mplot3d
from sort_population import sort_population_cornerfirst
from EI_problem import ego_believer
from paper1_refactor import get_paretofront, cornerplus_selectiveEvaluate, get_ndfront, get_ndfrontx


def cornersearch_onrealfunction(seed_index, target_problem, method_selection, search_ideal, max_eval, num_pop, num_gen, visual):

    enable_crossvalidation = False
    mp.freeze_support()
    np.random.seed(seed_index)

    target_problem = eval(target_problem)
    print('Problem %s, seed %d' % (target_problem.name(), seed_index))



    relayproblem = single_krg_optim.cornersearch_problem(target_problem)
    low = target_problem.xl
    up = target_problem.xu
    single_bounds = np.vstack((low, up)).T.tolist()

    plot_param = {'inner_problem': target_problem,
                 'visualize': True}

    pop_x, pop_f, _, _, _, _ = optimizer.optimizer_forcornersearch(relayproblem, relayproblem.n_obj, relayproblem.n_constr,
                                                                   single_bounds,
                                                                   0.2, 0.8, 200, 200, **plot_param)

    # check silhouette
    # pick top 6 from ND front
    pop_f = close_adjustment(pop_f)
    nd = get_ndfront(pop_f)
    ndx = get_ndfrontx(pop_x, pop_f)
    selected = sort_population_cornerfirst(nd.shape[0], relayproblem.n_obj, 0, [], [], [], nd, ndx)
    nd = nd[selected, :]
    ndx = ndx[selected, :]









if __name__ == "__main__":
    import json

    # problems_json = 'p/zdt_problems_hvnd.json'
    # problems_json = 'p/zdt_problems_hvndr.json'
    # problems_json = 'p/dtlz_problems_hvndr3.json'
    problems_json = 'p/maf_problems_hvndr3.json'

    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['MO_target_problems']
    method_selection = hyp['method_selection']
    search_ideal = hyp['search_ideal']
    search_ideal = 2

    max_eval = hyp['max_eval']
    num_pop = 200
    num_gen = 500

    target_problem = target_problems[12]
    method_selection = "normalization_with_nd"
    seed_index = 3
    visual = True
    cornersearch_onrealfunction(seed_index, target_problem, method_selection, search_ideal, max_eval, num_pop, num_gen, visual)


    print('0')