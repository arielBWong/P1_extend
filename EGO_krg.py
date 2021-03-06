import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop.factory import get_problem_from_func
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
    DTLZ1, DTLZ2, \
    BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances
import pyDOE
from cross_val_hyperp import cross_val_krg
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness

import os
import copy
import multiprocessing as mp
import pygmo as pg

from pymop.factory import get_uniform_weights
import EI_krg
from copy import deepcopy

import EI_problem


def return_current_extreme(train_x, train_y):
    '''
    :param train_x:
    :param train_y:
    :return: select n_obj x varibles for guide of each local search
    '''
    best_index = np.argmin(train_y, axis=0)
    guide_x = train_x[best_index, :]
    return guide_x


def saveNameConstr(problem_name, seed_index, method, run_signature):
    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + problem_name + '_' + run_signature
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)
    # else:
    # os.mkdir(result_folder)
    savename_x = result_folder + '\\best_x_seed_' + str(seed_index) + '_' + method + '.joblib'
    savename_y = result_folder + '\\best_f_seed_' + str(seed_index) + '_' + method + '.joblib'
    savename_FEs = result_folder + '\\FEs_seed_' + str(seed_index) + '_' + method + '.joblib'
    return savename_x, savename_y, savename_FEs


def lexsort_with_certain_row(f_matrix, target_row_index):
    '''
    problematic function, given lexsort, it does not matter how upper
    rows are shuffled, sort is according to the last row.
    sort matrix according to certain row in fact last row
    e.g. sort the last row, the rest rows move its elements accordingly
    however, the matrix except last row is also sorted row wise
    according to number of min values each row has
    '''

    # f_matrix should have the size of [n_obj * popsize]
    # determine min
    target_row = f_matrix[target_row_index, :].copy()
    f_matrix = np.delete(f_matrix, target_row_index, axis=0)  # delete axis is opposite to normal

    f_min = np.min(f_matrix, axis=1)
    f_min = np.atleast_2d(f_min).reshape(-1, 1)
    # according to np.lexsort, put row with largest min values last row
    f_min_count = np.count_nonzero(f_matrix == f_min, axis=1)
    f_min_accending_index = np.argsort(f_min_count)
    # adjust last_f_pop
    last_f_pop = f_matrix[f_min_accending_index, :]

    # add saved target
    last_f_pop = np.vstack((last_f_pop, target_row))

    # apply np.lexsort (works row direction)
    lexsort_index = np.lexsort(last_f_pop)
    # print(last_f_pop[:, lexsort_index])
    selected_x_index = lexsort_index[0]

    return selected_x_index


def lexsort_specify_baserow(matrix_in, row_index):
    '''
    this function sort matrix with
    lexicongraphic sort according to the row specified by row index
    it then returns a index pointing to smallest value of base row(row_index)
    if there is multiple smallest values, it returns the index where
    other row upper this values is the largest
    example:
    [ 1, 4, 7, 9, 0
      0, 0 ,0, 1, 2]
    return index is 2 (input row_index= 1, sort according to last row)
    '''

    matrix = copy.deepcopy(matrix_in)
    matrix = np.around(matrix, 16)
    target_row = matrix[row_index, :].copy()
    # how many min values in target row
    rowmin = np.min(target_row)
    n_min = np.count_nonzero(target_row == rowmin)

    matrix = np.delete(matrix, row_index, axis=0)  # delete axis is opposite to normal

    # prepare matrix to use np.lexsort
    matrix = np.vstack((matrix, target_row))

    # apply np.lexsort
    lexsort_index = np.lexsort(matrix)
    # return rule, largest value in other row
    selected_x_index = lexsort_index[n_min - 1]
    return selected_x_index


def check_krg_ideal_points(krg, n_var, n_constr, n_obj, low, up, guide_x):
    '''This function uses  krging model to search for a better x
    krg(list): krging model
    n_var(int): number of design variable for kriging
    n_constr(int): number of constraints
    n_obj(int): number of objective function
    low(list):
    up(list)
    guide_x(row vector): starting point to insert to initial population
    '''
    x_krg = []
    f_krg = []

    last_x_pop = []
    last_f_pop = []

    n_krg = len(krg)
    x_pop_size = 100
    x_pop_gen = 100
    add_info = {}

    # identify ideal x and f for each objective
    for k_i, k in enumerate(krg):
        problem = single_krg_optim.single_krg_optim(k, n_var, n_constr, 1, low, up)
        single_bounds = np.vstack((low, up)).T.tolist()

        guide = guide_x[k_i, :]
        # add_info['guide'] = guide

        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, record = optimizer_EI.optimizer(problem,
                                                                                              nobj=1,
                                                                                              ncon=0,
                                                                                              bounds=single_bounds,
                                                                                              recordFlag=False,
                                                                                              pop_test=None,
                                                                                              mut=0.1,
                                                                                              crossp=0.9,
                                                                                              popsize=x_pop_size,
                                                                                              its=x_pop_gen,
                                                                                              add_info=guide
                                                                                              )
        # save the last population for lexicon sort
        last_x_pop = np.append(last_x_pop, pop_x)
        last_f_pop = np.append(last_f_pop, pop_f)  # var for test

    # long x
    last_x_pop = np.atleast_2d(last_x_pop).reshape(n_obj, -1)

    x_estimate = []
    # lex sort because
    # considering situation when f1 min has multiple same values
    # choose the one with bigger f2 value, so that nd can expand

    for i in range(n_obj):
        x_pop = last_x_pop[i, :]
        x_pop = x_pop.reshape(x_pop_size, -1)
        all_f = []
        # all_obj_f under current x pop
        for k in krg:
            f_k, _ = k.predict(x_pop)
            all_f = np.append(all_f, f_k)

        # reorganise all f in obj * popsize shape
        all_f = np.atleast_2d(all_f).reshape(n_obj, -1)
        # select an x according to lexsort
        x_index = lexsort_with_certain_row(all_f, i)
        # x_index = lexsort_specify_baserow(all_f, i)

        x_estimate = np.append(x_estimate, x_pop[x_index, :])

    x_estimate = np.atleast_2d(x_estimate).reshape(n_obj, -1)

    return x_estimate


def update_nadir_with_estimate(train_x,  # warning not suitable for more than 3 fs
                               train_y,
                               norm_train_y,
                               cons_y,
                               next_y,
                               problem,
                               x_krg,
                               krg,
                               krg_g,
                               nadir,
                               ideal,
                               enable_crossvalidation,
                               methods_ops,
                               ):
    # add estimated f to train y samples to update ideal and nadir
    n_var = problem.n_var
    n_obj = problem.n_obj
    x1 = np.atleast_2d(x_krg[0]).reshape(-1, n_var)
    x2 = np.atleast_2d(x_krg[1]).reshape(-1, n_var)

    # add new evaluation when next_y is better in any direction compared with
    # current ideal

    if next_y is not None:
        if np.any(next_y < ideal, axis=1):

            # warning: version3, dace is trained on normalized f space
            # for eim version
            train_y_tmp = train_y.copy()
            f1_norm_esti = []
            f2_norm_esti = []
            for k in krg:
                y_norm1, _ = k.predict(x1)
                y_norm2, _ = k.predict(x2)

                f1_norm_esti = np.append(f1_norm_esti, y_norm1)
                f2_norm_esti = np.append(f2_norm_esti, y_norm2)

            # convert back to real scale with ideal and nadir
            f1_norm_esti = np.atleast_2d(f1_norm_esti).reshape(1, -1)
            f2_norm_esti = np.atleast_2d(f2_norm_esti).reshape(1, -1)

            # from this step hv-r3 and eim-3 start to use different processes
            if methods_ops == 'eim_r3':
                # de-normalize back to real range
                f1_esti = f1_norm_esti * (nadir - ideal) + ideal
                f2_esti = f2_norm_esti * (nadir - ideal) + ideal

                # add to existing samples to work out new nadir and ideal
                tmp_sample = np.vstack((train_y_tmp, f1_esti, f2_esti))

                tmp_sample = close_adjustment(tmp_sample)
                nd_front_index = return_nd_front(tmp_sample)
                nd_front = tmp_sample[nd_front_index, :]

                nadir = np.amax(nd_front, axis=0)
                ideal = np.amin(nd_front, axis=0)

                # update krg with one new x/f pair
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

            elif methods_ops == 'hv_r3':
                # hv krg operate on real scale
                # so _norm_ still refer to real scale
                tmp_sample = np.vstack((train_y_tmp, f1_norm_esti, f2_norm_esti))

                tmp_sample = close_adjustment(tmp_sample)
                nd_front_index = return_nd_front(tmp_sample)
                nd_front = tmp_sample[nd_front_index, :]

                nadir = np.amax(nd_front, axis=0)
                ideal = np.amin(nd_front, axis=0)

                # update krg
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None
        else:
            if methods_ops == 'eim_r3':
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
            else:
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None

    return train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal


def additional_evaluation(x_krg, train_x, train_y, problem,
                          ):
    '''
    this method only deal with unconstraint mo
    :return: add kriging estimated x to training data.
    '''
    n_var = problem.n_var
    n_obj = problem.n_obj
    x1 = np.atleast_2d(x_krg[0]).reshape(-1, n_var)
    x2 = np.atleast_2d(x_krg[1]).reshape(-1, n_var)

    out = {}
    problem._evaluate(x1, out)
    y1 = out['F']
    problem._evaluate(x2, out)
    y2 = out['F']

    train_x = np.vstack((train_x, x1, x2))
    train_y = np.vstack((train_y, y1, y2))
    train_y = close_adjustment(train_y)
    return train_x, train_y


def update_nadir(train_x,  # warning not suitable for more than 3 fs
                 train_y,
                 norm_train_y,
                 cons_y,
                 next_y,
                 problem,
                 x_krg,
                 krg,
                 krg_g,
                 nadir,
                 ideal,
                 enable_crossvalidation,
                 methods_ops,
                 ):
    '''this fuunction does what?
    deal with dominance resistance solution
    '''

    '''
    # plot train_y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_y[:, 0], train_y[:, 1], marker='x', c='blue')
    f1 = [nadir[0], nadir[0], ideal[0], ideal[0], nadir[0]]
    f2 = [nadir[1], ideal[1], ideal[1], nadir[1], nadir[1]]
    line = Line2D(f1, f2, c='green')
    ax.add_line(line)
    '''

    # check with new nadir and ideal point
    # update them if they do not meet ideal/nadir requirement
    n_var = problem.n_var
    n_obj = problem.n_obj
    x1 = np.atleast_2d(x_krg[0]).reshape(-1, n_var)
    x2 = np.atleast_2d(x_krg[1]).reshape(-1, n_var)

    # add new evaluation when next_y is better in any direction compared with
    # current ideal
    estimate_flag = False
    if next_y is not None:
        # either add new point and update krg
        # or no adding and only update krg
        if np.any(next_y < ideal, axis=1):
            estimate_flag = True
            # print('new next_y better than ideal')
            # print(next_y)
            # print(ideal)
            out = {}
            problem._evaluate(x1, out)
            y1 = out['F']

            if 'G' in out.keys():
                g1 = out['G']

            problem._evaluate(x2, out)
            y2 = out['F']

            if 'G' in out.keys():
                g2 = out['G']

            # whether there is smaller point than nadir
            train_x = np.vstack((train_x, x1, x2))
            train_y = np.vstack((train_y, y1, y2))
            if 'G' in out.keys():
                cons_y = np.vstack((cons_y, g1, g2))

            # solve the too small distance problem
            train_y = close_adjustment(train_y)
            nd_front_index = return_nd_front(train_y)
            nd_front = train_y[nd_front_index, :]

            nadir = np.amax(nd_front, axis=0)
            ideal = np.amin(nd_front, axis=0)

            # print('ideal update')
            # print(ideal)
            if methods_ops == 'eim_r' or methods_ops == 'eim_r3':
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

            else:  # hvr/hv_r3
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None
        else:
            if methods_ops == 'eim_r':
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
            else:  # hvr
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None

    '''
      
    ax.scatter(y1[:, 0], y1[:, 1], marker='o', c='m')
    ax.scatter(y2[:, 0], y2[:, 1], marker='o', c='red')
   
    # add new line
    f1 = [nadir_new[0], nadir_new[0], ideal_new[0], ideal_new[0], nadir_new[0]]
    f2 = [nadir_new[1], ideal_new[1], ideal_new[1], nadir_new[1], nadir_new[1]]

    line = Line2D(f1, f2, c='red')
    ax.add_line(line)

    ax.scatter(nd_front[:, 0], nd_front[:, 1], c='yellow')
    # ax.scatter(train_y[-1, 0], train_y[-1, 1], marker='D', c='g')
    ax.scatter(y1[:, 0], y1[:, 1], s=200, marker='_', c='m')
    ax.scatter(y2[:, 0], y2[:, 1], s=200, marker='_', c='red')

    up_lim = np.max(np.amax(train_y, axis=0))
    low_lim = np.min(np.amin(train_y, axis=0))
    ax.set(xlim=(low_lim-1, up_lim+1), ylim=(low_lim-1, up_lim+1))
    plt.show()
    '''

    return train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal, estimate_flag


def initNormalization_by_nd(train_y):
    '''
    this function returns nadir and ideal w.r.t. nd front
    :param train_y:
    :return: nadir and ideal
    '''

    nd_front_index = return_nd_front(train_y)
    nd_front = train_y[nd_front_index, :]
    nadir = np.amax(nd_front, axis=0)
    ideal = np.amin(nd_front, axis=0)

    return nadir, ideal


def init_xy(number_of_initial_samples, target_problem, seed):
    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples, criterion='maximin', iterations=1000)

    xu = np.atleast_2d(target_problem.xu).reshape(1, -1)
    xl = np.atleast_2d(target_problem.xl).reshape(1, -1)

    train_x = xl + (xu - xl) * train_x

    # test
    # lfile = 'sample_x' + str(seed) + '.csv'
    # train_x = np.loadtxt(lfile, delimiter=',')

    out = {}
    target_problem._evaluate(train_x, out)
    train_y = out['F']

    if 'G' in out.keys():
        cons_y = out['G']
        cons_y = np.atleast_2d(cons_y).reshape(-1, n_sur_cons)
    else:
        cons_y = None

    # test
    '''
    lfile = 'sample_x' + str(seed) + '.csv'
    train_x_1 = np.loadtxt(lfile, delimiter=',')
    out = {}
    target_problem._evaluate(train_x_1, out)
    train_y_1 = out['F']

    plt.scatter(train_y[:, 0], train_y[:, 1])
    plt.scatter(train_y_1[:, 0], train_y_1[:, 1])
    plt.legend(['python', 'matlab'])
    plt.show()
    '''

    return train_x, train_y, cons_y


def return_nd_front(train_y):
    '''
    :param train_y: np.2d
    :return: index of nd front points in train_y
    '''
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)

    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    return ndf_extend

def get_ndfront(train_y):
    '''
       :param train_y: np.2d
       :return: nd front points extracted from train_y
       '''
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    ndf_index = ndf[0]
    nd_front = train_y[ndf_index, :]
    return nd_front

def return_hv(nd_front, reference_point, target_problem):
    p_name = target_problem.name()
    if 'DTLZ' in p_name and int(p_name[-1]) < 5:
        ref_dir = get_uniform_weights(10000, 2)
        true_pf = target_problem.pareto_front(ref_dir)
    else:
        true_pf = target_problem.pareto_front(n_pareto_points=10000)

    max_by_f = np.amax(true_pf, axis=0)
    min_by_f = np.amin(true_pf, axis=0)

    # normalized to 0-1
    nd_front = (nd_front - min_by_f) / (max_by_f - min_by_f)

    n_obj = nd_front.shape[1]
    n_nd = nd_front.shape[0]

    reference_point_norm = reference_point

    nd_list = []
    for i in range(n_nd):
        if np.all(nd_front[i, :] < reference_point):
            nd_list = np.append(nd_list, nd_front[i, :])
    nd_list = np.atleast_2d(nd_list).reshape(-1, n_obj)

    if len(nd_list) > 0:
        hv = pg.hypervolume(nd_list)
        hv_value = hv.compute(reference_point_norm)
    else:
        hv_value = 0

    return hv_value


def return_igd(target_problem, number_pf_points, nd_front):
    # extract pareto front
    nd_front = check_array(nd_front)
    n_obj = target_problem.n_obj

    # for test
    # nd_front = np.loadtxt('non_dominated_front.csv', delimiter=',')

    if n_obj == 2:
        if 'DTLZ' not in target_problem.name():
            true_pf = target_problem.pareto_front(n_pareto_points=number_pf_points)
        else:
            ref_dir = get_uniform_weights(number_pf_points, 2)
            true_pf = target_problem.pareto_front(ref_dir)

    max_by_f = np.amax(true_pf, axis=0)
    min_by_f = np.amin(true_pf, axis=0)

    # normalized to 0-1
    nd_front = (nd_front - min_by_f) / (max_by_f - min_by_f)

    true_pf = np.atleast_2d(true_pf).reshape(-1, n_obj)
    true_pf = (true_pf - min_by_f) / (max_by_f - min_by_f)

    eu_dist = pairwise_distances(true_pf, nd_front, 'euclidean')
    eu_dist = np.min(eu_dist, axis=1)
    igd = np.mean(eu_dist)
    return igd


def save_hv_igd(train_x, train_y, hv_ref, seed_index, target_problem, method_selection):
    problem_name = target_problem.name()
    n_x = train_x.shape[0]
    nd_front_index = return_nd_front(train_y)
    nd_front = train_y[nd_front_index, :]
    hv = return_hv(nd_front, hv_ref, target_problem)

    # for igd, only consider first front
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    nd_front = train_y[ndf[0], :]
    igd = return_igd(target_problem, 10000, nd_front)

    save = [hv, igd]
    print('sample size %d, final save hv of current nd_front: %.4f, igd is: %.4f' % (n_x, hv, igd))

    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + problem_name + '_' + method_selection
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)
    saveName = result_folder + '\\hv_igd_' + str(seed_index) + '.csv'
    np.savetxt(saveName, save, delimiter=',')


def feasible_check(train_x, target_problem, evalparas):
    out = {}
    sample_n = train_x.shape[0]
    n_sur_cons = target_problem.n_constr
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    target_problem._evaluate(train_x, out)

    if 'G' in out.keys():
        mu_g = out['G']
        mu_g = np.atleast_2d(mu_g).reshape(-1, n_sur_cons)
        temp_mug = copy.deepcopy(out['G'])

        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)
        feasible_y = evalparas['train_y'][feasible, :]
        evalparas['feasible'] = feasible_y

        if feasible.size > 0:
            print('feasible solutions: ')
        else:
            print('No feasible solutions in this iteration %d')
    else:
        evalparas['feasible'] = -1

    return evalparas


def post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature):
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    # output best archive solutions
    sample_n = train_x.shape[0]
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    out = {}
    target_problem._evaluate(train_x, out)
    if 'G' in out.keys():
        mu_g = out['G']
        mu_g = np.atleast_2d(mu_g).reshape(-1, n_sur_cons)

        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)

        feasible_solutions = train_x[feasible, :]
        feasible_f = train_y[feasible, :]

        n = len(feasible_f)
        # print('number of feasible solutions in total %d solutions is %d ' % (sample_n, n))

        if n > 0:
            best_f = np.argmin(feasible_f, axis=0)
            print('Best solutions encountered so far')
            print(feasible_f[best_f, :])
            best_f_out = feasible_f[best_f, :]
            best_x_out = feasible_solutions[best_f, :]
            print(feasible_solutions[best_f, :])
        else:
            best_f_out = None
            best_x_out = None
            print('No best solutions encountered so far')
    elif n_sur_objs == 1:
        best_f = np.argmin(train_y, axis=0)
        best_f_out = train_y[best_f, :]
        best_x_out = train_x[best_f, :]
    else:
        # print('MO save pareto front from all y')
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        f_pareto = train_y[ndf[0], :]
        best_f_out = f_pareto
        best_x_out = train_x[ndf[0], :]

    savename_x, savename_f, savename_FEs = saveNameConstr(target_problem.name(), seed_index, method_selection,
                                                          run_signature)

    dump(best_x_out, savename_x)
    dump(best_f_out, savename_f)


def referece_point_check(train_x, train_y, cons_y, ideal_krg, x_out, target_problem, krg, krg_g,
                         enable_crossvalidation):
    # check whether there is any f that is even better/smaller than ideal
    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

    ideal_true_samples = np.atleast_2d(np.amin(train_y, axis=0))
    compare = np.any(ideal_true_samples < ideal_krg, axis=1)
    # print(ideal_true_samples)
    # print(ideal_krg)
    # print(compare)

    if sum(compare) > 0:
        print('New evaluation')
        # add true evaluation
        for x in x_out:
            x = np.atleast_2d(x).reshape(-1, n_vals)
            out = {}
            target_problem._evaluate(x, out)
            y = out['F']

            train_x = np.vstack((train_x, x))
            train_y = np.vstack((train_y, y))
            if 'G' in out:
                g = np.atleast_2d(out['G']).reshape(-1, n_sur_cons)
                cons_y = np.vstack((cons_y, g))
        # re-conduct krg training
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
    return krg, krg_g


def normalization_with_nadir_ideal(y, nadir, ideal):
    y = check_array(y)
    return (y - ideal) / (nadir - ideal)


def normalization_with_self(y):
    y = check_array(y)
    min_y = np.min(y, axis=0)
    max_y = np.max(y, axis=0)
    return (y - min_y) / (max_y - min_y)


def normalization_with_nd(y):
    y = check_array(y)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y)
    ndf = list(ndf)
    ndf_size = len(ndf)
    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = y[ndf_extend, :]

    # normalization boundary
    min_nd_by_feature = np.amin(nd_front, axis=0)
    max_nd_by_feature = np.amax(nd_front, axis=0)

    # rule out exception nadir and ideal are too close
    # add more fronts to nd front
    if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
        print('nd front aligned problem, re-select nd front')
        ndf_index = ndf[0]
        for k in np.arange(1, ndf_size):
            ndf_index = np.append(ndf_index, ndf[k])
            nd_front = y[ndf_index, :]
            min_nd_by_feature = np.amin(nd_front, axis=0)
            max_nd_by_feature = np.amax(nd_front, axis=0)
            if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
                continue
            else:
                break
    norm_y = (y - min_nd_by_feature) / (max_nd_by_feature - min_nd_by_feature)
    return norm_y


def normalization_with_nothing(y):
    # for experiment purpose
    return y


def idealsearch_update(train_x, train_y, krg, target_problem):
    n_vals = train_x.shape[1]
    n_sur_cons = 0
    n_sur_objs = train_y.shape[1]

    guide_x = return_current_extreme(train_x, train_y)  # return current ideal x of two objectives
    # run
    x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu,
                                   guide_x)
    train_x, train_y = additional_evaluation(x_out, train_x, train_y, target_problem)
    return train_x, train_y


def confirm_search(new_y, train_y):
    obj_min = np.min(train_y, axis=0)
    diff = new_y - obj_min
    if np.any(diff < 0):
        return True
    else:
        return False


def nd2csv(train_y, target_problem, seed_index, method_selection, search_ideal):
    # (5)save nd front under name \problem_method_i\nd_seed_1.csv
    path = os.getcwd()
    path = path + '\paper1_results'
    if not os.path.exists(path):
        os.mkdir(path)
    savefolder = path + '\\' + target_problem.name() + '_' + method_selection + '_' + str(int(search_ideal))
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    savename = savefolder + '\\nd_seed_' + str(seed) + '.csv'
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    ndindex = ndf[0]
    ndfront = train_y[ndindex, :]
    np.savetxt(savename, ndfront, delimiter=',')


def paper1_mainscript(seed_index, target_problem, method_selection, search_ideal):
    '''
    :param seed_index:
    :param target_problem:
    :param method_selection: function name string, normalization scheme
    :param ideal_search: whether to use kriging to search for ideal point
    :return:
    '''
    # steps
    # (1) init training data with number of initial_samples
    # (2) normalization on f
    # (3) train krg
    # (4) enter iteration, propose next x till number of iteration is met
    # (5) save nd front under name \problem_method_i\nd_seed_1.csv
    from scipy.optimize import differential_evolution
    from scipy.optimize import Bounds
    enable_crossvalidation = False
    mp.freeze_support()
    np.random.seed(seed_index)
    recordFlag = False

    target_problem = eval(target_problem)

    print('Problem %s, seed %d' % (target_problem.name(), seed_index))
    hv_ref = [1.1, 1.1]

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_vals = target_problem.n_var

    if n_sur_objs > 2:
        stop = 200
    else:
        stop = 100

    number_of_initial_samples = 11 * n_vals - 1
    n_iter = 300  # stopping criterion set

    if 'WFG' in target_problem.name():
        stop = 250
        number_of_initial_samples = 200
    # (1) init training data with number of initial_samples
    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem, seed_index)

    # (2) normalization scheme
    norm_scheme = eval(method_selection)
    norm_train_y = norm_scheme(train_y)  # compatible with no normalization/
    # nomalization with all y /
    # normalization with nd
    # (3) train krg
    krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    # (4-0) before enter propose x phase, conduct once krg search on ideal
    if search_ideal:
        train_x, train_y = idealsearch_update(train_x, train_y, krg, krg_g, target_problem)
        norm_train_y = norm_scheme(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    # (4) enter iteration, propose next x till number of iteration is met
    num_pop = 50
    num_gen = 50
    if 'ZDT' in target_problem.name():
        num_pop = 200
        num_gen = 200

    bounds = Bounds(lb=target_problem.xl, ub=target_problem.xu)
    for iteration in range(n_iter):
        print('iteration %d' % iteration)
        # (4-1) de search for proposing next x point
        nd_front = get_ndfront(norm_train_y)
        def obj(x):
            fitness = EI_problem.ego_believer(x, krg, nd_front, hv_ref)
            return fitness
        result = differential_evolution(obj, bounds)

        # propose next_x location
        next_x = result.x

        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)
        next_y = target_problem.evaluate(next_x, return_values_of=['F'])

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        # (4-2) according to configuration determine whether to estimate new point
        if search_ideal:
            if confirm_search(next_y, train_y[0:-1, :]):
                train_x, train_y = idealsearch_update(train_x, train_y, krg, target_problem)

        # retrain krg, normalization needed
        norm_train_y = norm_scheme(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    # (5) save nd front under name \problem_method_i\nd_seed_1.csv
    nd2csv(train_y, target_problem, seed_index, method_selection, search_ideal)


def main(seed_index, target_problem, enable_crossvalidation, method_selection, run_signature):
    # this following one line is for work around 1d plot in multiple-processing settings
    mp.freeze_support()
    np.random.seed(seed_index)
    recordFlag = False

    target_problem = eval(target_problem)

    print('Problem %s, seed %d' % (target_problem.name(), seed_index))
    hv_ref = [1.1, 1.1]

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    n_vals = target_problem.n_var

    if n_sur_objs > 2:
        stop = 200
    else:
        stop = 100

    number_of_initial_samples = 11 * n_vals - 1
    n_iter = 300  # stopping criterion set

    if 'WFG' in target_problem.name():
        stop = 250
        number_of_initial_samples = 200

    # (1) init training data with number of initial_samples
    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem, seed_index)

    # for evalparas compatibility across different algorithms
    # nadir/ideal initialization on nd front no 2d alignment fix
    nadir, ideal = initNormalization_by_nd(train_y)

    # kriging data preparision
    # initialization before infill interaction
    if method_selection == 'eim':
        norm_train_y = normalization_with_self(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    elif method_selection == 'eim_nd':
        norm_train_y = normalization_with_nd(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    elif method_selection == 'eim_r':
        norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    elif method_selection == 'eim_r3':
        norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    else:
        norm_train_y = None
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)

    # always conduct reference search on a initialized samples
    if method_selection == 'hvr' or method_selection == 'hv_r3' or method_selection == 'eim_r' or method_selection == 'eim_r3':
        guide_x = return_current_extreme(train_x, train_y)  # return current ideal
        x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu,
                                       guide_x)
        next_y = np.atleast_2d([ideal[0] - 1, ideal[1] - 1])  # force to estimate
        train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal, est_flag = update_nadir(train_x,
                                                                                                  train_y,
                                                                                                  norm_train_y,
                                                                                                  cons_y,
                                                                                                  next_y,
                                                                                                  # flag for initialization
                                                                                                  target_problem,
                                                                                                  x_out,
                                                                                                  krg,
                                                                                                  krg_g,
                                                                                                  nadir,
                                                                                                  ideal,
                                                                                                  enable_crossvalidation,
                                                                                                  method_selection,
                                                                                                  )

        # test
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        ndf_size = len(ndf)
        # extract nd for normalization
        if len(ndf[0]) > 1:
            ndf_extend = ndf[0]
        else:
            ndf_extend = np.append(ndf[0], ndf[1])

        nd_front = train_y[ndf_extend, :]
        min_pf_by_feature = ideal
        max_pf_by_feature = nadir
        norm_nd = (nd_front - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
        hv = pg.hypervolume(norm_nd)
        hv_value = hv.compute([1.1, 1.1])

    # create EI problem
    evalparas = {'train_x': train_x,
                 'train_y': train_y,
                 'norm_train_y': norm_train_y,
                 'krg': krg,
                 'krg_g': krg_g,
                 'nadir': nadir,
                 'ideal': ideal,
                 'feasible': np.array([]),
                 'ei_method': method_selection,
                 'problem_name': target_problem.name()}

    # construct ei problems
    ei_problem = get_problem_from_func(acqusition_function,
                                       target_problem.xl,  # row direction
                                       target_problem.xu,
                                       n_var=n_vals,
                                       func_args=evalparas)

    x_bounds = np.vstack((target_problem.xl, target_problem.xu)).T.tolist()  # for ea, column direction

    start_all = time.time()
    # start the searching process
    plot_flag = False
    plt.ion()
    for iteration in range(n_iter):
        print('iteration %d' % iteration)

        # check feasibility in main loop, update evalparas['feasible']
        evalparas = feasible_check(train_x, target_problem, evalparas)

        '''
        if train_x.shape[0] % 5 == 0:
            recordFlag = utilities.intermediate_save(target_problem, method_selection, seed_index, iteration, krg, train_y, nadir, ideal)
        '''

        start = time.time()
        # main loop for finding next x
        candidate_x = np.zeros((1, n_vals))
        candidate_y = []

        num_pop = 50
        num_gen = 50
        if 'ZDT' in target_problem.name():
            num_pop = 200
            num_gen = 200

        # test_iter = 203
        # if train_x.shape[0] == test_iter:
        plot_flag = False

        for restart in range(1):
            # DE
            pop_x, pop_f, _, _ = optimizer_EI.optimizer_DE(ei_problem,
                                                     # ei_problem.n_obj,
                                                     ei_problem.n_constr,
                                                     x_bounds,
                                                     # recordFlag,
                                                     # pop_test=pop_test,
                                                     insertpop=None,
                                                     F=0.8,
                                                     CR=0.8,
                                                     NP=num_pop,
                                                     itermax=num_gen,
                                                     visflag=plot_flag,
                                                     ax=None,
                                                     **evalparas)

            candidate_x = np.vstack((candidate_x, pop_x[0, :]))
            candidate_y = np.append(candidate_y, pop_f[0, :])

            '''
            if recordFlag:
                saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection+ '_seed_' + str(seed_index) + 'search_record_iteration_' + str(iteration) + '_restart_' + str(restart) + '.joblib'
                dump(record, saveName)
            '''
        end = time.time()
        lasts = (end - start)

        # print('propose to next x in iteration %d uses %.2f sec' % (iteration, lasts))
        w = np.argwhere(candidate_y == np.min(candidate_y))
        metric_opt = np.min(candidate_y)

        # propose next_x location
        next_x = candidate_x[w[0] + 1, :]

        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)

        # generate corresponding f and g
        out = {}
        target_problem._evaluate(next_x, out)
        next_y = out['F']
        # print(next_y)

        '''
        if train_x.shape[0] % 5 == 0:
            saveName  = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(seed_index) + 'nextF_iteration_' + str(iteration) + '.joblib'
            dump(next_y, saveName)
        '''
        recordFlag = False
        if 'G' in out.keys():
            next_cons_y = out['G']
            next_cons_y = np.atleast_2d(next_cons_y)
        else:
            next_cons_y = None

        # -----------plot -------------
        # plot progress

        plt.clf()
        if 'DTLZ' in target_problem.name() and int(target_problem.name()[-1]) < 5:
            ref_dir = get_uniform_weights(100, 2)
            true_pf = target_problem.pareto_front(ref_dir)
        else:
            true_pf = target_problem.pareto_front(n_pareto_points=100)

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        nd_front = train_y[ndf[0], :]

        f1_pred, _ = krg[0].predict(next_x)
        f2_pred, _ = krg[1].predict(next_x)
        f_pred = np.hstack((f1_pred, f2_pred))

        if method_selection == 'eim_r':
            f_pred = f_pred * (nadir - ideal) + ideal

        if method_selection == 'eim':
            f_min_by_feature = np.amin(train_y, axis=0)
            f_max_by_feature = np.max(train_y, axis=0)
            f_pred = f_pred * (f_max_by_feature - f_min_by_feature) + f_min_by_feature
        if method_selection == 'eim_nd':
            nd_front_index = return_nd_front(train_y)
            nd_front_plot = train_y[nd_front_index, :]
            f_min_by_feature = np.amin(nd_front_plot, axis=0)
            f_max_by_feature = np.max(nd_front_plot, axis=0)
            f_pred = f_pred * (f_max_by_feature - f_min_by_feature) + f_min_by_feature
        if method_selection == 'hv':
            nd_front_index = return_nd_front(train_y)
            nd_front_plot = train_y[nd_front_index, :]
            f_min_by_feature = np.amin(nd_front_plot, axis=0)
            f_max_by_feature = np.max(nd_front_plot, axis=0)
            reference_point = 1.1 * (f_max_by_feature - f_min_by_feature) + f_min_by_feature
        if method_selection == 'hvr' or method_selection == 'hv_r3':
            reference_point = 1.1 * (nadir - ideal) + ideal

        plt.scatter(true_pf[:, 0], true_pf[:, 1], s=0.2)

        plt.scatter(next_y[:, 0], next_y[:, 1], marker="D", c='red')
        text2 = 'truely evaluated next F'
        plt.text(next_y[:, 0] + 0.05, next_y[:, 1] + 0.08, text2)

        plt.scatter(train_y[:, 0], train_y[:, 1], marker="o", s=1, c='k')
        plt.scatter(nd_front[:, 0], nd_front[:, 1], marker='o', c='c')
        plt.scatter(f_pred[:, 0], f_pred[:, 1], marker="P")
        # text1 = ' predicted next y' + "{:4.2f}".format(metric_opt) + " [{:4.2f}".format(f_pred[0, 0]) + ' ' + "{:4.2f}]".format(f_pred[0, 1])
        text1 = 'predicted next F'
        plt.text(f_pred[:, 0] - 0.05, f_pred[:, 1] - 0.1, text1)

        if method_selection == 'eim_r' or method_selection == 'eim_r3' or method_selection == 'hv_r3' or method_selection == 'hvr':
            plt.scatter(nadir[0], nadir[1], marker='+', c='g')
            plt.text(nadir[0] + 0.01, nadir[1] - 0.05, 'f_max')
            plt.scatter(ideal[0], ideal[1], marker='+', c='g')
            plt.text(ideal[0] + 0.01, ideal[1] + 0.01, 'f_min')
            # tt = " [{:4.2f}".format(reference_point[0]) + ' ' + "{:4.2f}]".format(reference_point[1])
            tt = 'HV ref'
            plt.scatter(reference_point[0], reference_point[1], marker='+', c='red')
            plt.text(reference_point[0] + 0.02, reference_point[1] + 0.02, tt)

            if iteration == 0:
                plt.scatter(train_y[-1, 0], train_y[-1, 1], marker='x', c='black')
                plt.scatter(train_y[-2, 0], train_y[-2, 1], marker='x', c='black')

        if method_selection == 'eim' or method_selection == 'eim_nd' or method_selection == 'hv':
            plt.scatter(f_min_by_feature[0], f_min_by_feature[1], marker='+', c='g')
            plt.text(f_min_by_feature[0], f_min_by_feature[1], 'f_min')
            plt.scatter(f_max_by_feature[0], f_max_by_feature[1], marker='+', c='g')

            # tt = " [{:4.2f}".format(f_max_by_feature[0]) + ' ' + "{:4.2f}]".format(f_max_by_feature[1])
            plt.text(f_max_by_feature[0] + 0.08, f_max_by_feature[1] - 0.08, 'f_max')
            # tt = " [{:4.2f}".format(reference_point[0]) + ' ' + "{:4.2f}]".format(reference_point[1])
            plt.scatter(reference_point[0], reference_point[1], marker='+', c='red')
            plt.text(reference_point[0] + 0.08, reference_point[1] + 0.08, 'hv ref')

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        # print('train x  size %d' % train_x.shape[0])

        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))

        # ---------
        start = time.time()
        # use extended data to train krging model

        # output hv during the search
        n_x = train_x.shape[0]
        nd_front_index = return_nd_front(train_y)
        nd_front = train_y[nd_front_index, :]
        hv = return_hv(nd_front, hv_ref, target_problem)
        igd = return_igd(target_problem, 10000, nd_front)
        print('iteration: %d, number evaluation: %d, hv of current nd_front: %.4f, igd is: %.4f' % (
        iteration, n_x, hv, igd))

        # ---------plot--------------------------------
        # t = 'hv  after adding new point {:6.4f}'.format(hv)
        t = 'Initialization impact on f_min'
        plt.title(t)

        # kriging  update with newly added x/f
        if method_selection == 'eim':
            norm_train_y = normalization_with_self(train_y)
            krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        elif method_selection == 'eim_nd':
            norm_train_y = normalization_with_nd(train_y)
            krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        # elif method_selection == 'eim_r':
        # norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
        # krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        elif method_selection == 'hv':
            norm_train_y = None
            krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)

        # new evaluation added depending on condition
        if method_selection == 'hvr' or method_selection == 'eim_r':

            guide_x = return_current_extreme(train_x, train_y)
            x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu,
                                           guide_x)
            train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal, est_flag = update_nadir(train_x,
                                                                                                      train_y,
                                                                                                      norm_train_y,
                                                                                                      cons_y,
                                                                                                      next_y,
                                                                                                      target_problem,
                                                                                                      x_out,
                                                                                                      krg,
                                                                                                      krg_g,
                                                                                                      nadir,
                                                                                                      ideal,
                                                                                                      enable_crossvalidation,
                                                                                                      method_selection)

            savename = 'visualization\\' + target_problem.name() + '_' + method_selection + '_guide_' + str(
                seed_index) + '_iteration_' + str(train_x.shape[0]) + '.eps'
            if est_flag == True:
                plt.scatter(train_y[-1, 0], train_y[-1, 1], marker='x', c='red')
                plt.scatter(train_y[-2, 0], train_y[-2, 1], marker='x', c='red')
            plt.savefig(savename, format='eps')
            plt.pause(5)

            # -----------plot ends -------------

        # r3 does not add
        if method_selection == 'eim_r3' or method_selection == 'hv_r3':
            guide_x = return_current_extreme(train_x, train_y)
            x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu,
                                           guide_x)
            train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal = update_nadir_with_estimate(train_x,
                                                                                                          train_y,
                                                                                                          norm_train_y,
                                                                                                          cons_y,
                                                                                                          next_y,
                                                                                                          target_problem,
                                                                                                          x_out,
                                                                                                          krg,
                                                                                                          krg_g,
                                                                                                          nadir,
                                                                                                          ideal,
                                                                                                          enable_crossvalidation,
                                                                                                          method_selection)

            savename = 'visualization\\' + target_problem.name() + '_' + method_selection + '_guide_' + str(
                seed_index) + '_iteration_' + str(train_x.shape[0]) + '.png'
            plt.savefig(savename)
            # plt.pause(0.5)
            # -----------plot ends -------------

        savename = 'visualization3\\' + target_problem.name() + '_' + method_selection + '_guide_' + str(
            seed_index) + '_iteration_' + str(train_x.shape[0]) + '.eps'
        plt.savefig(savename, format='eps')
        plt.pause(0.5)

        # -----------plot ends -------------

        lasts = (end - start)
        # print('cross-validation %d uses %.2f sec' % (iteration, lasts))

        # update ea parameters
        evalparas['train_x'] = train_x
        evalparas['train_y'] = train_y
        evalparas['norm_train_y'] = norm_train_y
        evalparas['krg'] = krg
        evalparas['krg_g'] = krg_g
        evalparas['nadir'] = nadir
        evalparas['ideal'] = ideal

        # stopping criteria
        sample_n = train_x.shape[0]
        if sample_n == stop:
            break
        if sample_n > stop:
            vio_more = np.arange(stop, sample_n)
            train_y = np.delete(train_y, vio_more, 0)
            train_x = np.delete(train_x, vio_more, 0)
            break

    plt.ioff()
    end_all = time.time()
    print('overall time %.4f ' % (end_all - start_all))
    save_hv_igd(train_x, train_y, hv_ref, seed_index, target_problem, method_selection)
    post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature)

    # plot
    # result_processing.plot_pareto_vs_ouputs('ZDT3', [seed_index], 'eim', 'eim')
    # savename = 'sample_out_freensga_' + str(seed_index) + '.csv'
    # out = [eim_compare[0], hv, igd]
    # np.savetxt(savename, out, delimiter=',')


if __name__ == "__main__":

    MO_target_problems = [
        'ZDT1(n_var=6)',
        'ZDT2(n_var=6)',
        # 'ZDT3(n_var=6)',
        # 'WFG.WFG_1(n_var=2, n_obj=2, K=1)',
        #  'WFG.WFG_2(n_var=6, n_obj=2, K=4)',
        #   'WFG.WFG_3(n_var=6, n_obj=2, K=4)',
        #   'WFG.WFG_4(n_var=6, n_obj=2, K=4)',
        #  'WFG.WFG_5(n_var=6, n_obj=2, K=4)',
        #  'WFG.WFG_6(n_var=6, n_obj=2, K=4)',
        # 'WFG.WFG_7(n_var=6, n_obj=2, K=4)',
        # 'WFG.WFG_8(n_var=6, n_obj=2, K=4)',
        # 'WFG.WFG_9(n_var=6, n_obj=2, K=4)',
        'DTLZ1(n_var=6, n_obj=2)',
        #  'DTLZ2(n_var=6, n_obj=2)',
        # 'DTLZs.DTLZ5(n_var=6, n_obj=2)',
        # 'DTLZs.DTLZ7(n_var=6, n_obj=2)',
        # 'iDTLZ.IDTLZ1(n_var=6, n_obj=2)',
        # 'iDTLZ.IDTLZ2(n_var=6, n_obj=2)',
    ]

    # x = np.atleast_2d([.15, .25, .35, .45, .55, .65])
    # target_problem = eval(MO_target_problems[0])

    # out = target_problem.evaluate(x)
    # print(out)

    args = []
    run_sig = ['hv', 'hvr', ]  # 'hv_r3']  #'eim_nd', 'eim', 'eim_r', 'eim_r3']
    methods_ops = ['hv',
                   'hvr', ]  # 'hv_r3']  # 'eim_nd', 'eim', 'eim_r', 'eim_r3']  #, 'hv', 'eim_r', 'hvr',  'eim','eim_nd' ]

    for seed in range(0, 21):
        for target_problem in MO_target_problems:
            for method in methods_ops:
                args.append((seed, target_problem, False, method, method))

    # single processor run/for debugging
    # for seed in range(1, 16):
    # for target_problem in MO_target_problems:
    # for method in methods_ops:
    # main(seed, target_problem, False, method, method)

    main(0, MO_target_problems[1], False, 'hvr', 'hvr')
    # paper1_mainscript(0, MO_target_problems[1], 'normalization_with_nd', True)

    # num_workers = 21
    # pool = mp.Pool(processes=num_workers)
    # pool.starmap(main, ([arg for arg in args]))

    ''' 
    target_problems = [branin.new_branin_5(),
                       Gomez3.Gomez3(),
                       Mystery.Mystery(),
                       Reverse_Mystery.ReverseMystery(),
                       SHCBc.SHCBc(),
                       Haupt_schewefel.Haupt_schewefel(),
                       HS100.HS100(),
                       GPc.GPc()]
    '''
