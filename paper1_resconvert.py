import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
    DTLZ1, DTLZ2,DTLZ3,DTLZ4, \
    BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
import pyDOE
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness, EI, MAF
from matplotlib.lines import Line2D
import scipy
import statistics
import pickle

import os
import copy
import multiprocessing as mp
import pygmo as pg
import EI_problem
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds

def hv_convergeplot(k):
    '''this function needs specify a problem internally
    and plot its hv convergence
    find the following line to decide which problem to process
    problem = eval(target_problems[2])
    ** warning: dependent on running of hv_summary2csv or trainy_summary2csv
    ** to create seed selection
    *** or plot same seed change the following line
    '''
    import json
    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']

    method_selection = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    problem = eval(target_problems[k])
    pf = get_paretofront(problem, 100)
    nadir = np.max(pf, axis=0)
    ref = nadir * 1.1

    n_vals = problem.n_var
    if 'ZDT' in problem.name():
        number_of_initial_samples = 11 * n_vals - 1
        max_eval = 100
    if 'WFG' in problem.name():
        max_eval = 250
        number_of_initial_samples = 200

    num_plot = max_eval - number_of_initial_samples
    hvplot = np.zeros((3, num_plot))

    path = os.getcwd()
    path = path + '\\paper1_results\\paper1_resconvert\\median_id.joblib'
    median_id = load(path)
    seed = [int(median_id[problem.name() + '_' + method_selection[0]]),
            int(median_id[problem.name() + '_' + method_selection[1]]),
            int(median_id[problem.name() + '_' + method_selection[2]])]
    seed = [1] * 3

    path = os.getcwd()
    path = path + '\paper1_results'
    for m in range(3):
        savefolder = path + '\\' + problem.name() + '_' + method_selection[m]
        savename = savefolder + '\\trainy_seed_' + str(seed[m]) + '.csv'
        print(savename)
        trainy = np.loadtxt(savename, delimiter=',')
        trainy = np.atleast_2d(trainy)
        for i in range(number_of_initial_samples, max_eval):
            f = trainy[0:i, :]
            hvplot[m, i-number_of_initial_samples] = get_f2hv(f, ref)


    # m = ['normalization f', 'normalization nd', 'nd and ideal search']
    m = ['normalization nd', 'normalization nd and ideal search']

    style = ['dotted', '--']
    plt.ion()
    fig, ax = plt.subplots()
    # algs = {}
    for i in range(2):
        ax.plot(hvplot[i+1, :], linestyle=style[i])
    ax.legend(m)
    ax.set_xlabel('iterations')
    ax.set_ylabel('Hypervolume')
    plt.title(problem.name())
    plt.pause(1)


    path = os.getcwd()
    savefolder = path + '\\paper1_results\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + '_hvconverge.eps'
    savename2 = savefolder + '\\' + problem.name() + '_hvconverge.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)

    plt.ioff()


def hvconverge_averageplot(resultfolder, resultconver):
    '''
            this function reads parameter files to track experimental results
            read saved training data  from each seed, and cut the right number of evaluation
            calculate pareto front to generated reference point
            use this pf reference to calculate hv for each seed
            :return: two csv files (1) raw data hv collection, (2) mean median csv collection
            '''
    import json

    problems_json = 'p/resconvert_plot3.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29

    num_pro = len(target_problems)
    # methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_4']
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_2',
               'normalization_with_nd_4',
               'normalization_with_nd_3']  # 'normalization_with_nd_3']   'normalization_with_external_4'
    methods_title = ['Norm$R_A$', 'Norm$R_{ND}$', 'Norm$R_{NDE}$', 'Norm$R_{NDC1}$', 'Norm$R_{NDC2}$']

    num_methods = len(methods)

    path = os.getcwd()
    path = path + '\\' + resultfolder
    plt.ion()
    for problem_i, problem in enumerate(target_problems):
        # for each problem create three plots
        # problem_i = 11
        problem = target_problems[problem_i]
        problem = eval(problem)
        pf = get_paretofront(problem, 100)
        nadir = np.max(pf, axis=0)
        ref = nadir * 1.1
        if 'ZDT' in problem.name() or 'DTLZ' in problem.name() or 'MAF' in problem.name():
            evalnum = 300 # 100
            initsize = problem.n_var * 11 - 1
        if 'WFG' in problem.name() :
            evalnum = 300 # 250
            initsize = 200

        # each problem calculate mean and variance
        fig, ax = plt.subplots()

        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection

            # create raw data for each method
            rawhv = np.zeros((seedmax, evalnum -initsize))
            for seed in range(seedmax):
                # savename = savefolder + '\\hvconvg_seed_' + str(seed) + '.csv'
                # print(savename)
                # hv = np.loadtxt(savename, delimiter=',')
                # hv = np.atleast_2d(hv)
                # hv = hv[0:evalnum - initsize, 1]

                # fix bug
                savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
                print(savename)
                trainy = np.loadtxt(savename, delimiter=',')
                trainy = np.atleast_2d(trainy)
                hv = []
                for i in range(initsize, evalnum):
                    eval_hv = get_f2hvnorm(trainy[0:i, :], pf)
                    hv = np.append(hv, eval_hv)

                rawhv[seed, :] = hv
            # average over all seeds
            mean_hv1 = np.mean(rawhv, axis=0)
            std_hv1 = np.std(rawhv, axis=0)
            x = range(initsize, evalnum)

            ax.plot(x, mean_hv1, label=methods_title[j])
            ax.fill_between(x, mean_hv1 + std_hv1, mean_hv1-std_hv1, alpha=0.2)
            # plt.pause(2)
            a = 0
        ss = 16

        # bottom, top = ax.get_ylim()
        # top = top + 0.02
        # bottom =  bottom - 0.4
        # ax.set_ylim(bottom, top)
        ax.set_xlabel('evaluations', fontsize=ss)
        ax.set_ylabel('hypervolume', fontsize=ss)
        plt.title(problem.name(), fontsize=ss)
        plt.legend( fontsize=ss)

        # save ---
        paths = os.getcwd()
        savefolder = paths + '\\' + resultfolder + '\\process_plot'
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        savename1 = savefolder + '\\' + problem.name() + '_hvconverge.eps'
        savename2 = savefolder + '\\' + problem.name() + '_hvconverge.png'
        plt.savefig(savename1, format='eps')
        plt.savefig(savename2)
        plt.pause(5)
        plt.close()
        #break

    plt.ioff()
    # plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.5)



def nadirplot(resultfolder, convertfolder):
    # target_problems = "DTLZs.DTLZ7(n_var=6, n_obj=3)"
    target_problems = "WFG.WFG_5(n_var=6, n_obj=3, K=4)"

    method_selection = ['normalization_with_nd_0', 'normalization_with_nd_3']
    # method_selection = ['normalization_with_self_0', 'normalization_with_nd_3']
    problem = eval(target_problems)

    n_vals = problem.n_var
    if 'DTLZ' in problem.name():
        number_of_initial_samples = 11 * n_vals - 1
        max_eval = 200
    if 'WFG' in problem.name():
        max_eval = 300
        number_of_initial_samples = 200


    path = os.getcwd()
    path = path + '\\' + resultfolder + '\\'
    path = path + convertfolder
    if not os.path.exists(path):
        os.mkdir(path)


    filename1 = path + '\\' + problem.name() + '_' + method_selection[0] + 'nadir_check.csv'
    filename2 = path + '\\' + problem.name() + '_' + method_selection[1] + 'nadir_check.csv'



    n1 = np.loadtxt(filename1, delimiter=',')
    n2 = np.loadtxt(filename2, delimiter=',')

    # x = statistics.median(n2[:, -1])

    n1 = np.mean(n1, axis=0)
    n2 = np.mean(n2, axis=0)

    # n1 = n1[0, 0:10]
    # n2 = n2[0, 0:10]

    diff = abs(n1[-1] - n2[-1])
    diff = np.format_float_positional(diff, precision=3)
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(n1, linestyle='dotted', c='r')
    ax.plot(n2, linestyle='--', c='b')
    plt.title(problem.name() + ' ' + 'normalized nadir distance' + ' ' + str(diff))
    plt.legend(['nd', 'nde'])
    plt.show()



def get_hv(front, ref):
    '''
    this function calcuate hv with respect to ref
    :param front:
    :param ref:
    :return:
    '''
    front = np.atleast_2d(front)
    new_front = []
    n = front.shape[0]
    m = front.shape[1]

    # eliminate front instance which is beyond reference point
    for i in range(n):
        if np.any(front[i, :] - ref >= 0):
            continue
        else:
            new_front = np.append(new_front, front[i, :])

    # elimiate case where nd is far way off
    if len(new_front) == 0:
        hv = 0
    else:
        new_front = np.atleast_2d(new_front).reshape(-1, m)
        # calculate hv
        hv_class = pg.hypervolume(new_front)
        hv = hv_class.compute(ref)

    return hv



def mat2csv(target_problems, matrix, matrix_sts, methods, seedmax, ideal_stat, nadir_stat, resultfolder, result_convertfolder):
    '''
    this function iterate through problems and create csv with headers and index
    :param target_problems:
    :param matrix:
    :return:
    '''
    num_methods = len(methods)
    num_prob    = len(target_problems)

    # write header
    path = os.getcwd()
    path = path + '\\' + resultfolder + '\\'
    savefile = path + result_convertfolder + '\\results_convert.csv'

    with open(savefile, 'w') as f:
        # write header
        f.write(',')
        for i in range(num_prob):
            for j in range(num_methods):
                m = eval(target_problems[i]).name() + '_' + methods[j]
                f.write(m)
                f.write(',')
        f.write('\n')

        for seed in range(seedmax):
            # write seed
            f.write(str(seed))
            f.write(',')

            # write matrix
            for i in range(num_prob * num_methods):
                f.write(str(matrix[seed, i]))
                f.write(',')
            f.write('\n')

        # write statistics
        sts = ['mean', 'std', 'median', 'median_id']
        for s in range(4):
            f.write(sts[s])
            f.write(',')
            for i in range(num_prob * num_methods):
                f.write(str(matrix_sts[s, i]))
                f.write(',')
            f.write('\n')
    #-------------------------------------
    #----a lean results
    path = os.getcwd()
    path = path + '\\' + resultfolder + '\\'

    savefile = path + result_convertfolder + '\\results_convertlean.csv'
    with open(savefile, 'w') as f:
        f.write(',')
        for j in range(num_methods):
            f.write(methods[j])
            f.write(',')
        f.write('\n')
        for i in range(num_prob):
            f.write(eval(target_problems[i]).name())
            f.write(',')
            for j in range(num_methods):

                median_value = matrix_sts[2, i * num_methods + j]
                f.write(str(median_value))
                f.write(',')
            f.write('\n')

    # ----a lean results
    path = os.getcwd()
    path = path + '\\' + resultfolder +'\\'
    savefile = path + result_convertfolder + '\\results_ideallean.csv'
    with open(savefile, 'w') as f:
        f.write(',')
        for j in range(num_methods):
            f.write(methods[j])
            f.write(',')
        f.write('\n')
        for i in range(num_prob):
            f.write(eval(target_problems[i]).name())
            f.write(',')
            for j in range(num_methods):
                median_value = ideal_stat[2, i * num_methods + j]
                f.write(str(median_value))
                f.write(',')
            f.write('\n')

    # ----a nadir results
    path = os.getcwd()
    path = path + '\\' + resultfolder + '\\'
    savefile = path + result_convertfolder + '\\results_nadirlean.csv'
    with open(savefile, 'w') as f:
        f.write(',')
        for j in range(num_methods):
            f.write(methods[j])
            f.write(',')
        f.write('\n')
        for i in range(num_prob):
            f.write(eval(target_problems[i]).name())
            f.write(',')
            for j in range(num_methods):
                median_value = nadir_stat[2, i * num_methods + j]
                f.write(str(median_value))
                f.write(',')
            f.write('\n')


def plot_process3d(problem, train_y, method,seed):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    true_pf = get_paretofront(problem, 1000)
    nadir = np.max(true_pf, axis=0)
    ref = nadir * 1.1

    # ---------- visual check
    ax.cla()
    ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], c='green', alpha=0.2)
    # ax.scatter(train_y[:, 0], train_y[:, 1], train_y[:, 2], c='blue', s=0.2)

    nd_frontdn = get_ndfront(train_y)
    ax.scatter(nd_frontdn[:, 0], nd_frontdn[:, 1], nd_frontdn[:, 2], c='blue')

    # plot reference point
    ref_dn = ref
    ax.scatter(ref_dn[0], ref_dn[1], ref_dn[2], c='red', marker='D')

    # plt.legend(['PF', 'archive A', 'nd front', 'ref point'])
    plt.legend(['PF',  'nd front', 'ref point'])
    # -----------visual check--

    plt.title(problem.name())

    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    plt.show()
    a = 0

    # -----

    path = os.getcwd()
    savefolder = path + '\\paper1_results\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + '_' + method + '_'+ str(seed)+ '.eps'
    savename2 = savefolder + '\\' + problem.name() + '_' + method + '_' + str(seed)+  '.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)

def hv_medianplot():
    '''
    this function plot the final results of each problem

    :return: saved png and eps file in folder paper1_results\\ndplots\\prob_method_seed_*.eps/png
    (1) read the median seed
    (2) read nd of train y median seed
    (3) plot and save

    '''
    import json

    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    num_methods = len(methods)

    path = os.getcwd()
    pathsave = path + '\\paper1_results\\ndplots\\'
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)

    path_medianseed = os.getcwd() + '\\paper1_results\\paper1_resconvert\\median_id.joblib'
    seedmedian = load(path_medianseed)


    for prob_id in range(num_pro):
        # prob_id = 13

        problem = target_problems[prob_id]
        problem = eval(problem)

        for method_selection in methods:
            seed = seedmedian[problem.name() + '_' + method_selection]


            savefolder = path + '\\paper1_results\\' + problem.name() + '_' + method_selection
            savename = savefolder + '\\trainy_seed_' + str(int(seed)) + '.csv'

            trainy = np.loadtxt(savename, delimiter=',')
            nd_front = get_ndfront(trainy)
            # nd_front = np.loadtxt(savename, delimiter=',')
            nd_front = np.atleast_2d(nd_front)
            pf = get_paretofront(problem, 1000)


            plt.ion()
            fig, ax = plt.subplots()
            ax.scatter(pf[:, 0], pf[:, 1], s=1)
            ax.scatter(nd_front[:, 0], nd_front[:, 1], facecolors='white', edgecolors='red', alpha=1)
            plt.title(problem.name())
            plt.legend(['PF', 'Search result'])
            plt.xlabel('f1')
            plt.ylabel('f2')

            #-----
            paths = os.getcwd()
            savefolder = paths + '\\paper1_results\\plots'
            if not os.path.exists(savefolder):
                os.mkdir(savefolder)

            savename1 = savefolder + '\\' + problem.name() + '_' + method_selection + '_final_' + str(seed) + '.eps'
            savename2 = savefolder + '\\' + problem.name() + '_' + method_selection + '_final_' + str(seed) + '.png'
            plt.savefig(savename1, format='eps')
            plt.savefig(savename2)

            plt.pause(2)
            plt.close()


def hv_medianplot3in1():
    '''
    this function plot the final results of each problem

    :return: saved png and eps file in folder paper1_results\\ndplots\\prob_method_seed_*.eps/png
    (1) read the median seed
    (2) read nd of train y median seed
    (3) plot and save
    *** change this line to deal with single problem : prob_id = 7

    '''
    import json

    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    num_methods = len(methods)

    path = os.getcwd()
    pathsave = path + '\\paper1_results\\ndplots\\'
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)

    path_medianseed = os.getcwd() + '\\paper1_results\\paper1_resconvert\\median_id.joblib'
    seedmedian = load(path_medianseed)


    for prob_id in range(num_pro):
        prob_id = 0

        problem = target_problems[prob_id]
        problem = eval(problem)

        plt.ion()
        fig, ax = plt.subplots()

        pf = get_paretofront(problem, 1000)
        ax.scatter(pf[:, 0], pf[:, 1], s=1, color='green')
        edgecolors = ['red', 'black']

        for id, method_selection in enumerate(methods):
            seed = seedmedian[problem.name() + '_' + method_selection]


            savefolder = path + '\\paper1_results\\' + problem.name() + '_' + method_selection
            savename = savefolder + '\\trainy_seed_' + str(int(seed)) + '.csv'

            # create nd from trainy,
            trainy = np.loadtxt(savename, delimiter=',')
            nd_front = get_ndfront(trainy)
            # nd_front = np.loadtxt(savename, delimiter=',')
            nd_front = np.atleast_2d(nd_front)
            if id == 0:
                ax.scatter(nd_front[:, 0], nd_front[:, 1], marker='X', c='black')
            if id == 1:
                ax.scatter(nd_front[:, 0], nd_front[:, 1], marker='D')
            if id == 2:
                ax.scatter(nd_front[:, 0], nd_front[:, 1], facecolors='white', edgecolors='red', alpha=1)

        ss = 16
        plt.title(problem.name(), fontsize=ss)
        plt.legend(['PF', 'Norm$R_A$', 'Norm$R_{ND}$', 'Norm$R_{NDE}$'], fontsize=ss)
        plt.xlabel('f1', fontsize=ss)
        plt.ylabel('f2', fontsize=ss, rotation='horizontal')

        #-----
        paths = os.getcwd()
        savefolder = paths + '\\paper1_results\\plots'
        if not os.path.exists(savefolder):
             os.mkdir(savefolder)

        savename1 = savefolder + '\\' + problem.name()  + '_final.eps'
        savename2 = savefolder + '\\' + problem.name() + '_final.png'
        plt.savefig(savename1, format='eps')
        plt.savefig(savename2)

        plt.pause(5)
        plt.close()
        break

def get_paretofront(problem, n):
    from pymop.factory import get_uniform_weights
    n_obj = problem.n_obj
    if problem.name() == 'DTLZ2' or problem.name() == 'DTLZ1' or problem.name() == 'DTLZ3' or problem.name() == 'DTLZ4':
        ref_dir = get_uniform_weights(n, n_obj)
        return problem.pareto_front(ref_dir)
    else:
        return problem.pareto_front(n_pareto_points=n)

def hv_summary2csv():
    '''
    this function reads parameter files to track experimental results
    read nd front from each seed, and load pareto front to generated reference point
    use this pf reference to calculate hv for each seed
    :return: two csv files (1) raw data hv collection, (2) mean median csv collection
    '''
    import json

    problems_json = 'p/resconvert_dtlz.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_1']
    num_methods = len(methods)
    hv_raw = np.zeros((seedmax, num_pro * 3))
    path = os.getcwd()
    path = path + '\paper1_results'
    # plt.ion()
    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        pf = get_paretofront(problem, 100)
        nadir = np.max(pf, axis=0)
        ref = nadir * 1.1
        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection
            for seed in range(seedmax):
                savename = savefolder + '\\nd_seed_' + str(seed) + '.csv'
                print(savename)
                nd_front = np.loadtxt(savename, delimiter=',')
                nd_front = np.atleast_2d(nd_front)
                '''
                fig, ax = plt.subplots()
                ax.scatter(pf[:, 0], pf[:, 1])
                ax.scatter(nd_front[:, 0], nd_front[:, 1])
                plt.title(problem.name())
                plt.show()
                plt.pause(0.5)
                plt.close()
                '''
                hv = get_hv(nd_front, ref)
                print(hv)
                hv_raw[seed, problem_i * num_methods + j] = hv
    # (2) mean median collection
    hv_stat = np.zeros((4, num_pro*3))
    for i in range(num_pro*3):
        hv_stat[0, i] = np.mean(hv_raw[:, i])
        hv_stat[1, i] = np.std(hv_raw[:, i])
        sortlist = np.argsort(hv_raw[:, i])
        hv_stat[2, i] = hv_raw[:, i][sortlist[int(29/2)]]
        hv_stat[3, i] = sortlist[int(29/2)]


    plt.ioff()
    path = path + '\paper1_resconvert'
    if not os.path.exists(path):
        os.mkdir(path)
    saveraw = path + '\\hvrawdltz.csv'
    np.savetxt(saveraw, hv_raw, delimiter=',')

    target_problems = hyp['MO_target_problems']
    mat2csv(target_problems, hv_raw, hv_stat, methods, seedmax)

    #--- save
    median_id = {}
    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        for j in range(num_methods):
            method_selection = methods[j]
            name = problem.name() + '_' + method_selection
            median_id[name] = hv_stat[3, problem_i * 3 + j]
    saveName = path + '\\' + 'median_id_dtlz.joblib'
    dump(median_id, saveName)

    savestat = path + '\\hvstatdtlz.csv'
    np.savetxt(savestat, hv_stat, delimiter=',')

    print(0)


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

def get_f2hv(f, ref):
    '''
    this function takes training data as input calculate hv with ref
    :param f:
    :param ref:
    :return: one value
    '''
    nd_front = get_ndfront(f)
    hv = get_hv(nd_front, ref)
    return hv

def get_f2hvnorm(f, pf):
    '''
    this function takes training data as input calculate hv with pareto front
    and fix the problem of hv saved over generation is nomalized
    :param f:
    :param ref:
    :return: one value
    '''
    nd_front = get_ndfront(f)
    pf_endmax = np.max(pf, axis=0)
    pf_endmin = np.min(pf, axis=0)
    pf = (pf - pf_endmin) / (pf_endmax - pf_endmin)
    nd = (nd_front - pf_endmin) / (pf_endmax - pf_endmin)
    ref = [1.1] * f.shape[1]
    hv = get_hv(nd, ref)
    return hv

def get_idealdistance(train_y, problem):
    PF = get_paretofront(problem, 1000)
    # pname = problem.name()
    # PF = np.loadtxt('PF_' + pname + '.csv', delimiter=',')
    idealpoint = np.min(PF, axis=0)
    nadirpoint = np.max(PF, axis=0)

    ideal_now = np.min(train_y, axis=0)
    # normalization
    # normalize passed data first
    ideal_now = (ideal_now - idealpoint)/(nadirpoint - idealpoint)
    idealpoint = (idealpoint - idealpoint)/(nadirpoint - idealpoint)
    d = np.linalg.norm(idealpoint - ideal_now)
    return d

def trainy_summary2csv(resultfolder, result_convertfolder):
    '''
        this function reads parameter files to track experimental results
        read saved training data  from each seed, and cut the right number of evaluation
        calculate pareto front to generated reference point
        use this pf reference to calculate hv for each seed
        :return: two csv files (1) raw data hv collection, (2) mean median csv collection
    '''
    import json

    problems_json = 'p/resconvertonly1.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29
    median_visual = True

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_2', 'normalization_with_nd_5', 'normalization_with_nd_4','normalization_with_nd_6'] #'normalization_with_nd_3']   'normalization_with_external_4'
    # methods = [ 'normalization_with_nd_1']
    num_methods = len(methods)
    hv_raw = np.zeros((seedmax, num_pro * num_methods))
    ideal_distance = np.zeros((seedmax, num_pro * num_methods))
    nadir_distance = np.zeros((seedmax, num_pro * num_methods))

    path = os.getcwd()
    # path = path + '\paper1_results'
    path = path + '\\' + resultfolder

    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        pf = get_paretofront(problem, 1000)
        nadir = np.max(pf, axis=0)
        ref = nadir * 1.1

        evalnum = 300
        '''
        if 'ZDT' in problem.name():
            evalnum = 100
        if 'DTLZ' in problem.name():
            evalnum = 300  # 200   # 100
        if 'WFG' in problem.name() or 'MAF' in problem.name():
            evalnum = 300  # 300   # 250
        '''
        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection
            for seed in range(seedmax):
                savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
                print(savename)
                trainy = np.loadtxt(savename, delimiter=',')
                trainy = np.atleast_2d(trainy)

                # fixed bug on number of evaluations
                trainy = trainy[0:evalnum, :]
                # hv = get_f2hv(trainy, ref)
                hv = get_f2hvnorm(trainy, pf)
                # print(hv)
                hv_raw[seed, problem_i * num_methods + j] = hv
                d = get_idealdistance(trainy, problem)
                ideal_distance[seed, problem_i * num_methods + j] = d
                d2 = get_nadirdistance(trainy, problem, method_selection)
                nadir_distance[seed, problem_i * num_methods + j] = d2

    # (2) mean median collection
    hv_stat = np.zeros((4, num_pro * num_methods))
    for i in range(num_pro * num_methods):
    # for i in range(num_pro * 1):
        hv_stat[0, i] = np.mean(hv_raw[:, i])
        hv_stat[1, i] = np.std(hv_raw[:, i])
        sortlist = np.argsort(hv_raw[:, i])
        hv_stat[2, i] = hv_raw[:, i][sortlist[int(seedmax / 2)]]
        hv_stat[3, i] = sortlist[int(seedmax / 2)]

    # (3) ideal distance check
    d_stat = np.zeros((4, num_pro * num_methods))
    nadir_stat = np.zeros((4, num_pro * num_methods))
    for i in range(num_pro * num_methods):
        # for i in range(num_pro * 1):
        d_stat[0, i] = np.mean(ideal_distance[:, i])
        d_stat[1, i] = np.std(ideal_distance[:, i])
        sortlist = np.argsort(ideal_distance[:, i])
        d_stat[2, i] = ideal_distance[:, i][sortlist[int(seedmax / 2)]]
        d_stat[3, i] = sortlist[int(seedmax / 2)]
        if median_visual:
            problem_id = int(i/num_methods)
            method_id = int(np.mod(i, num_methods))
            problem_2visual = target_problems[problem_id]
            problem = eval(problem_2visual)
            savefolder = path + '\\' + problem.name() + '_' + methods[method_id]
            savename = savefolder + '\\trainy_seed_' + str(int(d_stat[3, i])) + '.csv'
            median_trainy = np.loadtxt(savename, delimiter=',')


            initsize = 150

            plotmedian_hv(d_stat[3, i], median_trainy, problem_2visual, method_id, initsize, resultfolder)

        nadir_stat[0, i] = np.mean(nadir_distance[:, i])
        nadir_stat[1, i] = np.std(nadir_distance[:, i])
        sortlist = np.argsort(nadir_distance[:, i])
        nadir_stat[2, i] = nadir_distance[:, i][sortlist[int(seedmax / 2)]]
        nadir_stat[3, i] = sortlist[int(seedmax / 2)]

    plt.ioff()
    # path = path + '\paper1_resconvert'
    path = path + '\\' + result_convertfolder
    if not os.path.exists(path):
        os.mkdir(path)
    saveraw = path + '\\hvraw.csv'
    np.savetxt(saveraw, hv_raw, delimiter=',')

    saveraw = path + '\\idealraw.csv'
    np.savetxt(saveraw, ideal_distance, delimiter=',')

    saveraw = path + '\\nadirraw.csv'
    np.savetxt(saveraw, nadir_distance, delimiter=',')

    target_problems = hyp['MO_target_problems']
    mat2csv(target_problems, hv_raw, hv_stat, methods, seedmax, d_stat, nadir_stat, resultfolder, result_convertfolder)

    # --- save
    median_id = {}
    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        for j in range(num_methods):
            method_selection = methods[j]
            name = problem.name() + '_' + method_selection
            median_id[name] = hv_stat[3, problem_i * 3 + j]
    saveName = path + '\\' + 'median_id.joblib'
    dump(median_id, saveName)

    savestat = path + '\\hvstat.csv'

    np.savetxt(savestat, hv_stat, delimiter=',')

    print(0)

def get_nadirdistance(trainy, problem, method_selection):

    PF = get_paretofront(problem, 1000)
    # pname = problem.name()
    # PF = np.loadtxt('PF_' + pname + '.csv', delimiter=',')
    nadir = np.max(PF, axis=0)
    ideal = np.min(PF, axis=0)

    if method_selection in 'normalization_with_self_0':
        # normalization method is with all population
        nadir_algo = np.max(trainy, axis=0)
    else:
        nd = get_ndfront(trainy)
        nadir_algo = np.max(nd, axis=0)

    # normalization
    # normalize trained algorithm first and then PF
    nadir_algo = (nadir_algo - ideal) / (nadir - ideal)
    nadir = (nadir - ideal)/(nadir - ideal)
    return np.linalg.norm(nadir - nadir_algo)

def plot_for_paper(resultfolder, result_convertfolder):

    import json
    problems_json = 'p/resconvert_plot3.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29
    median_visual = True

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_2', 'normalization_with_nd_4', 'normalization_with_nd_3'] #'normalization_with_nd_3']   'normalization_with_external_4'
    # methods = [ 'normalization_with_nd_1']
    num_methods = len(methods)
    hv_raw = np.zeros((seedmax, num_pro * num_methods))
    ideal_distance = np.zeros((seedmax, num_pro * num_methods))
    nadir_distance = np.zeros((seedmax, num_pro * num_methods))

    path = os.getcwd()
    # path = path + '\paper1_results'
    path = path + '\\' + resultfolder

    for problem_i, problem in enumerate(target_problems):
        problem = eval(problem)
        pf = get_paretofront(problem, 1000)
        nadir = np.max(pf, axis=0)
        ref = nadir * 1.1
        evalnum = 300

        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection
            for seed in range(seedmax):
                savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
                print(savename)
                trainy = np.loadtxt(savename, delimiter=',')
                trainy = np.atleast_2d(trainy)

                # fixed bug on number of evaluations
                trainy = trainy[0:evalnum, :]
                # hv = get_f2hv(trainy, ref)
                hv = get_f2hvnorm(trainy, pf)
                # print(hv)
                hv_raw[seed, problem_i * num_methods + j] = hv
                d = get_idealdistance(trainy, problem)
                ideal_distance[seed, problem_i * num_methods + j] = d
                d2 = get_nadirdistance(trainy, problem, method_selection)
                nadir_distance[seed, problem_i * num_methods + j] = d2

    # (2) mean median collection
    hv_stat = np.zeros((4, num_pro * num_methods))
    for i in range(num_pro * num_methods):
        # for i in range(num_pro * 1):
        hv_stat[0, i] = np.mean(hv_raw[:, i])
        hv_stat[1, i] = np.std(hv_raw[:, i])
        sortlist = np.argsort(hv_raw[:, i])
        hv_stat[2, i] = hv_raw[:, i][sortlist[int(seedmax / 2)]]
        hv_stat[3, i] = sortlist[int(seedmax / 2)]

    # (3) ideal distance check
    d_stat = np.zeros((4, num_pro * num_methods))
    nadir_stat = np.zeros((4, num_pro * num_methods))
    for i in range(num_pro * num_methods):
        # for i in range(num_pro * 1):
        d_stat[0, i] = np.mean(ideal_distance[:, i])
        d_stat[1, i] = np.std(ideal_distance[:, i])
        sortlist = np.argsort(ideal_distance[:, i])
        d_stat[2, i] = ideal_distance[:, i][sortlist[int(seedmax / 2)]]
        d_stat[3, i] = sortlist[int(seedmax / 2)]
        if median_visual:
            problem_id = int(i / num_methods)
            method_id = int(np.mod(i, num_methods))
            problem_2visual = target_problems[problem_id]
            problem = eval(problem_2visual)
            savefolder = path + '\\' + problem.name() + '_' + methods[method_id]
            savename = savefolder + '\\trainy_seed_' + str(int(d_stat[3, i])) + '.csv'
            median_trainy = np.loadtxt(savename, delimiter=',')

            if 'DTLZ' in problem.name() or 'MAF' in problem.name():
                initsize = 11 * problem.n_var - 1
            if 'WFG' in problem.name():
                initsize = 200

            plotmdeidan_hv_forpaper(d_stat[3, i], median_trainy, problem_2visual, method_id, initsize, resultfolder)

def plotmdeidan_hv_forpaper(median_id, trainy, problem_str, method_id, init_size, resultfolder):
    # this method plot the median results  of final ND and PF
    # methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_external_4']#'normalization_with_nd_3']
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_2',
               'normalization_with_nd_4',
               'normalization_with_nd_3']  # 'normalization_with_nd_3']   'normalization_with_external_4'
    ss = 16
    # methods_title = ['archive',  'ND front', 'external corner'] # 'corner search']
    methods_title = ['Norm$R_A$', 'Norm$R_{ND}$', 'Norm$R_{NDE}$', 'Norm$R_{NDC1}$', 'Norm$R_{NDC2}$']
    problem = eval(problem_str)
    true_pf = get_paretofront(problem, 1000)
    # ---------- visual check
    # fig, (ax1, ax2) = plt.subplots(1, 2, projection='3d')
    f1= plt.figure(figsize=(5.5, 5.5))
    ax1 = f1.add_subplot(111, projection=Axes3D.name)

    ax1.scatter3D(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=10, c='g', alpha=0.2,
                  label='PF')  # facecolors='none', edgecolors='g',
    nd_front = get_ndfront(trainy)

    ax1.scatter3D(nd_front[:, 0], nd_front[:, 1], nd_front[:, 2], c='red', label='ND')
    ax1.legend(fontsize=ss)
    ax1.set_xlabel('F1')
    ax1.set_ylabel('F2')
    ax1.set_zlabel('F3')
    t = problem.name() # + ' ' + methods_title[method_id]
    ax1.set_title(t, fontsize=ss)
    ax1.view_init(20, 340)
    # for angle in range(0, 360):
    #     ax1.view_init(30, angle)
    #    plt.pause(.001)

    path = os.getcwd()
    savefolder = path + '\\' + resultfolder + '\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + methods[method_id] + '_medianFinalnd_seed_' + str(
        int(median_id)) + '_nd.eps'
    savename2 = savefolder + '\\' + problem.name() + methods[method_id] + '_medianFinalnd_seed_' + str(
        int(median_id)) + '_nd.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)

    savename3 = savefolder + '\\' + problem.name() + methods[method_id] + '_median_seed_' + str(
        int(median_id)) + '_nd.fig.pickle'
    pickle.dump(f1, open(savename3, 'wb'))
    plt.close()




def plotmedian_hv(median_id, trainy, problem_str, method_id, init_size, resultfolder):
    # this method plot the median results  of final ND and PF
    # methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_external_4']#'normalization_with_nd_3']
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_2', 'normalization_with_nd_5',
     'normalization_with_nd_4', 'normalization_with_nd_4',  ]  # 'normalization_with_nd_3']   'normalization_with_external_4'

    # methods_title = ['archive',  'ND front', 'external corner'] # 'corner search']
    methods_title = ['Archive',  'ND', 'Extreme point', 'Corner 1 all evaluate', 'Corner 1 selective', 'Corner 1 Sel+Sil']
    problem = eval(problem_str)
    true_pf = get_paretofront(problem, 1000)
    # ---------- visual check
    # fig, (ax1, ax2) = plt.subplots(1, 2, projection='3d')
    f1 = plt.figure(figsize=(10.5, 5.5))
    ax1 = f1.add_subplot(121, projection=Axes3D.name)

    ax1.scatter3D(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=10, c='g', alpha=0.2, label='pareto front') # facecolors='none', edgecolors='g',
    nd_front = get_ndfront(trainy)

    ax1.scatter3D(nd_front[:, 0], nd_front[:, 1], nd_front[:, 2], c='red', label='ND front')
    ax1.legend()
    ax1.set_xlabel('F1')
    ax1.set_ylabel('F2')
    ax1.set_zlabel('F3')
    ax1.set_title('Final ND')
    ax1.view_init(20, 340)
    # for angle in range(0, 360):
    #     ax1.view_init(30, angle)
    #    plt.pause(.001)

    ax2 = f1.add_subplot(122, projection=Axes3D.name)
    ax2.scatter3D(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=10, c='g', alpha=0.2, label='pareto front')
    ax2.scatter3D(trainy[0:init_size, 0], trainy[0:init_size, 1], trainy[0:init_size, 2], c='orange', label='Initialization')
    ax2.legend()
    ax2.set_title('Initial archive')
    ax2.view_init(20, 340)
    title = problem.name() + ' normalization with ' + methods_title[method_id]
    f1.suptitle(title)


    path = os.getcwd()
    savefolder = path + '\\' + resultfolder + '\\process_plot'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    savename1 = savefolder + '\\' + problem.name() + methods[method_id] + '_median_seed_' + str(int(median_id)) + '_nd.eps'
    savename2 = savefolder + '\\' + problem.name() + methods[method_id] + '_median_seed_' + str(int(median_id)) + '_nd.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)


    savename3 = savefolder + '\\' + problem.name() + methods[method_id] + '_median_seed_' + str(int(median_id)) + '_nd.fig.pickle'
    pickle.dump(f1, open(savename3, 'wb'))
    plt.close()



def create_ideal_dis_file(resultfolder, resultconver):
    import json
    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_3']
    # methods = [ 'normalization_with_nd_1']
    num_methods = len(methods)

    path = os.getcwd()
    path = path + '\\' + resultfolder

    path2 = os.getcwd()
    savefolder2 = path2 + '\\' + resultfolder + '\\distance_plot'
    if not os.path.exists(savefolder2):
        os.mkdir(savefolder2)


    for problem_i, problem in enumerate(target_problems):
        problem_i = 8
        problem = target_problems[problem_i]
        problem = eval(problem)
        pf = get_paretofront(problem, 1000)
        nadir = np.max(pf, axis=0)
        ideal = np.min(pf, axis=0)

        ref = nadir * 1.1
        if 'DTLZ' in problem.name():
            evalnum = 300  # 100
            initnum = problem.n_var * 11 - 1
        if 'WFG' in problem.name():
            evalnum = 400  # 250
            initnum = 200

        for j in range(num_methods):
            method_selection = methods[j]
            savefolder = path + '\\' + problem.name() + '_' + method_selection
            n = evalnum - initnum
            ideal_distance = np.ones((seedmax, n)) * (-1)  # [seed, ideal distance of each iteration]
            nadir_distance = np.ones((seedmax, n)) * (-1)  # [seed, nadir distance of each iteration]

            for seed in range(seedmax):
                savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
                print(savename)
                trainy = np.loadtxt(savename, delimiter=',')
                trainy = np.atleast_2d(trainy)

                for iter in np.arange(initnum, evalnum):
                    iter_y = trainy[:iter, :]
                    d_ideal = get_idealdistance(iter_y, problem)
                    d_nadir = get_nadirdistance(iter_y, problem, method_selection)
                    ideal_distance[seed, iter-initnum] = d_ideal
                    nadir_distance[seed, iter-initnum] = d_nadir


            savefile1 = savefolder2 + '\\' + problem.name() + '_' + method_selection + 'ideal_check.csv'
            savefile2 = savefolder2 + '\\' + problem.name() + '_' + method_selection + 'nadir_check.csv'
            np.savetxt(savefile1, ideal_distance, delimiter=',')
            np.savetxt(savefile2, nadir_distance, delimiter=',')
        break


def extract_nadir_ideal(problem, num_methods, methods, seedmax, path, savefolder2):
    pf = get_paretofront(problem, 1000)
    nadir = np.max(pf, axis=0)
    ideal = np.min(pf, axis=0)

    ref = nadir * 1.1
    if 'DTLZ' in problem.name():
        evalnum = 300  # 100
        initnum = problem.n_var * 11 - 1
    if 'WFG' in problem.name():
        evalnum = 400  # 250
        initnum = 200

    for j in range(num_methods):
        method_selection = methods[j]
        savefolder = path + '\\' + problem.name() + '_' + method_selection
        n = evalnum - initnum
        ideal_distance = np.ones((seedmax, n)) * (-1)  # [seed, ideal distance of each iteration]
        nadir_distance = np.ones((seedmax, n)) * (-1)  # [seed, nadir distance of each iteration]
        print(j)
        for seed in range(seedmax):
            savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
            print(savename)
            trainy = np.loadtxt(savename, delimiter=',')
            trainy = np.atleast_2d(trainy)

            for iter in np.arange(initnum, evalnum):
                iter_y = trainy[:iter, :]
                d_ideal = get_idealdistance(iter_y, problem)
                d_nadir = get_nadirdistance(iter_y, problem, method_selection)
                ideal_distance[seed, iter - initnum] = d_ideal
                nadir_distance[seed, iter - initnum] = d_nadir

        savefile1 = savefolder2 + '\\' + problem.name() + '_' + method_selection + 'ideal_check.csv'
        savefile2 = savefolder2 + '\\' + problem.name() + '_' + method_selection + 'nadir_check.csv'

        np.savetxt(savefile1, ideal_distance, delimiter=',')
        np.savetxt(savefile2, nadir_distance, delimiter=',')
    return None

def parallel_extract_nadir_ideal():
    import json
    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]
    seedmax = 29

    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_3']
    # methods = [ 'normalization_with_nd_1']
    num_methods = len(methods)

    path = os.getcwd()
    path = path + '\paper1_results'

    path2 = os.getcwd()
    savefolder2 = path2 + '\\paper1_results\\distance_plot'
    if not os.path.exists(savefolder2):
        os.mkdir(savefolder2)

    args = []
    for problem_i, problem in enumerate(target_problems):
       problem = eval(problem)
       args.append((problem, num_methods, methods, seedmax, path, savefolder2))
    num_workers = 48
    pool = mp.Pool(processes=num_workers)
    pool.starmap(extract_nadir_ideal, ([arg for arg in args]))

def savePF():
    import json
    problems_json = 'p/resconvert.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    for problem_i, problem in enumerate(target_problems):
       problem = eval(problem)
       PF = get_paretofront(problem, 1000)
       path = os.getcwd()
       path = path + '\paper1_results\PF_'
       savename = path + problem.name() + '.csv'
       np.savetxt(savename, PF, delimiter=',')





def plot3dresults():
    import json

    problems_json = 'p/resconvert_dtlz.json'

    # (1) load parameter settings
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)

    target_problems = hyp['MO_target_problems']
    # target_problems = target_problems[4:5]


    num_pro = len(target_problems)
    methods = ['normalization_with_self_0', 'normalization_with_nd_0', 'normalization_with_nd_3']
    num_methods = len(methods)

    prob_id = 2
    med_id = 2
    seed = 4
    path = os.getcwd()
    path = path + '\paper1_results'
    # plt.ion()
    problem = target_problems[prob_id]
    problem = eval(problem)
    pf = get_paretofront(problem, 100)
    nadir = np.max(pf, axis=0)
    ref = nadir * 1.1

    if 'ZDT' in problem.name():
        evalnum = 100
    if 'WFG' in problem.name():
        evalnum = 250
    if 'DTLZ' in problem.name():
        evalnum = 200


    method_selection = methods[med_id]
    savefolder = path + '\\' + problem.name() + '_' + method_selection

    savename = savefolder + '\\trainy_seed_' + str(seed) + '.csv'
    print(savename)
    trainy = np.loadtxt(savename, delimiter=',')
    trainy = np.atleast_2d(trainy)

    # fix bug
    trainy = trainy[0:evalnum, :]
    plot_process3d(problem, trainy, method_selection, seed)


def activation_count():
    path = os.getcwd()
    problem = eval("MAF.MAF3(n_var=6, n_obj=3)")
    resultfolder = 'results_OBJ3'
    path = path + '\\' + resultfolder
    savefolder = path + '\\' + problem.name() + '_' + 'normalization_with_nd_3\\activationcheck_seed_0.joblib'

    d = load(savefolder)
    print(d)



if __name__ == "__main__":
    # resultfolder = 'paper1_results'
    # resultconver = 'paper1_convert'

    # activation_count()

    resultfolder = 'results_OBJ3'
    resultconver = 'paper1_convert'

    # plot_for_paper(resultfolder, resultconver)
    # import pickle

    # path = os.getcwd()
    # savefolder = path + '\\paper1_results\\process_plot'
    # savename3 = savefolder + '\\DTLZ1normalization_with_self_0_median_seed_2_nd.fig.pickle'


    # figx = pickle.load(open(savename3, 'rb'))
    # figx.show()  # Show the figure, edit it, etc.!
    # savePF()
    # nadirplot(resultfolder, 'distance_plot')
    # parallel_extract_nadir_ideal()
    # create_ideal_dis_file(resultfolder, resultconver)
    trainy_summary2csv(resultfolder, resultconver)
    # prediction_accuracy()

    # hv_summary2csv()
    # hv_medianplot()
    # plot3dresults()
    # hvconverge_averageplot(resultfolder, resultconver)

    # nadirplot()
    # hv_medianplot3in1()
