import numpy as np
from create_child import create_child, create_child_c
from sort_population import sort_population
import time
from scipy.optimize import differential_evolution
import pygmo as pg
import matplotlib.pyplot as plt
import pyDOE
from EI_krg import normalization_with_nd, EI_hv, EI_hv_contribution
from sklearn.utils.validation import check_array
import os
from matplotlib import cm

def optimizer(problem, nobj, ncon, bounds, recordFlag, pop_test, mut, crossp, popsize, its,  **kwargs):

    record_f = list()
    record_x = list()

    dimensions = len(bounds)
    pop_g = []
    archive_g = []
    all_cv = []
    pop_cv = []
    a = np.linspace(0, 2 * popsize - 1, 2 * popsize, dtype=int)

    if len(kwargs) != 0:
        # print(kwargs['add_info'])
        guide_x = kwargs['add_info']

    all_cv = np.zeros((2 * popsize, 1))
    all_g = np.zeros((2 * popsize, ncon))
    pop_g = np.zeros((popsize, ncon))
    pop_cv = np.zeros((2 * popsize, 1))
    child_g = np.zeros((popsize, ncon))
    archive_g = pop_g
    all_x = np.zeros((2 * popsize, dimensions))
    all_f = np.zeros((2 * popsize, nobj))
    pop_f = np.zeros((popsize, nobj))
    child_f = np.zeros((popsize, nobj))
    pop = np.random.rand(popsize, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_x = min_b + pop * diff
    archive_x = pop
    archive_f = pop_f

    # print(pop)
    if len(kwargs) != 0:
        pop_x[0, :] = guide_x
        pop[0, :] = (guide_x - min_b)/diff

    if pop_test is not None:
        pop = pop_test
        pop_x = min_b + pop * diff


    if ncon != 0:
        pop_f, pop_g = problem.evaluate(pop_x, return_values_of=["F", "G"], **kwargs)
        tmp = pop_g
        tmp[tmp <= 0] = 0
        pop_cv = tmp.sum(axis=1)

    if ncon == 0:
        # np.savetxt('test_x.csv', pop_x, delimiter=',')
        pop_f = problem.evaluate(pop_x, return_values_of=["F"], **kwargs)


    # Over the generations
    for i in range(its):

        start = time.time()
        child_x = create_child_c(dimensions, bounds, popsize, crossp, mut, pop, pop_f, 20, 30)
        end = time.time()
        # print('create child time used %.4f' % (end - start))

        start = time.time()
        trial_denorm = min_b + child_x * diff
        if ncon != 0:
            child_f, child_g = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
        if ncon == 0:
            child_f = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)
        end = time.time()
        # print(' evaluation time used %.4f' % (end - start))

        # Parents and offspring
        all_x = np.append(pop, child_x, axis=0)
        all_f = np.append(pop_f, child_f, axis=0)
        if ncon != 0:
            all_g = np.append(pop_g, child_g, axis=0)
            all_g[all_g <= 0] = 0
            all_cv = all_g.sum(axis=1)
            infeasible = np.nonzero(all_cv)
            feasible = np.setdiff1d(a, infeasible)
        if ncon == 0:
            feasible = a
            infeasible = []

        feasible = np.asarray(feasible)
        feasible = feasible.flatten()

        start = time.time()

        # Selecting the parents for the next generation
        selected = sort_population(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)
        end = time.time()
        # print('sort time used %.4f' % (end - start))

        pop = all_x[selected, :]
        pop_f = all_f[selected, :]

        # insert a crossvalidation

        if ncon != 0:
            pop_g = all_g[selected, :]

        # Storing all solutions in tha archive
        archive_x = np.append(archive_x, child_x, axis=0)
        archive_f = np.append(archive_f, child_f)
        if ncon != 0:
            archive_g = np.append(archive_g, child_g)

        if recordFlag:
            # record all best_individual
            record_f = np.append(record_f, pop_f[0, :])
            record_x = np.append(record_x, min_b + diff * pop[0, :])

    # Getting the variables in appropriate bounds
    pop_x = min_b + pop * diff
    archive_x = min_b + archive_x * diff

    return pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x)

def plot_infill_landscape(train_x, train_y, norm_train_y, krg, krg_g, nadir, ideal, feasible, ei_method, problem_name):
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)

    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = train_y[ndf_extend, :]

    plt.ion()
    plt.clf()
    min_by_feature = np.min(train_y)
    max_by_feature = np.max(train_y)
    n_vals = train_y.shape[1]
    number_of_initial_samples = 10000

    generated_samples = pyDOE.lhs(n_vals, number_of_initial_samples)
    generated_samples = min_by_feature + (max_by_feature - min_by_feature) * generated_samples

    norm_mu, norm_nd, point_reference = normalization_with_nd(generated_samples, train_y)
    ei = EI_hv_contribution(norm_mu, norm_nd, point_reference)
    cm1 = plt.cm.get_cmap('RdYlBu')
    ei = ei.ravel()
    plt.scatter(generated_samples[:, 0].ravel(), generated_samples[:, 1].ravel(), c=ei, cmap=cm1)
    plt.scatter(nd_front[:, 0], nd_front[:, 1])
    plt.pause(0.5)
    plt.ioff()


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

def plotde_gens(ax, popf, denormalize, norm_orig, real_prob):
    '''
    This function takes current f population of de, denormalize them
    and plot to ax
    :param ax: plot ax
    :param popf: population to be ploted
    :param denormalize denomalization function name
    :param norm_orig  list of data that is used for creating  normalization bounds
    :return:
    '''

    #----restore background
    ax.cla()
    true_pf = real_prob.pareto_front(n_pareto_points=100)
    ax.scatter(true_pf[:, 0], true_pf[:, 1], c='red', s=0.2)
    ax.scatter(norm_orig[:, 0], norm_orig[:, 1], c='blue')
    nd_front = get_ndfront(norm_orig)
    ax.scatter(nd_front[:, 0], nd_front[:, 1], c='red')
    # -------restore back ground

    popf_denorm = denormalize(popf, norm_orig)
    ax.scatter(popf_denorm[:, 0], popf_denorm[:, 1], facecolors='none', edgecolors='black')
    plt.pause(0.1)

def visualize_egobelieverde(ax, pop_x, **kwargs):
    denormalize = kwargs['denorm']
    norm_orig = kwargs['normdata']
    pred_model = kwargs['pred_model']
    real_prob = kwargs['real_prob']
    pred_f = model_pred(pop_x, pred_model)
    plotde_gens(ax, pred_f, denormalize, norm_orig, real_prob)

def visualize_firstgenlandscape(pop_x, **kwargs):
    '''
    this function creates a landscape plot of first generation for de
    it uses ref (reference point, hard coded),  to form boundary for plot
    then uses meshgrid type data to form a color plot of search landscape
    :param pop_x:
    :param kwargs:
    :return:
    '''

    denormalize = kwargs['denorm']
    norm_orig = kwargs['normdata']
    pred_model = kwargs['pred_model']
    real_prob = kwargs['real_prob']

    pred_f = model_pred(pop_x, pred_model)
    n_obj = pred_f.shape[1]
    if n_obj > 2:
        raise ( "not compatible with objective more than 2")

    # landscape value is based on contribution on hv
    # so nd front is the base line
    # in order to plot clearly, convert plot to original space
    true_pf = real_prob.pareto_front(n_pareto_points=1000)
    ref = [1.1] * n_obj
    ideal_zerodn = np.min(true_pf, axis=0)
    ref_dn = denormalize(np.atleast_2d(ref), norm_orig)

    #----create mesh grip to plot landscape
    n = 100
    f1 = np.linspace(ideal_zerodn[0], ref_dn[0, 0], n)
    f2 = np.linspace(ideal_zerodn[1], ref_dn[0, 1], n)


    nd_front = get_ndfront(norm_orig)  # original space
    hv_class = pg.hypervolume(nd_front)
    ref_dn = ref_dn.flatten()
    ndhv_value = hv_class.compute(ref_dn)

    # meshgrid
    f = np.zeros((n, n))
    f1_m, f2_m = np.meshgrid(f1, f2)
    for i in range(n):
        for j in range(n):
            pred_instance = [f1_m[i, j], f2_m[i, j]]
            if np.any(pred_instance - ref_dn >= 0):
                f[i, j] = 0
            else:
                hv_class = pg.hypervolume(np.vstack((nd_front, pred_instance)))
                f[i, j] = hv_class.compute(ref_dn) - ndhv_value

    plt.ion()
    figure, ax = plt.subplots()
    # plot search landscape
    ms = ax.pcolormesh(f1_m, f2_m, f, shading='auto', cmap='RdYlBu')

    # plot training for surrogate
    # and its nd front
    # plot pareto front
    ax.scatter(true_pf[:, 0], true_pf[:, 1], c='green', s=1)
    # plot training data of surrogate
    ax.scatter(norm_orig[:, 0], norm_orig[:, 1], c='black', s=1)
    nd_front = get_ndfront(norm_orig)
    # plot nd front
    ax.scatter(nd_front[:, 0], nd_front[:, 1], c='blue')
    # plot reference point
    ax.scatter(ref_dn[0], ref_dn[1], marker='X', c='green')
    left, right = plt.xlim()
    ax.set_xlim(left, right + 0.1)
    bottom, top = plt.ylim()
    ax.set_ylim(bottom, top + 0.1)
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    plt.colorbar(ms, ax=ax)
    plt.legend(['PF', 'Archive', 'Init nd', 'Ref'])
    plt.title(real_prob.name())

    # plt.pause(5)
    # -- plot save verbose---
    path = os.getcwd()
    path = path + '\paper1_results'
    savefolder = path + '\\' + real_prob.name()
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    method_part1 = kwargs['method']
    method_part2 = kwargs['ideal_search']
    seed = kwargs['seed']
    savename1 = savefolder + '\\' + str(method_part1) + '_' + str(method_part2) + '_firstgenlandscape' + str(seed) + '.eps'
    savename2 = savefolder + '\\' + str(method_part1) + '_' + str(method_part2) + '_firstgenlandscape' + str(seed) + '.png'
    plt.savefig(savename1, format='eps')
    plt.savefig(savename2)
    plt.pause(1)
    plt.close()
    plt.ioff()
    a = 0






def model_pred(x, models):
    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    n_obj = len(models)
    pred_obj = []
    for model in models:
        y, _ = model.predict(x)
        pred_obj = np.append(pred_obj, y)

    pred_obj = np.atleast_2d(pred_obj).reshape(-1, n_obj, order='F')
    return pred_obj



def optimizer_DE(problem, ncon, bounds, insertpop, F, CR, NP, itermax, visflag, ax, **kwargs):
    #  NP: number of population members/popsize
    #  itermax: number of generation
    #  kwargs for this method is for plot, keys are train_y
    ax = None
    dimensions = len(bounds)
    # Check input variables
    VTR = -np.inf
    refresh = 0
    F = 0.8
    CR = 0.8
    strategy = 6
    use_vectorize = 1

    if NP < 5:
        NP = 5
        print('pop size is increased to minimize size 5')

    if CR < 0 or CR > 1:
        CR = 0.5
        print('CR should be from interval [0,1]; set to default value 0.5')

    if itermax <= 0:
        itermax = 200
        print('generation size is set to default 200')

    # insert guide population
    if insertpop is not None:
        check_array(insertpop)
        n_insertpop = len(insertpop)
        n_rest = NP - n_insertpop
        if n_insertpop > NP:  # rear situation where nd is larger than evolution population size
            n_rest = 1
            insertpop = insertpop[0:NP-1, :]
    else:
        n_rest = NP

    # Initialize population and some arrays
    # if pop is a matrix of size NPxD. It will be initialized with random
    # values between the min and max values of the parameters
    min_b, max_b = np.asarray(bounds).T
    pop = np.random.rand(n_rest, dimensions)
    pop_x = min_b + pop * (max_b - min_b)
    if insertpop is not None:  # attach guide population
        pop_x = np.vstack((pop_x, insertpop))

    XVmin = np.repeat(np.atleast_2d(min_b), NP, axis=0)
    XVmax = np.repeat(np.atleast_2d(max_b), NP, axis=0)

    if ncon != 0:
        pop_f, pop_g = problem.evaluate(pop_x, return_values_of=["F", "G"], **kwargs)
        tmp = pop_g.copy()
        tmp[tmp <= 0] = 0
        pop_cv = tmp.sum(axis=1)

    if ncon == 0:
        # np.savetxt('test_x.csv', pop_x, delimiter=',')
        pop_f = problem.evaluate(pop_x, return_values_of=["F"], **kwargs)

    # best member of current iteration
    bestval = np.min(pop_f)  # single objective only
    ibest = np.where(pop_f == bestval)  # what if multiple best values?
    bestmemit = pop_x[ibest[0][0]]  # np.where return tuple of (row_list, col_list)

    # save best_x ever
    bestmem = bestmemit

    # DE-Minimization
    # popold is the population which has to compete. It is static through one
    # iteration. pop is the newly emerging population
    # initialize bestmember  matrix
    bm = np.zeros((NP, dimensions))

    # intermediate population of perturbed vectors
    ui = np.zeros((NP, dimensions))

    # rotating index array (size NP)
    rot = np.arange(0, NP)

    # rotating index array (size D)
    rotd = np.arange(0, dimensions)  # (0:1:D-1);

    iter = 1
    while iter < itermax and bestval > VTR:
        if visflag and (problem.name not in 'all but one'):
            # visflag and ax come in pairs
            visualize_egobelieverde(ax, pop_x, **kwargs)
            if iter == 1:
                visualize_firstgenlandscape(pop_x, **kwargs)
                exit(1)
        elif visflag and (problem.name in 'all but one'):
            # show visualizaiton landscape and final result
            x1 = np.arange(bounds[0][0], bounds[0][1], 0.01)
            nn = len(x1)
            x2 = np.arange(bounds[1][0], bounds[1][1], 0.01)
            X, Y = np.meshgrid(x1, x2)
            X1 = np.atleast_2d(X).reshape(-1, 1)
            Y1 = np.atleast_2d(Y).reshape(-1, 1)
            xx = np.hstack((X1, Y1))
            yy = problem.evaluate(xx, return_values_of=["F"])
            Z = np.atleast_2d(yy).reshape(-1, nn)

            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.1)
            ax.scatter3D(pop_x[:, 0], pop_x[:, 1], pop_f[:, 0], marker=7, c='black')



        # save the old population
        # print('iteration: %d' % iter)
        oldpop_x = pop_x.copy()

        # index pointer array
        ind = np.random.permutation(4) + 1

        # shuffle locations of vectors
        a1 = np.random.permutation(NP)

        # rotate indices by ind(1) positions
        rt = np.remainder(rot + ind[0], NP)
        # rotate vector locations
        a2 = a1[rt]

        rt = np.remainder(rot + ind[1], NP)
        a3 = a2[rt]

        rt = np.remainder(rot + ind[2], NP)
        a4 = a3[rt]

        rt = np.remainder(rot + ind[3], NP)
        a5 = a4[rt]
        # for test
        # a5 = np.loadtxt('a5.csv', delimiter=',')
        # a5 = np.array(list(map(int, a5)))-1

        # shuffled population 1
        pm1 = oldpop_x[a1, :]
        pm2 = oldpop_x[a2, :]
        pm3 = oldpop_x[a3, :]
        pm4 = oldpop_x[a4, :]
        pm5 = oldpop_x[a5, :]

        # population filled with the best member of the last iteration
        # print(bestmemit)
        for i in range(NP):
            bm[i, :] = bestmemit

        mui = np.random.rand(NP, dimensions) < CR
        if strategy > 5:
            st = strategy - 5
        else:
            # exponential crossover
            st = strategy
            # transpose, collect 1's in each column
            # did not implement following strategy process

        # inverse mask to mui
        # mpo = ~mui same as following one
        mpo = mui < 0.5

        if st == 1:  # DE/best/1
            # differential variation
            ui = bm + F * (pm1 - pm2)  # permutate best member population
            # crossover
            ui = oldpop_x * mpo + ui * mui  # partially old population, partially new population

        if st == 2:  # DE/rand/1
            # differential variation
            ui = pm3 + F * (pm1 - pm2)
            # crossover
            ui = oldpop_x * mpo + ui * mui
        if st == 3:  # DE/rand-to-best/1
            ui = oldpop_x + F * (bm - oldpop_x) + F * (pm1 - pm2)
            ui = oldpop_x * mpo + ui * mui
        if st == 4:  # DE/best/2
            ui = bm + F * (pm1 - pm2 + pm3 - pm4)
            ui = oldpop_x * mpo + ui * mui
        if st == 5:  #DE/rand/2
            ui = pm5 + F * (pm1 - pm2 + pm3 - pm4)
            ui = oldpop_x * mpo + ui * mui


        # correcting violations on the lower bounds of the variables
        # validate components
        maskLB = ui > XVmin
        maskUB = ui < XVmax

        # part one: valid points are saved, part two/three beyond bounds are set as bounds
        ui = ui * maskLB * maskUB + XVmin * (~maskLB) + XVmax * (~maskUB)

        # Select which vectors are allowed to enter the new population
        if use_vectorize == 1:

            if ncon != 0:
                pop_f_temp, pop_g_temp = problem.evaluate(ui, return_values_of=["F", "G"], **kwargs)
                tmp = pop_g_temp.copy()
                tmp[tmp <= 0] = 0
                pop_cv_temp = tmp.sum(axis=1)

            if ncon == 0:
                # np.savetxt('test_x.csv', pop_x, delimiter=',')
                pop_f_temp = problem.evaluate(ui, return_values_of=["F"], **kwargs)

            # if competitor is better than value in "cost array"
            indx = pop_f_temp <= pop_f
            # replace old vector with new one (for new iteration)
            change = np.where(indx)
            pop_x[change[0], :] = ui[change[0], :]
            pop_f[change[0], :] = pop_f_temp[change[0], :]

            # we update bestval only in case of success to save time
            indx = pop_f_temp < bestval
            if np.sum(indx) != 0:
                # best member of current iteration
                bestval = np.min(pop_f_temp)  # single objective only
                ibest = np.where(pop_f_temp == bestval)  # what if multiple best values?
                if len(ibest[0]) > 1:
                    print(
                        "multiple best values, selected first"
                    )
                bestmem = ui[ibest[0][0], :]
            # freeze the best member of this iteration for the coming
            # iteration. This is needed for some of the strategies.
            bestmemit = bestmem.copy()

            if visflag and (problem.name in 'all but one'):
                ax.cla()
                ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
                ax.scatter3D(pop_x[:, 0], pop_x[:, 1], pop_f[:, 0], marker=7, c='black')


        if refresh == 1:
            print('Iteration: %d,  Best: %.4f,  F: %.4f,  CR: %.4f,  NP: %d' % (iter, bestval, F, CR, NP))

        iter = iter + 1
        del oldpop_x

    return np.atleast_2d(bestmem), np.atleast_2d(bestval), pop_x, pop_f






