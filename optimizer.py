import numpy as np
#from test_function import fobj
from create_child import create_child, create_child_c
from sort_population import sort_population, sort_population_cornerfirst, sort_population_NDcorner, sort_population_cornerlexicon
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
from paper1_refactor import get_ndfront, get_ndfrontx
from EI_krg import acqusition_function, close_adjustment
import copy


from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score


def cross_val(val_x, val_y, **kwargs):
    gpr = kwargs['gpr']
    val_x = val_x.reshape(-1, 1)
    pred_y = gpr.predict(val_x)
    mse = mean_squared_error(val_y, pred_y)

    val_x  = val_x[0, 0]
    val_y  = val_y[0]
    pred_y = pred_y[0, 0]

    print('cross validation on x %.2f, real_y is %0.2f, predicted_y is %0.2f, mse is %0.2f' % (val_x, val_y, pred_y, mse))
    return mse

def optimizer(problem, nobj, ncon, bounds, mut, crossp, popsize, its,  **kwargs):
    '''

    :param problem:
    :param nobj:
    :param ncon:
    :param bounds: upper and lower bounds of problem variables
    :param mut: mutation rate
    :param crossp:  crossover rate
    :param popsize:  population size
    :param its:
    :param val_data:
    :return:
    '''
    dimensions = len(bounds)
    pop_g = []
    archive_g = []
    all_cv = []
    pop_cv = []
    a = np.linspace(0, 2*popsize-1, 2*popsize, dtype=int)
   
    all_cv = np.zeros((2*popsize, 1))
    all_g = np.zeros((2*popsize, ncon))
    pop_g = np.zeros((popsize, ncon))
    pop_cv = np.zeros((2*popsize, 1))
    child_g = np.zeros((popsize, ncon))
    archive_g = pop_g
    all_x = np.zeros((2*popsize, dimensions))
    all_f = np.zeros((2*popsize, nobj))
    pop_f = np.zeros((popsize, nobj))
    child_f = np.zeros((popsize, nobj))
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_x = min_b + pop * diff
    archive_x = pop
    archive_f = pop_f
    for ind in range(popsize):
        if ncon != 0:
            pop_f[ind, :], pop_g[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F", "G"], **kwargs)
            tmp = pop_g
            tmp[tmp <= 0] = 0
            pop_cv = tmp.sum(axis=1)

        if ncon == 0:
            # print('initialization loglikelihood check send in %d th theta: %0.4f ' % (ind, pop_x[ind, :]))
            pop_f[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F"], **kwargs)
       
    # Over the generations
    for i in range(its):
        # child_x = create_child(dimensions, bounds, popsize, crossp, mut, pop)
        start = time.time()
        child_x = create_child_c(dimensions, bounds, popsize, crossp, mut, pop, pop_f, 20, 30)
        end = time.time()
        # print('create child time used %.4f' % (end-start))

        start = time.time()
        # Evaluating the offspring
        for ind in range(popsize):
            trial_denorm = min_b + child_x[ind, :] * diff
            if ncon != 0:
                child_f[ind, :], child_g[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
            if ncon == 0:
                # print('over generation %d send in %d th theta: ' % (i, ind))
                child_f[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)
        end = time.time()
        # print('population evaluation time used %.4f' % (end - start))


        start = time.time()
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
        # Selecting the parents for the next generation
        selected = sort_population(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)



        end = time.time()
        # print('sorting  time used %.4f' % (end - start))
        
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

    # Getting the variables in appropriate bounds    
    pop_x = min_b + pop * diff
    archive_x = min_b + archive_x * diff
    return pop_x, pop_f, pop_g, archive_x, archive_f, archive_g

def optimizer_forcornersearch(problem, nobj, ncon, bounds, mut, crossp, popsize, its, **kwargs):
    '''

    :param problem:
    :param nobj:
    :param ncon:
    :param bounds: upper and lower bounds of problem variables
    :param mut: mutation rate
    :param crossp:  crossover rate
    :param popsize:  population size
    :param its:
    :param val_data:
    :return:
    '''

    visualize = False
    if 'visualize' in kwargs.keys():
        ax = plt.axes(projection='3d')
        visualize = True

    dimensions = len(bounds)
    pop_g = []
    archive_g = []
    all_cv = []
    pop_cv = []
    a = np.linspace(0, 2 * popsize - 1, 2 * popsize, dtype=int)

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

    # bring real ND front into search process
    if 'insert_x' in kwargs.keys():
        nd_x = kwargs['insert_x']
        len_nd = nd_x.shape[0]
        # delete len_nd randomly generated solutions
        if len_nd > popsize:
            len_nd = popsize
        del_id = range(len_nd)


        pop_x = np.delete(pop_x, del_id, axis=0)
        pop = np.delete(pop, del_id, axis=0)

        # add  ND front to population
        # f does not matter
        pop_x = np.vstack((pop_x, nd_x))
        norm_ndx = (nd_x - min_b)/diff
        pop = np.vstack((pop, norm_ndx))


    archive_x = pop
    archive_f = pop_f
    for ind in range(popsize):
        if ncon != 0:
            pop_f[ind, :], pop_g[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F", "G"], **kwargs)
            tmp = pop_g
            tmp[tmp <= 0] = 0
            pop_cv = tmp.sum(axis=1)

        if ncon == 0:
            # print('initialization loglikelihood check send in %d th theta: %0.4f ' % (ind, pop_x[ind, :]))
            pop_f[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F"], **kwargs)

    # Over the generations
    for i in range(its):
        # child_x = create_child(dimensions, bounds, popsize, crossp, mut, pop)
        start = time.time()
        child_x = create_child_c(dimensions, bounds, popsize, crossp, mut, pop, pop_f, 20, 30)
        end = time.time()
        # print('create child time used %.4f' % (end-start))

        start = time.time()
        # Evaluating the offspring
        for ind in range(popsize):
            trial_denorm = min_b + child_x[ind, :] * diff
            if ncon != 0:
                child_f[ind, :], child_g[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
            if ncon == 0:
                # print('over generation %d send in %d th theta: ' % (i, ind))
                child_f[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)
        end = time.time()
        # print('population evaluation time used %.4f' % (end - start))

        start = time.time()
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


        # remove DRS
        # this is not an efficient way to deal with DRS
        # because only 1st ND front needs DRS remove
        # the result dominated solutions do not need
        all_f = close_adjustment(all_f)

        # Selecting the parents for the next generation
        # selected = sort_population(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)
        selected = sort_population_cornerfirst(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f, all_x)
        # selected = sort_population_NDcorner(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f, all_x)
        # selected = sort_population_cornerlexicon(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f, all_x)

        end = time.time()
        # print('sorting  time used %.4f' % (end - start))

        pop = all_x[selected, :]
        pop_f = all_f[selected, :]

        if 'trainy' in kwargs.keys() and visualize:
            plot_cornersearch(ax, problem, pop_f, pop_x, False, **kwargs)
        elif 'trainy' not in kwargs.keys() and visualize:
            plot_cornersearch_nosurrogate(ax, problem, pop_f, pop_x, False, **kwargs)

        # insert a crossvalidation
        if ncon != 0:
            pop_g = all_g[selected, :]

        # Storing all solutions in tha archive
        archive_x = np.append(archive_x, child_x, axis=0)
        archive_f = np.append(archive_f, child_f)
        if ncon != 0:
            archive_g = np.append(archive_g, child_g)

    # Getting the variables in appropriate bounds
    pop_x = min_b + pop * diff
    archive_x = min_b + archive_x * diff


    if 'trainy' in kwargs.keys() and visualize:
        plot_cornersearch(ax, problem, pop_f, pop_x, True, **kwargs)
    elif 'trainy' not in kwargs.keys() and visualize:
        plot_cornersearch_nosurrogate(ax, problem, pop_f, pop_x, True, **kwargs)


    # plt.close()
    return pop_x, pop_f, pop_g, archive_x, archive_f, archive_g



def plot_cornersearch(ax, problem, pop_f, pop_x, last, **kwargs):
    # ax = plt.axes(projection='3d')
    ax.cla()
    inner_problem = kwargs['inner_problem']
    true_pf = get_paretofront(inner_problem, 1000)
    ax.scatter3D(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], c='green', alpha=0.2, label='PF')
    trainy = kwargs['trainy']
    pop_f3 = copy.deepcopy(pop_f[:, 0:3])

    denormalization = kwargs['denorm']


    if 'external_y' in kwargs.keys():
        corner = kwargs['external_y']
        pop_f3 = denormalization(pop_f3, trainy, corner)
    else:
        pop_f3 = denormalization(pop_f3, trainy)  # careful with denormalization
        ref = [1.1, 1.1, 1.1]
        ref = denormalization(ref, trainy)

    ax.scatter3D(pop_f3[:, 0], pop_f3[:, 1], pop_f3[:, 2], c='tab:pink', s=10, label='population')
    ax.scatter3D(ref[0], ref[1], ref[2], c='tab:purple', s=50, label='ref')

    ndf3 = get_ndfront(pop_f3)
    ax.scatter3D(ndf3[:, 0], ndf3[:, 1], ndf3[:, 2], c='blue', s=20, marker='x', alpha=0.2)

    plt.title(problem.name)
    if last:
        kmeans = False
        if 'external_y' in kwargs.keys():
            pop_f = close_adjustment(pop_f)

            # pick top 6 from ND front
            nd = get_ndfront(pop_f)
            ndx = get_ndfrontx(pop_x, pop_f)

            selected = sort_population_cornerfirst(nd.shape[0], problem.n_obj, 0, [], [], [], nd, ndx)
            nd = nd[selected, :]
            ndx = ndx[selected, :]
            nd3 = nd[:, 0:3]
            nd3 = denormalization(nd3, trainy, corner)

            kmeans = cluster.k_means(nd3, n_clusters=3)

            id1 = np.where(kmeans[1] == 0)
            batch1 = nd3[id1[0], :]
            ax.scatter3D(batch1[:, 0], batch1[:, 1], batch1[:, 2], c='tab:orange', s=80)

            id2 = np.where(kmeans[1] == 1)
            batch2 = nd3[id2[0], :]
            ax.scatter3D(batch2[:, 0], batch2[:, 1], batch2[:, 2], c='tab:purple', s=80)

            id3= np.where(kmeans[1] == 2)
            batch3 = nd3[id3[0], :]
            ax.scatter3D(batch3[:, 0], batch3[:, 1], batch3[:, 2], c='tab:olive', s=80)
        elif kmeans:
            pop_f = close_adjustment(pop_f)
            nd = get_ndfront(pop_f)
            ndx = get_ndfrontx(pop_x, pop_f)
            selected = sort_population_cornerfirst(nd.shape[0], problem.n_obj, 0, [], [], [], nd, ndx)
            nd = nd[selected, :]
            ndx = ndx[selected, :]
            x_out, f_out = kmeans_selection(nd, ndx, 6)
        else:
            # pick top 6 from ND front
            pop_f = close_adjustment(pop_f)
            nd = get_ndfront(pop_f)
            ndx = get_ndfrontx(pop_x, pop_f)
            selected = sort_population_cornerfirst(nd.shape[0], problem.n_obj, 0, [], [], [], nd, ndx)
            nd = nd[selected, :]
            ndx = ndx[selected, :]

            if nd.shape[0] >= problem.n_obj:
                x_out = ndx[0: problem.n_obj, :]
                f_out = nd[0:problem.n_obj, :]
            else:
                x_out = ndx
                f_out = nd



        f_out = f_out[:, 0:3]
        f_out = denormalization(f_out, trainy)
        nd_f = get_ndfront(trainy)
        ax.scatter3D(f_out[:, 0], f_out[:, 1],  f_out[:, 2], c='tab:orange', marker='x', s=80, label='selected')
        ax.scatter3D(nd_f[:, 0], nd_f[:, 1], nd_f[:, 2], c='tab:olive', marker='d', s=80, label='normalization nd')


    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')

    plt.legend()
    plt.pause(0.1)


def plot_cornersearch_nosurrogate(ax, problem, pop_f, pop_x, last, **kwargs):
    # this method plots corner search process without surrogate
    ax.cla()
    PF = get_paretofront(problem, 1000)
    ax.scatter3D(pop_f[:, 0], pop_f[:, 1], pop_f[:, 2], c='orange', marker='x', s=50, label='population')
    ax.scatter3D(PF[:, 0], PF[:, 1], PF[:, 2], c='g', alpha=0.2, label='PF')

    if last:
        # pick up nd front
        pop_f = close_adjustment(pop_f)
        nd = get_ndfront(pop_f)
        ndx = get_ndfrontx(pop_x, pop_f)
        selected = sort_population_cornerfirst(nd.shape[0], problem.n_obj, 0, [], [], [], nd, ndx)
        nd = nd[selected, :]
        ndx = ndx[selected, :]


        # Silhouette selection
        sil_score = []
        n = problem.n_obj
        colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:olive']
        true_nobj = 3
        clusters = np.arange(2, n + 1)
        for i in clusters:
            labels = cluster.KMeans(n_clusters=i).fit_predict(nd)
            s = silhouette_score(nd, labels)
            sil_score = np.append( sil_score,s)
        k = np.argsort(sil_score)
        best_cluster = clusters[k[-1]]  # max sort value
        print('best cluster number: '+ str(best_cluster))
        labels = cluster.KMeans(n_clusters=best_cluster).fit_predict(nd)

        out_x = []
        out_f = []
        n_var = ndx.shape[1]
        f_var = nd.shape[1]

        for k in range(best_cluster):
            idk = np.where(labels == k)
            batchkf = nd[idk[0], :]

            ax.scatter3D(batchkf[:, 0], batchkf[:, 1], batchkf[:, 2], c=colors[k])

            batchkx = ndx[idk[0], :]
            batchkd = np.linalg.norm(batchkf[:, 0:true_nobj], axis=1)
            dist_orderk = np.argsort(batchkd)

            x = batchkx[dist_orderk[0], :]
            f = batchkf[dist_orderk[0], :]
            out_x = np.append(out_x, x)
            out_f = np.append(out_f, f)
            ax.scatter3D(f[0], f[ 1], f[2], c=colors[k], marker='x', s=200)
    plt.legend()
    plt.pause(0.1)

def kmeans_selection(nd, ndx, n):
    from sklearn import cluster
    kmeans = cluster.k_means(nd, n_clusters=n)
    out_x = []
    out_f = []
    n_var = ndx.shape[1]
    f_var = nd.shape[1]
    for k in range(n):
        idk = np.where(kmeans[1] == k)
        batchkf = nd[idk[0], :]
        batchkd = np.linalg.norm(batchkf[:, 0:3], axis=1)
        dist_orderk = np.argsort(batchkd)
        batchkx = ndx[idk[0], :]
        x = batchkx[dist_orderk[0], :]
        f = batchkf[dist_orderk[0], :]
        out_x = np.append(out_x, x)
        out_f = np.append(out_f, f)

    out_x = np.atleast_2d(out_x).reshape(-1, n_var)
    out_f = np.atleast_2d(out_f).reshape(-1, f_var)
    return out_x, out_f


def get_paretofront(problem, n):
    from pymop.factory import get_uniform_weights
    n_obj = problem.n_obj
    if problem.name() == 'DTLZ1' or problem.name() == 'DTLZ2' or problem.name() == 'DTLZ3'\
            or problem.name() == 'DTLZ4':
        ref_dir = get_uniform_weights(n, n_obj)
        return problem.pareto_front(ref_dir)
    else:
        return problem.pareto_front(n_pareto_points=n)
