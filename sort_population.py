import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import copy

def sort_population(popsize,nobj,ncon,infeasible,feasible,all_cv,all_f):
    l2=[]
    l1=[]
    sl=[]
    ff=[]
    if ncon!=0:
        infeasible=np.asarray(infeasible)
        infeasible=infeasible.flatten()
        index1 = all_cv[infeasible].argsort()
        index1=index1.tolist()
        l2=infeasible[index1]
    if len(feasible)>=1:
        ff = all_f[feasible, :]
        if nobj==1:
            ff=ff.flatten()
            index1 = ff.argsort()
            index1=index1.tolist()
            l1=feasible[index1]
        if nobj>1:
            sl = pg.sort_population_mo(ff)
            l1 = feasible[sl]
    order=np.append(l1, l2, axis=0)
    order=order.flatten()
    selected=order[0:popsize]
    selected=selected.flatten()
    selected=selected.astype(int)
    return selected


def sort_population_cornerfirst(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f, all_x):
    # return ordered ID
    l1 = []
    order = []
    n_var = all_x.shape[1]

    if ncon != 0:
        print('this sorting does not deal with constraints')
    else:
        # sort all f assume all f is of form  [f1, f2, f3, f1^2+f2^2, f1^2+f3^2, f1^2+f3^2]
        n = all_f.shape[0]
        a = np.linspace(0, n - 1, n, dtype=int)
        uniq_f, indx_unique = np.unique(all_f, return_index=True, axis=0)
        indx_same = np.setdiff1d(a, indx_unique)

        #
        for i in range(nobj):
            single_colid = np.argsort(uniq_f[:, i])
            l1 = np.append(l1, single_colid)

        # l1 is unique objectives' sorted ID
        l1 = np.atleast_2d(l1).reshape(-1, nobj, order='F')

        # corner sort unique objective's ID
        # candidate_id is selecting from each objective
        # order is the list of selected ID
        i = 0
        while len(order) < uniq_f.shape[0]:
            # print(order)
            candidate_id = l1[0, i]

            order = np.append(order, candidate_id)
            # rearrange l1 remove candidate id
            l1 = l1.flatten(order='F')
            remove_id = np.where(l1 == candidate_id)
            l1 = np.delete(l1, remove_id)
            l1 = np.atleast_2d(l1).reshape(-1, nobj, order='F')

            # cycled pointer
            i = i + 1
            if i >= nobj:
                i = 0

    # convert back to original population ID
    selected = indx_unique[order.astype(int)]
    selected = np.append(selected, indx_same).flatten()
    selected = selected[0:popsize]
    selected = selected.astype(int)
    return selected


def sort_population_cornerlexicon(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f, all_x):
    # return ordered ID
    l1 = []
    order = []
    n_var = all_x.shape[1]

    if ncon != 0:
        print('this sorting does not deal with constraints')
    else:
        # sort all f assume all f is of form  [f1, f2, f3, f1^2+f2^2, f1^2+f3^2, f1^2+f3^2]
        n = all_f.shape[0]
        a = np.linspace(0, n - 1, n, dtype=int)
        uniq_f, indx_unique = np.unique(all_f, return_index=True, axis=0)
        indx_same = np.setdiff1d(a, indx_unique)

        #
        for i in range(nobj):
            # single_colid = np.argsort(uniq_f[:, i])
            # sort with lexcon style
            indx = np.arange(nobj - 1, -1, -1)
            indx = np.delete(indx, nobj - 1 - i)
            indx = np.append(indx, i)

            #
            rearrange_uniquef = uniq_f[:, indx]
            rearrange_uniquef = rearrange_uniquef.transpose()
            single_colid = np.lexsort(
                rearrange_uniquef)  # lex sort key is last row, so need to transpose original matrix
            l1 = np.append(l1, single_colid)

        # l1 is unique objectives' sorted ID
        l1 = np.atleast_2d(l1).reshape(-1, nobj, order='F')

        # corner sort unique objective's ID
        # candidate_id is selecting from each objective
        # order is the list of selected ID
        i = 0
        while len(order) < uniq_f.shape[0]:
            # print(order)
            candidate_id = l1[0, i]

            order = np.append(order, candidate_id)
            # rearrange l1 remove candidate id
            l1 = l1.flatten(order='F')
            remove_id = np.where(l1 == candidate_id)
            l1 = np.delete(l1, remove_id)
            l1 = np.atleast_2d(l1).reshape(-1, nobj, order='F')

            # cycled pointer
            i = i + 1
            if i >= nobj:
                i = 0

    # convert back to original population ID
    selected = indx_unique[order.astype(int)]
    selected = np.append(selected, indx_same).flatten()
    selected = selected[0:popsize]
    selected = selected.astype(int)
    return selected




def sort_population_NDcorner(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f, all_x):
    # this corner sorting method will consider sorting in ND front
    # note: all_f and all_x passed in are combined population of parents and children
    # (1) DRS should be removed before this method is called
    # (2) separate ND front from each ND tier
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(all_f)
    ndf = list(ndf)
    sorted_id = []
    num_fronts = len(ndf)

    # ax = plt.axes(projection='3d')
    for k in range(num_fronts):
        ndf_index = ndf[k]
        nd_frontsize = len(ndf_index)
        nd_frontf = all_f[ndf_index, :]
        nd_frontx = all_x[ndf_index, :]

        # for each tier of fronts do corner sorting
        nd_sortedid = sort_population_cornerfirst(nd_frontsize, nobj, ncon, infeasible, feasible, all_cv, nd_frontf, nd_frontx)
        # cornersort_plot(ax, nd_sortedid, nd_frontf)

        # should convert back to  original
        front_sortedid = copy.deepcopy(ndf_index)
        front_sortedid = front_sortedid[nd_sortedid]

        sorted_id = np.append(sorted_id, front_sortedid)

    # plt.close()
    # (3) select population size ID as returned value
    selected = sorted_id[0:popsize]
    selected = selected.astype(int)
    return selected


def cornersort_plot(ax, sorted_id, nd_frontf):

    nd_frontf = nd_frontf[sorted_id, :]
    ax.scatter(nd_frontf[:, 0], nd_frontf[:, 1],  nd_frontf[:, 1],  c='blue', s=10)
    a = 0
