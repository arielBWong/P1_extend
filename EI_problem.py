import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array
import pygmo as pg
from scipy import special

import copy

def gaussiancdf(x):
    # x = check_array(x)
    y = 0.5 * (1 + special.erf(x / np.sqrt(2)))
    return y

def gausspdf(x):
    # x = check_array(x)
    y = 1/np.sqrt(2*np.pi) * np.exp(-np.square(x)/2)
    return y

def ego_eim(x, krg, nd_front, ref):
    '''
    this method calculate eim fitness
    ** warning: constraint problem not considered
    '''

    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    n_obj = len(krg)
    pred_obj = []
    pred_sigma = []
    for model in krg:
        y, s = model.predict(x)
        pred_obj = np.append(pred_obj, y)
        pred_sigma = np.append(pred_sigma, s)

    pred_obj = np.atleast_2d(pred_obj).reshape(-1, n_obj, order='F')
    pred_sigma = np.atleast_2d(pred_sigma).reshape(-1, n_obj, order='F')

    fit = EIM_hv(pred_obj, pred_sigma, nd_front, ref)
    return fit

def EIM_hv(mu, sig, nd_front, reference_point):
    '''
    this method calculate expected improvement with regard to hypervolume
    '''
    # mu sig nu_front has to be np_2d
    mu = check_array(mu)
    sig = check_array(sig)
    nd_front = check_array(nd_front)
    reference_point = check_array(reference_point)

    n_nd = nd_front.shape[0]
    n_mu = mu.shape[0]

    mu_extend = np.repeat(mu, n_nd, axis=0)
    sig_extend = np.repeat(sig, n_nd, axis=0)
    r_extend = np.repeat(reference_point, n_nd * n_mu, axis=0)

    nd_front_extend = np.tile(nd_front, (n_mu, 1))

    imp = (nd_front_extend - mu_extend)/sig_extend
    EIM = (nd_front_extend - mu_extend) * gaussiancdf(imp) + \
            sig_extend * gausspdf(imp)

    y1 = np.prod((r_extend - nd_front_extend + EIM), axis=1)
    y2 = np.prod((r_extend - nd_front_extend), axis=1)

    y = np.atleast_2d((y1 - y2)).reshape(-1, n_nd)
    y = np.min(y, axis=1)
    y = np.atleast_2d(y).reshape(-1, 1)
    return y

def eim_eu(mu, sig, nd_front, ref):
    '''
    this function uses euclidean distance to calcuate fitness
    for each point, calculate eim w.r.t. each nd
    squared sum over objectives and then sqrt (euclidean distance to each nd front),
    select the minimum
    '''

    mu = check_array(mu)
    sig = check_array(sig)
    nd_front = check_array(nd_front)
    ref = check_array(ref)

    n_nd = nd_front.shape[0]
    n_mu = mu.shape[0]

    mu_extend = np.repeat(mu, n_nd, axis=0)
    sig_extend = np.repeat(sig, n_nd, axis=0)
    r_extend = np.repeat(ref, n_nd * n_mu, axis=0)
    nd_front_extend = np.tile(nd_front, (n_mu, 1))

    eim = (nd_front_extend - mu_extend) * gaussiancdf((nd_front_extend - mu_extend)/sig_extend) + \
           sig_extend * gausspdf((nd_front_extend - mu_extend)/sig_extend)

    eim = np.sum(eim ** 2, axis=1)  # squared sum over objectives
    eim = np.sqrt(eim)              # sqrt over objectives, get euclidean distance
    eim = eim.reshape(n_mu, -1)     # arrange 'c type', row: nd fronts size

    out = np.min(eim, axis=1)        # minimum one over nd fronts
    return out

def eim_maxmin(mu, sig, nd_front, ref):
    '''
       this function uses max min metric to calcuate fitness
       for each point, calculate eim w.r.t. each nd results should be [size_nd, size_obj]
       for each point max over objectives and then min over nd ,
       return the min value as fitness
       '''
    mu = check_array(mu)
    sig = check_array(sig)
    nd_front = check_array(nd_front)
    ref = check_array(ref)

    n_nd = nd_front.shape[0]
    n_mu = mu.shape[0]

    mu_extend = np.repeat(mu, n_nd, axis=0)
    sig_extend = np.repeat(sig, n_nd, axis=0)
    r_extend = np.repeat(ref, n_nd * n_mu, axis=0)
    nd_front_extend = np.tile(nd_front, (n_mu, 1))

    eim = (nd_front_extend - mu_extend) * gaussiancdf((nd_front_extend - mu_extend) / sig_extend) + \
          sig_extend * gausspdf((nd_front_extend - mu_extend) / sig_extend)

    eim = np.atleast_2d(eim)
    eim = np.max(eim, axis=1)   # max over objectives
    eim = eim.reshape(n_mu, -1)  # re-arrange to row being nd front size
    out = np.min(eim, axis=1)   #  min over nd front
    return out



def ego_believer(x, krg, nd_front, ref):
    '''
    This function return x's evaluation results using kriging believer and hypervolume
    :param x: population of design variables to be evaluated with kriging
    :param krg:  (list) kriging models for each objective
    :param nd_front:  current nd front for mo problems
    :param ref:  reference point for calculating hypervolume
    :return:
    '''

    # only for predicted normalization situation idealsearch = 4
    org = nd_front.shape[0]
    nd_front_check = copy.deepcopy(nd_front)
    nd_front_check[nd_front_check <= 1.1] = 0
    nd_front_check = nd_front_check.sum(axis=1)
    deleterows = np.nonzero(nd_front_check)

    nd_front = np.delete(nd_front, deleterows, 0)
    if nd_front.shape[0] == 0:
        ndhv_value = 0
        nd_front = None
        print('external normalization caused ALL real ND front fall out of normalization boundary')
    else:
        if nd_front.shape[0] < org:
            print('external normalization caused SOME real ND front fall out of normalization boundary')
        hv_class = pg.hypervolume(nd_front)
        ndhv_value = hv_class.compute(ref)

    # --------------------
    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    n_obj = len(krg)
    pred_obj = []
    for model in krg:
        y, _ = model.predict(x)
        pred_obj = np.append(pred_obj, y)

    pred_obj = np.atleast_2d(pred_obj).reshape(-1, n_obj,  order='F')

    fit = np.zeros((n_samples, 1))
    for i in range(n_samples):
        pred_instance = pred_obj[i, :]
        if np.any(pred_instance - ref >= 0):
            fit[i] = 0
        else:
            if nd_front is not None:
                hv_class = pg.hypervolume(np.vstack((nd_front, pred_instance)))
            else:
                pred_instance = np.atleast_2d(pred_instance)
                hv_class = pg.hypervolume(pred_instance)

            fit[i] = hv_class.compute(ref) - ndhv_value

    fit = np.atleast_2d(fit)
    return fit




def ego_eim(x, krg, nd_front, ref):
    '''
    multiple objective, no constraint problems
    :param x:
    :param krg:
    :param nd_front:
    :param ref:
    :return:
    '''
    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    n_obj = len(krg)
    pred_obj = []
    pred_sig = []
    for model in krg:
        y, s = model.predict(x)
        pred_obj = np.append(pred_obj, y)
        pred_sig = np.append(pred_sig, s)
    pred_obj = np.reshape(pred_obj,(-1, n_obj), order='F')
    pred_sig = np.reshape(pred_sig, (-1, n_obj), order='F')

    eim = EIM_hv(pred_obj, pred_sig, nd_front, ref)
    return eim


def expected_improvement(X,
                         X_sample,
                         Y_sample,
                         y_mean,
                         y_std,
                         cons_g_mean,
                         cons_g_std,
                         feasible, gpr,
                         gpr_g=None,
                         xi=0.01):

    # X_sample/Y_sample, in the denormalized range
    n_samples = X.shape[0]
    n_obj = len(gpr)
    # mu, sigma = gpr.predict(X, return_std=True)
    mu_temp = np.zeros((n_samples, 1))
    sigma_temp = np.zeros((n_samples, 1))


    convert_index = 0
    for g in gpr:
        mu, sigma = g.predict(X, return_cov=True)

        # convert back to denormalized range
        sigma = np.atleast_2d(sigma)
        sigma = sigma * y_std[convert_index] + y_mean[convert_index]
        sigma_temp = np.hstack((sigma_temp, sigma))

        # convert back to denormalized range
        mu = np.atleast_2d(mu)
        mu = mu * y_std[convert_index] + y_mean[convert_index]
        mu_temp = np.hstack((mu_temp, mu))

        convert_index = convert_index + 1

    mu = np.delete(mu_temp, 0, 1).reshape(n_samples, n_obj)
    sigma = np.delete(sigma_temp, 0, 1).reshape(n_samples, n_obj)


    pf = 1.0

    if len(gpr_g) > 0:
        # with constraint
        n_g = len(gpr_g)
        mu_temp = np.zeros((n_samples, 1))
        sigma_temp = np.zeros((n_samples, 1))
        convert_index = 0
        for g in gpr_g:
            mu_gx, sigma_gx = g.predict(X, return_cov=True)

            # pf operate on denormalized range
            mu_gx = np.atleast_2d(mu_gx)
            mu_gx = mu_gx * cons_g_std[convert_index] + cons_g_mean[convert_index]
            mu_temp = np.hstack((mu_temp, mu_gx))

            # gpr prediction on sigma is not the same dimension as the mu
            # details have not been checked, here just make a conversion
            # on sigma
            sigma_gx = np.atleast_2d(sigma_gx)
            sigma_gx = sigma_gx * cons_g_std[convert_index] + cons_g_mean[convert_index]
            sigma_temp = np.hstack((sigma_temp, sigma_gx))

            convert_index = convert_index + 1

        # re-organise, and delete zero volume
        mu_gx = np.delete(mu_temp, 0, 1)
        sigma_gx = np.delete(sigma_temp, 0, 1)

        with np.errstate(divide='warn'):

            if sigma_gx == 0:
                z = 0
            pf = norm.cdf((0 - mu_gx) / sigma_gx)
            # create pf on multiple constraints (multiply over all constraints)
            pf_m = pf[:, 0]
            for i in np.arange(1, n_g):
                pf_m = pf_m * pf[:, i]
            pf = np.atleast_2d(pf_m).reshape(-1, 1)

        if feasible.size > 0:
            # If there is feasible solutions
            # EI to look for both feasible and EI preferred solution
            mu_sample_opt = np.min(feasible)
        else:
            # If there is no feasible solution,
            # then EI go look for feasible solutions
            return pf
    else:
        # without constraint
        mu_sample_opt = np.min(Y_sample)

    if len(gpr) > 1:
        # multi-objective situation
        if len(gpr_g) > 0:
            # this condition means mu_gx has been calculated
            if feasible.size > 0:
                ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(feasible)
                f_pareto = feasible[ndf, :]
                point_nadir = np.max(f_pareto, axis=0)
                point_reference = point_nadir * 1.1

                # calculate hyper volume
                point_list = np.vstack((f_pareto, mu))
                if mu[0][0] > point_reference[0][0] or mu[0][1] > point_reference[0][1]:
                    ei = 1e-5
                else:
                    hv = pg.hypervolume(point_list)
                    hv_value = hv.compute(point_reference)
                    ei = hv_value

            else:
                return pf
        else:
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(Y_sample)
            ndf = list(ndf)
            f_pareto = Y_sample[ndf[0], :]
            point_nadir = np.max(f_pareto, axis=0)
            point_reference = point_nadir * 1.1

            # calculate hyper volume
            point_list = np.vstack((f_pareto, mu))
            if mu[0, 0] > point_reference[0] or mu[0, 1] > point_reference[1]:
                ei = 1e-5
            else:
                hv = pg.hypervolume(point_list)
                hv_value = hv.compute(point_reference)
                ei = hv_value

    else:
        # single objective situation
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu
            # print(imp.shape)
            # print(sigma.shape)
            Z = imp / sigma
            ei1 = imp * norm.cdf(Z)
            ei1[sigma == 0.0] = 0.0
            ei2 = sigma * norm.pdf(Z)
            ei = (ei1 + ei2)

    pena_ei = ei * pf
    pena_ei = np.atleast_2d(pena_ei)
    # print('return penalized ei')

    return pena_ei


# this acqusition function on G should be refactored
def acqusition_function(x,
                        out,
                        X_sample,
                        Y_sample,
                        y_mean,
                        y_std,
                        cons_g_mean,
                        cons_g_std,
                        gpr,
                        gpr_g,
                        feasible,
                        xi=0.01):

    dim = X_sample.shape[1]
    x = np.atleast_2d(x).reshape(-1, dim)

    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x,
                                     X_sample,
                                     Y_sample,
                                     y_mean,
                                     y_std,
                                     cons_g_mean,
                                     cons_g_std,
                                     feasible,
                                     gpr,
                                     gpr_g,
                                     xi=0.01)



