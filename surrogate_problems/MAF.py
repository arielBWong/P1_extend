
import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
from scipy.special import comb
from functools import reduce
from math import fabs, ceil, floor, sin, cos, pi
from operator import mul
from copy import deepcopy
from itertools import combinations
from scipy.spatial.distance import cdist
from matplotlib import path

import copy

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def vec_nchoosek(v, k):
    """v is a vector, k is the number of elements to be chosen"""
    k_obj = combinations(v, k)
    out = []
    for ki in k_obj:
        out = np.append(out, ki)

    out = np.atleast_2d(out).reshape(-1, k)
    return out

def uniform_points(n_samples, n_obj):
    samples = None
    h1 = 1
    while comb(h1 + n_obj, n_obj - 1) <= n_samples:
        h1 = h1 + 1

    compo1 = vec_nchoosek(np.arange(1, h1 + n_obj), n_obj - 1)
    t = np.atleast_2d(np.arange(0, n_obj - 1))
    compo2 = np.repeat(t, comb(h1 + n_obj - 1, n_obj - 1), axis=0)
    W = compo1 - compo2 - 1

    t = np.atleast_2d(np.zeros((W.shape[0], 1))) + h1
    w1 = np.hstack((W, t))
    t2 = np.atleast_2d(np.zeros((W.shape[0], 1)))
    w2 = np.hstack((t2, W))
    W = (w1 - w2) / h1

    if h1 < n_obj:
        h2 = 0
        while comb(h1 + n_obj - 1, n_obj - 1) + comb(h2 + n_obj, n_obj - 1) <= n_samples:
            h2 = h2 + 1

        if h2 > 0:
            t1 = vec_nchoosek(np.arange(1, h2 + n_obj), n_obj - 1)
            t2 = np.atleast_2d(np.arange(0, n_obj - 1))
            t2 = np.repeat(t2, comb(h2 + n_obj - 1, n_obj - 1), axis=0)
            W2 = t1 - t2 - 1

            t1 = np.atleast_2d(np.zeros((W2.shape[0], 1))) + h2
            t1 = np.hstack((W2, t1))
            t2 = np.atleast_2d(np.zeros((W2.shape[0], 1)))
            t2 = np.hstack((t2, W2))
            W2 = (t1 - t2) / h2

            t3 = W2 / 2 + 1 / (2 * n_obj)
            W = np.vstack((W, t3))

    # fix 0
    W = np.maximum(W, 1e-6)
    new_sample_size = W.shape[0]
    return new_sample_size, W

def Rastrigin(x):
    x = check_array(x)
    f = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10, 1)
    return np.atleast_2d(f).reshape(-1, 1)

def Rosenbrock(x):
    x = check_array(x)
    n, d = x.shape
    f = np.sum(100 * ((x[:, np.arange(0, d-1)])**2 - x[:, 1:d]) ** 2 + (x[:, 0:d-1]-1)**2, 1)
    return np.atleast_2d(f).reshape(-1, 1)

'''function f = Rastrigin(x)
    f = sum(x.^2-10.*cos(2.*pi.*x)+10,2);
end

function f = Rosenbrock(x)
    f = sum(100.*(x(:,1:size(x,2)-1).^2-x(:,2:size(x,2))).^2+(x(:,1:size(x,2)-1)-1).^2,2);
end'''



class MAF1(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array([1] * self.n_var)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):

        # g      = sum((PopDec(:,obj.M:end)-0.5).^2,2);
        # PopObj = repmat(1+g,1,obj.M) - repmat(1+g,1,obj.M).*fliplr(cumprod([ones(size(g,1),1),PopDec(:,1:obj.M-1)],2)).*[ones(size(g,1),1),1-PopDec(:,obj.M-1:-1:1)];

        x = check_array(x)
        n = x.shape[0]

        g = np.sum((x[:, self.n_obj-1:] - 0.5)**2, 1)
        g = np.atleast_2d(g).reshape(n, 1)


        f1 = np.hstack((np.ones((n, 1)), x[:, 0:self.n_obj-1]))

        f2 = 1 - np.fliplr(x[:, 0:self.n_obj-1])
        f2 = np.hstack((np.ones((n, 1)), f2))
        f = (1+g) - (1+g) * np.fliplr(np.cumprod(f1, axis=1)) * f2
        f = np.atleast_2d(f)

        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        # R = 1 - UniformPoint(N, obj.M);
        p_size, p =  uniform_points(n_pareto_points, self.n_obj)
        return 1-p

class MAF2(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array([1] * self.n_var)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        ''' matlab implementation
        g = zeros(size(PopDec, 1), obj.M);
        for m = 1: obj.M
        if m < obj.M
            g(:, m) = sum(((PopDec(:, obj.M+(m-1) * floor((obj.D - obj.M + 1) / obj.M)
                           :obj.M+m * floor((obj.D-obj.M+1) / obj.M) - 1) / 2 + 1 / 4)-0.5).^ 2, 2);
            else
            g(:, m) = sum(
                ((PopDec(:, obj.M+(obj.M-1) * floor((obj.D - obj.M + 1) / obj.M):obj.D) / 2 + 1 / 4) - 0.5).^ 2, 2);
            end
        end
        PopObj = (1 + g). * fliplr(
            cumprod([ones(size(g, 1), 1), cos((PopDec(:, 1:obj.M-1) / 2 + 1 / 4) * pi / 2)], 2)).*[ones(size(g, 1), 1),
                                                                                                   sin((PopDec(
                                                                                                       :, obj.M-1:-1:1) / 2 + 1 / 4) * pi / 2)];
        '''
        x = check_array(x)

        n = x.shape[0]
        xx = np.hstack((np.zeros((n, 1)), x))     # dummy x for index adjustment
        g = np.zeros((n, self.n_obj + 1))      # adjust index  lazy solution,  add a dummy first column  so that index same as matlab implementation
        for m in np.arange(1, self.n_obj + 1): # range 1-M (obj)
            if m < self.n_obj:
                idx_start = self.n_obj + (m-1) * np.floor((self.n_var - self.n_obj + 1)/self.n_obj)
                idx_end = self.n_obj + m * np.floor((self.n_var - self.n_obj + 1)/self.n_obj) - 1
                if idx_start <= idx_end:
                    g[:, m] = np.sum((xx[:, int(idx_start):int(idx_end) + 1]/2 + 0.25 - 0.5)**2, 1)  # +1 is for slicing convention (right open bracket)
                # else keep g column as zeros same as matlab procedure
            else:
                idx_start = self.n_obj + (self.n_obj-1) * np.floor((self.n_var - self.n_obj + 1)/self.n_obj)
                idx_end = self.n_var
                if idx_start <= idx_end:
                    g[:, m] = np.sum((xx[:, int(idx_start): int(idx_end) + 1]/2 + 0.25 - 0.5)**2, 1)

        #
        g = np.delete(g, 0, 1)  # return to python indexing
        f1 = np.hstack((np.ones((n, 1)), np.cos(np.pi/2 * (x[:, 0:self.n_obj-1]/2 + 0.25))))
        f1 = np.fliplr(np.cumprod(f1, axis=1))
        f2 = np.hstack((np.ones((n, 1)), np.sin(np.pi/2 * (np.fliplr(x[:, 0: self.n_obj-1])/2 + 0.25))))

        out['F'] = (1 + g) * (f1 * f2)




    def _calc_pareto_front(self, n_pareto_points=100):
        '''
        R = UniformPoint(N,obj.M);
            c = zeros(size(R,1),obj.M-1);
            for i = 1 : size(R,1)
                for j = 2 : obj.M
                    temp = R(i,j)/R(i,1)*prod(c(i,obj.M-j+2:obj.M-1));
                    c(i,obj.M-j+1) = sqrt(1/(1+temp^2));
                end
            end
            if obj.M > 5
                c = c.*(cos(pi/8)-cos(3*pi/8)) + cos(3*pi/8);
            else
                c(any(c<cos(3*pi/8)|c>cos(pi/8),2),:) = [];
            end
            R = fliplr(cumprod([ones(size(c,1),1),c(:,1:obj.M-1)],2)).*[ones(size(c,1),1),sqrt(1-c(:,obj.M-1:-1:1).^2)];

        '''
        n, R = uniform_points(n_pareto_points, self.n_obj)
        c = np.zeros((n, self.n_obj))     # convert to matlab index
        RR = np.hstack((np.zeros((n, 1)), R))
        for i in np.arange(n):                    # row index does not change
            for j in np.arange(2, self.n_obj + 1):  # column index change
                indx_start = self.n_obj - j + 2
                indx_end = self. n_obj
                if indx_start < indx_end:
                    cc = c[i, indx_start: indx_end]
                else:
                    cc = [1]

                tmp = RR[i, j]/RR[i, 1] * np.prod(cc)
                c[i, self.n_obj - j + 1] = np.sqrt(1 / (1 + tmp ** 2))

        # convert back python index
        c = np.delete(c, 0, 1)
        if self.n_obj > 5:
            c = c * (np.cos(np.pi/8) - np.cos(3*np.pi/8)) + np.cos(3*np.pi/8)
        else:
            cc = copy.deepcopy(c)
            cc[cc < np.cos(3*np.pi/8)] = -1
            cc[cc > np.cos(np.pi/8)] = -1
            cc[cc > 0] = 0
            cc = np.sum(cc, 1)
            del_idx = np.nonzero(cc)
            c = np.delete(c, del_idx, 0)
        n = c.shape[0]
        #
        f1 = np.hstack((np.ones((n, 1)), c[:, 0:self.n_obj-1]))
        f2 = np.hstack((np.ones((n, 1)), np.sqrt(1 - np.fliplr(c[:, 0:self.n_obj-1])**2)))
        R = np.fliplr(np.cumprod(f1, axis=1)) * f2
        return R


class MAF3(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array([1] * self.n_var)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        '''g = 100*(obj.D-obj.M+1+sum((PopDec(:,obj.M:end)-0.5).^2-cos(20.*pi.*(PopDec(:,obj.M:end)-0.5)),2));
            PopObj = repmat(1+g,1,obj.M).*fliplr(cumprod([ones(size(g,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)];
            PopObj = [PopObj(:,1:obj.M-1).^4,PopObj(:,obj.M).^2];'''

        x = check_array(x)
        n = x.shape[0]
        # f = (x[:, self.n_obj-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[:, self.n_obj-1:] - 0.5))
        # g = 100 * (self.n_var - self.n_obj + 1 + np.sum(f, 1))

        f = (x[:, self.n_obj - 1:] - 0.5) ** 2 - np.cos(1 * np.pi * (x[:, self.n_obj - 1:] - 0.5))
        g = 1 * (self.n_var - self.n_obj + 1 + np.sum(f, 1))

        g = np.atleast_2d(g).reshape(n, -1)

        f1 = np.hstack((np.ones((n, 1)), np.cos(x[:, 0:self.n_obj-1] * np.pi/2)))
        f2 = np.hstack((np.ones((n, 1)), np.sin(np.fliplr(x[:, 0:self.n_obj-1]) * np.pi/2)))

        f = (1+g) * np.fliplr(np.cumprod(f1, axis=1)) * f2
        f = np.hstack((f[:, 0:self.n_obj-1]**4, np.atleast_2d(f[:, self.n_obj-1]**2).reshape(n, -1)))

        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        ''' R    = UniformPoint(N,obj.M).^2;
            temp = sum(sqrt(R(:,1:end-1)),2) + R(:,end);
            R    = R./[repmat(temp.^2,1,size(R,2)-1),temp];'''
        n, R = uniform_points(n_pareto_points, self.n_obj)
        R = R**2
        tmp = np.sum(np.sqrt(R[:, 0:-1]), 1) + R[:, -1]
        tmp = np.atleast_2d(tmp).reshape(-1, 1)
        m = R.shape[1]
        d = np.hstack((np.tile(tmp**2, (1, m-1)), tmp))
        R = R/d

        return R

class MAF4(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array([1] * self.n_var)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        ''' matlab platemo
         g = 100*(obj.D-obj.M+1+sum((PopDec(:,obj.M:end)-0.5).^2-cos(20.*pi.*(PopDec(:,obj.M:end)-0.5)),2));
         PopObj = repmat(1+g,1,obj.M) - repmat(1+g,1,obj.M).*fliplr(cumprod([ones(size(PopDec,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)],2)).*[ones(size(PopDec,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)];
         PopObj = PopObj.*repmat(2.^(1:obj.M),size(PopDec,1),1);
        '''
        x = check_array(x)
        n = x.shape[0]
        # f = (x[:, self.n_obj-1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[:, self.n_obj-1:] - 0.5))
        # g = 100 * (self.n_var - self.n_obj + 1 + np.sum(f, 1))

        f = (x[:, self.n_obj - 1:] - 0.5) ** 2 - np.cos(1* np.pi * (x[:, self.n_obj - 1:] - 0.5))
        g = 1 * (self.n_var - self.n_obj + 1 + np.sum(f, 1))

        g = np.atleast_2d(g).reshape(n, 1)

        # [ones(size(PopDec,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)]
        f1 = np.hstack((np.ones((n, 1)), np.cos(x[:, 0:self.n_obj-1] * np.pi/2)))
        # [ones(size(PopDec,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)]
        f2 = np.hstack((np.ones((n, 1)), np.sin(np.fliplr(x[:, 0:self.n_obj-1]) * np.pi/2)))

        f3 = (1+g) - (1+g) * np.fliplr(np.cumprod(f1, axis=1)) * f2
        f4 = np.power(np.ones((1, self.n_obj)) * 2, np.arange(1, self.n_obj+1))
        f = f3 * f4
        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        ''' platemo
        R = UniformPoint(N,obj.M);
        R = R./repmat(sqrt(sum(R.^2,2)),1,obj.M);
        R = (1-R).*repmat(2.^(1:obj.M),size(R,1),1);'''

        n, R = uniform_points(n_pareto_points, self.n_obj)
        d1 = np.sqrt(np.sum(R**2, axis=1))
        d1 = np.atleast_2d(d1).reshape(-1, 1)
        d1 = np.tile(d1, (1, self.n_obj))
        R = R/d1
        d2 = np.power(np.ones((1, self.n_obj)) * 2, np.arange(1, self.n_obj+1))
        R = (1-R) * d2
        return R


class MAF5(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array([1] * self.n_var)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        ''' platemo
         PopDec(:,1:obj.M-1) = PopDec(:,1:obj.M-1).^100;
            g      = sum((PopDec(:,obj.M:end)-0.5).^2,2);
            PopObj = repmat(1+g,1,obj.M).*fliplr(cumprod([ones(size(g,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)];
            PopObj = PopObj.*repmat(2.^(obj.M:-1:1),size(g,1),1);
        '''
        x[:, 0:self.n_obj-1] = x[:, 0:self.n_obj-1] ** 100
        g = np.sum((x[:, self.n_obj-1:]-0.5) ** 2, axis=1)
        g = np.atleast_2d(g).reshape(-1, 1)
        n = g.shape[0]
        # [ones(size(g,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)]
        f1 = np.hstack((np.ones((n, 1)), np.cos(x[:, 0:self.n_obj - 1] * np.pi / 2)))
        # [ones(size(g,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)];
        f2 = np.hstack((np.ones((n, 1)), np.sin(np.fliplr(x[:, 0:self.n_obj - 1]) * np.pi / 2)))

        f3 = (1+g) * np.fliplr(np.cumprod(f1, axis=1)) * f2
        f4 = np.power(np.ones((1, self.n_obj)) * 2, np.arange(self.n_obj, 0, -1))
        f = f3 * f4
        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        '''
        R = UniformPoint(N,obj.M);
        R = R./repmat(sqrt(sum(R.^2,2)),1,obj.M);
        R = R.*repmat(2.^(obj.M:-1:1),size(R,1),1);
        '''
        n, R = uniform_points(n_pareto_points, self.n_obj)
        d1 = np.sqrt(np.sum(R ** 2, axis=1))
        d1 = np.atleast_2d(d1).reshape(-1, 1)
        d1 = np.tile(d1, (1, self.n_obj))
        R = R / d1
        d2 = np.power(np.ones((1, self.n_obj)) * 2, np.arange(self.n_obj, 0, -1))
        R = R * d2
        return R


class MAF6(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array([1] * self.n_var)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double,
                         evaluation_of="manual")

    def _evaluate(self, x, out, *args, **kwargs):
        ''' platemo
         I       = 2;
         g       = sum((PopDec(:,obj.M:end)-0.5).^2,2);
         Temp    = repmat(g,1,obj.M-I);
         PopDec(:,I:obj.M-1) = (1+2*Temp.*PopDec(:,I:obj.M-1))./(2+2*Temp);
         PopObj = repmat(1+100*g,1,obj.M).*fliplr(cumprod([ones(size(g,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)];
        '''
        if self.n_obj >= self.n_var:
            print("number of objective should be higher than number of variables")
            return

        I = 2
        g = np.sum((x[:, self.n_obj-1:] - 0.5)**2, 1)
        g = np.atleast_2d(g).reshape(-1, 1)

        tmp = np.tile(g, (1, self.n_obj - I))
        # PopDec(:,I:obj.M-1)--> [I-1, M-2]--> [I-1, M-1)
        # (1+2*Temp.*PopDec(:,I:obj.M-1))./(2+2*Temp)
        if I <= self.n_obj-1:
            x[:, I-1: self.n_obj-1] = (1 + 2 * tmp * x[:, I-1: self.n_obj - 1]) / (2 + 2 *tmp)

        n = g.shape[0]
        # [ones(size(g,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)]
        f1 = np.hstack((np.ones((n, 1)), np.cos(x[:, 0:self.n_obj - 1] * np.pi / 2)))
        # [ones(size(PopDec,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)]
        f2 = np.hstack((np.ones((n, 1)), np.sin(np.fliplr(x[:, 0:self.n_obj - 1]) * np.pi / 2)))
        f = (1 + 100*g) * np.fliplr(np.cumprod(f1, axis=1)) * f2

        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        ''' platemo
        I = 2;
        R = UniformPoint(N, I);
        R = R. / repmat(sqrt(sum(R. ^ 2, 2)), 1, size(R, 2));
        R = [R(:, ones(1, obj.M - size(R, 2))), R];
        R = R. / sqrt(2). ^ repmat(max([obj.M - I, obj.M - I:-1: 2 - I], 0), size(R, 1), 1);
        '''
        I = 2
        n, R = uniform_points(n_pareto_points, I)
        #
        d1 = np.sqrt(np.sum(R ** 2, axis=1))
        d1 = np.atleast_2d(d1).reshape(-1, 1)
        d1 = np.tile(d1, (1, R.shape[1]))
        R = R/d1

        idx = np.zeros((1, self.n_obj - R.shape[1]))
        R1 = np.atleast_2d(R[:, int(idx)]).reshape(-1, self.n_obj - R.shape[1])
        R = np.hstack((R1, R))

        # max([obj.M - I, obj.M - I:-1: 2 - I], 0)
        ord = np.arange(2-I, self.n_obj - I + 1)
        ord = np.atleast_2d(ord)
        ord = np.fliplr(ord)
        c1 = np.hstack((np.atleast_2d(self.n_obj-I), ord))
        c2 = np.zeros((1, 1+ord.shape[1]))
        d2 = np.maximum(c1, c2)
        d2 = np.tile(d2, (R.shape[0], 1))
        R = R / np.power(np.ones((1, d2.shape[1]))*np.sqrt(2) , d2)
        return R


class MAF13(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj
        #  [zeros(1,2),zeros(1,obj.D-2)-2]
        xl = np.zeros((1, 2))
        xl = np.append(xl, np.zeros((1, (self.n_var-2))) - 2)
        self.xl = anp.array(xl)

        # [ones(1,2),zeros(1,obj.D-2)+2]
        xu = np.ones((1, 2))
        xu = np.append(xu, np.zeros((1, (self.n_var-2))) + 2)
        self.xu = anp.array(xu)

        if self.n_obj < 3:
            print('minimum number of objective is 3')
            exit(1)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        '''platemo
        [N,D] = size(X);
        Y = X - 2*repmat(X(:,2),1,D).*sin(2*pi*repmat(X(:,1),1,D)+repmat(1:D,N,1)*pi/D);
        PopObj(:,1) = sin(X(:,1)*pi/2)                   + 2*mean(Y(:,4:3:D).^2,2);
        PopObj(:,2) = cos(X(:,1)*pi/2).*sin(X(:,2)*pi/2) + 2*mean(Y(:,5:3:D).^2,2);
        PopObj(:,3) = cos(X(:,1)*pi/2).*cos(X(:,2)*pi/2) + 2*mean(Y(:,3:3:D).^2,2);
        PopObj(:,4:obj.M) = repmat(PopObj(:,1).^2+PopObj(:,2).^10+PopObj(:,3).^10+2*mean(Y(:,4:D).^2,2),1,obj.M-3);
        '''
        x = check_array(x)
        N, D = x.shape
        y = x - 2 * np.tile(x[:, 1:2], (1, self.n_var)) * np.sin(2 * np.pi * np.tile(x[:, 0:1], (1, self.n_var)) + np.arange(1, self.n_var+1) * np.pi/self.n_var)

        idx = np.arange(3, self.n_var, 3)
        n = len(idx)
        if n == 1:
            f1 = np.sin(x[:, 0:1] * np.pi/2) + 2 * np.atleast_2d(np.mean(y[:, idx[0]: idx[0]+1]**2, 1)).reshape(-1, 1)
        else:
            f1 = np.sin(x[:, 0:1] * np.pi / 2) + 2 * np.atleast_2d(np.mean(y[:, idx] ** 2, 1)).reshape(-1, 1)
        f1 = np.atleast_2d(f1).reshape(-1, 1)

        idx = np.arange(4, self.n_var, 3)
        n = len(idx)
        if n == 1:
            f2 = np.cos(x[:, 0:1] * np.pi/2) * np.sin(x[:, 1:2] * np.pi/2) + 2 * np.atleast_2d(np.mean(y[:, idx[0]: idx[0]+1]**2, 1)).reshape(-1, 1)
        else:
            f2 = np.cos(x[:, 0:1] * np.pi / 2) * np.sin(x[:, 1:2] * np.pi / 2) + 2 * np.atleast_2d(np.mean(y[:, idx] ** 2, 1)).reshape(-1, 1)
        f2 = np.atleast_2d(f2).reshape(-1, 1)

        idx = np.arange(2, self.n_var, 3)
        n = len(idx)
        if n == 1:
            f3 = np.cos(x[:, 0: 1] * np.pi/2) * np.cos(x[:, 1:2] * np.pi/2) + 2 * np.atleast_2d(np.mean(y[:, idx[0]:idx[0]+1]**2, 1)).reshape(-1, 1)
        else:
            f3 = np.cos(x[:, 0: 1] * np.pi / 2) * np.cos(x[:, 1:2] * np.pi / 2) + 2 * np.atleast_2d(np.mean(y[:, idx] ** 2, 1)).reshape(-1, 1)
        f3 = np.atleast_2d(f3).reshape(-1, 1)

        if self.n_var <= 3:
            print('error on number of variable should be larger than 3')

        f4 = np.tile(f1**2 + f2**10 + f3**10 + 2 * np.atleast_2d(np.mean(y[:, 3: self.n_var]**2, 1)).reshape(-1, 1), (1, self.n_obj-3))
        if f4.shape[1] == 0:
            f = np.hstack((f1, f2, f3))
        else:
            f = np.hstack((f1, f2, f3, f4))
        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        '''platemo
        R = UniformPoint(N,3);
        R = R./repmat(sqrt(sum(R.^2,2)),1,3);
        R = [R,repmat(R(:,1).^2+R(:,2).^10+R(:,3).^10,1,obj.M-3)];
        '''
        n, R = uniform_points(n_pareto_points, 3)
        d = np.atleast_2d(np.sqrt(np.sum(R**2, 1))).reshape(-1, 1)
        R = R/d

        R1 = R[:, 0:1]**2 + R[:, 1:2]**10 + R[:, 2:3]**10
        R1 = np.tile(R1, (1, self.n_obj-3))
        if R1.shape[1] != 0:
            R = np.hstack((R, R1))

        return R


class MAF14(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = n_var
        self.n_constr = 0
        self.n_obj = n_obj



        #  obj.lower    = zeros(1,obj.D);
        self.xl = np.zeros((1, self.n_var))
        # #  obj.upper = [ones(1,obj.M-1), 10.*ones(1,obj.D-obj.M+1)];
        xu = np.ones((1, self.n_obj-1))
        xu = np.append(xu, np.ones((1, (self.n_var-self.n_obj+1))) * 10)
        self.xu = anp.array(xu)

        '''
        nk = 2;
        c = 3.8 * 0.1 * (1 - 0.1);
        for i = 1: obj.M - 1
        c = [c, 3.8. * c(end). * (1 - c(end))];

        end
        obj.sublen = floor(c. / sum(c). * (obj.D - obj.M + 1) / nk);
        obj.len = [0, cumsum(obj.sublen * nk)]; '''

        nk = 2
        c = [3.8 * 0.1 * (1 - 0.1)]
        for i in range(self.n_obj-1):
            c = np.append(c, 3.8 * c[-1] *(1-c[-1]))

        self.sublen = np.floor(c / np.sum(c) * (self.n_var - self.n_obj + 1) / nk)
        self.len = np.append(0, np.cumsum(self.sublen * nk))

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)



    def _evaluate(self, x, out, *args, **kwargs):
        '''
         [N,D] = size(PopDec);
            M     = obj.M;
            nk    = 2;
            PopDec(:,M:D) = (1+repmat((M:D)./D,N,1)).*PopDec(:,M:D) - repmat(PopDec(:,1)*10,1,D-M+1);
            G = zeros(N,M);
            for i = 1 : 2 : M
                for j = 1 : nk
                    G(:,i) = G(:,i) + Rastrigin(PopDec(:,obj.len(i)+M-1+(j-1)*obj.sublen(i)+1:obj.len(i)+M-1+j*obj.sublen(i)));
                end
            end
            for i = 2 : 2 : M
                for j = 1 : nk
                    G(:,i) = G(:,i) + Rosenbrock(PopDec(:,obj.len(i)+M-1+(j-1)*obj.sublen(i)+1:obj.len(i)+M-1+j*obj.sublen(i)));
                end
            end
            G      = G./repmat(obj.sublen,N,1)./nk;
            PopObj = (1+G).*fliplr(cumprod([ones(N,1),PopDec(:,1:M-1)],2)).*[ones(N,1),1-PopDec(:,M-1:-1:1)];
        '''
        print('un tested')
        x = check_array(x)
        N, D = x.shape
        M = self.n_obj
        nk = 2
        d = np.arange(self.n_obj, self.n_var+1)/self.n_var
        x[:, M-1:] = (1 + np.tile(d, (N, 1))) * x[:, M-1:] - np.tile(x[:, 0:1] * 10, (1, D-M+1))
        G = np.zeros((N, M))

        for i in np.arange(0, M, 2):
            for j in np.arange(0, nk):
                # obj.len(i) + M - 1 + (j-1)*obj.sublen(i)+1
                # obj.len(i) + M - 1 + j*obj.sublen(i)
                front_idx = self.len[i] + M - 1 + j * self.sublen[i] + 1
                end_idx = self.len[i] + M - 1 + (j+1) * self.sublen[i]
                if front_idx <= end_idx:
                    ix = x[:, int(front_idx)-1 : int(end_idx)]
                    G[:, i:i+1] = G[:, i] + Rastrigin(ix)

        for i in np.arange(1, M, 2):
            for j in range(nk):
                # obj.len(i)+M-1+(j-1)*obj.sublen(i)+1
                # obj.len(i)+M-1+j*obj.sublen(i)
                front_idx = self.len[i] + M - 1 + j * self.sublen[i] + 1
                end_idx = self.len[i] + M - 1 + (j+1) * self.sublen[i]
                ix = x[:, int(front_idx)-1 : int(end_idx)]
                G[:, i:i+1] = G[:, i] + Rosenbrock(ix)

        kk = np.tile(self.sublen, (N, 1))/nk
        G = G/kk

        # [ones(N,1),PopDec(:,1:M-1)]
        f1 = np.hstack((np.ones(N, 1), x[:, 0:M]))
        # [ones(N,1),1-PopDec(:,M-1:-1:1)]
        f2 = np.hstack((np.ones(N, 1), 1 - np.fliplr(x[:, 0:M])))
        f = (1+G) * np.fliplr(np.cumprod(f1, 1)) * f2

        print('0')
        out['F'] = f

    def _calc_pareto_front(self, n_pareto_points=100):
        #  R = UniformPoint(N,obj.M);

        _, R = uniform_points(n_pareto_points, self.n_obj)
        return R




class MAF8(Problem):
    def __init__(self, n_var=6, n_obj=3):
        self.n_var = 2
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([-10000] * self.n_var)
        self.xu = anp.array([10000] * self.n_var)

        rho, thera = cart2pol(0, 1)
        #  [obj.Points(:,1),obj.Points(:,2)] = pol2cart(thera-(1:obj.M)*2*pi/obj.M,rho);
        p1, p2 = pol2cart(thera-np.arange(1, self.n_obj+1)* 2 * np.pi/self.n_obj, rho)
        self.points = np.hstack((np.atleast_2d(p1).reshape(-1, 1), np.atleast_2d(p2).reshape(-1, 1)))

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        #  PopObj = pdist2(PopDec,obj.Points);
        x = check_array(x)
        out['F'] = cdist(x, self.points)

    def _calc_pareto_front(self, n_pareto_points=100):
        '''   [X,Y] = ndgrid(linspace(-1,1,ceil(sqrt(N))));
            ND    = ((X(:),Y(:),obj.Points(:,1),obj.Points(:,2));
            R     = pdist2([X(ND),Y(ND)],obj.Points); '''

        X, Y = np.meshgrid(np.linspace(-1, 1,  int(np.ceil(np.sqrt(n_pareto_points)))), np.linspace(-1, 1, int(np.ceil(np.sqrt(n_pareto_points)))), indexing ='ij')
        p =  path.Path(self.points)
        x = X.flatten(order='F')
        y = Y.flatten(order='F')
        r = np.hstack((np.atleast_2d(x).reshape(-1, 1), np.atleast_2d(y).reshape(-1, 1)))
        true_indx = p.contains_points(r)
        r = r[true_indx, :]
        R = cdist(r, self.points)
        return R



if __name__ == "__main__":


    pro = MAF8(n_var=6, n_obj=3)
    x = np.atleast_2d([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.3, 0.3, 0.7, 0.7, 0.7]])



    print(np.prod([1]))

    # y = pro.evaluate(x)
    # print(y)

    k = pro.pareto_front(n_pareto_points = 100)
    print(k)
    a = 0