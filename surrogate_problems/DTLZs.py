import autograd.numpy as anp

from pymop.problem import Problem
import numpy as np
from scipy.special import comb
from itertools import combinations
from pymop.factory import get_uniform_weights


class DTLZ(Problem):

    def __init__(self, n_var, n_obj, k=None):
        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)

    def g1(self, X_M):
        return 1 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(2 * anp.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:, :X_.shape[1] - i], alpha) * anp.pi / 2.0), axis=1)
            if i > 0:
                _f *= anp.sin(anp.power(X_[:, X_.shape[1] - i], alpha) * anp.pi / 2.0)

            f.append(_f)
        f = anp.column_stack(f)

        return f


def generic_sphere(ref_dirs):
    return ref_dirs / anp.tile(anp.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        return 0.5 * ref_dirs

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        f = []
        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= anp.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        out["F"] = anp.column_stack(f)


class minusDTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, n_pareto_points = 100):
        # g = 11.0125 # g1's magnitude is reduced
        if self.n_var is not 6:
            raise('number of variable is not 6, g needs recalculation, this version does not cover')

        if self.n_obj == 3:
            g = 9
        elif self.n_obj == 2:
            g = 11.25
        elif self.n_obj == 5:
            g = 4.5
        else:
            raise('pareto front is not implemented')

        # g = 9
        # P = UniformPoint(N, obj.M) / 2 * (1 + g) * (-1);
        ref_dir = get_uniform_weights(n_pareto_points, self.n_obj)
        pf = ref_dir/2 * (1 + g) * (-1)
        return pf

    def _evaluate(self, x, out, *args, **kwargs):
        original_problem = DTLZ1(n_var=self.n_var, n_obj=self.n_obj)
        obj_F = original_problem.evaluate(x, return_values_of=['F'])
        out["F"] = -1 * obj_F



class invertedDTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, n_pareto_points = 100):

        ref_dir = get_uniform_weights(n_pareto_points, self.n_obj)
        pf = (1 - ref_dir)/2
        return pf

    def _evaluate(self, x, out, *args, **kwargs):
        original_problem = DTLZ1(n_var=self.n_var, n_obj=self.n_obj)
        obj_F = original_problem.evaluate(x, return_values_of=['F'])

        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        g = np.atleast_2d(g).reshape(-1, 1)
        g = np.tile(g, (1, self.n_obj))

        out["F"] =  0.5 * (1+g) - obj_F





class DTLZ2(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class minusDTLZ2(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        original_problem = DTLZ2(n_var=self.n_var, n_obj=self.n_obj)
        obj_F = original_problem.evaluate(x, return_values_of=['F'])
        out["F"] = -1 * obj_F

    def _calc_pareto_front(self, n_pareto_points = 100):
        # g = 0.5 ^ 2 * 10;
        # P = UniformPoint(N, obj.M);
        # P = P. / repmat(sqrt(sum(P. ^ 2, 2)), 1, obj.Global.M) * (1 + g) * (-1);

        # g = 0.5 ** 2 * 10
        g = 0.5 ** 2 * (self.n_var - self.n_obj + 1)

        ref_dir = get_uniform_weights(n_pareto_points, self.n_obj)
        d = np.sqrt(np.sum(ref_dir ** 2, 1))
        d = np.atleast_2d(d).reshape(-1, 1)
        d = np.tile(d, (1, self.n_obj))

        pf = ref_dir/d  *  (1 + g) * (-1)
        return pf



class invertedDTLZ2(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        original_problem = DTLZ2(n_var=self.n_var, n_obj=self.n_obj)
        obj_F = original_problem.evaluate(x, return_values_of=['F'])

        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        g = np.atleast_2d(g).reshape(-1, 1)
        g = np.tile(g, (1, self.n_obj))
        out["F"] = (1+g) - obj_F

    def _calc_pareto_front(self, n_pareto_points = 100):

        ref_dir = get_uniform_weights(n_pareto_points, self.n_obj)
        d = np.sqrt(np.sum(ref_dir ** 2, 1))
        d = np.atleast_2d(d).reshape(-1, 1)
        d = np.tile(d, (1, self.n_obj))

        pf = 1 - ref_dir/d
        return pf



class DTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class minusDTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        original_problem = DTLZ3(n_var=self.n_var, n_obj=self.n_obj)
        obj_F = original_problem.evaluate(x, return_values_of=['F'])
        out["F"] = -1 * obj_F

    def _calc_pareto_front(self, n_pareto_points = 100):
        # g = 2202.5;
        # P = UniformPoint(N, obj.M);
        # P = P. /  repmat(sqrt(sum(P. ^ 2, 2)), 1, obj.M) * (1 + g) * (-1);
        if self.n_var is not 6:
            raise('number of variable is not 6, g needs recalculation')
        if self.n_obj == 3:
            g = 9
        elif self.n_obj == 2:
            g = 11.25
        elif self.n_obj == 5:
            g = 4.5
        else:
            raise ('pareto front is not implemented')
        ref_dir = get_uniform_weights(n_pareto_points, self.n_obj)
        d = np.sqrt(np.sum(ref_dir ** 2, 1))
        d = np.atleast_2d(d).reshape(-1, 1)
        d = np.tile(d, (1, self.n_obj))
        pf = ref_dir / d * (1 + g) * (-1)

        return pf


class DTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.alpha = alpha
        self.d = d

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=self.alpha)




class minusDTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        original_problem = DTLZ4(n_var=self.n_var, n_obj=self.n_obj)
        obj_F = original_problem.evaluate(x, return_values_of=['F'])
        out["F"] = -1 * obj_F

    def _calc_pareto_front(self, n_pareto_points = 100):
        # g = 0.5 ^ 2 * 10;
        # P = UniformPoint(N, obj.M);
        # P = P. / repmat(sqrt(sum(P. ^ 2, 2)), 1, obj.M) * (1 + g) * (-1);
        g = 0.5 ** 2 * (self.n_var - self.n_obj + 1)
        ref_dir = get_uniform_weights(n_pareto_points, self.n_obj)
        d = np.sqrt(np.sum(ref_dir ** 2, 1))
        d = np.atleast_2d(d).reshape(-1, 1)
        d = np.tile(d, (1, self.n_obj))

        pf = ref_dir / d * (1 + g) * (-1)
        return pf


class DTLZ5(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        # raise Exception("Not implemented yet.")
        p1 = np.atleast_2d(np.arange(0, 1 + 1/n_pareto_points, 1/(n_pareto_points-1))).reshape(-1, 1)
        p2 = np.atleast_2d(np.arange(1, 0-1/n_pareto_points, -1/(n_pareto_points-1))).reshape(-1, 1)
        p = np.hstack((p1, p2))
        p3 = np.atleast_2d(np.sqrt(np.sum(p**2, axis=1))).reshape(-1, 1)
        p3 = np.repeat(p3, p.shape[1], axis=1)
        p = p/p3
        # p = np.hstack((p[:, ]))
        a = 0
        if self.n_obj - 2 > 0:
            select_columes = list(map(int, np.zeros(self.n_obj-2)))
            p = np.hstack((p[:, select_columes], p))
            n_p = len(p)  ## number of rows
            p4 = [self.n_obj-2]
            p4 = np.append(p4, np.arange(self.n_obj - 2, -0.01, -1))
            p4 = np.repeat(np.atleast_2d(p4), n_p, axis=0)
            p = p/np.sqrt(2)**p4
        return p




    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ6(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        raise Exception("Not implemented yet.")

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = anp.sum(anp.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1]-interval[0])/(interval[3]-interval[2]+interval[1]-interval[0])
        X = self._replicatepoint(n_pareto_points, self.n_obj-1)
        X[X <= median] = X[X <= median]*(interval[1]-interval[0])/median+interval[0]
        X[X > median] = (X[X > median] - median) * (interval[3]-interval[2])/(1-median)+interval[2]
        p2 = 2 * (self.n_obj - np.sum(X/2 * (1 + np.sin(3 * np.pi * X)), axis=1))
        p2 = np.atleast_2d(p2).reshape(-1, 1)
        p = np.hstack((X, p2))
        return p



    def _replicatepoint(self, sample_num, M):
        if M > 1 and M < 3:
            sample_num = np.ceil(sample_num**(1/M))**M
            gap = np.arange(0, 1 + 1e-7, 1/(sample_num**(1/M)-1))
            c1, c2 = np.meshgrid(gap, gap, indexing='ij')
            W = np.hstack((np.atleast_2d(c1.flatten(order='F')).reshape(-1, 1),
                           np.atleast_2d(c2.flatten(order='F')).reshape(-1, 1)))

        elif M == 1:
            W = np.arange(0, 1 + 1e-5, 1/(sample_num-1))
            W = np.atleast_2d(W).reshape(-1, 1)

        else:
            raise(
                "for number objectives greater than 3, not implemented"
            )
        return W




    def _evaluate(self, x, out, *args, **kwargs):
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = anp.column_stack(f)

        g = 1 + 9 / self.k * anp.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - anp.sum(f / (1 + g[:, None]) * (1 + anp.sin(3 * anp.pi * f)), axis=1)

        out["F"] = anp.column_stack([f, (1 + g) * h])


class ScaledProblem(Problem):

    def __init__(self, problem, scale_factor):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj, n_constr=problem.n_constr,
                         xl=problem.xl, xu=problem.xu, type_var=problem.type_var)
        self.problem = problem
        self.scale_factor = scale_factor

    @staticmethod
    def get_scale(n, scale_factor):
        return anp.power(anp.full(n, scale_factor), anp.arange(n))

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = t[0] * ScaledProblem.get_scale(self.n_obj, self.scale_factor)
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem.pareto_front(*args, **kwargs) * ScaledProblem.get_scale(self.n_obj, self.scale_factor)


class ConvexProblem(Problem):

    def __init__(self, problem):
        super().__init__(problem.n_var, problem.n_obj, problem.n_constr, problem.xl, problem.xu)
        self.problem = problem

    @staticmethod
    def get_power(n):
        p = anp.full(n, 4.0)
        p[-1] = 2.0
        return p

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = anp.power(t[0], ConvexProblem.get_power(self.n_obj))
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        F = self.problem.pareto_front(ref_dirs)
        return anp.power(F, ConvexProblem.get_power(self.n_obj))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pyDOE

    # pro = minusDTLZ4(n_var=6, n_obj=3)
    pro = DTLZ4(n_var=6, n_obj=3)

    print(pro.name())

    train_x = pyDOE.lhs(pro.n_var, 66, criterion='maximin')  # , iterations=1000)

    xu = np.atleast_2d(pro.xu).reshape(1, -1)
    xl = np.atleast_2d(pro.xl).reshape(1, -1)

    trgx = xl + (xu - xl) * train_x
    trgy = pro.evaluate(trgx, return_values_of=['F'])

    ref_dir = get_uniform_weights(1000, pro.n_obj)
    pf = pro.pareto_front(ref_dir)
    # pf = pro.pareto_front(n_pareto_points = 1000)

    plt.ion()
    f1 = plt.figure(figsize=(5.5, 5.5))
    ax1 = f1.add_subplot(111, projection=Axes3D.name)
    ax1.scatter3D(trgy[:, 0], trgy[:, 1], trgy[:, 2], s=10, c='r', alpha=0.8,
                      label='init')
    ax1.scatter3D(pf[:, 0], pf[:, 1], pf[:, 2], s=10, c='g', alpha=0.2,
                  label='PF')
    ax1.view_init(20, 340)
    plt.close()
    # bj = pro.evaluate(x)
    # print(obj)