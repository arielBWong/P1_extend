import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import copy



class single_krg_optim(Problem):

    def __init__(self, krg, n_var, n_constr, n_obj, low, up):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(low)
        self.xu = anp.array(up)
        self.model = krg
        self.name = 'optimization on kriging'
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        y, _ = self.model.predict(x)
        out["F"] = y
        return out["F"]

class all_but_one_krgopt(Problem):
    def __init__(self, krg, n_var, n_constr, n_obj, low, up, oneID):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(low)
        self.xu = anp.array(up)
        self.model = krg
        self.oneOut = oneID
        self.name = 'all but one'
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        n = len(self.model)
        f = []
        for i in range(n):
            if i is not self.oneOut:
                y, _ = self.model[i].predict(x)
                f = np.append(f, y)

        f = np.atleast_2d(f).reshape(-1, n-1, order='F')
        f = np.linalg.norm(f, axis=1)
        f = np.atleast_2d(f).reshape(-1, 1)
        out["F"] = f
        return out["F"]

class cornersearch_krgopt(Problem):
    def __init__(self, krg, n_var, n_constr, n_obj, low, up):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(low)
        self.xu = anp.array(up)
        self.model = krg
        self.name = 'corner search'
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        n = len(self.model)
        f = []
        for i in range(n):
                y, _ = self.model[i].predict(x)
                f = np.append(f, y)

        f1 = copy.deepcopy(f)
        f1 = np.atleast_2d(f1).reshape(-1, n, order='F')
        f2 = []
        f = np.atleast_2d(f).reshape(-1, n, order='F')
        for i in range(n):
            indx = np.arange(n)
            indx = np.delete(indx, i)
            f2 = np.append(f2, np.linalg.norm(f[:, indx], axis=1))

        f2 = np.atleast_2d(f2).reshape(-1, n, order='F')
        out["F"] = np.hstack((f1, f2))
        return out["F"]



class cornersearch_krgoptminus(Problem):
    def __init__(self, krg, n_var, n_constr, n_obj, low, up):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(low)
        self.xu = anp.array(up)
        self.model = krg
        self.name = 'corner search minus'
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        n = len(self.model)
        f = []
        for i in range(n):
                y, _ = self.model[i].predict(x)
                f = np.append(f, y)

        f1 = copy.deepcopy(f)
        f1 = np.atleast_2d(f1).reshape(-1, n, order='F')
        out["F"] = f1
        return out["F"]


class cornersearch_krgoptplus(Problem):
    # surrogate problem
    def __init__(self, krg, n_var, n_constr, n_obj, low, up):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(low)
        self.xu = anp.array(up)
        self.model = krg
        self.name = 'corner search'
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        n = len(self.model)
        f = []
        for i in range(n):
                y, _ = self.model[i].predict(x)
                f = np.append(f, y)

        f1 = copy.deepcopy(f)
        f1 = np.atleast_2d(f1).reshape(-1, n, order='F')
        f2 = []
        f = np.atleast_2d(f).reshape(-1, n, order='F')
        for i in range(n):
            indx = np.arange(n)
            indx = np.delete(indx, i)
            f2 = np.append(f2, np.linalg.norm(f[:, indx], axis=1))

        f2 = np.atleast_2d(f2).reshape(-1, n, order='F')
        f3 = np.atleast_2d(np.linalg.norm(f, axis=1))

        out["F"] = np.hstack((f1, f2, f3))
        return out["F"]


class cornersearch_problem(Problem):
    # real problem
    def __init__(self, relay_problem):
        self.n_var = relay_problem.n_var
        self.n_constr = relay_problem.n_constr
        self.n_obj = relay_problem.n_obj * 2
        self.xl = anp.array(relay_problem.xl)
        self.xu = anp.array(relay_problem.xu)
        self.model = None
        self.prob = relay_problem
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        n = self.prob.n_obj


        f1 = self.prob.evaluate(x, return_values_of='F')
        f2 = []
        f = copy.deepcopy(f1)

        for i in range(n):
            indx = np.arange(n)
            indx = np.delete(indx, i)
            f2 = np.append(f2, np.linalg.norm(f[:, indx], axis=1))

        f2 = np.atleast_2d(f2).reshape(-1, n, order='F')

        out["F"] = np.hstack((f1, f2))
        return out["F"]

    def name(self):
        return self.prob.__class__.__name__

    def _calc_pareto_front(self, n_pareto_points=100):
        return self.prob.pareto_front(n_pareto_points)

