import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg


from scipy.stats import norm
from sklearn.utils.validation import check_array

from EI_problem import  ego_believer, ego_eim




class ego_fit(Problem):

    def __init__(self, n_var, n_obj, n_constr, upper_bound, lower_bound, name):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(lower_bound)
        self.xu = anp.array(upper_bound)
        self.name = name
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        krg = kwargs['krg']
        nd_front = kwargs['nd_front']
        ref = kwargs['ref']
        fit = ego_believer(x, krg, nd_front, ref)

        out["F"] = -fit


class ego_fiteim(Problem):

    def __init__(self, n_var, n_obj, n_constr, upper_bound, lower_bound, name):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(lower_bound)
        self.xu = anp.array(upper_bound)
        self.name = name
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        krg = kwargs['krg']
        nd_front = kwargs['nd_front']
        ref = kwargs['ref']
        fit = ego_eim(x, krg, nd_front, ref)

        out["F"] = -fit



