import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI, optimizer
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
import pyDOE
from cross_val_hyperp import cross_val_krg
from surrogate_problems import DTLZs, iDTLZ
from matplotlib.lines import Line2D
from matplotlib import cm
from joblib import dump, load
from matplotlib.lines import Line2D
import os
import copy
import multiprocessing as mp
import pygmo as pg
# from mpl_toolkits import mplot3d
from sort_population import sort_population_cornerfirst
from EI_problem import ego_believer
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score
import time

if __name__ == "__main__":
    prob = DTLZs.DTLZ2(n_var=6, n_obj=3)
    # x = np.atleast_2d([0, 0, 0, 0, 0, 0, 0])
    # f = prob.evaluate(x, return_values_of=['F'])
    np.random.seed(4)
    bounds = np.vstack((prob.xl, prob.xu)).T.tolist()
    plot_param = {'visualize':True}
    optimizer.optimizer(prob, prob.n_obj, 0, bounds, 0.2, 0.8,100, 100, visualization=True,  **plot_param)
