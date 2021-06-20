# check how to plot 3D objective space

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
    DTLZ1, DTLZ2, DTLZ3, DTLZ4, \
    BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
import pyDOE
from cross_val_hyperp import cross_val_krg
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
    MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, ego_fitness, EI
from matplotlib.lines import Line2D

import os
import copy
import multiprocessing as mp
import pygmo as pg
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



from pymop.factory import get_uniform_weights



x = np.array([[1,3], [3,4],[4, 1]])
print(x)
x = x.flatten(order='F')
print(x)
id = np.where(x == 1)
print(id)
x = np.delete(x, id)
x = np.atleast_2d(x).reshape(-1, 2, order='F')
print(x)

a = np.arange(3)
s = []
if s is not []:
    print('full')
else:
    print('empty')
