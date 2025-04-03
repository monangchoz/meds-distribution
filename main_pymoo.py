import pathlib
import random

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from pymoo_interface.arr1 import ARR1
from problem.hvrp3l import HVRP3L
from pymoo_interface.hvrp3l_opt import HVRP3L_OPT


def run():
    filename = "JK2_nc_30_ni__1363_nv_20_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    problem_intf = HVRP3L_OPT(problem, ARR1())
    algo = DE(pop_size=4)
    termination = DefaultSingleObjectiveTermination(n_max_gen=1)
    res = minimize(problem_intf, algo, termination=termination, verbose=True)

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    run()