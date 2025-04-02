import pathlib

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

from problem.hvrp3l import HVRP3L
from pymoo_interface.hvrp3l_opt import HVRP3L_OPT

def run():
    filename = "JK2_nc_30_ni__1363_nv_20_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)
    problem_intf = HVRP3L_OPT(problem)
    algo = DE()
    res = minimize(problem_intf, algo)

if __name__ == "__main__":
    run()