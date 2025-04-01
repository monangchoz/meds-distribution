import pathlib

from pymoo_interface.hvrp3l_opt import HVRP3L_OPT

from problem.hvrp3l import HVRP3L

def run():
    filename = "JK2_nc_30_ni__1363_nv_20_0.json"
    instance_filepath = pathlib.Path()/"instances"/filename
    problem = HVRP3L.read_from_json(instance_filepath)


if __name__ == "__main__":
    run()