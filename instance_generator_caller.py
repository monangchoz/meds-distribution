import subprocess
from typing import Tuple, Optional
import multiprocessing as mp

def call_generate_instance(region: str,
                           num_customers: int,
                           demand_mode: str,
                           num_clusters: int,
                           num_normal_trucks: int = 10,
                           num_reefer_trucks: int = 10,
                           ratio:Optional[Tuple[float,float,float]]=None):
    
    # print(region, num_customers, demand_mode)
    cmd_args = ["python",
                "generate_instance.py",
                "--region",
                region,
                "--num-customers",
                str(num_customers),
                "--demand-mode",
                demand_mode,
                "--num-clusters",
                str(num_clusters),
                "--num-normal-trucks",
                str(num_normal_trucks),
                "--num-reefer-trucks",
                str(num_reefer_trucks)]
    if demand_mode == "generated":
        if ratio is None:
            raise ValueError("demand mode generated must have ratio")
        cmd_args += ["--small-items-ratio",
                     str(ratio[0]),
                     "--large-items-ratio",
                     str(ratio[2])]
    subprocess.call(cmd_args)
        
if __name__ == "__main__":
    regions = [#"JK2","SBY",
               "MKS"]
    num_clusters_list = [#1, 
                         3, 
                         #5
                         ]
    num_customers_list = [
                        #   15,
                        #   30,
                          50
                          ]
    repetitions = 2
    # generate historical
    args = []
    for i in range(repetitions):
        for region in regions:
            for nc in num_customers_list:
                for ncl in num_clusters_list:
                    args += [(region, nc, "historical", ncl, 10, 10, None)]
                    # call_generate_instance(region, nc, "historical", ncl)
    # for arg in args:
    #     call_generate_instance(*arg)
    
    with mp.Pool(8) as p:
        p.starmap(call_generate_instance, args)
    
    
    # # generate generated with ratio
    # ratio_list = [(1/3, 1/3, 1/3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)]
    # args = []
    # for i in range(repetitions):
    #     for region in regions:
    #         for nc in num_customers_list:
    #             for ncl in num_clusters_list:
    #                 for r in ratio_list:
    #                     args += [(region, nc, "generated", ncl, 10, 10, r)]
    # with mp.Pool(8) as p:
    #     p.starmap(call_generate_instance, args)
                        # call_generate_instance(region, nc, "generated", ncl, ratio=r)
    # for arg in args:
    #     call_generate_instance(*arg)