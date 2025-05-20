import os
import subprocess
from typing import Tuple, Optional
import multiprocessing as mp


def call_main_pymoo(algo_name, instance_file_name, patience):
    cmd_args = ["python",
                "main_pymoo.py",
                "--instance-file-name",
                instance_file_name,
                "--algo-name",
                algo_name,
                "--patience",
                str(patience)]
    subprocess.run(cmd_args)
        
if __name__ == "__main__":
    import pathlib
    instance_dir = pathlib.Path()/"instances" 
    instance_file_names = os.listdir(instance_dir.absolute())
    algo_names = ["brkga","pso","ga","de"]
    patience = 30
    for instance_file_name in instance_file_names:
        for algo_name in algo_names:
            call_main_pymoo(algo_name, instance_file_name, patience)