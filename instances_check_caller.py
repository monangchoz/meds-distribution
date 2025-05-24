import os
import subprocess
from typing import Tuple, Optional
import multiprocessing as mp


def call_instance_check(instance_file_name):
    cmd_args = ["python",
                "instance_check.py",
                "--instance-file-name",
                instance_file_name]
    subprocess.run(cmd_args)
        
if __name__ == "__main__":
    import pathlib
    instance_dir = pathlib.Path()/"instances" 
    instance_file_names = os.listdir(instance_dir.absolute())
    
    for instance_file_name in instance_file_names:
        print(instance_file_name)
        call_instance_check(instance_file_name)