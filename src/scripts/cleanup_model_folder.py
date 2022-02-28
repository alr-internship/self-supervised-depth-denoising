
from pathlib import Path
import shutil

from natsort import natsorted
import numpy as np


dir = Path("resources/models")

for subdir in dir.iterdir():
    if not subdir.is_dir():
        continue 
        
    dirs = [d for d in subdir.iterdir() if d.is_dir()]
    print("#" * 10 + f" Checking {subdir.name}" + "#" * 10)

    obj_in_dirs = np.sum([len(list(d.iterdir())) for d in dirs])

    if len(dirs) == 0 or obj_in_dirs == 0:
        print(f"removing {subdir}")
        shutil.rmtree(subdir)

    else:
        for dir in dirs:
            checkpoints = natsorted(dir.glob("*pth"), key=lambda l: l.stem)
            if len(checkpoints) == 0:
                continue

            without_last = checkpoints[:-1]
            for checkpoint in without_last:
                print(f"deleting checkpoint {checkpoint.name}")
                checkpoint.unlink()
            print(f"deleted all checkpoints except {checkpoints[-1].name}")
            

