
from pathlib import Path
import shutil


dir = Path("resources/models")

for subdir in dir.iterdir():
    if not subdir.is_dir():
        continue 

    print(subdir.name)
    dirs = [d for d in subdir.iterdir() if d.is_dir()]

    if len(dirs) == 0:
        shutil.rmtree(subdir)
