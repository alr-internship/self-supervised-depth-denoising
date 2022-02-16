from pathlib import Path

from tqdm import tqdm

from dataset.dataset_interface import DatasetInterface


dir = Path("resources/images/calibrated_masked/not-cropped")
files = list(dir.rglob("*.npz"))

for file in tqdm(files):
    rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)
    if mask.size == 0:
        file.unlink()
        print(file)