
import numpy as np
from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface


files = DatasetInterface.get_paths_in_dir("resources/images")
for file in tqdm(files):
    rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)

    rs_depth = np.where(rs_depth == 0, np.nan, rs_depth)

    DatasetInterface.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask)