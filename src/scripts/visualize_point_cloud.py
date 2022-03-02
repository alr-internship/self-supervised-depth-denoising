
from pathlib import Path

import numpy as np

from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import imgs_to_pcd, rs_ci
import open3d as o3d


path = Path("resources/images/calibrated_masked")
apply_mask = True

files = path.rglob("*.npz")


for file in files:
    rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)

    if apply_mask:
        rs_depth = np.where(mask, rs_depth, np.nan)

    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    o3d.visualization.draw_geometries([rs_pcd])