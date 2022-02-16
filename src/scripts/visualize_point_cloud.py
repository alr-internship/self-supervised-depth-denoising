
from pathlib import Path

from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import imgs_to_pcd, rs_ci
import open3d as o3d


path = Path("resources/images/calibrated_masked/not-cropped")

files = path.rglob("*.npz")

for file in files:
    rs_rgb, rs_depth, zv_rgb, zv_depth, _ = DatasetInterface.load(file)
    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    o3d.visualization.draw_geometries([rs_pcd])