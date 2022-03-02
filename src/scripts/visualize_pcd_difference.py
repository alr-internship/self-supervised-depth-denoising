# %%
import random
from re import A
from utils.transformation_utils import imgs_to_pcd, pcd_to_imgs, pcd_to_imgs_old, rs_ci, zv_ci
import yaml
from utils.transformation_utils import imgs_to_pcd, rs_ci, unnormalize_depth, to_rgb
from utils.visualization_utils import visualize_depth, visualize_geometries, visualize_mask
import open3d as o3d
from networks.UNet.unet_model import UNet
from dataset.dataset_interface import DatasetInterface
from dataset.data_loading import BasicDataset
from matplotlib import pyplot as plt
from torch import nn
from natsort import natsorted
import torch
import numpy as np
from pathlib import Path
import logging
import sys
sys.path.append("../src")


path_1 = Path("resources/images/calibrated/not-cropped/ycb_video/10022022")
path_2 = Path("resources/images/calibrated_new/not-cropped/ycb_video/10022022")

files_1 = DatasetInterface.get_files_by_path(path_1)
random.shuffle(files_1)

# %%
for file_1 in files_1:
    rs_rgb_1, rs_depth_1, zv_rgb_1, zv_depth_1, mask_1 = DatasetInterface.load(file_1)
    rs_rgb_2, rs_depth_2, zv_rgb_2, zv_depth_2, mask_2 = DatasetInterface.load(path_2 / file_1.relative_to(path_1))

    if mask_1 is not None:
        zv_depth_1 = np.where(mask_1, zv_depth_1, np.nan)
        zv_depth_2 = np.where(mask_2, zv_depth_2, np.nan)

    print("mean depths: zv:", np.nanmean(zv_depth_1), "rs:", np.nanmean(zv_depth_2))

    diff = np.abs(zv_depth_1 - zv_depth_2)
    print("maximal difference in zv depth frames:", np.nanmax(diff))
    # green > small diff, red > large diff
    equals = np.where(np.nan_to_num(diff > 1)[..., None], [0, 0, 255], [0, 255, 0]).astype(np.uint8)

    # rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd_1 = imgs_to_pcd(zv_rgb_1, zv_depth_1, rs_ci)
    zv_pcd_2 = imgs_to_pcd(zv_rgb_2, zv_depth_2, rs_ci)
    diff = imgs_to_pcd(equals, zv_depth_2, rs_ci)

    zv_pcd_1.paint_uniform_color([1, 0, 0])
    zv_pcd_2.paint_uniform_color([0, 0, 1])

    # visualize_geometries([zv_pcd_1, zv_pcd_2])
    visualize_geometries([diff])

# %%
#