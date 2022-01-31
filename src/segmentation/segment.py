import pickletools
from re import A
import time
from unittest import result

from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from matplotlib import pyplot as plt, transforms
import copy

from utils.visualization_utils import to_bgr, to_rgb

zv_ci = o3d.camera.PinholeCameraIntrinsic(
    width=1920, height=1200,
    fx=2.76012e+03, fy=2.75978e+03,
    cx=9.51680e+02, cy=5.94779e+02,
)

rs_ci = o3d.camera.PinholeCameraIntrinsic(
    width=1920, height=1080,
    fx=1377.6448974609375, fy=1375.7239990234375,
    cx=936.4846801757812, cy=537.48779296875,
)

threshold = 1
trans_init = np.array(
    [[0.99807603, -0.01957923, -0.05882929,  0.17911331],
        [0.0212527, 0.99938319, 0.02795643, -0.01932425],
        [0.05824564, -0.02915292, 0.99787652, -0.03608739],
        [0., 0., 0., 1.]]
)
# trans_init_charuco = np.array(
#     [[0.99912644, -0.00259379, -0.04170882,  0.17370972],
#      [0.00228578, 0.99996978, -0.00743066,  0.02393987],
#      [0.04172684, 0.00732883,  0.99910218, -0.04262905],
#      [0., 0.,  0.,  1.]]
# )
final_size = (1920, 1080)




def __ask_to_annotate_points(
        target: o3d.geometry.PointCloud):
    vis_target = o3d.visualization.VisualizerWithEditing()
    vis_target.create_window(window_name="Select corresponding points, alternating between the both point clouds")
    vis_target.add_geometry(target)
    vis_target.run()
    vis_target.destroy_window()
    target_points_idx = vis_target.get_picked_points()
    target_points = np.asarray(target.points)[target_points_idx]

    return target_points


def align(rs_rgb, rs_depth, zv_rgb, zv_depth):
    crop_region_mask = np.zeros((rs_rgb.shape[:2]), dtype=np.uint8)
    crop_region_mask[:, 500:1600] = 1

    rs_rgb = rs_rgb * crop_region_mask[..., None]
    rs_depth = rs_depth * crop_region_mask
    zv_rgb = zv_rgb * crop_region_mask[..., None]
    zv_depth = zv_depth * crop_region_mask

    zv_depth = np.where(zv_depth < 1.13, zv_depth, np.nan)
    # rs_depth = np.where(rs_depth < 1.13, rs_depth, np.nan)

    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, rs_ci)

    o3d.visualization.draw_geometries([zv_pcd, rs_pcd])

    _, rs_inliers = rs_pcd.segment_plane(distance_threshold=0.001,
                         ransac_n=3, num_iterations=1000)
    inlier_rs_pcd =rs_pcd.select_by_index(rs_inliers)
    inlier_rs_pcd.paint_uniform_color([1, 0, 0])
    rs_pcd = rs_pcd.select_by_index(rs_inliers, invert=True)
    o3d.visualization.draw_geometries([rs_pcd, inlier_rs_pcd])

    _, zv_inliers = zv_pcd.segment_plane(distance_threshold=0.000005,
                         ransac_n=3, num_iterations=1000)
    inlier_zv_pcd =zv_pcd.select_by_index(zv_inliers)
    inlier_zv_pcd.paint_uniform_color([1, 0, 0])
    zv_pcd = zv_pcd.select_by_index(zv_inliers, invert=True)
    o3d.visualization.draw_geometries([zv_pcd, inlier_zv_pcd])


    zv_rgb, zv_depth, zv_ul_corner, zv_lr_corner = pcd_to_imgs(zv_pcd, rs_ci)
    rs_rgb, rs_depth, rs_ul_corner, rs_lr_corner = pcd_to_imgs(rs_pcd, rs_ci)

    ul_corner = np.max([zv_ul_corner, rs_ul_corner], axis=0)
    lr_corner = np.min([zv_lr_corner, rs_lr_corner], axis=0)

    zv_rgb = zv_rgb[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    rs_rgb = rs_rgb[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    zv_depth = zv_depth[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    rs_depth = rs_depth[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]

    zv_rgb = cv2.resize(zv_rgb, final_size)
    zv_depth = cv2.resize(zv_depth, final_size)
    rs_rgb = cv2.resize(rs_rgb, final_size)
    rs_depth = cv2.resize(rs_depth, final_size)

    print(zv_rgb.shape)

    return rs_rgb, rs_depth, zv_rgb, zv_depth


def main():
    uncal_dir = Path("resources/images/calibrated/3d_aligned")
    dataset_interface = DatasetInterface(uncal_dir)
    cal_dir = Path("resources/images/calibrated/segmented")
    aligned_dataset_interface = DatasetInterface(cal_dir)

    for idx, image_tuple in tqdm(enumerate(dataset_interface)):
        aligned_image_tuple = align(*image_tuple)
        rel_dir_path = dataset_interface.data_file_paths[idx].relative_to(uncal_dir)
        # aligned_dataset_interface.append_and_save(*aligned_image_tuple, rel_dir_path)


if __name__ == "__main__":
    main()
