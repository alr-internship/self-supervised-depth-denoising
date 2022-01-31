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


def imgs_to_pcd(bgr, depth, ci):
    rgb = to_rgb(bgr)
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, ci)
    return pcd


def pcd_to_imgs(pcd, ci):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # camera parameters
    fx, fy = ci.get_focal_length()
    cx, cy = ci.get_principal_point()
    f = np.array([fx, fy])
    c = np.array([cx, cy])

    # convert to 3d to 2d space
    depths = points[:, 2]
    pixels = points[:, :2] * f / np.expand_dims(depths, axis=1) + c
    pixels = pixels.astype(np.uint16)

    # create empty frames for final rgb and depth images
    ul_corner = np.min(pixels, axis=0)
    lr_corner = np.max(pixels, axis=0)
    picture_size = np.round_(lr_corner).astype(np.uint16).T
    rgb_frame = np.zeros((picture_size[1] + 1, picture_size[0] + 1, 3), dtype=np.uint8)
    # depth frame filled with NaNs
    depth_frame = np.empty((picture_size[1] + 1, picture_size[0] + 1), dtype=np.float32)
    depth_frame[:] = np.nan

    # fill respective pixels with depth and rgb info
    pixels = pixels.T
    pixels = (pixels[1], pixels[0])
    rgb_frame[pixels] = colors * 255
    depth_frame[pixels] = depths

    bgr_frame = to_bgr(rgb_frame)

    return bgr_frame, depth_frame, ul_corner, lr_corner


def __ask_to_annotate_points(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud):
    vis_target = o3d.visualization.VisualizerWithEditing()
    vis_target.create_window(window_name="Select corresponding points, alternating between the both point clouds")
    vis_target.add_geometry(target)
    vis_target.run()
    vis_target.destroy_window()
    target_points_idx = vis_target.get_picked_points()

    vis_source = o3d.visualization.VisualizerWithEditing()
    vis_source.create_window(window_name="Select corresponding points, alternating between the both point clouds")
    vis_source.add_geometry(source)
    vis_source.run()
    vis_source.destroy_window()
    source_points_idx = vis_source.get_picked_points()

    assert len(source_points_idx) == len(target_points_idx) > 0

    source_points = np.asarray(source.points)[source_points_idx]
    target_points = np.asarray(target.points)[target_points_idx]

    return source_points, target_points


def __compute_transform_matrix(A, B):
    # https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def refine_transformation_matrix(trans_init, rs_pcd, zv_pcd):
    zv_pcd.transform(trans_init)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        zv_pcd, rs_pcd, threshold, trans_init)
    print("Initial Alignment", evaluation)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        zv_pcd, rs_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    trans_final = reg_p2p.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        zv_pcd, rs_pcd, threshold, trans_final)
    print("Final Alignment", evaluation)
    print(trans_final)
    return trans_final


def compute_initial_transformation_matrix(zv_pcd, rs_pcd):
    source_pts, target_pts = __ask_to_annotate_points(zv_pcd, rs_pcd)
    print(source_pts)
    print(target_pts)
    R, t = __compute_transform_matrix(source_pts.T, target_pts.T)
    trans = np.identity(4)
    trans[:3, :3] = R
    trans[:3, 3] = t[:, 0]
    print(trans)
    return trans


def align(rs_rgb, rs_depth, zv_rgb, zv_depth):
    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, zv_ci)

    # trans_init = compute_initial_transformation_matrix(zv_pcd, rs_pcd)
    zv_pcd.transform(trans_init)
    o3d.visualization.draw_geometries([zv_pcd, rs_pcd])

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
    uncal_dir = Path("resources/images/old/c_dataset_h_1")
    dataset_interface = DatasetInterface(uncal_dir)
    cal_dir = Path("resources/images/calibrated/3d_aligned")
    aligned_dataset_interface = DatasetInterface(cal_dir)

    for idx, image_tuple in tqdm(enumerate(dataset_interface)):
        aligned_image_tuple = align(*image_tuple)
        rel_dir_path = dataset_interface.data_file_paths[idx].relative_to(uncal_dir)
        # aligned_dataset_interface.append_and_save(*aligned_image_tuple, rel_dir_path)


if __name__ == "__main__":
    main()
