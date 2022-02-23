from argparse import ArgumentParser
from pathlib import Path
import open3d as o3d
import numpy as np
from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import imgs_to_pcd, rs_ci, zv_ci

threshold = 1
final_size = (1920, 1080)


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


def main(args):
    file = args.raw_file
    assert file.exists(), f"file {file.absolute} does not exist"

    rs_rgb, rs_depth, zv_rgb, zv_depth, _ = DatasetInterface.load(file)

    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, zv_ci)
    trans = compute_initial_transformation_matrix(zv_pcd, rs_pcd)

    print(f"""

    Computed extrinsic transformation matrix
        {trans}

    """)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("raw_file", type=Path, help="file path to the data the features picking is executed on")

    main()
