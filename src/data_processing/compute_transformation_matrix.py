from typing import List
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from pathlib import Path
import open3d as o3d
import numpy as np
from joblib import Parallel, delayed
import cv2
from utils.general_utils import split
from utils.transformation_utils import image_points_to_camera_points, imgs_to_pcd, pcd_to_imgs, rs_ci, zv_ci

threshold = 1
final_size = (1920, 1080)


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


def main():
    # get all charuco images with small charuco board
    dir_with_charuco_files = Path("resources/images/_old")
    files = list(dir_with_charuco_files.glob("c_dataset_small*/**/*.npz"))

    print(f"files for calibration: {len(files)}")

    charuco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # charuco_board = cv2.aruco.CharucoBoard_create(7, 5, 7.5, 5.625, charuco_dict) # large
    charuco_board = cv2.aruco.CharucoBoard_create(5, 3, 6.5, 4.875, charuco_dict)  # small (prob. not correct)
    common_points_threshold = 10

    # Reference: https://stackoverflow.com/questions/64612924/opencv-stereocalibration-of-two-cameras-using-charuco
    rs_imgs_xyz = []
    zv_imgs_xyz = []

    imgs_tuples = [DatasetInterface.load(file)[:4] for file in files]

    for rs_rgb, rs_depth, zv_rgb, zv_depth in imgs_tuples:
        try:
            # convert BGR to Gray
            rs_rgb = cv2.cvtColor(rs_rgb, cv2.COLOR_BGR2GRAY)
            zv_rgb = cv2.cvtColor(zv_rgb, cv2.COLOR_BGR2GRAY)

            # Find markers corners
            rs_corners, rs_ids, _ = cv2.aruco.detectMarkers(rs_rgb, charuco_dict)
            zv_corners, zv_ids, _ = cv2.aruco.detectMarkers(zv_rgb, charuco_dict)
            if not rs_corners or not zv_corners:
                raise Exception("No markers detected")

            # find charcuo corners # TODO: try with cameraMatrix/distCoeffs
            retA, rs_corners, rs_ids = cv2.aruco.interpolateCornersCharuco(rs_corners, rs_ids, rs_rgb, charuco_board)
            retB, zv_corners, zv_ids = cv2.aruco.interpolateCornersCharuco(zv_corners, zv_ids, zv_rgb, charuco_board)
            if not retA or not retB:
                raise Exception("Can't interpolate corners")

            # Find common points in both frames (is there a nicer way?)
            rs_obj_points, rs_points = cv2.aruco.getBoardObjectAndImagePoints(charuco_board, rs_corners, rs_ids)
            zv_obj_points, zv_points = cv2.aruco.getBoardObjectAndImagePoints(charuco_board, zv_corners, zv_ids)

            # Create dictionary for each frame objectPoint:imagePoint to get common markers detected
            rs_obj_to_points = {
                tuple(a): tuple(b)
                for a, b in zip(rs_obj_points[:, 0], rs_points[:, 0])
            }
            zv_obj_to_points = {
                tuple(a): tuple(b)
                for a, b in zip(zv_obj_points[:, 0], zv_points[:, 0])
            }
            common = set(rs_obj_to_points.keys()) & set(zv_obj_to_points.keys())

            if len(common) < common_points_threshold:
                raise Exception(
                    f"To few respective points found in images ({len(common)})"
                )

            # fill arrays where each index specifies one markers objectPoint and both
            # respective imagePoints
            rs_points = []
            zv_points = []
            for objP in common:
                rs_points.append(rs_obj_to_points[objP])
                zv_points.append(zv_obj_to_points[objP])

            rs_points = np.array(rs_points)[:, [1, 0]]  # swap height, width
            zv_points = np.array(zv_points)[:, [1, 0]]  # swap height, width

            rs_indices = tuple(rs_points.astype(np.uint16).T)
            zv_indices = tuple(zv_points.astype(np.uint16).T)

            rs_rgb = np.repeat(rs_rgb[:, :, None], repeats=3, axis=2)
            zv_rgb = np.repeat(zv_rgb[:, :, None], repeats=3, axis=2)

            rs_depths = rs_depth[rs_indices]
            zv_depths = zv_depth[zv_indices]

            rs_xyz = np.concatenate((rs_points, rs_depths[:, None]), axis=1)
            zv_xyz = np.concatenate((zv_points, zv_depths[:, None]), axis=1)

            # remove nan points
            no_nans = ~np.logical_or(np.isnan(zv_xyz), np.isnan(rs_xyz)).any(axis=1)
            rs_imgs_xyz.extend(rs_xyz[no_nans])
            zv_imgs_xyz.extend(zv_xyz[no_nans])

        except Exception as e:
            print(f"Exception: {e}")
            continue

    if len(rs_imgs_xyz) == 0:
        raise Exception("No image with enough feature points given")

    rs_imgs_xyz = np.array(rs_imgs_xyz)
    zv_imgs_xyz = np.array(zv_imgs_xyz)
    
    rs_camera_xyz = image_points_to_camera_points(rs_imgs_xyz, rs_ci)
    zv_camera_xyz = image_points_to_camera_points(zv_imgs_xyz, zv_ci)

    R, t = __compute_transform_matrix(zv_camera_xyz.T, rs_camera_xyz.T)
    trans = np.identity(4)
    trans[:3, :3] = R
    trans[:3, 3] = t[:, 0]

    print(trans)


if __name__ == "__main__":
    main()
