import pickletools
import time
from unittest import result
from models.dataset.dataset_interface import DatasetInterface
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

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


def imgs_to_pcd(bgr, depth, ci):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        ci
    )
    return pcd

def pcd_to_imgs(pcd, ci, filename: str):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    fx, fy = ci.get_focal_length()
    cx, cy = ci.get_principal_point()
    w_h = np.asarray([ci.width - 1, ci.height - 1])

    pixels = []
    depths = []
    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]

        u = x * fx / z + cx
        v = y * fy / z + cy
        d = z
        pixels.append([u, v])
        depths.append([d])
    
    pixels = np.asarray(pixels)
    depths = np.asarray(depths)

    min = np.min(pixels, axis=0)
    max = np.max(pixels, axis=0)
    print("before", min, max)

    # mapped_pixels = (pixels - min) * w_h / (max - min)
    mapped_pixels = pixels - min

    min = np.min(mapped_pixels, axis=0)
    max = np.max(mapped_pixels, axis=0)
    print("after", min, max)

    mapped_pixels = mapped_pixels.astype(np.uint16)

    picture_size = np.round_(max - min).astype(np.uint16).T
    print(picture_size)
    rgb = np.zeros((picture_size[1] + 1, picture_size[0] + 1, 3))
    for idx, mapped_pixel in enumerate(mapped_pixels):
       rgb[mapped_pixel[1], mapped_pixel[0]] = colors[idx]

    print(np.min(rgb, axis=(0, 1)))

    rgb = rgb * 255
    rgb = rgb.astype(np.uint8)

    cv2.imwrite(f"{filename}.png", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))


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


def main():
    dataset_interface = DatasetInterface(Path("resources/images/uncalibrated/dataset_4"))

    rs_rgb, rs_depth, zv_rgb, zv_depth = dataset_interface[20]

    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, zv_ci)

    source_ci = zv_ci
    target_ci = rs_ci
    source_pcd = zv_pcd
    target_pcd = rs_pcd

    # source_pts, target_pts = __ask_to_annotate_points(zv_pcd, rs_pcd)
    # print(source_pts)
    # print(target_pts)

    source_pts = np.array([[-0.09624184, -0.14684464,  0.73445874],
                  [0.04727544, -0.11387316,  0.68202949],
                  [-0.08901674,  0.01387794,  0.86610579],
                  [0.0559292,  0.04645218,  0.81539875],
                  [-0.25726869, -0.2150067,   1.18018293]])
    target_pts = np.array([[0.03753771, -0.14300872, 0.69400001],
                  [0.18737095, -0.11891152, 0.66100001],
                  [0.04112452, 0.0180759, 0.815],
                  [0.18834673, 0.05571066, 0.778],
                  [-0.14103209, -0.21123472, 1.13300002]])

    R, t = __compute_transform_matrix(source_pts.T, target_pts.T)

    trans = np.identity(4)
    trans[:3, :3] = R
    trans[:3, 3] = t[:, 0]

    # o3d...
    source_pcd.transform(trans)

    dists = target_pcd.compute_point_cloud_distance(source_pcd)
    ind = np.where(np.asarray(dists) <= 0.1)[0]
    target_pcd = target_pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([source_pcd, target_pcd])


    # pcd_to_imgs(source_pcd, source_ci)
    pcd_to_imgs(source_pcd, target_ci, "proj_source")
    # pcd_to_imgs(target_pcd, source_ci)
    pcd_to_imgs(target_pcd, target_ci, "proj_target")

    cv2.imwrite("source.png", zv_rgb)
    cv2.imwrite("target.png", rs_rgb)


if __name__ == "__main__":
    main()
