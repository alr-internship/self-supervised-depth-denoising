from argparse import ArgumentParser
from typing import List
from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from pathlib import Path
import open3d as o3d
import numpy as np
from joblib import Parallel, delayed
import cv2
from utils.general_utils import split
from utils.transformation_utils import imgs_to_pcd, pcd_to_imgs, rs_ci, zv_ci


transformation = np.array(
    [[0.99807603, -0.01957923, -0.05882929,  0.17911331],
        [0.0212527, 0.99938319, 0.02795643, -0.01932425],
        [0.05824564, -0.02915292, 0.99787652, -0.03608739],
        [0., 0., 0., 1.]]
)


def align_cropped(rs_rgb, rs_depth, zv_rgb, zv_depth, debug: bool):
    if debug:
        rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, zv_ci)
    final_size = (rs_rgb.shape[1], rs_rgb[0])

    if debug:
        o3d.visualization.draw_geometries([zv_pcd, rs_pcd], "raw pcds")
    zv_pcd.transform(transformation)
    zv_rgb, zv_depth, ul_corner, lr_corner = pcd_to_imgs(zv_pcd, rs_ci)
    if debug:
        o3d.visualization.draw_geometries([zv_pcd, rs_pcd], "calibrated pcds")

    zv_rgb = zv_rgb[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    rs_rgb = rs_rgb[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    zv_depth = zv_depth[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    rs_depth = rs_depth[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]

    zv_rgb = cv2.resize(zv_rgb, final_size)
    zv_depth = cv2.resize(zv_depth, final_size)
    rs_rgb = cv2.resize(rs_rgb, final_size)
    rs_depth = cv2.resize(rs_depth, final_size)

    return rs_rgb, rs_depth, zv_rgb, zv_depth


def align_uncropped(rs_rgb, rs_depth, zv_rgb, zv_depth, debug: bool):
    if debug:
        rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, zv_ci)

    if debug:
        o3d.visualization.draw_geometries([zv_pcd, rs_pcd], "raw pcds")
    zv_pcd.transform(transformation)
    if debug:
        o3d.visualization.draw_geometries([zv_pcd, rs_pcd], "calibrated pcds")

    zv_rgb, zv_depth, ul_corner, lr_corner = pcd_to_imgs(zv_pcd, rs_ci)

    zv_rgb_large = np.zeros_like(rs_rgb)
    zv_depth_large = np.zeros_like(rs_depth)

    zv_rgb_large[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]
                 ] = zv_rgb[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    zv_depth_large[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]
                   ] = zv_depth[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
    zv_rgb = zv_rgb_large
    zv_depth = zv_depth_large

    assert zv_rgb.shape == rs_rgb.shape and zv_depth.shape == rs_depth.shape

    # mask out not overlapping area
    rs_mask = np.zeros(rs_depth.shape, dtype=np.uint8)
    rs_mask[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]] = 1

    rs_rgb = rs_rgb * rs_mask[..., None]
    rs_depth = rs_depth * rs_mask

    return rs_rgb, rs_depth, zv_rgb, zv_depth


def align(cropped: bool, out_path: Path, in_path: Path, debug: bool, files: List[Path]):
    for file in tqdm(files):
        image_tuple = DatasetInterface.load(file)[:4]  # skip mask
        if cropped:
            aligned_image_tuple = align_cropped(*image_tuple, debug=debug)
        else:
            aligned_image_tuple = align_uncropped(*image_tuple, debug=debug)
        rel_file_path = file.relative_to(in_path)
        DatasetInterface.save(*aligned_image_tuple, None, out_path / rel_file_path)


def main(args):
    files = DatasetInterface.get_paths_in_dir(args.raw_dir)
    print(f"files to process: {len(files)}")

    if args.jobs > 1:
        files_chunked = split(files, args.jobs)

        Parallel(n_jobs=args.jobs)(
            delayed(align)(args.cropped, args.out_dir, args.raw_dir, args.debug, files_chunk)
            for files_chunk in files_chunked
        )

    else:
        for file in tqdm(files):
            image_tuple = DatasetInterface.load(file)[:4]  # no mask present
            if args.cropped:
                aligned_image_tuple = align_cropped(*image_tuple, args.debug)
            else:
                aligned_image_tuple = align_uncropped(*image_tuple, args.debug)
            rel_dir_path = file.relative_to(args.raw_dir)
            DatasetInterface.save(*aligned_image_tuple, mask=None, file_name=rel_dir_path)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("raw_dir", type=Path, help="path to raw, uncalibrated dataset")
    argparse.add_argument("out_dir", type=Path, help="path where the calibrated files will be saved to")
    argparse.add_argument("--jobs", type=int, default=1, help="number of jobs")
    argparse.add_argument("--cropped", action="store_true",
                          help="if the resulting images should be cropped to remove black regions")
    argparse.add_argument("--debug", action="store_true", help="visualized uncalibrated and calibrated results")
    main(argparse.parse_args())
