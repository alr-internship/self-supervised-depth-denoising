
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import imgs_to_pcd, rs_ci


def main(args):
    files = DatasetInterface.get_files_by_path(args.dir)
    assert len(files) > 0, "no files in directory"

    for file in files:
        rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)

        rs_depth = np.where(mask, rs_depth, np.nan)
        zv_depth = np.where(mask, zv_depth, np.nan)

        diff = np.abs(rs_depth - zv_depth)
        mean_diff = np.nanmean(diff)
        max_diff = np.nanmax(diff)
        print(max_diff, file)
        if max_diff > 20:
            color = np.where(diff[..., None] > 20, [255, 0, 0], [0, 0, 0]).astype(np.uint8)
            depth = rs_depth # np.where(diff <= 50, np.nan, rs_depth)
            pcd = imgs_to_pcd(color, depth.astype(np.float32), rs_ci)
            import open3d as o3d
            o3d.visualization.draw_geometries([pcd])
        # rs_depth = np.where(diff > 3 * mean_diff, np.nan, rs_depth)
        # zv_depth = np.where(diff > 3 * mean_diff, np.nan, zv_depth)

        # rs_rgb = np.where(diff[..., None] > 3 * mean_diff, [255, 0, 0], rs_rgb).astype(np.uint8)
        # from utils.transformation_utils import imgs_to_pcd, rs_ci
        # pcds = imgs_to_pcd(rs_rgb, rs_depth.astype(np.float32), rs_ci)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("dir", type=Path, help="dataset directory the bounds should computed for")
    main(argparse.parse_args())

# 