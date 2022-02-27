
from argparse import ArgumentParser
from pathlib import Path
from sys import stdin
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed

from dataset.dataset_interface import DatasetInterface
from utils.general_utils import split


def compute_bounds(files):
    min_depth = np.inf
    max_depth = -np.inf
    # for file in tqdm(files, desc='computing depth bounds for normalization'):
    for file in tqdm(files):
        rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)

        rs_depth = np.where(mask, rs_depth, np.nan)
        zv_depth = np.where(mask, zv_depth, np.nan)

        diff = np.abs(rs_depth - zv_depth)
        mean_diff = np.nanmean(diff)
        rs_depth = np.where(diff > 3 * mean_diff, np.nan, rs_depth)
        zv_depth = np.where(diff > 3 * mean_diff, np.nan, zv_depth)

        min_t = np.nanmin([rs_depth, zv_depth])
        max_t = np.nanmax([rs_depth, zv_depth])
        min_depth = min(min_t, min_depth)
        max_depth = max(max_t, max_depth)

        if min_t < 700: # or max_t > 1200:
            print(file)
            from utils.transformation_utils import imgs_to_pcd, rs_ci
            import open3d as o3d
            rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
            zv_pcd = imgs_to_pcd(zv_rgb, zv_depth, rs_ci)
            o3d.visualization.draw_geometries([rs_pcd, zv_pcd])

            if input("should remove [y/N]") == "y":
                file.unlink()
                print("file removed")

    return min_depth, max_depth

def main(args):
    files = DatasetInterface.get_paths_in_dir(args.dir)
    assert len(files) > 0, "no files in directory"
    jobs = args.jobs

    files_chunked = split(files, jobs)
    min_max_depth_list = Parallel(n_jobs=jobs)(
        delayed(compute_bounds)(files_chunk)
        for files_chunk in files_chunked
    )

    min_depth = np.min(min_max_depth_list, axis=0)[0]
    max_depth = np.max(min_max_depth_list, axis=0)[1]
    print(f"computed normalization bounds: min {min_depth}, max {max_depth}")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("dir", type=Path, help="dataset directory the bounds should computed for")
    argparse.add_argument("--jobs", type=int, default=1) # default=cpu_count())
    main(argparse.parse_args())

# 