
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed

from dataset.dataset_interface import DatasetInterface
from utils.general_utils import split


def compute_bounds(files):
    min_depth = np.inf
    max_depth = -np.inf
    for file in tqdm(files, desc='computing depth bounds for normalization'):
        _, rs_depth, _, zv_depth, _ = DatasetInterface.load(file)

        print(np.nanmin(zv_depth))
        print(np.nanmin(np.where(zv_depth == 0, np.nan, zv_depth)))
        min_t = np.nanmin([rs_depth, zv_depth])
        max_t = np.nanmax([rs_depth, zv_depth])
        min_depth = min(min_t, min_depth)
        max_depth = max(max_t, max_depth)
        print(min_depth, max_depth)
    return min_depth, max_depth

def main(args):
    files = DatasetInterface.get_paths_in_dir(args.dir)
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
    argparse.add_argument("--jobs", type=int, default=cpu_count())
    main(argparse.parse_args())