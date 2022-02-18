
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed
from wandb import visualize

from dataset.dataset_interface import DatasetInterface
from utils.general_utils import split
from utils.visualization_utils import visualize_depth


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

    for file in tqdm(files, desc='computing depth bounds for normalization'):
        rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)

        rs_min_depth = np.nanmin(rs_depth)
        zv_min_depth = np.nanmin(zv_depth)

        # if rs_min_depth <= 400:
        #     print(file)
        #     plt.imshow(rs_depth)
        #     plt.imshow(np.where(rs_depth <= 400, 1, 0))
        #     plt.show()

        if zv_min_depth == 0:
            print(file)
            file.unlink()
            zv_depth = np.where(zv_depth == 0, np.nan, zv_depth)
            DatasetInterface.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, file)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("dir", type=Path, help="dataset directory the bounds should computed for")
    argparse.add_argument("--jobs", type=int, default=cpu_count())
    main(argparse.parse_args())