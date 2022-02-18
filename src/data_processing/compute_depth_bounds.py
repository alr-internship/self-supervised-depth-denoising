
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm

from dataset.dataset_interface import DatasetInterface


def main(args):
    files = DatasetInterface.get_paths_in_dir(args.dir)
    min_depth = np.inf
    max_depth = -np.inf
    for file in tqdm(files, desc='computing depth bounds for normalization'):
        _, rs_depth, _, zv_depth, _ = DatasetInterface.load(file)

        plt.imshow(rs_depth)
        plt.imshow(zv_depth)
        plt.show()
        min_t = np.nanmin([rs_depth, zv_depth])
        max_t = np.nanmax([rs_depth, zv_depth])
        print(min_t, max_t)
        min_depth = min(min_t, min_depth)
        max_depth = max(max_t, max_depth)
    print(f"computed normalization bounds: min {min_depth}, max {max_depth}")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("dir", type=Path, 
    help="dataset directory the bounds should computed for")
    main(argparse.parse_args())