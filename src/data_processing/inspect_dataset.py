
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


def main(args):
    files = DatasetInterface.get_paths_in_dir(args.dir)

    for file in tqdm(files, desc='inspecting files'):
        rs_rgb, rs_depth, zv_rgb, zv_depth, mask = DatasetInterface.load(file)

        rs_min_depth = np.nanmin(rs_depth)
        zv_min_depth = np.nanmin(zv_depth)

        if rs_min_depth == 0:
            print("rs depth 0", file)
            file.unlink()
            rs_depth = np.where(rs_depth == 0, np.nan, rs_depth)
            DatasetInterface.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, file)

        if zv_min_depth == 0:
            print("zv depth 0", file)
            file.unlink()
            zv_depth = np.where(zv_depth == 0, np.nan, zv_depth)
            DatasetInterface.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, file)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("dir", type=Path, help="dataset directory the bounds should computed for")
    argparse.add_argument("--jobs", type=int, default=cpu_count())
    main(argparse.parse_args())