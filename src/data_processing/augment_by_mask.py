
from argparse import ArgumentParser
import itertools
from pathlib import Path
from typing import List
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

import numpy as np

from dataset.dataset_interface import DatasetInterface
from utils.general_utils import split


def augment(data: List[np.array]):
    rs_rgb, rs_depth, zv_rgb, zv_depth, masks = data
    assert masks is not None, "mask must be present to augment data"
    num_masks = masks.shape[-1]
    # generate binary combinations
    masks_combinations = list(itertools.product([0, 1], repeat=num_masks))
    print(f"generate {len(masks_combinations)} augmentations for image")

    augmented_dataset = []
    for masks_combination in masks_combinations:
        masks_indices = np.nonzero(masks_combination)[0]
        if len(masks_indices) == 0:
            continue
        mask = np.expand_dims(np.sum(masks[:, :, masks_indices], axis=2) > 0, axis=2)
        rs_rgb_masked = rs_rgb * mask
        rs_depth_masked = np.expand_dims(rs_depth, axis=2) * mask
        zv_rgb_masked = zv_rgb * mask
        zv_depth_masked = np.expand_dims(zv_depth, axis=2) * mask
        augmented_dataset.append([rs_rgb_masked, rs_depth_masked, zv_rgb_masked, zv_depth_masked, mask])
    return augmented_dataset


def augment_files(in_dir: Path, out_dir: Path, files: List[Path]):
    num_augmented_files = 0
    num_files = len(files)
    for idx, file in enumerate(files):
        print(f"processing {file}, {idx}/{num_files}")
        relative_dir_path = file.relative_to(in_dir).parent
        data = DatasetInterface.load(file)

        augmented_dataset = augment(data)
        num_augmented_files += len(augmented_dataset)

        for idx, augmented_data in enumerate(augmented_dataset):
            DatasetInterface.save(
                *augmented_data, 
                file_name=out_dir / relative_dir_path / f"{file.stem}_{idx}{file.suffix}"
            )

    return num_augmented_files



def main(args):
    print(f"""
    input directory:  {args.in_dir}
    output directory: {args.out_dir}
    number of jobs:   {args.jobs}
    """)
    files = DatasetInterface.get_paths_in_dir(args.in_dir, recursive=True)
    jobs = args.jobs

    if jobs > 1:
        files_chunked = split(files, jobs)
        num_augmented_files = Parallel(n_jobs=jobs)(
            delayed(augment_files)(args.in_dir, args.out_dir, files_chunk)  for files_chunk in files_chunked
        )
    
    else:
        num_augmented_files = 0
        for file in tqdm(files):
            num_augmented_files += augment_files(args.in_dir, args.out_dir, [file])

    print(f"Generated {np.sum(num_augmented_files)} augmented files")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("in_dir", type=Path, help='datset, the augmentation should be computed for')
    argparse.add_argument("out_dir", type=Path, help='directory the dataset including augmentations should be saved to')
    argparse.add_argument("--jobs", type=int, default=cpu_count(), help='number of processes to use')
    main(argparse.parse_args())

