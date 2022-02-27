
import copy
from pathlib import Path
from typing import Tuple
from joblib import Parallel, delayed
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from psutil import cpu_count

from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from utils.general_utils import split
from utils.transformation_utils import imgs_to_pcd, pcd_to_imgs, rs_ci, fill_to_shape
from utils.visualization_utils import to_rgb


def sample_transformation():
    # rotation settings
    rotate_low = -5 # in degree
    rotate_high = 5 # in degree
    rotate = np.random.rand(3) * (rotate_high - rotate_low) + rotate_low
    rotate = np.deg2rad(rotate)
    R = o3d.geometry.get_rotation_matrix_from_xyz(rotate)

    # translation
    translate_min = [-0.05, -0.05, -0.05] # in meter
    translate_max = [0.05, 0.05, 0.05]
    translation = [np.random.rand(1)\
        * (t_max - t_min) + t_min
        for t_min, t_max in zip (translate_min, translate_max)
    ]


    def transform(pcds):
        transformed_pcds = []
        for pcd in pcds:
            pcd = copy.deepcopy(pcd)
            pcd.rotate(R, center=pcds[0].get_center())
            pcd.translate(translation)
            transformed_pcds.append(pcd)
        return transformed_pcds

    return transform    


def augment_files(files, in_dir, out_dir, num_augs_per_img):
    for file in tqdm(files):
        imgs = DatasetInterface.load(file)
        out_dir_path = out_dir / file.relative_to(in_dir).parent

        # augmented_img_sets = generate_augmentations(imgs, num_augs=num_augs_per_img)
        rs_rgb, rs_depth, zv_rgb, zv_depth, mask = imgs

        if len(mask.shape) == 2:
            mask = mask[..., None]

        mask = np.sum(mask, axis=2, keepdims=True) > 0

        rgb_mask = np.repeat(mask.astype(np.uint8), repeats=3, axis=2)

        rs_pcd = imgs_to_pcd(rs_rgb, rs_depth.astype(np.float32), rs_ci)
        zv_pcd = imgs_to_pcd(zv_rgb, zv_depth.astype(np.float32), rs_ci)
        mask_pcd = imgs_to_pcd(rgb_mask, zv_depth.astype(np.float32), rs_ci)
        # o3d.visualization.draw_geometries([rs_pcd, zv_pcd])

        # _, ax = plt.subplots(3, num_augs + 1)

        # ax[0][0].imshow(rs_rgb)
        # ax[1][0].imshow(zv_rgb)
        # ax[2][0].imshow(mask)
        orignal_imgs = (rs_rgb, rs_depth, zv_rgb, zv_depth, mask)
        original_out_path = out_dir_path / file.name
        if not original_out_path.exists():
            DatasetInterface.save(*orignal_imgs, file_name=original_out_path)

        for idx in range(num_augs_per_img):
            out_file_name = out_dir_path / f"{file.stem}_{idx}.npz"
            if out_file_name.exists():
                continue

            transform = sample_transformation()
            t_rs_pcd, t_zv_pcd, t_mask_pcd = transform([rs_pcd, zv_pcd, mask_pcd])
            # o3d.visualization.draw_geometries([t_rs_pcd, t_zv_pcd])

            t_rs_rgb, t_rs_depth, _, _ = pcd_to_imgs(t_rs_pcd, rs_ci)
            t_zv_rgb, t_zv_depth, _, _ = pcd_to_imgs(t_zv_pcd, rs_ci)
            t_mask, _, _, _ = pcd_to_imgs(t_mask_pcd, rs_ci)

            t_mask = t_mask[..., 0]
            t_mask = t_mask[..., None]
            
            t_rs_rgb = fill_to_shape(t_rs_rgb, rs_rgb.shape, 0, dtype=np.uint8)
            t_rs_depth = fill_to_shape(t_rs_depth, rs_depth.shape, np.nan, dtype=np.float32)
            t_zv_rgb = fill_to_shape(t_zv_rgb, zv_rgb.shape, 0, dtype=np.uint8)
            t_zv_depth = fill_to_shape(t_zv_depth, zv_depth.shape, np.nan, dtype=np.float32)
            t_mask = fill_to_shape(t_mask, mask.shape, False, dtype=bool)
            augmented_imgs = (t_rs_rgb, t_rs_depth, t_zv_rgb, t_zv_depth, t_mask)

            DatasetInterface.save(*augmented_imgs, file_name=out_file_name)

            # ax[0][i].imshow(to_rgb(t_rs_rgb))
            # ax[1][i].imshow(to_rgb(t_zv_rgb))
            # ax[2][i].imshow(t_mask)

        # plt.show()



def main(args):
    img_dir = args.in_dir
    out_dir = args.out_dir
    num_augs_per_img = args.num_augs
    jobs = args.jobs

    files = list(img_dir.rglob("*.npz"))

    if jobs > 1:
        files_chunked = split(files, jobs)

        Parallel(jobs)(
            delayed(augment_files)(files_chunk, img_dir, out_dir, num_augs_per_img)
            for files_chunk in files_chunked
        )

    else:
        augment_files(files)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("in_dir", type=Path)
    argparse.add_argument("out_dir", type=Path)
    argparse.add_argument("--num-augs", type=int, default=5)
    argparse.add_argument("--jobs", type=int, default=cpu_count())
    main(argparse.parse_args())