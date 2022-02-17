
import copy
from pathlib import Path
from typing import Tuple
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import imgs_to_pcd, pcd_to_imgs, rs_ci
from utils.visualization_utils import to_rgb

def fill_to_shape(array, shape, fill_value, dtype):
    s = (min(shape[0], array.shape[0]), min(shape[1], array.shape[1]))
    array_large = np.full(shape, fill_value=fill_value, dtype=dtype)
    array_large[:s[0], :s[1]] = array[:s[0], :s[1]]
    return array_large


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


def generate_augmentations(imgs: Tuple[np.array, np.array, np.array, np.array], num_augs: int):
    rs_rgb, rs_depth, zv_rgb, zv_depth = imgs

    rs_pcd = imgs_to_pcd(rs_rgb, rs_depth.astype(np.float32), rs_ci)
    zv_pcd = imgs_to_pcd(zv_rgb, zv_depth.astype(np.float32), rs_ci)
    # o3d.visualization.draw_geometries([rs_pcd, zv_pcd])

    # _, ax = plt.subplots(2, num_augs + 1)

    # ax[0][0].imshow(rs_rgb)
    # ax[1][0].imshow(zv_rgb)
    augmented_imgs = [(rs_rgb, rs_depth, zv_rgb, zv_depth)]
    for i in range(1, num_augs + 1):
        transform = sample_transformation()
        t_rs_pcd, t_zv_pcd = transform([rs_pcd, zv_pcd])
        # o3d.visualization.draw_geometries([t_rs_pcd, t_zv_pcd])

        t_rs_rgb, t_rs_depth, _, _ = pcd_to_imgs(t_rs_pcd, rs_ci)
        t_zv_rgb, t_zv_depth, _, _ = pcd_to_imgs(t_zv_pcd, rs_ci)
        
        t_rs_rgb = fill_to_shape(t_rs_rgb, rs_rgb.shape, 0, dtype=np.uint8)
        t_rs_depth = fill_to_shape(t_rs_depth, rs_depth.shape, np.nan, dtype=np.float32)
        t_zv_rgb = fill_to_shape(t_zv_rgb, zv_rgb.shape, 0, dtype=np.uint8)
        t_zv_depth = fill_to_shape(t_zv_depth, zv_depth.shape, np.nan, dtype=np.float32)
        augmented_imgs.append((t_rs_rgb, t_rs_depth, t_zv_rgb, t_zv_depth))

        # ax[0][i].imshow(to_rgb(t_rs_rgb))
        # ax[1][i].imshow(to_rgb(t_zv_rgb))

    # plt.show()

    return augmented_imgs


def main():
    img_dir = Path("resources/images/calibrated")
    out_dir = Path("resources/images/calibrated_augmented")
    num_augs_per_img = 5

    files = list(img_dir.rglob("*.npz"))

    for file in tqdm(files):
        imgs = DatasetInterface.load(file)[:4] # skip mask
        augmented_img_sets = generate_augmentations(imgs, num_augs=num_augs_per_img)

        out_dir_path = out_dir / file.relative_to(img_dir).parent

        for idx, augmented_img_set in enumerate(augmented_img_sets):
            out_file_name = out_dir_path / f"{file.stem}_{idx}.npz"
            DatasetInterface.save(*augmented_img_set, mask=None, file_name=out_file_name)


if __name__ == "__main__":
    main()