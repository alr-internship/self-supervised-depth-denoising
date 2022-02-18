"""
Reference: https://github.com/milesial/Pytorch-UNet
"""

import logging
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.dataset_interface import DatasetInterface


class BasicDataset(Dataset):

    class Config:
        def __init__(
                self,
                scale: float = 1.0,
                add_nan_mask_to_input: bool = True,
                add_region_mask_to_input: bool = True,
                normalize_depths: bool = False):
            assert 0 < scale <= 1, 'Scale must be between 0 and 1'
            self.scale = scale
            self.add_nan_mask_to_input = add_nan_mask_to_input
            self.add_region_mask_to_input = add_region_mask_to_input
            self.normalize_depths = normalize_depths

        def __iter__(self):
            yield 'img_scale', self.scale
            yield 'add_nan_mask_to_input', self.add_nan_mask_to_input
            yield 'add_region_mask_to_input', self.add_region_mask_to_input
            yield 'normalize_depths', self.normalize_depths

        def num_in_channels(self):
            return 4 + self.add_region_mask_to_input + self.add_nan_mask_to_input

        def get_printout(self):
            return f"""
                Images scaling:      {self.scale}
                NaNs Mask:           {self.add_nan_mask_to_input}
                Region Mask:         {self.add_region_mask_to_input}
                Normalize Depths:    {self.normalize_depths}
            """

    def __init__(
            self,
            dataset_path: Path,
            config: Config):
        assert dataset_path.exists(), "file at dataset path does not exist"
        self.files = DatasetInterface.get_files_by_path(dataset_path)
        self.dataset_config = config

        if self.dataset_config.normalize_depths:
            self.depth_normalization, _ = self.compute_depth_bounds_normalization(self.files)

        if len(self.files) == 0:
            raise RuntimeError(f'Dataset {dataset_path} contains no images, make sure you put your images there')

        logging.info(f'Creating dataset with size {len(self.files)}')


    def __len__(self):
        return len(self.files)

    @staticmethod
    def compute_depth_bounds_normalization(files):
        min_depth = np.inf
        max_depth = -np.inf
        for file in tqdm(files, desc='computing depth bounds for normalization'):
            _, rs_depth, _, zv_depth, _ = DatasetInterface.load(file)
            min_depth = min(np.nanmin([rs_depth, zv_depth]), min_depth)
            max_depth = max(np.nanmax([rs_depth, zv_depth]), max_depth)
        print(f"computed normalization bounds: min {min_depth}, max {max_depth}")

        norm = lambda depth: (depth - min_depth) / (max_depth - min_depth)
        unnorm = lambda depth: depth * (max_depth - min_depth) + min_depth
        return norm, unnorm


    @classmethod
    def resize(cls, img: np.array, scale: float):
        if scale == 1:
            return img

        assert 0 < scale < 1, f"invalid scale {scale}"

        h, w = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)

        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        if img.dtype == np.bool:
            resized_img = cv2.resize(img.astype(np.uint8), (newW, newH), cv2.INTER_NEAREST).astype(np.bool)
        else:
            resized_img = cv2.resize(img, (newW, newH), cv2.INTER_AREA)

        return resized_img

    @classmethod
    def preprocess(cls, img: np.array, scale: float):
        img_ndarray = cls.resize(img, scale)
        if len(img_ndarray.shape) == 2:  # expand dim on depth maps
            img_ndarray = img_ndarray[..., None]
        return img_ndarray.transpose((2, 0, 1))  # move WxHxC -> CxWxH

    @classmethod
    def preprocess_set(cls, rs_rgb, rs_depth, region_mask, zv_depth,
                       dataset_config: Config, depth_normalization = None):

        if region_mask.shape[2] > 1:
            # mask is still on a per object basis => union
            region_mask = np.sum(region_mask, axis=2) > 0

        assert rs_rgb.shape[:2] == zv_depth.shape[:2], \
            f'Image and mask should be the same size, but are {rs_rgb.shape[:2]} and {zv_depth[:2]}'

        # resize and transpose 
        processed_rs_rgb = cls.preprocess(rs_rgb, dataset_config.scale)
        processed_rs_depth = cls.preprocess(rs_depth, dataset_config.scale)
        processed_region_mask = cls.preprocess(region_mask, dataset_config.scale)
        processed_zv_depth = cls.preprocess(zv_depth, dataset_config.scale)

        # normalize rgb and depth
        processed_rs_rgb = processed_rs_rgb.astype(np.float32) / 255
        if dataset_config.normalize_depths:
            mean1 = np.nanmean(processed_zv_depth)
            processed_rs_depth = depth_normalization(processed_rs_depth)
            processed_zv_depth = depth_normalization(processed_zv_depth)
            mean1 = np.nanmean(processed_zv_depth)

        # map nan to 0 and add mask to inform net about where nans are located
        nan_mask = ~np.logical_or(np.isnan(processed_rs_depth), np.isnan(processed_zv_depth))
        processed_rs_depth = np.nan_to_num(processed_rs_depth)
        processed_zv_depth = np.nan_to_num(processed_zv_depth)

        # some assertions to check channel correctness
        assert processed_rs_rgb.shape[0] == 3
        assert len(processed_rs_depth.shape) == 2 or processed_rs_depth.shape[0] == 1
        assert len(processed_region_mask.shape) == 2 or processed_region_mask.shape[0] == 1
        assert len(nan_mask.shape) == 2 or nan_mask.shape[0] == 1

        # setup input and label
        input_tuple = (processed_rs_rgb, processed_rs_depth)

        if dataset_config.add_nan_mask_to_input:
            input_tuple += (nan_mask,)
        if dataset_config.add_region_mask_to_input:
            input_tuple += (processed_region_mask,)

        input = np.concatenate(input_tuple, axis=0)
        label = processed_zv_depth

        assert input.shape[0] == 4 + dataset_config.add_nan_mask_to_input + dataset_config.add_region_mask_to_input

        return {
            'image': torch.as_tensor(input.copy()).float().contiguous(),
            'label': torch.as_tensor(label.copy().astype(np.float32)).float().contiguous(),
            'nan-mask': torch.as_tensor(nan_mask.copy()),
            'region-mask': torch.as_tensor(processed_region_mask.copy())
        }

    def __getitem__(self, idx):
        rs_rgb, rs_depth, _, zv_depth, region_mask = DatasetInterface.load(self.files[idx])

        try:
            return self.preprocess_set(rs_rgb, rs_depth, region_mask, zv_depth, self.dataset_config, self.depth_normalization)
        except AssertionError as e:
            raise RuntimeError(f"AssertionError in file {self.files[idx]}: {e}")
