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
from utils.transformation_utils import normalize_depth


class BasicDataset(Dataset):

    class Config:
        def __init__(
                self,
                scale: float = 1.0,
                add_nan_mask_to_input: bool = True,
                add_region_mask_to_input: bool = True,
                normalize_depths: bool = False,
                normalize_depths_min: float = 0,
                normalize_depths_max: float = 3000
        ):
            assert 0 < scale <= 1, 'Scale must be between 0 and 1'
            self.scale = scale
            self.add_nan_mask_to_input = add_nan_mask_to_input
            self.add_region_mask_to_input = add_region_mask_to_input
            self.normalize_depths = normalize_depths
            self.normalize_depths_min = normalize_depths_min
            self.normalize_depths_max = normalize_depths_max

        @staticmethod
        def from_config(config: dict):
            return BasicDataset.Config(
                scale=config['scale_images'],
                add_nan_mask_to_input=config['add_nan_mask_to_input'],
                add_region_mask_to_input=config['add_region_mask_to_input'],
                normalize_depths=config['normalize_depths']['active'],
                normalize_depths_min=config['normalize_depths']['min'],
                normalize_depths_max=config['normalize_depths']['max']
            )

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

        if len(self.files) == 0:
            raise RuntimeError(f'Dataset {dataset_path} contains no images, make sure you put your images there')

        logging.info(f'Creating dataset with size {len(self.files)}')

    def __len__(self):
        return len(self.files)

    @classmethod
    def resize_to(cls, img: np.array, newH: int, newW: int):
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        assert len(img.shape) == 3, 'image must be 3 dimensional'

        if img.dtype == np.bool:
            fill_value = False
        elif img.dtype == np.uint8:
            fill_value = 0
        else: # float or else
            fill_value = np.nan

        currH, currW = img.shape[:2]
        if currW / currH == newW / newH: # new aspect ration does match current aspect ratio
            if img.dtype == np.bool:
                return cv2.resize(img.astype(np.uint8), (newW, newH), cv2.INTER_NEAREST).astype(np.bool)
            else:
                return cv2.resize(img, (newW, newH), cv2.INTER_AREA)

        else: # aspect ratio does not match -> add padding
            if currW / currH > newW / newH: # aspect ratio larger -> add padding top, bottom
                scale_factor = newW / currW 
            elif currW / currH < newW / newH: # add padding to the left right
                scale_factor = newH / currH

            if img.dtype == np.bool:
                r = cv2.resize(img.astype(np.uint8), None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST).astype(np.bool)
            else:
                r = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            if len(r.shape) == 2:
                r = r[..., None]

            resizedH, resizedW = r.shape[:2]
            startH = (newH - resizedH) // 2
            startW = (newW - resizedW) // 2

            r_full = np.full((newH, newW, img.shape[2]), fill_value=fill_value, dtype=img.dtype)
            r_full[startH:(resizedH + startH), startW:(resizedW + startW)] = r

            return r_full

    @classmethod
    def resize(cls, img: np.array, scale: float):
        if scale == 1:
            return img

        assert 0 < scale < 1, f"invalid scale {scale}"

        h, w = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)

        return cls.resize_to(img, newH, newW)

    @classmethod
    def preprocess(cls, img: np.array, scale: float):
        img_ndarray = cls.resize(img, scale)
        if len(img_ndarray.shape) == 2:  # expand dim on depth maps
            img_ndarray = img_ndarray[..., None]
        return img_ndarray.transpose((2, 0, 1))  # move WxHxC -> CxWxH

    @classmethod
    def preprocess_set(cls, rs_rgb, rs_depth, region_mask, zv_depth,
                       dataset_config: Config):

        # extend region mask to 3 dims to equal the images
        if len(region_mask.shape) == 2:
            region_mask = region_mask[..., None]

        # if mask is still on a per object basis => union
        if region_mask.shape[2] > 1:
            region_mask = np.sum(region_mask, axis=2, keepdims=True) > 0

        assert rs_rgb.shape[:2] == zv_depth.shape[:2], \
            f'Image and mask should be the same size, but are {rs_rgb.shape[:2]} and {zv_depth[:2]}'

        original_size = rs_rgb.shape[:2]

        # apply region mask
        rs_rgb = np.where(region_mask, rs_rgb, 0)
        rs_depth = np.where(region_mask, rs_depth[..., None], np.nan)
        zv_depth = np.where(region_mask, zv_depth[..., None], np.nan)

        # scale to intersting region
        # mask_indices = region_mask.nonzero()
        # min = np.min(mask_indices, axis=1) - 5 # padding 5
        # min = np.where(min < 0, 0, min)
        # max = np.max(mask_indices, axis=1) + 5 # padding 5
        # max = np.where(max > region_mask.shape, region_mask.shape, max)
        # region_mask = region_mask[min[0]:max[0], min[1]:max[1]]
        # rs_depth = rs_depth[min[0]:max[0], min[1]:max[1]]
        # rs_rgb = rs_rgb[min[0]:max[0], min[1]:max[1]]
        # zv_depth = zv_depth[min[0]:max[0], min[1]:max[1]]
        # region_mask = cls.resize_to(region_mask, *original_size)
        # zv_depth = cls.resize_to(zv_depth, *original_size)
        # rs_rgb = cls.resize_to(rs_rgb, *original_size)
        # rs_depth = cls.resize_to(rs_depth, *original_size)

        # resize and transpose
        processed_rs_rgb = cls.preprocess(rs_rgb, dataset_config.scale)
        processed_rs_depth = cls.preprocess(rs_depth, dataset_config.scale)
        processed_region_mask = cls.preprocess(region_mask, dataset_config.scale)
        processed_zv_depth = cls.preprocess(zv_depth, dataset_config.scale)

        # normalize rgb and depth
        processed_rs_rgb = processed_rs_rgb.astype(np.float32) / 255
        if dataset_config.normalize_depths:
            params = dict(min=dataset_config.normalize_depths_min, max=dataset_config.normalize_depths_max)
            processed_rs_depth = normalize_depth(processed_rs_depth, **params)
            processed_zv_depth = normalize_depth(processed_zv_depth, **params)

        # map nan to 0
        # add nan mask. 0 implies nan value in input region
        nan_mask = ~np.logical_and(np.isnan(processed_rs_depth), processed_region_mask)
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
            return self.preprocess_set(rs_rgb, rs_depth, region_mask, zv_depth, self.dataset_config)
        except AssertionError as e:
            raise RuntimeError(f"AssertionError in file {self.files[idx]}: {e}")
