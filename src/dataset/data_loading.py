"""
Reference: https://github.com/milesial/Pytorch-UNet
"""

import logging
from pathlib import Path
from re import A
from typing import List, Tuple

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import normalize_depth, unnormalize_depth


class BasicDataset(Dataset):

    class Config:
        def __init__(
                self,
                scale: float = 1.0,
                add_nan_mask_to_input: bool = True,
                add_region_mask_to_input: bool = True,
                normalize_depths: bool = False,
                normalize_depths_min: float = 0,
                normalize_depths_max: float = 3000,
                resize_region_to_fill_input: bool = False,
        ):
            assert 0 < scale <= 1, 'Scale must be between 0 and 1'
            self.scale = scale
            self.add_nan_mask_to_input = add_nan_mask_to_input
            self.add_region_mask_to_input = add_region_mask_to_input
            self.normalize_depths = normalize_depths
            self.normalize_depths_min = normalize_depths_min
            self.normalize_depths_max = normalize_depths_max
            self.resize_region_to_fill_input = resize_region_to_fill_input

        @staticmethod
        def from_config(config: dict):
            return BasicDataset.Config(
                scale=config['scale_images'],
                add_nan_mask_to_input=config['add_nan_mask_to_input'],
                add_region_mask_to_input=config['add_region_mask_to_input'],
                resize_region_to_fill_input=config['resize_region_to_fill_input'] if 'resize_region_to_fill_input' in config else False,
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

    @staticmethod
    def get_fill_type(dtype):
        if dtype == np.bool:
            return False
        elif dtype == np.uint8:
            return 0
        else:  # float or else
            return np.nan

    @classmethod
    def resize_to(cls, img: np.array, newH: int, newW: int):
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        assert len(img.shape) == 3, 'image must be 3 dimensional'

        currH, currW = img.shape[:2]
        if currW / currH == newW / newH:  # new aspect ration does match current aspect ratio
            if img.dtype == np.bool:
                return cv2.resize(img.astype(np.uint8), (newW, newH), cv2.INTER_NEAREST).astype(np.bool)
            else:
                return cv2.resize(img, (newW, newH), cv2.INTER_AREA)

        else:  # aspect ratio does not match -> add padding
            if currW / currH > newW / newH:  # aspect ratio larger -> add padding top, bottom
                scale_factor = newW / currW
            elif currW / currH < newW / newH:  # add padding to the left right
                scale_factor = newH / currH

            if img.dtype == np.bool:
                r = cv2.resize(img.astype(np.uint8), None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_NEAREST).astype(np.bool)
            else:
                r = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            if len(r.shape) == 2:
                r = r[..., None]

            resizedH, resizedW = r.shape[:2]
            startH = (newH - resizedH) // 2
            startW = (newW - resizedW) // 2

            r_full = np.full((newH, newW, img.shape[2]),
                             fill_value=cls.get_fill_type(img.dtype), dtype=img.dtype)
            r_full[startH:(resizedH + startH), startW:(resizedW + startW)] = r

            return r_full

    @classmethod
    def resize_by(cls, img: np.array, scale: float):
        if scale == 1:
            return img

        assert 0 < scale < 1, f"invalid scale {scale}"

        h, w = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)

        return cls.resize_to(img, newH, newW)

    @classmethod
    def preprocess(cls, img: np.array, scale: float):
        img_ndarray = cls.resize_by(img, scale)
        if len(img_ndarray.shape) == 2:  # expand dim on depth maps
            img_ndarray = img_ndarray[..., None]
        return img_ndarray.transpose((2, 0, 1))  # move WxHxC -> CxWxH

    @classmethod
    def postprocess(cls, img: np.array, scale: float):
        if len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
        img_ndarray = cls.resize_by(img, 1 / scale)
        return img_ndarray

    @classmethod
    def resize_to_fill(cls, mask: np.array, arrays_to_fill: List[np.array]):
        mask_indices = mask.nonzero()
        original_size = mask.shape[:2]
        min = np.min(mask_indices, axis=1) - 5  # padding 5
        min = np.where(min < 0, 0, min)[:2]
        max = np.max(mask_indices, axis=1) + 5  # padding 5
        max = np.where(max > mask.shape, mask.shape, max)[:2]

        return (min, max), [
            cls.resize_to(array[min[0]:max[0], min[1]:max[1]], *original_size)
            for array in arrays_to_fill
        ]

    @classmethod
    def undo_resize_to_fill(cls, bbox: Tuple[np.array, np.array], mask, arrays_to_undo: Tuple[np.array]):
        original_size = arrays_to_undo[0].shape[:2]
        min, max = bbox
        bbox_size = max - min

        arrays_to_undo = [
            array if len(array.shape) == 3 else array[..., None]
            for array in arrays_to_undo
        ]
        
        bbox_aspect_ratio = bbox_size[1] / bbox_size[0]
        orig_aspect_ration = original_size[1] / original_size[0]
        if bbox_aspect_ratio > orig_aspect_ration:
            scale_factor = original_size[1] / bbox_size[1]
        else:
            scale_factor = original_size[0] / bbox_size[0]
        
        large_bbox_size = np.round(bbox_size * scale_factor).astype(np.int64)

        startH = (original_size[0] - large_bbox_size[0]) // 2
        startW = (original_size[1] - large_bbox_size[1]) // 2

        unresized_arrays = [
            cls.resize_to(array[startH:(startH +large_bbox_size[0]), startW:(large_bbox_size[1] + startW)], *bbox_size) 
            for array in arrays_to_undo
        ]

        final_arrays = []
        for unresized_array in unresized_arrays:
            full_array = np.full(original_size + (unresized_array.shape[2],),
                                 fill_value=cls.get_fill_type(unresized_array.dtype),
                                 dtype=unresized_array.dtype)
            full_array[min[0]:max[0], min[1]:max[1]] = unresized_array
            final_arrays.append(full_array)
        return tuple(final_arrays)

    @classmethod
    def postprocess_set(cls, set, orig_rm, unprocessed_prediction, dataset_config: Config):
        nan_mask = set['nan-mask'].numpy()
        processed_region_mask = set['region-mask'].numpy()
        fill_input_bbox = set['fill-input-bbox']

        # if dataset_config.add_nan_mask_to_input:
        #     nan_mask = image[4]
        # if dataset_config.add_region_mask_to_input:
        #     processed_region_mask = image[4 + dataset_config.add_nan_mask_to_input]

        # map 0 to nan
        unprocessed_prediction = np.where(nan_mask, unprocessed_prediction, np.nan)

        # unnormalize depths
        if dataset_config.normalize_depths:
            params = dict(min=dataset_config.normalize_depths_min, max=dataset_config.normalize_depths_max)
            unprocessed_prediction = unnormalize_depth(unprocessed_prediction, **params)

        # resize and tranpose
        region_mask = cls.postprocess(processed_region_mask, dataset_config.scale)
        prediction = cls.postprocess(unprocessed_prediction, dataset_config.scale)

        # undo scale image to intersting region
        if dataset_config.resize_region_to_fill_input:
            prediction, region_mask = cls.undo_resize_to_fill(fill_input_bbox, region_mask, [prediction, region_mask])

        # apply region mask (set to NaN)
        prediction = np.where(region_mask, prediction, np.nan)
        prediction = prediction.squeeze()

        return prediction

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

        # apply region mask
        rs_rgb = np.where(region_mask, rs_rgb, 0)
        rs_depth = np.where(region_mask, rs_depth[..., None], np.nan)
        zv_depth = np.where(region_mask, zv_depth[..., None], np.nan)

        # clean depths
        # diff_depth = np.abs(rs_depth - zv_depth)
        # diff_mean = np.nanmean(diff_depth)
        # diff_std = np.nanstd(diff_depth)
        # clean_mask = diff_depth > (diff_mean * 5)
        # rs_depth = np.where(clean_mask, np.nan, rs_depth)
        # rs_rgb = rs_rgb * ~clean_mask
        # zv_depth = np.where(clean_mask, np.nan, zv_depth)

        # print(f'Removed: {np.sum(clean_mask)} from {np.sum(region_mask)}')

        # TODO: back sizing for inference (+norm+augm)
        # scale image to intersting region (region mask bounding box)
        if dataset_config.resize_region_to_fill_input:
            bbox, outputs = cls.resize_to_fill(region_mask, [region_mask, rs_depth, rs_rgb, zv_depth])
            region_mask, rs_depth, rs_rgb, zv_depth = outputs
        else:
            bbox = None

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
            'region-mask': torch.as_tensor(processed_region_mask.copy()),
            'fill-input-bbox': bbox
        }

    def __getitem__(self, idx):
        rs_rgb, rs_depth, _, zv_depth, region_mask = DatasetInterface.load(self.files[idx])

        try:
            return self.preprocess_set(rs_rgb, rs_depth, region_mask, zv_depth, self.dataset_config)
        except AssertionError as e:
            raise RuntimeError(f"AssertionError in file {self.files[idx]}: {e}")
