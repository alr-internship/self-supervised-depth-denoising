"""
Reference: https://github.com/milesial/Pytorch-UNet
"""

import imghdr
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.utils import data
from torch.utils.data import Dataset 

from models.dataset.dataset_interface import DatasetInterface


class BasicDataset(Dataset):
    def __init__(self, dataset_path: Path, scale: float = 1.0):
        self.dataset_interface = DatasetInterface(dataset_path)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        if len(self.dataset_interface) == 0:
            raise RuntimeError(f'Dataset {dataset_path} contains no images, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.dataset_interface)} examples')

    def __len__(self):
        return len(self.dataset_interface)

    @classmethod
    def preprocess_input(cls, rgb: np.array, depth: np.array, scale: float):
        processed_rgb = cls.preprocess(rgb, scale)
        processed_depth = cls.preprocess(depth, scale)

        img_ndarray = np.concatenate((processed_rgb, processed_depth), axis=0)
        return img_ndarray

    @classmethod
    def preprocess(cls, img: np.array, scale: float):
        # scale images 
        h, w = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)

        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        img_ndarray= cv2.resize(img, (newW, newH), cv2.INTER_LINEAR)
        # pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)

        # normalize images
        if len(img_ndarray.shape) == 2:
            img_ndarray = np.expand_dims(img_ndarray, axis=2)

        img_ndarray = img_ndarray.transpose((2, 0, 1))  # move WxHxC -> CxWxH
        # img_ndarray = img_ndarray / 255                 # [0, 255] -> [0, 1]
        return img_ndarray

    def __getitem__(self, idx):
        rs_rgb, rs_depth, _, zv_depth = self.dataset_interface[idx]

        assert rs_rgb.shape[:2] == zv_depth.shape[:2], \
            f'Image and mask {idx} should be the same size, but are {rs_rgb.shape[:2]} and {zv_depth[:2]}'

        # map nan to numbers
        rs_depth = np.nan_to_num(rs_depth)
        zv_depth = np.nan_to_num(zv_depth)

        input = self.preprocess_input(rs_rgb / 255, rs_depth, self.scale)
        label = self.preprocess(zv_depth, self.scale)

        return {
            'image': torch.as_tensor(input.copy()).float().contiguous(),
            'mask': torch.as_tensor(label.copy()).float().contiguous()
        }
