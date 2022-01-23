"""
Reference: https://github.com/milesial/Pytorch-UNet
"""

import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
    def preprocess(cls, pil_img, scale: float):
        # scale images 
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST)
        # pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)

        # normalize images
        img_ndarray = np.asarray(pil_img, dtype=np.float32)
        img_ndarray = np.expand_dims(img_ndarray, axis=2)
        img_ndarray = img_ndarray.transpose((2, 0, 1))  # move WxHxD -> DxWxH
        img_ndarray = img_ndarray / 255                 # [0, 255] -> [0, 1]
        return img_ndarray

    def __getitem__(self, idx):
        _ , img, _, mask = self.dataset_interface[idx]

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }
