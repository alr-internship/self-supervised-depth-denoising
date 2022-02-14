"""
Reference: https://github.com/milesial/Pytorch-UNet
"""

import logging
from pathlib import Path

from imgaug import augmenters as iaa
from imgaug import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from dataset.dataset_interface import DatasetInterface


class BasicDataset(Dataset):

    class Config:
        def __init__(
                self,
                scale: float = 1.0,
                enable_augmentation: bool = True,
                add_nan_mask_to_input: bool = True,
                add_region_mask_to_input: bool = True,
                normalize_depths: bool = False):
            assert 0 < scale <= 1, 'Scale must be between 0 and 1'
            self.scale = scale
            self.enable_augmentation = enable_augmentation
            self.add_nan_mask_to_input = add_nan_mask_to_input
            self.add_region_mask_to_input = add_region_mask_to_input
            self.normalize_depths = normalize_depths

        def __iter__(self):
            yield 'scale', self.scale
            yield 'enable_augmentation', self.enable_augmentation
            yield 'add_nan_mask_to_input', self.add_nan_mask_to_input
            yield 'add_region_mask_to_input', self.add_region_mask_to_input
            yield 'normalize_depths', self.normalize_depths

        def num_in_channels(self):
            return 4 + self.add_region_mask_to_input + self.add_nan_mask_to_input

        def get_printout(self):
            return f"""
                Images scaling:      {self.scale}
                Augmentation:        {self.enable_augmentation}
                NaNs Mask:           {self.add_nan_mask_to_input}
                Region Mask:         {self.add_region_mask_to_input}
                Normalize Depths:    {self.normalize_depths}
            """

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8),
        )
    ], random_order=True)  # apply augmenters in random order

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
    def augment(cls, rs_rgb, rs_depth, zv_depth, crop_region_mask):
        depths = np.concatenate((rs_depth[..., None], zv_depth[..., None]), axis=2)

        heatmaps = HeatmapsOnImage(depths, min_value=np.nanmin(depths),
                                   max_value=np.nanmax(depths),
                                   shape=rs_rgb.shape)

        # augmentation adds some random padding to depth maps
        # this segmap sets this padding to NaN again
        segmap = crop_region_mask
        segmaps = SegmentationMapsOnImage(segmap, shape=rs_rgb.shape[:2])

        augmented = cls.seq(image=rs_rgb, heatmaps=heatmaps,
                             segmentation_maps=segmaps)

        aug_rs_rgb = augmented[0]
        aug_heatmaps = augmented[1].get_arr()
        aug_segmap = augmented[2].get_arr()

        aug_rs_depth = aug_heatmaps[..., 0]
        aug_zv_depth = aug_heatmaps[..., 1]

        return aug_rs_rgb, aug_rs_depth, aug_zv_depth, aug_segmap

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
                       dataset_config: Config):

        if dataset_config.enable_augmentation:
            rs_rgb, rs_depth, zv_depth, region_mask = cls.augment(rs_rgb, rs_depth,
                                                                   zv_depth, region_mask)

        assert rs_rgb.shape[:2] == zv_depth.shape[:2], \
            f'Image and mask should be the same size, but are {rs_rgb.shape[:2]} and {zv_depth[:2]}'

        processed_rs_rgb = cls.preprocess(rs_rgb, dataset_config.scale)
        processed_rs_depth = cls.preprocess(rs_depth, dataset_config.scale)
        processed_region_mask = cls.preprocess(region_mask, dataset_config.scale)
        processed_zv_depth = cls.preprocess(zv_depth, dataset_config.scale)

        # normalize
        processed_rs_rgb = processed_rs_rgb.astype(np.float32) / 255

        # map nan to 0 and add mask to inform net about where nans are located
        rs_nan_mask = np.logical_not(np.isnan(processed_rs_depth))
        zv_nan_mask = np.logical_not(np.isnan(processed_zv_depth))
        nan_mask = np.logical_and(rs_nan_mask, zv_nan_mask)
        processed_rs_depth = np.nan_to_num(processed_rs_depth)
        processed_zv_depth = np.nan_to_num(processed_zv_depth)

        # normalize depth
        if dataset_config.normalize_depths:
            eps = np.finfo(float).eps
            processed_rs_depth = (processed_rs_depth - np.mean(processed_rs_depth)) / (np.std(processed_rs_depth + eps))
            processed_zv_depth = (processed_zv_depth - np.mean(processed_zv_depth)) / (np.std(processed_zv_depth + eps))

        input_tuple = (processed_rs_rgb, processed_rs_depth)

        if dataset_config.add_nan_mask_to_input:
            input_tuple += (nan_mask,)
        if dataset_config.add_region_mask_to_input:
            input_tuple += (processed_region_mask,)

        input = np.concatenate(input_tuple, axis=0)
        label = processed_zv_depth

        return {
            'image': torch.as_tensor(input.copy()).float().contiguous(),
            'label': torch.as_tensor(label.copy()).float().contiguous(),
            'nan-mask': torch.as_tensor(nan_mask.copy()),
            'region-mask': torch.as_tensor(processed_region_mask.copy())
        }

    def __getitem__(self, idx):
        rs_rgb, rs_depth, _, zv_depth, region_mask = DatasetInterface.load(self.files[idx])

        if region_mask.shape[2] > 1:
            # mask is still on a per object basis => union
            region_mask = np.sum(region_mask, axis=2) > 0

        return self.preprocess_set(rs_rgb, rs_depth, region_mask, zv_depth, self.dataset_config)
