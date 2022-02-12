from ctypes import addressof
from pathlib import Path
import torch
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet
from trainers.trainer import Trainer


class BasicTrainer(Trainer):

    def __init__(
        self,
        device: torch.device,
        train_path: Path,
        val_path: Path,
        scale: float,
        enable_augmentation: bool,
        add_nan_mask_to_input: bool,
        add_region_mask_to_input: bool,
        bilinear: bool,
    ):
        super().__init__(device, scale, enable_augmentation,
                         add_nan_mask_to_input, add_region_mask_to_input)

        dataset_params = dict(scale=scale, add_nan_mask_to_input=add_nan_mask_to_input,
                              add_region_mask_to_input=add_region_mask_to_input)
        self.train_dataset = BasicDataset(train_path,
                                          enable_augmentation=enable_augmentation, 
                                          **dataset_params)
        self.val_dataset = BasicDataset(val_path,
                                        enable_augmentation=False, **dataset_params)


        n_input_channels = 4 + add_nan_mask_to_input + add_region_mask_to_input
        n_output_channels = 1

        self.M_total = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            bilinear=bilinear,
            name='M_total'
        )
