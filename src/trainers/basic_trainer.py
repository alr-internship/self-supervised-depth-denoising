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
        add_mask_for_nans: bool,
        bilinear: bool,
    ):
        super().__init__(device, scale, enable_augmentation, add_mask_for_nans)

        dataset_params = dict(scale=scale, add_mask_for_nans=add_mask_for_nans)
        self.train_dataset = BasicDataset(train_path,
                                          enable_augmentation=enable_augmentation, **dataset_params)
        self.val_dataset = BasicDataset(val_path,
                                        enable_augmentation=False, **dataset_params)

        n_input_channels = 5 if add_mask_for_nans else 4
        n_output_channels = 1

        self.M_total = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            bilinear=bilinear,
            name='M_total'
        )