from pathlib import Path
import numpy as np
import torch
from torch.utils.data import random_split
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet
from trainers.trainer import Trainer


class OutOfFoldTrainer(Trainer):

    @staticmethod
    def get_oof_dataset(dataset_path: Path, oof_p: float, scale: float, enable_augmentation: bool, add_mask_for_nans: bool):
        dataset = BasicDataset(dataset_path, scale, enable_augmentation, add_mask_for_nans)
        lens = np.floor([len(dataset) * oof_p for _ in range(3)]).astype(np.int32)
        lens[-1] += len(dataset) - lens[-1] // oof_p
        assert(sum(lens) == len(dataset))
        return random_split(dataset, lens)

    @staticmethod
    def get_validation_subset(dataset: torch.utils.data.Dataset, percent: float = 0.1):
        indices = np.random.choice(range(len(dataset)), size=(len(dataset) // int(1 / percent)), replace=False)
        return torch.utils.data.Subset(dataset, indices)

    def __init__(
        self,
        device: torch.device,
        dataset_path: Path,
        scale: float,
        enable_augmentation: bool,
        add_mask_for_nans: bool,
        oof_p: float,
        bilinear: bool,
    ):
        super.__init__(device, scale, enable_augmentation, add_mask_for_nans)

        dataset_params = dict(dataset_path=dataset_path, oof_p=oof_p, scale=scale, add_mask_for_nans=add_mask_for_nans)
        self.P_1, self.P_2, self.P_test = self.get_oof_dataset(
            enable_augmentation=enable_augmentation, **dataset_params)
        self.P_1_and_P_2 = torch.utils.data.ConcatDataset([self.P_1, self.P_2])
        P_1_val, P_2_val, P_test_val = self.get_oof_dataset(enable_augmentation=False, **dataset_params)
        self.P_1_val = self.get_validation_subset(P_1_val)
        self.P_2_val = self.get_validation_subset(P_2_val)
        self.P_test_val = self.get_validation_subset(P_test_val)

        n_input_channels = 5 if add_mask_for_nans else 4
        n_output_channels = 1

        self.M_11 = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            name='M_11'
        )
        self.M_12 = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            bilinear=bilinear,
            name='M_12'
        )
        self.M_1 = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            bilinear=bilinear,
            name='M_1'
        )
        # self.M_2 = LSTMUNet(
        #     n_channels=Args.n_channels,
        #     bilinear=Args.bilinear,
        #     name='M_2'
        # )