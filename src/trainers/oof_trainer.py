import copy
from ctypes import addressof
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import random_split
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet
from trainers.trainer import Trainer


class OutOfFoldTrainer(Trainer):

    @staticmethod
    def get_oof_dataset(dataset_path: Path, oof_p: float, dataset_config: BasicDataset.Config):
        dataset = BasicDataset(dataset_path, dataset_config)
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
        trainer_id: str,
        device: torch.device,
        dataset_path: Path,
        dataset_config: BasicDataset.Config,
        oof_p: float,
        initial_channels: int,
        bilinear: bool,
    ):
        super.__init__(trainer_id, device, dataset_config)

        dataset_params = dict(dataset_path=dataset_path, oof_p=oof_p)

        val_dataset_config = copy.deepcopy(dataset_config)
        val_dataset_config.enable_augmentation = False

        # train sets
        self.P_1, self.P_2, self.P_test = self.get_oof_dataset(
            dataset_config=dataset_config, **dataset_params)
        self.P_1_and_P_2 = torch.utils.data.ConcatDataset([self.P_1, self.P_2])

        # validation sets
        P_1_val, P_2_val, P_test_val = self.get_oof_dataset(
            dataset_config=val_dataset_config, **dataset_params)
        self.P_1_val = self.get_validation_subset(P_1_val)
        self.P_2_val = self.get_validation_subset(P_2_val)
        self.P_test_val = self.get_validation_subset(P_test_val)

        n_input_channels = dataset_config.num_in_channels()
        n_output_channels = 1

        self.M_11 = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            initial_channels=initial_channels,
            name='M_11'
        )
        self.M_12 = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            initial_channels=initial_channels,
            bilinear=bilinear,
            name='M_12'
        )
        self.M_1 = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            initial_channels=initial_channels,
            bilinear=bilinear,
            name='M_1'
        )
        # self.M_2 = LSTMUNet(
        #     n_channels=Args.n_channels,
        #     bilinear=Args.bilinear,
        #     name='M_2'
        # )
