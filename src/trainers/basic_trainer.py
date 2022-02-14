import copy
from pathlib import Path
import torch
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet
from trainers.trainer import Trainer


class BasicTrainer(Trainer):

    def __init__(
        self,
        trainer_id: str,
        device: torch.device,
        train_path: Path,
        val_path: Path,
        dataset_config: BasicDataset.Config,
        bilinear: bool,
    ):
        super().__init__(trainer_id, device, dataset_config)

        val_dataset_config = copy.deepcopy(dataset_config)
        val_dataset_config.enable_augmentation = False

        self.train_dataset = BasicDataset(train_path, dataset_config)
        self.val_dataset = BasicDataset(val_path, dataset_config)
        
        n_input_channels = dataset_config.num_in_channels()
        n_output_channels = 1

        self.M_total = UNet(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            bilinear=bilinear,
            name='M_total'
        )
