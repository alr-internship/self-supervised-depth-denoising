import os
from pathlib import Path
import torch
from models.networks.LSTMUNet.lstm_unet_model import LSTMUNet

from models.dataset.data_loading import BasicDataset
from models.networks.UNet.unet_model import UNet


class Args:
    epochs = 5                  # Number of epochs
    batch_size = 1              # Batch size
    learning_rate = 0.00001     # Learning rate
    load = None                 # Load model from a .pth file (path)
    scale = 0.5                 # Downscaling factor of the images
    # Percent of the data that is used as validation (0-100)
    validation = 10.0
    amp = False                 # Use mixed precision
    wandb = True                # toggle the usage of wandb for logging purposes
    save = False                # save trained model
    n_channels = 4              # rgbd
    bilinear = True             # unet using bilinear
    dataset_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources', 'images')


class OutOfFoldTrainer:
    def __init__(
        self,
        device: torch.device,
        args: Args,
    ):
        self.device = device
        self.args = args,

        # TODO: split up dataset to the P's randomly
        dataset = BasicDataset(args.dataset_path, args.scale)

        self.P_1 = None
        self.P_2 = None
        self.P_test = None

        self.M_11 = UNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
        )
        self.M_12 = UNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
        )
        self.M_1 = UNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
        )
        self.M_2 = LSTMUNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
        )

    # TODO: implement train, evaluate, update_dataset
    def train():
