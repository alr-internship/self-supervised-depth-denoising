from argparse import ArgumentParser
from cgitb import enable
from colorsys import rgb_to_yiq
from distutils.util import strtobool
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import not_equal, optim
from torch.utils.data import DataLoader, random_split
import wandb
from networks.LSTMUNet.lstm_unet_model import LSTMUNet
from utils.visualization_utils import to_rgb, visualize_depth, visualize_mask

from tqdm import tqdm
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet


class OutOfFoldTrainer:

    @staticmethod
    def get_oof_dataset(dataset_path: Path, oof_p: float, scale: float, enable_augmentation: bool, add_mask_for_nans: bool):
        dataset = BasicDataset(Path(dataset_path), scale, enable_augmentation, add_mask_for_nans)
        lens = np.floor([len(dataset) * oof_p for _ in range(3)]).astype(np.int32)
        lens[-1] += len(dataset) - lens[-1] // oof_p
        assert(sum(lens) == len(dataset))
        return random_split(dataset, lens)

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
        self.device = device
        self.scale = scale
        self.enable_augmentation = enable_augmentation
        self.add_mask_for_nans = add_mask_for_nans

        dataset_params = dict(dataset_path=dataset_path, oof_p=oof_p, scale=scale, add_mask_for_nans=add_mask_for_nans)
        self.P_1, self.P_2, self.P_test = self.get_oof_dataset(
            enable_augmentation=enable_augmentation, **dataset_params)
        self.P_1_and_P_2 = torch.utils.data.ConcatDataset([self.P_1, self.P_2])
        self.P_1_val, self.P_2_val, self.P_test_val = self.get_oof_dataset(enable_augmentation=False, **dataset_params)

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

    def evaluate(
        self,
        net: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        net.eval()
        num_val_batches = len(dataloader)
        criterion = nn.L1Loss()
        loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                # predict the mask
                mask_pred = net(image)
                loss += criterion(mask_pred, mask_true)

        net.train()

        # Fixes a potential division by zero error
        if num_val_batches == 0:
            return loss

        return loss / num_val_batches

    def train(
        self,
        net: nn.Module,
        train_set: list,
        val_set: list,
        dir_checkpoint: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        save_checkpoint: bool,
        amp: bool,
        activate_wandb: bool,
    ):
        if save_checkpoint:
            dir_checkpoint = dir_checkpoint\
                / f"{net.name}" / f"bs{batch_size}_aug{self.enable_augmentation}_nanmask{self.add_mask_for_nans}_sc{self.scale}"

        # (Initialize logging)
        if activate_wandb:
            experiment = wandb.init(project=net.name, resume='allow',
                                    entity="depth-denoising", reinit=True)
            experiment.config.update(
                dict(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    save_checkpoint=save_checkpoint,
                    img_scale=self.scale,
                    enable_augmentation=self.enable_augmentation,
                    add_mask_for_nans=self.add_mask_for_nans,
                    amp=amp)
            )

        net.to(self.device)

        n_train = len(train_set)
        n_val = len(val_set)

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {self.device.type}
            Images scaling:  {self.scale}
            Augmentation:    {self.enable_augmentation}
            NaNs Mask:       {self.add_mask_for_nans}
            Mixed Precision: {amp}
            ''')

        loader_args = dict(
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            **loader_args
        )
        val_loader = DataLoader(
            val_set,
            shuffle=False,
            drop_last=True,
            **loader_args
        )

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.RMSprop(
            net.parameters(),
            lr=learning_rate,
            weight_decay=1e-8,
            momentum=0.9
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'max',
            patience=2
        )
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.L1Loss()
        global_step = 0

        # Begin training
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images = batch['image']
                    label = batch['label']
                    nan_mask = batch['nan-mask']

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device, dtype=torch.float32)
                    nan_mask = nan_mask.to(device=self.device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        prediction = net(images)
                        # sum loss only where no nan is present
                        loss = torch.mean(torch.abs(prediction - label) * nan_mask)
                        # loss = criterion(prediction, label)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    if activate_wandb:
                        experiment.log({
                            'train loss': loss.item(),
                            'step': global_step,
                            'epoch': epoch
                        })

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (n_train // (10 * batch_size))
                    if division_step > 0 and global_step % division_step == 0:
                        val_loss = self.evaluate(net, val_loader, self.device)
                        scheduler.step(val_loss)

                        logging.info('Validation Loss: {}'.format(val_loss))

                        if activate_wandb:
                            histograms = {}
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' +
                                           tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' +
                                           tag] = wandb.Histogram(value.grad.data.cpu())

                            vis_image = images[0].cpu().detach().numpy().transpose((1, 2, 0))
                            vis_true_mask = label[0, 0].float().cpu().detach().numpy()
                            vis_pred_mask = prediction[0, 0].float().cpu().detach().numpy()

                            experiment_log = {
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_loss,
                                'input': {
                                    'rgb': wandb.Image(to_rgb(vis_image[..., :3])),
                                    'depth': wandb.Image(visualize_depth(vis_image[..., 3])),
                                },
                                'masks': {
                                    'true': wandb.Image(visualize_depth(vis_true_mask)),
                                    'pred': wandb.Image(visualize_depth(vis_pred_mask)),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            }

                            if self.add_mask_for_nans:
                                experiment_log['input']['mask'] = wandb.Image(visualize_mask(vis_image[..., 4]))

                            experiment.log(experiment_log)

            if save_checkpoint:
                dir_checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'e{epoch+1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if activate_wandb:
            wandb.finish()


def get_validation_subset(dataset: torch.utils.data.Dataset, percent: float = 0.1):
    indices = np.random.choice(range(len(dataset)), size=(len(dataset) // int(1 / percent)), replace=False)
    return torch.utils.data.Subset(dataset, indices)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    oof = OutOfFoldTrainer(
        device=device,
        dataset_path=Path(args.dataset_path),
        scale=args.scale_images,
        enable_augmentation=args.enable_augmentation,
        add_mask_for_nans=args.add_mask_for_nans,
        oof_p=args.oof_p,
        bilinear=args.bilinear
    )

    params = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_checkpoint=args.save,
        amp=args.amp,
        dir_checkpoint=Path(args.dir_checkpoint),
        activate_wandb=args.wandb
    )

    # Training M_12
    oof.train(
        net=oof.M_12,
        train_set=oof.P_2,
        val_set=get_validation_subset(oof.P_1_val),
        **params
    )
    # Training M_1
    oof.train(
        net=oof.M_1,
        train_set=oof.P_1_and_P_2,
        val_set=get_validation_subset(oof.P_test_val),
        **params
    )
    # Training M_11
    oof.train(
        net=oof.M_11,
        train_set=oof.P_1,
        val_set=get_validation_subset(oof.P_2_val),
        **params
    )


if __name__ == '__main__':
    file_dir = Path(__file__).parent

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)  # Number of epochs
    parser.add_argument("--batch_size", type=int, default=1)  # Batch size
    parser.add_argument("--learning_rate", type=float, default=0.00001)  # Learning rate
    parser.add_argument("--load_from_model", type=str, default=None)  # Load model from a .pth file (path)
    parser.add_argument("--scale_images", type=float, default=0.5)  # Downscaling factor of the images
    parser.add_argument("--enable_augmentation", type=lambda x: bool(strtobool(x)), nargs='?',
                        const=True, default=True)  # enable image augmentation
    parser.add_argument("--add_mask_for_nans", type=lambda x: bool(strtobool(x)), nargs='?',
                        const=True, default=True)
    parser.add_argument("--validation_percentage", type=float, default=10.0)
    # Percent of the data that is used as validation (0-100)
    parser.add_argument("--oof_p", type=float, default=1/3)  # length of each P for OOF training)
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        default=True)  # toggle the usage of wandb for logging purposes
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), nargs='?',
                        const=True, default=True)   # save trained model
    parser.add_argument("--dataset_path", type=Path, default=file_dir / "../../resources/images/calibrated/3d_aligned")
    parser.add_argument("--dir_checkpoint", type=Path, default=file_dir / "../../resources/networks")
    parser.add_argument("--bilinear", type=lambda x: bool(strtobool(x)), nargs='?',
                        const=True, default=True)      # unet using bilinear
    parser.add_argument("--amp", type=lambda x: bool(strtobool(x)), nargs='?',
                        const=True, default=False)  # Use mixed precision
    main(parser.parse_args())
