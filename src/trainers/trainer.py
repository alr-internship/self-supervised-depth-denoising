from argparse import ArgumentParser
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
from utils.visualization_utils import to_rgb, visualize_depth

from tqdm import tqdm
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet


class OutOfFoldTrainer:
    def __init__(
        self,
        device: torch.device,
        dataset_path: str,
        scale: float,
        oof_p: float,
        n_input_channels: int,
        n_output_channels: int,
        bilinear: bool,
    ):
        self.device = device

        # TODO: split up dataset to the P's randomly
        dataset = BasicDataset(Path(dataset_path), scale)
        lens = np.floor([len(dataset) * oof_p for _ in range(3)]).astype(np.int32)
        lens[-1] += len(dataset) - lens[-1] // oof_p
        assert(sum(lens) == len(dataset))
        self.P_1, self.P_2, self.P_test = random_split(dataset, lens)

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

    # TODO: implement train, evaluate, update_dataset

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
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
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
        device: torch.device,
        dir_checkpoint: Path,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 0.001,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        activate_wandb: bool = False,
    ):

        # (Initialize logging)
        if activate_wandb:
            experiment = wandb.init(project=net.name, resume='allow', entity="depth-denoising", reinit=True)
            experiment.config.update(
                dict(epochs=epochs, batch_size=batch_size,
                     learning_rate=learning_rate, save_checkpoint=save_checkpoint, img_scale=img_scale,
                     amp=amp)
            )

        net.to(device)

        n_train = len(train_set)
        n_val = len(val_set)

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
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
                    true_masks = batch['mask']

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        loss = criterion(masks_pred, true_masks)

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
                        val_loss = self.evaluate(net, val_loader, device)
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

                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_loss,
                                'input': {
                                    'rgb': wandb.Image(to_rgb(np.transpose(np.asarray(images[0, :3].cpu()), axes=(1, 2, 0)))),
                                    'depth': wandb.Image(visualize_depth(images[0, 3].cpu()))
                                },
                                'masks': {
                                    'true': wandb.Image(visualize_depth(true_masks[0].float().cpu())),
                                    'pred': wandb.Image(visualize_depth(masks_pred[0].float().cpu())),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })

            if save_checkpoint:
                dir_checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'{net.name}_e{epoch+1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if activate_wandb:
            wandb.finish()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof = OutOfFoldTrainer(
        device=device,
        dataset_path=args.dataset_path,
        scale=args.scale_images,
        oof_p=args.oof_p,
        n_input_channels=args.n_input_channels,
        n_output_channels=args.n_output_channels,
        bilinear=args.bilinear
    )

    params = dict(
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_checkpoint=args.save,
        img_scale=args.scale_images,
        amp=args.amp,
        dir_checkpoint=Path(args.dir_checkpoint),
        activate_wandb=args.wandb
    )

    # Training M_11
    oof.train(
        net=oof.M_11,
        train_set=oof.P_1,
        val_set=oof.P_2,
        **params
    )
    # Training M_12
    oof.train(
        net=oof.M_12,
        train_set=oof.P_2,
        val_set=oof.P_1,
        **params
    )
    # Training M_1
    # oof.train(
    #     net=oof.M_1,
    #     train_set=None,  # TODO
    #     val_set=None,  # TODO
    # )
    # TODO: function to create M_x(P_x)
    # TODO: save net weights


if __name__ == '__main__':
    file_dir = Path(__file__).parent

    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)  # Number of epochs
    parser.add_argument("--batch_size", type=int, default=1)  # Batch size
    parser.add_argument("--learning_rate", type=float, default=0.00001)  # Learning rate
    parser.add_argument("--load_from_model", type=str, default=None)  # Load model from a .pth file (path)
    parser.add_argument("--scale_images", type=float, default=0.5)  # Downscaling factor of the images
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
    parser.add_argument("--n_input_channels", type=int, default=4)          # rgbd as input
    parser.add_argument("--n_output_channels", type=int, default=1)     # depth as output
    parser.add_argument("--amp", type=lambda x: bool(strtobool(x)), nargs='?',
                        const=True, default=False)  # Use mixed precision
    main(parser.parse_args())
