import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import wandb
from models.networks.LSTMUNet.lstm_unet_model import LSTMUNet

from tqdm import tqdm
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
    p = 1/3                     # length of each P for OOF training
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
        lens = torch.floor([len(dataset) * args.p for _ in range(3)])

        self.P_1, self.P_2, self.P_test = random_split(dataset, lens)

        self.M_11 = UNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
            name='M_11'
        )
        self.M_12 = UNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
            name='M_12'
        )
        self.M_1 = UNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
            name='M_1'
        )
        self.M_2 = LSTMUNet(
            n_in_channels=Args.n_channels,
            bilinear=Args.bilinear,
            name='M_2'
        )

    # TODO: implement train, evaluate, update_dataset

    def evaluate(
        net: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        net.eval()
        num_val_batches = len(dataloader)
        loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(
                device=device, dtype=torch.float32)            # NxWxH

            with torch.no_grad():
                # predict the mask
                mask_pred = net(image)
                loss += torch.abs(mask_pred - mask_true).sum()

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
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 0.001,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
    ):
        n_train = len(train_set)
        n_val = len(val_set)

        # (Initialize logging)
        if self.args.wandb:
            experiment = wandb.init(
                project=net.name, resume='allow', entity="depth-denoising")
            experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                          save_checkpoint=save_checkpoint, img_scale=img_scale,
                                          amp=amp))
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
                    true_masks = true_masks.to(
                        device=device,
                        dtype=torch.float32
                    )

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        loss = criterion(masks_pred, true_masks)
                        """
                        loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(
                                            0, 3, 1, 2).float(),
                                        multiclass=True)
                        """

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    if self.args.wandb:
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

                        if self.args.wandb:
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
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint /
                           'checkpoint_epoch{}.pth'.format(epoch + 1)))
                logging.info(f'Checkpoint {epoch + 1} saved!')


def main():
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof = OutOfFoldTrainer(
        device=device,
        args=args,
    )
    # Training M_11
    oof.train(
        net=oof.M_11,
        train_set=oof.P_1,
        val_set=oof.P_2
    )
    # Training M_12
    oof.train(
        net=oof.M_12,
        train_set=oof.P_2,
        val_set=oof.P_1
    )
    # Training M_1
    oof.train(
        net=oof.M_1,
        train_set=None,  # TODO
        val_set=None,  # TODO
    )
    # TODO: function to create M_x(P_x)
    # TODO: save net weights


if __name__ == '__main__':
    main()
