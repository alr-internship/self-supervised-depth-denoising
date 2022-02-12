import logging
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import wandb
from utils.visualization_utils import to_rgb, visualize_depth, visualize_mask
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        device: torch.device,
        scale: float,
        enable_augmentation: bool,
        add_nan_mask_to_input: bool,
        add_region_mask_to_input: bool
    ):
        self.device = device
        self.scale = scale
        self.enable_augmentation = enable_augmentation
        self.add_nan_mask_to_input = add_nan_mask_to_input
        self.add_region_mask_to_input = add_region_mask_to_input

    def evaluate(
        self,
        net: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        net.eval()
        num_val_batches = len(dataloader)
        assert num_val_batches != 0, "at least one batch must be selected for evaluation"

        criterion = nn.L1Loss()
        loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
            images = batch['image']
            labels = batch['label']
            nan_mask = batch['nan-mask']
            region_mask = batch['region-mask']

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            nan_mask = nan_mask.to(device=self.device)
            region_mask = region_mask.to(device=self.device)

            with torch.no_grad():
                predictions = net(images)
                loss += torch.mean(torch.abs(predictions - labels) * nan_mask * region_mask)

        net.train()

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
        val_interval: int,
        save_checkpoint: bool,
        amp: bool,
        activate_wandb: bool,
    ):
        train_id = id(time.time())
        if save_checkpoint:
            dir_checkpoint = dir_checkpoint / f"{net.name}"

        division_step = (val_interval // batch_size)
        n_train = len(train_set)
        n_val = len(val_set)

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
                    add_mask_for_nans=self.add_nan_mask_to_input,
                    train_id=train_id,
                    amp=amp,
                    training_size=n_train,
                    validation_size=n_val,
                    evaluation_interval=division_step
                )
            )

        net = nn.DataParallel(net)
        net.to(self.device)

        logging.info(f'''Starting training:
            WandB:               {activate_wandb}
            Epochs:              {epochs}
            Batch size:          {batch_size}
            Learning rate:       {learning_rate}
            Training size:       {n_train}
            Validation size:     {n_val}
            Validation Interval: {val_interval} (in samples) 
            Checkpoints:         {save_checkpoint}
            Device:              {self.device.type}
            Images scaling:      {self.scale}
            Augmentation:        {self.enable_augmentation}
            NaNs Mask:           {self.add_nan_mask_to_input}
            Region Mask:         {self.add_region_mask_to_input}
            Mixed Precision:     {amp}
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
            #             drop_last=True,
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
                    nan_masks = batch['nan-mask']
                    region_masks = batch['region-mask']

                    # assert images.shape[1] == net.n_channels, \
                    #     f'Network has been defined with {net.n_channels} input channels, ' \
                    #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    #     'the images are loaded correctly.'

                    images = images.to(device=self.device, dtype=torch.float32)
                    label = label.to(device=self.device, dtype=torch.float32)
                    nan_masks = nan_masks.to(device=self.device)
                    region_masks = region_masks.to(device=self.device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        prediction = net(images)
                        # sum loss only where no nan is present
                        loss = torch.mean(torch.abs(prediction - label) * nan_masks * region_masks)
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
                            'step': global_step,
                            'epoch': epoch,
                            'train loss': loss.item(),
                            'learning rate': optimizer.param_groups[0]['lr'],
                        })

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    if global_step % division_step == 0:
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

                            # visualization images
                            # multiply prediction mask with nan mask to remove pixels where nans are
                            # in the input or target
                            vis_image = images[0].cpu().detach().numpy().transpose((1, 2, 0))
                            vis_true_mask = label[0, 0].float().cpu().detach().numpy()
                            vis_pred_mask = (prediction * nan_masks)[0, 0].float().cpu().detach().numpy()

                            experiment_log = {
                                'step': global_step,
                                'epoch': epoch,
                                'validation loss': val_loss,
                                'input': {
                                    'rgb': wandb.Image(to_rgb(vis_image[..., :3])),
                                    'depth': wandb.Image(visualize_depth(vis_image[..., 3])),
                                },
                                'masks': {
                                    'true': wandb.Image(visualize_depth(vis_true_mask)),
                                    'pred': wandb.Image(visualize_depth(vis_pred_mask)),
                                },
                                **histograms
                            }

                            if self.add_nan_mask_to_input:
                                nan_mask = nan_masks[0].cpu().detach().numpy()
                                experiment_log['input']['nan-mask'] = wandb.Image(visualize_mask(nan_mask))
                            if self.add_region_mask_to_input:
                                region_mask = region_masks[0].cpu().detach().numpy()
                                experiment_log['input']['region-mask'] = wandb.Image(visualize_mask(region_mask))

                            experiment.log(experiment_log)

            if save_checkpoint:
                dir_checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'e{epoch+1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if activate_wandb:
            wandb.finish()
