from enum import Enum
import logging
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import wandb
from dataset.data_loading import BasicDataset
from networks._original.common import act
from utils.transformation_utils import unnormalize_depth
from utils.visualization_utils import to_rgb, visualize_depth, visualize_mask
from tqdm import tqdm

class Trainer:

    def __init__(
        self,
        trainer_id: str,
        device: torch.device,
        dataset_config: BasicDataset.Config
    ):
        self.trainer_id = trainer_id
        self.device = device
        self.dataset_config = dataset_config

    def infer(
        self,
        net,
        batch,
        loss_criterion
    ):
        images = batch['image']
        label = batch['label']
        nan_masks = batch['nan-mask']
        region_masks = batch['region-mask']

        images = images.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device, dtype=torch.float32)
        nan_masks = nan_masks.to(device=self.device)
        region_masks = region_masks.to(device=self.device)

        predictions = net(images)

        # apply loss only on relevant regions
        # loss = loss_criterion(
        #     prediction * nan_masks * region_masks,
        #     label * nan_masks * region_masks
        # ) / len(images)

        # loss = torch.sum(torch.abs(prediction - label) * nan_masks * region_masks) / torch.sum(nan_masks * region_masks)
        loss = torch.sum(((prediction - label) ** 2) * nan_masks * region_masks) / torch.sum(nan_masks * region_masks)

        # loss = batch_loss / len(images)

        return predictions, loss

    def evaluate(
        self,
        net: nn.Module,
        dataloader: DataLoader,
        loss_criterion
    ):
        net.eval()
        num_val_batches = len(dataloader)
        assert num_val_batches != 0, "at least one batch must be selected for evaluation"

        loss = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, desc='Validation round', unit='batch', leave=False):
            with torch.no_grad():
                _, batch_loss = self.infer(net, batch, loss_criterion)
                loss += batch_loss

        net.train()

        return loss / num_val_batches

    def __evaluate_for_visualization(self, dataloader: DataLoader, net, loss_criterion):
        batch = dataloader[0]
        predictions, _ = self.infer(net, batch, loss_criterion)

        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        # visualizate images
        vis_image = batch['image'][0].numpy().transpose((1, 2, 0))
        nan_mask = batch['nan-mask'][0].numpy().squeeze()
        region_mask = batch['region-mask'][0].numpy().squeeze()
        vis_depth_input = vis_image[..., 3].squeeze()
        vis_depth_label = batch['label'][0].numpy().squeeze()
        depth_prediction = predictions[0].float().cpu().detach().numpy().squeeze()

        # apply masks
        mask = np.logical_and(nan_mask, region_mask)
        vis_depth_prediction = np.where(mask, depth_prediction, np.nan)
        vis_depth_input = np.where(mask, vis_depth_input, np.nan)
        vis_depth_label = np.where(mask, vis_depth_label, np.nan)

        # undo normalization
        if self.dataset_config.normalize_depths:
            params = dict(min=self.dataset_config.normalize_depths_min, max=self.dataset_config.normalize_depths_max)
            vis_depth_input = unnormalize_depth(vis_depth_input, **params) 
            vis_depth_label = unnormalize_depth(vis_depth_label, **params) 
            vis_depth_prediction = unnormalize_depth(vis_depth_prediction, **params) 

        experiment_log = {
            'input': {
                'rgb': wandb.Image(to_rgb(vis_image[..., :3])),
                'depth': wandb.Image(visualize_depth(vis_depth_input)),
            },
            'output': {
                'label': wandb.Image(visualize_depth(vis_depth_label)),
                'pred': wandb.Image(visualize_depth(vis_depth_prediction)),
            },
            **histograms
        }

        if self.dataset_config.add_nan_mask_to_input:
            experiment_log['input']['nan-mask'] = wandb.Image(visualize_mask(nan_mask))
        if self.dataset_config.add_region_mask_to_input:
            experiment_log['input']['region-mask'] = wandb.Image(visualize_mask(region_mask))

        return experiment_log

    def train(
        self,
        net: nn.Module,
        train_set: list,
        val_set: list,
        dir_checkpoint: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        lr_patience: int,
        val_interval: int,
        save_checkpoint: bool,
        amp: bool,
        activate_wandb: bool,
        optimizer_name: str,
    ):
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
                    lr_patience=lr_patience,
                    save_checkpoint=save_checkpoint,
                    **dict(self.dataset_config),
                    trainer_id=self.trainer_id,
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
            LR patience:         {lr_patience}
            Training size:       {n_train}
            Validation size:     {n_val}
            Validation Interval: {val_interval} (in samples) 
            Checkpoints:         {save_checkpoint}
            Device:              {self.device.type}
            Dataset Config:      
                {self.dataset_config.get_printout()}
            Mixed Precision:     {amp}
            Optimizer:           {optimizer_name}
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
            **loader_args
        )

        ##############
        #### LOSS ####
        ##############
        loss_criterion = nn.L1Loss(reduction='mean')

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        if optimizer_name == 'rmpsprop':
            optimizer = optim.RMSprop(
                net.parameters(),
                lr=learning_rate,
                weight_decay=1e-8,
                momentum=0.9
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                net.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False
            )
        else:
            RuntimeError(f"invalid optimizer name given {optimizer_name}")

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=lr_patience
        )
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        global_step = 0

        # Begin training
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    with torch.cuda.amp.autocast(enabled=amp):
                        _, loss = self.infer(net, batch, loss_criterion)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(batch_size)
                    global_step += 1
                    epoch_loss += loss.item()

                    if activate_wandb:
                        experiment.log({
                            'step': global_step * batch_size,
                            'epoch': epoch,
                            'train loss': loss.item(),
                            'learning rate': optimizer.param_groups[0]['lr'],
                        })

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Visualization round
                    if global_step % division_step == 0 and activate_wandb:
                        experiment_log = self.__evaluate_for_visualization(val_loader, net, loss_criterion)
                        experiment_log['step'] = global_step * batch_size,
                        experiment_log['epoch'] = epoch
                        experiment.log(experiment_log)

                # validation round to adapt learning rate
                epoch_val_loss = self.evaluate(net, val_loader, loss_criterion)
                lr_scheduler.step(epoch_val_loss)
                logging.info('Validation Loss: {}'.format(epoch_val_loss))

                if activate_wandb:
                    experiment.log({
                                'step': global_step * batch_size,
                                'epoch': epoch,
                                'validation loss': epoch_val_loss,
                    })

            if save_checkpoint:
                dir_checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(net.module.state_dict(), str(dir_checkpoint / f'e{epoch+1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')

        if activate_wandb:
            wandb.finish()
