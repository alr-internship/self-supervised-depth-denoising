import csv
import logging
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from dataset.data_loading import BasicDataset
from networks.UNet.unet_model import UNet
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader


def main(args):
    ds = BasicDataset(args.dataset_path, scale=0.5,
                      enable_augmentation=False, add_nan_mask_to_input=True)
    dl = DataLoader(ds, shuffle=False, batch_size=15,
                    num_workers=4, pin_memory=True)
    dl_size = len(dl)
    assert dl_size > 0, "test dataset is empty"

    # model_files = list(args.models_dir.glob("**/*.pth"))
    models_dir = sorted([model for model in args.models_path.rglob("*") if model.is_dir()])
    assert len(models_dir) > 0, "models dir has no models of type .pth"

    net = UNet(n_input_channels=5, n_output_channels=1)
    net = nn.DataParallel(net)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.eval()

    metrics = []
    for model_dir in tqdm(models_dir, desc='evaluated model dirs'):
        model_name = model_dir.relative_to(args.models_dir).as_posix()

        models = sorted(model_dir.glob("*.pth"))
        if not args.all_epochs:
            models = models[-1] # pick model with highest epoch

        for model in tqdm(models, desc='evaluated models'):
            epoch = model.stem.split('e')[1]

            logging.info(f'Loading model {model}, epoch {epoch}')
            net.load_state_dict(torch.load(model, map_location=device))

            mse = []
            for batch in dl:
                images = batch['image']
                labels = batch['label']
                nan_mask = batch['nan-mask']

                # move images and labels to correct device and type
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                nan_mask = nan_mask.to(device=device)

                with torch.no_grad():
                    predictions = net(images)
                    mse.append(torch.mean(((predictions - labels) ** 2) * nan_mask))

            metrics.append({
                "model": model_name,
                "epoch": epoch,
                "mean_mse": np.mean(mse),
                "std_mse": np.std(mse)
            })

    with open(f'{args.models_dir}/eval.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("models_path", type=Path)
    argparse.add_argument("dataset_path", type=Path)
    argparse.add_argument("--all-epochs", type=bool, default=True)
    main(argparse.parse_args())
