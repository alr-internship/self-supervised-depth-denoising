from typing import Dict, List
from natsort import natsorted
import yaml
import csv
import logging
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from dataset.data_loading import BasicDataset
from dataset.dataset_interface import ROOT_DIR, DatasetInterface
from networks.UNet.unet_model import UNet
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader

from trainers.trainer import get_loss_criterion

ROOT_DIR = Path(__file__).parent.parent.parent

def get_l1loss_in_region(lower_bound: float, upper_bound: float):
    def l1_loss_in_region(a, i, t, r) -> float:
        diff = np.abs(i - t)
        bound_mask = np.logical_and(lower_bound <= diff, diff < upper_bound)
        mask = np.logical_and(bound_mask, r)
        loss = np.sum(np.abs(a - t) * mask) / np.sum(mask)
        return loss
    return l1_loss_in_region


def main(args):
    # load models
    models_dir = ROOT_DIR / args.models_dir
    model_dirs = [model for model in models_dir.glob("*") if model.is_dir()]
    model_dirs = natsorted(model_dirs, key=lambda l: l.name)
    assert len(model_dirs) > 0, "models dir has no models of type .pth"

    print(f"found model directories: {len(model_dirs)}")

    losses = {
        'total_L1': lambda a, _, t, r: get_loss_criterion('mean_l1_loss')(a, t, r),
        '0_to10mm_L1': get_l1loss_in_region(lower_bound=0, upper_bound=10),
        '10_to20mm_L1': get_l1loss_in_region(lower_bound=10, upper_bound=20),
        'above20mm_L1': get_l1loss_in_region(lower_bound=20, upper_bound=np.inf)
    }

    def compute_metrics(i, o, t, r):
        metrics = {}
        for key, loss in losses.items():
            metrics[f"{key}_it"] = loss(i, i, t, r).cpu().detach().numpy().item()
            metrics[f"{key}_ot"] = loss(o, i, t, r).cpu().detach().numpy().item()
        return metrics

    def compute_metric_moments(metrics_list: List[dict]) -> Dict:
        list_metrics = {k: [metric[k] for metric in metrics_list] for k in metrics_list[0]}
        metrics_moments = {}
        for key, values in list_metrics.items():
            metrics_moments[f"{key}_mean"] = np.mean(values)
            metrics_moments[f"{key}_std"] = np.std(values)
        return metrics_moments

    metrics = []
    for model_dir in tqdm(model_dirs, desc='evaluated model dirs'):
        model_name = model_dir.relative_to(models_dir).as_posix()

        config_path = model_dir / "config.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        network_config = config['network_config']
        trainer_config = config['basic_trainer']
        dataset_config_yaml = config['dataset_config']
        dataset_config = BasicDataset.Config.from_config(dataset_config_yaml)
        network_config['n_input_channels'] = dataset_config.num_in_channels()
        network_config['n_output_channels'] = 1
        unet_config = UNet.Config.from_config(network_config)

        test_dataset_path = ROOT_DIR / Path(trainer_config['train_path']).parent / "test_dataset.json"
        if not test_dataset_path.exists():
            print(f"can't evaluation model {model_name}. Dataset does not exist")
            continue
        ds = BasicDataset(test_dataset_path, dataset_config)
        dl = DataLoader(ds, shuffle=False, batch_size=args.batch_size,
                        num_workers=4, pin_memory=True)
        dl_size = len(dl)
        assert dl_size > 0, "test dataset is empty"

        net = UNet(unet_config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.eval()

        models = natsorted(model_dir.rglob("*.pth"), key=lambda l: l.name)
        if len(models) == 0:
            continue

        if not args.all_epochs:
            models = [models[-1]]  # pick model with highest epoch

        for model in tqdm(models, desc='evaluated models'):
            epoch = model.stem.split('e')[1]

            logging.info(f'Loading model {model}, epoch {epoch}')
            net.load_state_dict(torch.load(model, map_location=device))

            model_metrics = []
            for batch in dl:
                images = batch['image']
                labels = batch['label']
                nan_masks = batch['nan-mask']
                region_masks = batch['region-mask']

                # move images and labels to correct device and type
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                nan_masks = nan_masks.to(device=device)
                region_masks = region_masks.to(device=device)

                with torch.no_grad():
                    predictions = net(images)

                    model_metrics.append(compute_metrics(
                        images[:, 3], predictions, labels, nan_masks * region_masks))

            metrics.append({
                "model": model_name,
                "epoch": epoch,
                "metrics": model_metrics
            })

    with open(f'{models_dir}/eval.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("models_dir", type=Path)
    argparse.add_argument("--all-epochs", action="store_true")
    argparse.add_argument("--batch-size", type=int, default=1)
    main(argparse.parse_args())
