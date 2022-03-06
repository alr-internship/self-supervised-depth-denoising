from argparse import ArgumentParser
import copy
import json
import os
from pathlib import Path
from random import sample
import subprocess

import yaml

variations_config = {
    'network_config': {
        "learning_rate": lambda: sample([0.01, 0.05, 0.1], 1)[0],
        "loss_type": lambda: sample(['huber_loss', 'mean_l1_loss', 'mean_l2_loss'], 1)[0],
        "bilinear": lambda: sample([True, False], 1)[0],
        "lr_patience": lambda: sample([1, 3], 1)[0],
        "output_activation": lambda: sample(['relu', 'none'], 1)[0],
        "initial_channels": lambda: sample([8, 16, 32], 1)[0],
        "skip_connections": lambda: sample([False, True], 1)[0],
    },
    'dataset_config': {
        "depth_difference_threshold": lambda: sample([0, 1, 2, 3], 1)[0],
        "scale_images": lambda: sample([0.5, 1], 1)[0],
        'add_region_mask_to_input': lambda: sample([False, True], 1)[0],
    }
}


def generate_all_adaptions(num_adaptions: int):
    adaptions = []
    while len(adaptions) < num_adaptions:
        adaption = {}
        for upper_key, params in variations_config.items():
            adaption[upper_key] = {}
            for key, config in params.items():
                adaption[upper_key][key] = config()

        if adaption['dataset_config']['scale_images'] == 0.5:
            if adaption['network_config']['initial_channels'] == 8:
                adaption['network_config']['batch_size'] = 70
            elif adaption['network_config']['initial_channels'] == 16:
                adaption['network_config']['batch_size'] = 30
            else:
                adaption['network_config']['batch_size'] = 16
        else:  # 1.0
            if adaption['network_config']['initial_channels'] == 8:
                adaption['network_config']['batch_size'] = 35
            elif adaption['network_config']['initial_channels'] == 16:
                adaption['network_config']['batch_size'] = 16
            else:
                adaption['network_config']['batch_size'] = 8

        if adaption not in adaptions:
            adaptions.append(adaption)

    return adaptions


def get_adapted_config(cfg, adaption):
    adapted_config = copy.deepcopy(cfg)
    for key in adapted_config.keys():
        if key in adaption and isinstance(adapted_config[key], dict):
            adapted_config[key].update(adaption[key])
    return adapted_config


ROOT_DIR = Path(__file__).parent.parent.parent
MAIN_SCRIPT = ROOT_DIR / "src/trainers/train_models.py"


def main(args):
    default_config = ROOT_DIR / args.default_config
    num_configurations = args.num_configurations
    adaptions_file = ROOT_DIR / args.adaptions_file
    tmp_config = ROOT_DIR / f"tmp_config_for_{adaptions_file.stem}.yml"

    # get adaptions
    if not adaptions_file.exists():
        adaptions = generate_all_adaptions(num_configurations)
        adaptions_file.parent.mkdir(parents=True, exist_ok=True)
        # file keeping track of processed adaptations
        with open(adaptions_file, 'w') as f:
            json.dump(adaptions, f)
        # file saving all adaptions
        with open(adaptions_file.parent / "backup_adaptations.json", 'w') as f:
            json.dump(adaptions, f)
    else:
        with open(adaptions_file, 'r') as f:
            adaptions = json.load(f)

    with open(default_config, 'r') as f:
        cfg = yaml.safe_load(f)

    adaptions_without_processed = adaptions
    for adaption in adaptions:
        adapted_config = get_adapted_config(cfg, adaption)

        with open(tmp_config, 'w') as f:
            yaml.safe_dump(adapted_config, f)

        subprocess.call(["python", MAIN_SCRIPT, tmp_config.as_posix()], cwd=os.getcwd())

        adaptions_without_processed.remove(adaption)

        with open(adaptions_file, 'w') as f:
            json.dump(adaptions_without_processed, f)

    tmp_config.unlink()


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("default_config", type=Path, help="default config to use")
    argparse.add_argument("--num-configurations", type=int, default=100)
    argparse.add_argument("--adaptions-file", type=Path, default="adaptions.json",
                          help='where the adaptions to iterate through are saved')
    main(argparse.parse_args())
