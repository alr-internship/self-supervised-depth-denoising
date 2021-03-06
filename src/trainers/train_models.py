from argparse import ArgumentParser
import logging
from pathlib import Path
import time
import torch
import yaml
from dataset.data_loading import BasicDataset
from trainers.basic_trainer import BasicTrainer

from trainers.oof_trainer import OutOfFoldTrainer

ROOT_DIR = Path(__file__).parent.parent.parent

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    network_config_yml = config['network_config']
    dataset_config = config['dataset_config']
    oof_trainer = config['oof_trainer']
    basic_trainer = config['basic_trainer']

    trainer_id = str(time.time())

    evaluation_dir = ROOT_DIR / network_config_yml['evaluation_dir'] / trainer_id
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # write config to evaluation directory
    with open(evaluation_dir / "config.yml", 'w') as f:
        yaml.safe_dump(config, f)

    trainer_config = BasicTrainer.Config.from_config(network_config_yml)

    params = dict(
        evaluation_dir=evaluation_dir,
        config=trainer_config
    )

    trainer_params = dict(
        device=device,
        dataset_config=BasicDataset.Config.from_config(dataset_config),
        trainer_id=trainer_id,
        network_config=network_config_yml
    )

    if basic_trainer['active']:
        basic = BasicTrainer(
            train_path=ROOT_DIR / basic_trainer['train_path'],
            val_path=ROOT_DIR / basic_trainer['val_path'],
            **trainer_params
        )

        # Training M_11
        basic.train(
            net=basic.M_total,
            train_set=basic.train_dataset,
            val_set=basic.val_dataset,
            **params
        )

    # depricated
    if oof_trainer['active']:
        oof = OutOfFoldTrainer(
            dataset_path=ROOT_DIR / oof_trainer['dataset_path'],
            oof_p=oof_trainer['oof_p'],
            **trainer_params
        )

        # Training M_11
        oof.train(
            net=oof.M_11,
            train_set=oof.P_1,
            val_set=oof.P_2_val,
            **params
        )
        # Training M_12
        oof.train(
            net=oof.M_12,
            train_set=oof.P_2,
            val_set=oof.P_1_val,
            **params
        )
        # Training M_1
        oof.train(
            net=oof.M_1,
            train_set=oof.P_1_and_P_2,
            val_set=oof.P_test_val
            **params
        )
    


if __name__ == '__main__':
    file_dir = Path(__file__).parent

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = ArgumentParser()
    parser.add_argument('config_path', type=str)
    main(parser.parse_args())
