import copy
import json
from pathlib import Path
import random
from utils.general_utils import split
import yaml


def main():
    model_dir = Path("local_resources/hp_models")
    adaptations = []
    dry_run = False

    if dry_run:
        print("$$$$$$ DRY RUN ACTIVE $$$$$$$")

    for dir in model_dir.iterdir():
        if not dir.is_dir():
            continue

        with open(dir / "config.yml") as f:
            config = yaml.safe_load(f)

        nc = config["network_config"]
        dc = config["dataset_config"]

        adaptation = {
            "network_config": {
                "learning_rate": nc['learning_rate'],
                "loss_type": nc['loss_type'],
                'bilinear': nc['bilinear'],
                'lr_patience': nc['lr_patience'],
                'output_activation': nc['output_activation'],
                'initial_channels': nc['initial_channels'],
                'skip_connections': nc['skip_connections'],
                'batch_size': nc['batch_size']
            }, 
            "dataset_config": {
                'depth_difference_threshold': dc['depth_difference_threshold'],
                'scale_images': dc['scale_images'],
                'add_region_mask_to_input': dc['add_region_mask_to_input']
            }
        }

        if adaptation in adaptations:
            print("adapation already included", dir.stem)
            continue

        adaptations.append(adaptation)

    print("len adapatations:", len(adaptations))
    processed_adaptations = copy.deepcopy(adaptations)

    jsons = [
        "bwunicluster/hp_train/bwunicluster/hp_train/adaptations_0.json",
        "bwunicluster/hp_train/bwunicluster/hp_train/adaptations_1.json",
        "bwunicluster/hp_train/bwunicluster/hp_train/adaptations_2.json",
        "bwunicluster/hp_train/bwunicluster/hp_train/adaptations_3.json",
    ]

    new_adaptations = []
    for json_path in jsons:
        with open(json_path, 'r') as f:
            json_adaptations = json.load(f)

        for json_adaptation in json_adaptations:
            if json_adaptation in processed_adaptations or json_adaptation in new_adaptations:
                print("adaptation already processed or in json")
                continue
            new_adaptations.append(json_adaptation)

    print("new_adapataions", len(new_adaptations), "processed adaptations", len(processed_adaptations))
    random.shuffle(new_adaptations)
    if len(new_adaptations) < 100 - len(processed_adaptations):
        print("appending backup adaptations since new adaptations are to few")
        with open("adaptions.json", 'r') as f:
            backup_adaptations = json.load(f)

        for backup_adaptation in backup_adaptations:
            if backup_adaptation not in new_adaptations and backup_adaptation not in processed_adaptations:
                new_adaptations.append(backup_adaptation)
    elif len(new_adaptations) + len(processed_adaptations) == 100:
        print("Nothing to do. JSONs are correct")
        return
            
    new_adaptations = new_adaptations[:100-len(processed_adaptations)]

    print("final new adapatations", len(new_adaptations))
    new_adaptations_chunked = split(new_adaptations, 4)

    for file, new_adaptations_chunk in zip(jsons, new_adaptations_chunked):
        print("writing adaptations files")
        if not dry_run:
            with open(Path(file), 'w') as f:
                json.dump(new_adaptations_chunk, f)

    print("writing initial adaptations file")
    if not dry_run:
        with open(Path("bwunicluster/hp_train/bwunicluster/hp_train/initial_adaptations.json"), 'w') as f:
            adaptations = []
            adaptations.extend(new_adaptations)
            adaptations.extend(processed_adaptations)
            json.dump(adaptations, f)

if __name__ == "__main__":
    main()