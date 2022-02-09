from argparse import ArgumentParser
import json
from pathlib import Path
import random
from typing import List

def save_paths_to_json(paths: List[Path], basepath: Path, file_name: str):
    paths = [path.relative_to(basepath).as_posix() for path in paths]
    dataset = {
        'base_path': basepath.resolve().as_posix(),
        'files': paths
    }

    with open(f"{basepath}/{file_name}.json", 'w') as f:
        json.dump(dataset, f)

def main(args):
    dataset_files = list(args.dataset_path.rglob("*.npz"))
    random.shuffle(dataset_files)

    ds_len = len(dataset_files)
    test_border = int(ds_len // (1 / args.test_percentage))
    val_border = int(ds_len // (1 / (args.test_percentage + args.val_percentage)))

    test_ds_files = dataset_files[:test_border]
    val_ds_files = dataset_files[test_border:val_border]
    train_ds_files = dataset_files[val_border:]

    save_paths_to_json(test_ds_files, args.dataset_path, "test_dataset")
    save_paths_to_json(val_ds_files, args.dataset_path, "val_dataset")
    save_paths_to_json(train_ds_files, args.dataset_path, "train_dataset")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("dataset_path", type=Path)
    argparse.add_argument("--test-percentage", type=float, default=0.05)
    argparse.add_argument("--val-percentage", type=float, default=0.05)
    main(argparse.parse_args())