import numpy as np
from pathlib import Path
import time


class DatasetInterface:

    def __init__(self, dir_path: Path):
        if not dir_path.exists():
            dir_path.mkdir()

        assert dir_path.is_dir()
        self.dir_path = dir_path

        self.data_file_paths = list(dir_path.glob("**/*.npz"))

    def __getitem__(self, arg):
        with np.load(self.data_file_paths[arg]) as data:
            rs_rgb = data['rs_rgb']
            rs_depth = data['rs_depth']
            zv_rgb = data['zv_rgb']
            zv_depth = data['zv_depth']

        return rs_rgb, rs_depth, zv_rgb, zv_depth

    def __len__(self):
        return len(self.data_file_paths)

    def append_and_save(self, rs_rgb, rs_depth, zv_rgb, zv_depth):
        file_name = Path(str(time.time()))

        np.savez_compressed(
            self.dir_path / file_name,
            rs_rgb=rs_rgb,
            rs_depth=rs_depth,
            zv_rgb=zv_rgb,
            zv_depth=zv_depth
        )

        self.data_file_paths.append(file_name)

    # def convert_from_old(self, dataset: DatasetContainer):
    #     for files in dataset:
    #         self.append_and_save(*files)