import numpy as np
from pathlib import Path
import time


class DatasetInterface:

    def __init__(self, dir_path: Path, recursive: bool = True):
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        assert dir_path.is_dir()
        self.dir_path = dir_path

        if recursive:
            self.data_file_paths = list(dir_path.glob("**/*.npz"))
        else:
            self.data_file_paths = list(dir_path.glob("*.npz"))

        self.data_file_paths = sorted(self.data_file_paths)

    def __getitem__(self, arg):
        items_to_get = self.data_file_paths[arg]
        if isinstance(items_to_get, list):
            process_list = True
        else:
            items_to_get = [items_to_get]
            process_list = False

        files = []
        for item in items_to_get:
            with np.load(item) as data:
                rs_rgb = data['rs_rgb']
                rs_depth = data['rs_depth']
                zv_rgb = data['zv_rgb']
                zv_depth = data['zv_depth']
                files.append(((rs_rgb, rs_depth, zv_rgb, zv_depth)))

        if not process_list:
            files = files[0]

        return files

    def __len__(self):
        return len(self.data_file_paths)

    def append_and_save(self, rs_rgb, rs_depth, zv_rgb, zv_depth, file_name: Path = None):
        if file_name == None:
            file_name = str(time.time())

        save_path = self.dir_path / file_name
    
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        np.savez_compressed(
            save_path,
            rs_rgb=rs_rgb,
            rs_depth=rs_depth,
            zv_rgb=zv_rgb,
            zv_depth=zv_depth
        )

        self.data_file_paths.append(file_name)

    # def convert_from_old(self, dataset: DatasetContainer):
    #     for files in dataset:
    #         self.append_and_save(*files)
