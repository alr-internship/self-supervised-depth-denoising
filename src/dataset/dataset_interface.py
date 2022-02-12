import json
import numpy as np
from pathlib import Path
import time


class DatasetInterface:

    def __init__(self, path: Path, recursive: bool = True):
        """creates dataset interface
        
        Arguments
        -----
        path: Path
            can be either a path to a JSON file or a path to a directory.
            If a path to a JSON file is passed,
            the files stored in JSON will be loaded
            If a path to a directory is given,
            the .npz files in the directory will be loaded

        """
        if not path.exists():
            path.mkdir(parents=True)
        
        if path.is_file():
            with open(path.as_posix(), 'r') as f:
                data = json.load(f)
            self.dir_path = Path(data['base_path'])
            self.data_file_paths = [
                self.dir_path / file 
                for file in data['files']
            ]

        else:
            self.dir_path = path
            self.data_file_paths = self.get_paths_in_dir(path, recursive)


    @staticmethod
    def get_paths_in_dir(dir: Path, recursive: bool):
        if recursive:
            data_file_paths = list(dir.glob("**/*.npz"))
        else:
            data_file_paths = list(dir.glob("*.npz"))

        return sorted(data_file_paths)

    @staticmethod
    def load(path: Path):
        with np.load(path) as data:
            rs_rgb = data['rs_rgb']
            rs_depth = data['rs_depth']
            zv_rgb = data['zv_rgb']
            zv_depth = data['zv_depth']
            if 'mask' in data:
                mask = data['mask']
            else:
                mask = None
        return rs_rgb, rs_depth, zv_rgb, zv_depth, mask

    def __getitem__(self, arg):
        items_to_get = self.data_file_paths[arg]
        if isinstance(items_to_get, list):
            process_list = True
        else:
            items_to_get = [items_to_get]
            process_list = False

        files = []
        for item in items_to_get:
            rs_rgb, rs_depth, zv_rgb, zv_depth, mask = self.load(item)
            if mask != None:
                files.append(((rs_rgb, rs_depth, zv_rgb, zv_depth, mask)))
            else:
                files.append(((rs_rgb, rs_depth, zv_rgb, zv_depth)))

        if not process_list:
            files = files[0]

        return files

    def __len__(self):
        return len(self.data_file_paths)

    @staticmethod
    def save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, file_name: Path):
        if not file_name.parent.exists():
            file_name.parent.mkdir(parents=True, exist_ok=True)

        params = dict(
            rs_rgb=rs_rgb,
            rs_depth=rs_depth,
            zv_rgb=zv_rgb,
            zv_depth=zv_depth
        )

        if mask is None:
            np.savez_compressed(file_name, **params)
        else:
            np.savez_compressed(file_name, mask=mask, **params)

    def append_with_mask_and_save(self, rs_rgb, rs_depth, zv_rgb, zv_depth, mask, file_name: Path = None):
        if file_name == None:
            file_name = str(time.time())

        save_path = self.dir_path / file_name
        self.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, save_path)

        self.data_file_paths.append(file_name)

    def append_and_save(self, rs_rgb, rs_depth, zv_rgb, zv_depth, file_name: Path = None):
        self.append_with_mask_and_save(rs_rgb, rs_depth, zv_rgb, zv_depth, None, file_name)