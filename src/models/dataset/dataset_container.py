from typing import List, Tuple
import numpy as np
from pathlib import Path


class CameraDataset:
    rgb: List[np.array]
    depth: List[np.array]
    camera_matrix: np.array
    distortion_coefficients: np.array

    def __init__(
        self,
        camera_matrix: np.array,
        distortion_coefficients: np.array,
        rgb: List[np.array] = None,
        depth: List[np.array] = None,
    ) -> None:
        """
        camera_matrix: np.array (shape: (3,3))
            intrinsic camera matrix of lense
        distortion_coefficients: np.array
            distortion coefficients of lense
        rgb: List[np.array] (optional, default: [])
            list of rgb frames
        depth: List[np.array] (optional, default: [])
            list of depth frames
        """
        assert camera_matrix.shape == (3, 3) \
            and distortion_coefficients.shape[0] >= 4
        assert (rgb is None and depth is None) or len(rgb) == len(depth)

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.rgb = [] if rgb is None else rgb
        self.depth = [] if depth is None else depth


class DatasetContainer:

    realsense: CameraDataset = None
    zivid: CameraDataset = None

    def load_from_intrinsics(self, rs_cm: np.array, rs_dc: np.array,
                             zv_cm: np.array, zv_dc: np.array) -> None:
        """
        Parameters
        ----------
        rs_cm: np.array (shape: (3,3))
            intrinsic camera matrix of realsense rgb lense
        rs_dc: np.array 
            distortion coefficients of realsense rgb lense
        zv_cm: np.array (shape: (3,3))
            intrinsic camera matrix of zivid rgb lense
        zv_dc: np.array
            distortion coefficients of zivid rgb lense 
        """
        self.realsense = CameraDataset(camera_matrix=rs_cm,
                                       distortion_coefficients=rs_dc)
        self.zivid = CameraDataset(camera_matrix=zv_cm,
                                   distortion_coefficients=zv_dc)

    def load_from_dataset(self, dataset_path: Path) -> None:
        """
        Parameters
        ----------
        dataset_path: str
            file path the compressed .npz file should be loaded from 
        """
        with np.load(dataset_path) as dataset:
            self.realsense = CameraDataset(
                camera_matrix=dataset["realsense_cm"],
                distortion_coefficients=dataset["realsense_dc"],
                rgb=dataset["realsense_color"],
                depth=dataset["realsense_depth"],
            )
            self.zivid = CameraDataset(
                camera_matrix=dataset['zivid_cm'],
                distortion_coefficients=dataset['zivid_dc'],
                rgb=dataset["zivid_color"],
                depth=dataset["zivid_depth"],
            )

    def __getitem__(self, arg):
        return self.realsense.rgb[arg], self.realsense.depth[arg],\
             self.zivid.rgb[arg], self.zivid.depth[arg]

    def get_camera_intrinscs(self) -> Tuple[np.array]:
        """
        Returns:
            rs_camera_matrix: np.array
            rs_distortion_coefficient: np.array
            zv_camera_matrix: np.array
            zv_distortion_coefficient: np.array
        """
        return self.realsense.camera_matrix, self.realsense.distortion_coefficients,\
            self.zivid.camera_matrix, self.zivid.distortion_coefficients

    def size(self):
        """returns amount of frame tuples collected"""
        return len(self.realsense.rgb)

    def append(self, rs_rgb, rs_depth, zv_rgb, zv_depth) -> None:
        """
        Parameters
        -----------
        rs_rgb: np.array
            color image of realsense camera
        rs_depth: np.array
            depth image of realsense camera
        zv_rgb: np.array 
            color image of zivid camera
        zv_depth: np.array
            depth image of zivid camera
        """
        self.realsense.rgb.append(rs_rgb)
        self.realsense.depth.append(rs_depth)
        self.zivid.rgb.append(zv_rgb)
        self.zivid.depth.append(zv_depth)

    def save(self, dataset_path: Path):
        """
        Parameters
        ----------
        dataset_path: str
            file path the compressed .npz file should be saved
        """
        np.savez_compressed(
            dataset_path,
            realsense_color=self.realsense.rgb,
            realsense_depth=self.realsense.depth,
            realsense_cm=self.realsense.camera_matrix,
            realsense_dc=self.realsense.distortion_coefficients,
            zivid_color=self.zivid.rgb,
            zivid_depth=self.zivid.depth,
            zivid_cm=self.zivid.camera_matrix,
            zivid_dc=self.zivid.distortion_coefficients)
