import zivid
import datetime
import cv2
import numpy as np
from pathlib import Path

from zivid import camera


class Zivid:
    def __init__(self) -> None:
        """Initializes the camera and takes photos to setup camera settings automatically
        Args:
            capture_time: the capture time of the camera in milliseconds
        """
        self.app = zivid.Application()

    def connect(self):
        """connect to camera
        """
        print("connecting to camera")
        self.camera = self.app.connect_camera()

    def configure_manual(self, path: Path):
        """configures the camera by taking multiple frames
        """
        print("manually configuring settings")
        self.settings = zivid.Settings.load(path)
        print("camera configured")

    def configure_automatically(self, capture_time: int = 1200):
        """configures the camera by taking multiple frames
        """
        print("automatically configuring settings")
        suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=capture_time),
            ambient_light_frequency=zivid.capture_assistant.
            SuggestSettingsParameters.AmbientLightFrequency.none,
        )

        self.settings = zivid.capture_assistant.suggest_settings(
            self.camera, suggest_settings_parameters)
        print("camera configured")

    def get_camera_matrix_and_distortion(self):
        """returns hard coded camera intrinsics

        Returns
        -------
        camera_matrix: np.array (shape: (3, 3))
            cameras camera matrix
        distortion_coefficients: np.array (shape: (5,))
            cameras distortion coefficients
        """
        camera_matrix = np.array(
            [[2.74373118e+03, 0.00000000e+00, 9.46713918e+02],
             [0.00000000e+00, 2.73588758e+03, 5.82900437e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        distortion_coefficients = np.array([
            -1.89468196e-01, -1.39553226e+00, -4.97692120e-04, 2.41059098e-03,
            1.30000987e+01
        ])
        return camera_matrix, distortion_coefficients

    def collect_frame(self):
        """captures a point cloud and converts it into bgr and depth image

        Returns
        -------
        rgb_image: np.array
            rgb image captured
        depth_image: np.array
            depth image captured
        """
        with self.camera.capture(self.settings) as frame:
            point_cloud = frame.point_cloud()

            rgb_image = Zivid._convert_2_bgr_image(point_cloud=point_cloud)
            depth_image = Zivid._convert_2_depth_image(point_cloud=point_cloud)

            return rgb_image, depth_image

    def _convert_2_bgr_image(point_cloud: zivid.PointCloud):
        """Convert from point cloud to rgb image.

        Arguments
        --------- 
            point_cloud: zivid.PointCloud
                a handle to point cloud in the GPU memory

        Returns
        --------
        bgr_image: np.array
        """
        rgba_image = point_cloud.copy_data("rgba")
        bgr_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
        return bgr_image

    def _convert_2_depth_image(point_cloud: zivid.PointCloud):
        """Convert from point cloud to depth image

        Arguments
        --------- 
            point_cloud: zivid.PointCloud
                a handle to point cloud in the GPU memory

        Returns
        --------
        depth_image: np.array
        """
        return np.array(point_cloud.copy_data("z"))