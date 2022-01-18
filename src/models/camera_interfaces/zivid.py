import zivid
import datetime
import cv2
import numpy as np
from pathlib import Path

"""
provides an interface to the zivid camera and has
some handy methods to configure the camera.
The interface can printout hard coded intrinsic camera parameters
and also make rgb and depth images converted to opencv.
"""
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
        self.camera = self.app.connect_camera()

        self._print_camera_specs()

    def configure_manual(self, path: Path):
        """configures the camera by taking multiple frames
        """
        print("manually configuring zivid camera settings")
        self.settings = zivid.Settings.load(path)

    def configure_automatically(self, capture_time: int = 1200):
        """configures the camera by taking multiple frames
        """
        print("automatically configuring zivid camera settings")
        suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=capture_time),
            ambient_light_frequency=zivid.capture_assistant.
            SuggestSettingsParameters.AmbientLightFrequency.none,
        )

        self.settings = zivid.capture_assistant.suggest_settings(
            self.camera, suggest_settings_parameters)

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
            [[2760.12, 0, 951.68],
             [0, 2759.78, 594.779],
             [0, 0, 1]])
        # k1, k2, p1, p2, k3
        distortion_coefficients = np.array([
            -0.27315, 0.354379, -0.000344441, 0.000198413, -0.322515
        ])
        return camera_matrix, distortion_coefficients

    def _print_camera_specs(self):
        """prints camera matrix and distortion coefficients to stdout
        """
        camera_matrix, distortion_coefficients = self.get_camera_matrix_and_distortion()
        print("=============================================================")
        print("Zivid Configuration")
        print("Color Camera Intrinsics")
        print(f"Principal Point (ppx ppy):    {camera_matrix[0, 2]} {camera_matrix[1, 2]}")
        print(f"Focal Length (fx fy):         {camera_matrix[0, 0]} {camera_matrix[1, 1]}")
        print(f"Distortion Coeffs:            {distortion_coefficients}")
        print("=============================================================")

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