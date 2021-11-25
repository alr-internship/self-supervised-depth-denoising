import zivid
import datetime
import cv2
import math
import numpy as np
from zivid import point_cloud


class Zivid:

    def __init__(self, capture_time: int = 1200) -> None:
        """Initializes the camera and takes photos to setup camera settings automatically
        Args:
            capture_time: the capture time of the camera in milliseconds
        """
        app = zivid.Application()

        """cameras = app.cameras()
        for cam in cameras:
            print(f"Connecting camera: {cam.serialNumber()}")"""

        print("connecting to camera")
        self.camera = app.connect_camera()

        self.configure(capture_time=capture_time)

    def configure(self, capture_time: int = 1200):
        """configures the camera by taking multiple frames
        """
        print("automatically configuring settings")
        suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=capture_time),
            ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
        )

        self.settings = zivid.capture_assistant.suggest_settings(
            self.camera, suggest_settings_parameters
        )
        print("camera configured")

    def collect_frame(self):
        print("Capturing frame")
        with self.camera.capture(self.settings) as frame:
            point_cloud = frame.point_cloud()

            rgb_image = Zivid._convert_2_bgr_image(point_cloud=point_cloud)
            depth_image = Zivid._convert_2_depth_image(point_cloud=point_cloud)

            return rgb_image, depth_image

    def _convert_zdf_to_png(zdf_path: str, color_path: str, depth_path: str):
      """
      @param zdf_path str: file path the .zdf file is located
      @param color_path str: path the rgb image should be saved with cv2.imwrite()
      @param depth_path str: path the depth image should be saved with cv2.imwrite()
      @return None
      """
      frame = zivid.Frame(zdf_path)
      point_cloud = frame.point_cloud()
      rgb_image = Zivid._convert_2_bgr_image(point_cloud=point_cloud)
      depth_image = Zivid._convert_2_depth_image(point_cloud=point_cloud)
      cv2.imwrite(color_path, rgb_image)
      cv2.imwrite(depth_path, depth_image)

    def _convert_2_bgr_image(point_cloud: zivid.PointCloud):
        """Convert from point cloud to 2D image.
        Args:
            point_cloud: a handle to point cloud in the GPU memory
        Returns rgb opencv image 
        """
        rgba = point_cloud.copy_data("rgba")
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return bgr

    def _convert_2_depth_image(point_cloud: zivid.PointCloud):
        return point_cloud.copy_data("z")
