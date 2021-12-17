import pyrealsense2 as rs
import numpy as np
import cv2


class RealSense:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()

        # setup depth and color stream
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth,
                                  width=1280,
                                  height=720,
                                  format=rs.format.z16,
                                  framerate=30)
        self.config.enable_stream(rs.stream.color,
                                  width=1920,
                                  height=1080,
                                  format=rs.format.bgr8,
                                  framerate=30)
        # align depth frame to color frame
        self.align = rs.align(rs.stream.color)

    def connect(self):
        # start streaming
        self.pipeline.start(self.config)

        # print camera specifications including intrinsic parameters 
        self._print_camera_specs()

    def _print_camera_specs(self):
        pipeline_profile = self.pipeline.get_active_profile()
        print("=============================================================")
        print("RealSense Configuration")
        product_line = pipeline_profile.get_device().get_info(
            info=rs.camera_info.product_line)
        product_id = pipeline_profile.get_device().get_info(
            info=rs.camera_info.product_id)
        print(f"camera product line: {product_line}")
        print(f"camera product id:   {product_id}")
        print("=============================================================")
        depth_stream_profile = pipeline_profile.get_stream(
            rs.stream.depth).as_video_stream_profile()
        color_stream_profile = pipeline_profile.get_stream(
            rs.stream.color).as_video_stream_profile()
        depth_i = depth_stream_profile.get_intrinsics()
        color_i = color_stream_profile.get_intrinsics()
        print("Depth Camera Intrinsics")
        print(f"Principal Point (ppx ppy):    {depth_i.ppx} {depth_i.ppy}")
        print(f"Focal Length (fx fy):         {depth_i.fx} {depth_i.fy}")
        print(f"Distortion Coeffs:            {depth_i.coeffs}")
        print(f"Distortion Model:             {depth_i.model}")
        print(
            f"HeightxWidth:                 {depth_i.height}x{depth_i.width}")
        print("Color Camera Intrinsics")
        print(f"Principal Point (ppx ppy):    {color_i.ppx} {color_i.ppy}")
        print(f"Focal Length (fx fy):         {color_i.fx} {color_i.fy}")
        print(f"Distortion Coeffs:            {color_i.coeffs}")
        print(f"Distortion Model:             {color_i.model}")
        print(
            f"HeightxWidth:                 {color_i.height}x{color_i.width}")
        print("=============================================================")

    def get_camera_matrix_and_distortion(self):
        """
        returns the camera matrix (intrinsic matrix) and distortion coefficients 
        of the realsenses rgb lense.

        Returns
        -------
            camera_matrix: np.array (shape: (3,3))
                rgb cameras intrinsic camera matrix
            distortion_coefficients: np.array
                rgb cameras distortion coefficients
        """

        pipeline_profile = self.pipeline.get_active_profile()
        color_stream_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_i = color_stream_profile.get_intrinsics()
        camera_matrix = np.array([[color_i.fx, 0, color_i.ppx],
                                  [0, color_i.fy, color_i.ppy], [0, 0, 1]])
        distortion_coefficients = np.array(color_i.coeffs)
        return camera_matrix, distortion_coefficients

    def collect_frame(self):
        """
        return values are None when at least one frame collected from camera is None

        Returns
        -------
        color_frame: np.array 
        depth_frame: np.array
        """
        # get frames from camera
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # convert frames to np arrays
        depth_image = np.asanyarray(frames.get_depth_frame().get_data())
        color_image = np.asanyarray(frames.get_color_frame().get_data())

        return color_image, depth_image
