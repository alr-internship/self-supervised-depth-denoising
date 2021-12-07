from os import pipe
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSense:

    def __init__(self) -> None:
        self.pipeline = rs.pipeline()

        # setup depth and color stream
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width=1280, height=720, format=rs.format.z16, framerate=30)
        self.config.enable_stream(rs.stream.color, width=1920, height=1080, format=rs.format.bgr8, framerate=30)
        self.align = rs.align(rs.stream.color)

        # print intrinsics
        self._print_camera_specs()

    def connect(self):
        # start streaming
        self.pipeline.start(self.config)


    def _print_camera_specs(self):
        print("=============================================================")
        print("RealSense Configuration")
        pipeline_profile = self.pipeline.get_active_profile()
        product_line = pipeline_profile.get_device().get_info(info=rs.camera_info.product_line)
        product_id = pipeline_profile.get_device().get_info(info=rs.camera_info.product_id)
        print(f"camera product line: {product_line}")
        print(f"camera product id:   {product_id}")
        print("=============================================================")
        depth_stream_profile = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_i = depth_stream_profile.get_intrinsics()
        color_i = color_stream_profile.get_intrinsics()
        print("Depth Camera Intrinsics")
        print(f"Principal Point (ppx ppy):    {depth_i.ppx} {depth_i.ppy}")
        print(f"Focal Length (fx fy):         {depth_i.fx} {depth_i.fy}")
        print(f"Distortion Coeffs:            {depth_i.coeffs}")
        print(f"Distortion Model:             {depth_i.model}")
        print(f"HeightxWidth:                 {depth_i.height}x{depth_i.width}")
        print("Color Camera Intrinsics")
        print(f"Principal Point (ppx ppy):    {color_i.ppx} {color_i.ppy}")
        print(f"Focal Length (fx fy):         {color_i.fx} {color_i.fy}")
        print(f"Distortion Coeffs:            {color_i.coeffs}")
        print(f"Distortion Model:             {color_i.model}")
        print(f"HeightxWidth:                 {color_i.height}x{color_i.width}")
        print("=============================================================")

    id = 0

    def collect_frame(self):
        """
        return values are None when at least one frame collected from camera is None

        :returns depth frame, color frame as numpy arrays
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

        cv2.imwrite(str(self.id) + "_realsense_depth.png", depth_image)
        cv2.imwrite(str(self.id) + "_realsense_color.png", color_image)
        self.id = self.id + 1

        return depth_image, color_image

    def __del__(self):
        self.pipeline.stop()
