from os import pipe
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSense:

    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()

        # setup depth and color stream
        config.enable_stream(rs.stream.depth, width=1280, height=720, format=rs.format.z16, framerate=30)
        config.enable_stream(rs.stream.color, width=1920, height=1080, format=rs.format.bgr8, framerate=30)
        self.align = rs.align(rs.stream.color)

        # start streaming
        self.pipeline.start(config)

        # print intrinsics
        self._print_camera_specs()

    def _print_camera_specs(self):
        pipeline_profile = self.pipeline.get_active_profile()
        product_line = pipeline_profile.get_device().get_info(info=rs.camera_info.product_line)
        product_id = pipeline_profile.get_device().get_info(info=rs.camera_info.product_id)
        print(f"camera product line: {product_line}")
        print(f"camera product id: {product_id}")

        depth_stream_profile = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_i = depth_stream_profile.get_intrinsics()
        color_i = color_stream_profile.get_intrinsics()
        print("Depth Camera Intrinsics")
        print(f"Principal Point: {depth_i.ppx} {depth_i.ppy}")
        print(f"Focal Length:    {depth_i.fx} {depth_i.fy}")
        print("Color Camera Intrinsics")
        print(f"Principal Point: {color_i.ppx} {color_i.ppy}")
        print(f"Focal Length:    {color_i.fx} {color_i.fy}")

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

        return depth_image, color_image

    def collect_and_save_pointcloud(self, path: str):
        color_frame, depth_frame = self.collect_frame()
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        pointcloud = pc.calculate(depth_frame)
        pointcloud.export_to_ply(path, color_frame)

    def __del__(self):
        self.pipeline.stop()
