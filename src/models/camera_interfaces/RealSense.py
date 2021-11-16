import pyrealsense2 as rs
import numpy as np


class RealSense:

    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()

        # setup depth and color stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # start streaming
        self.pipeline.start(config)

    def __del__(self):
        self.pipeline.stop()

    def collect_frame(self):
        """
        return values are None when at least one frame collected from camera is None

        :returns depth frame, color frame as numpy arrays
        """
        # get frames from camera
        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # convert frames to np arrays
        depth_image = np.asanyarray(frames.get_depth_frame().get_data())
        color_image = np.asanyarray(frames.get_color_frame().get_data())
        
        return depth_image, color_image
