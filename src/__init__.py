from models import *
from models import camera_interfaces
from visualization.visualize_images import visualize_depth_image

if __name__ == "__main__":
    realSense = camera_interfaces.RealSense()

    while True:
        # collect frames from realSense
        depth_frame, color_frame = realSense.collect_frame()

        # check if depth frame valid
        if not depth_frame:
            print("got invalid depth frame")
            break

        # visualize depth frame
        visualize_depth_image(depth_frame=depth_frame)
