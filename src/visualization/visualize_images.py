import numpy as np
import cv2


def get_visualized_depth_image(depth_image: np.array):
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return depth_colormap

def convert_bgr_to_grb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)