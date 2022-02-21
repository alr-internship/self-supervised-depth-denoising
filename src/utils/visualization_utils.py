import numpy as np
import cv2

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def visualize_mask(mask):
    return cv2.applyColorMap(mask.astype(np.uint8) * 255, cv2.COLORMAP_BONE)

def visualize_depth(depth):
    # ignore nans for max,min
    # mean = np.nanmean(depth)
    # std = np.nanstd(depth)
    # min = mean - 1 * std
    # max = mean + 1 * std
    max = np.nanmax(depth)
    min = np.nanmin(depth)
    # set nan to mean, +inf to max, -inf to min
    # map from [min, max] to [0, 255] linearly
    depth = ((depth - min) / (max - min)) * 255
    isnan = np.isnan(depth)
    depth = np.nan_to_num(depth)
    depth = np.where(depth > 255, 255, depth)
    depth = np.where(depth < 0, 0, depth)
    depth = depth.astype(np.uint8)
    depth = np.where(isnan, 0, depth)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth
