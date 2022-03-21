from typing import List
from matplotlib.colors import LightSource
import numpy as np
import cv2
import open3d as o3d
from torch import fill_

from utils.transformation_utils import combine_point_clouds


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def visualize_mask(mask):
    return cv2.applyColorMap(mask.astype(np.uint8) * 255, cv2.COLORMAP_BONE)


def visualize_depths(*depths):
    res_depths = []
    ls = LightSource(azdeg=315, altdeg=45)
    for depth in depths:
        # shaded = ls.hillshade(depth, vert_exag=5)
        grays = np.full((depth.shape[0], depth.shape[1], 3),
                        fill_value=[0.6, 0.6, 0.6])
        shaded = ls.shade_rgb(grays, depth, blend_mode='overlay')
        res_depths.append(shaded)

    #max = -np.inf
    #min = np.inf
    # for depth in depths:
    #    max = np.max([np.nanmax(depth), max])
    #    min = np.min([np.nanmin(depth), min])

    #res_depths = []
    # for depth in depths:
    #    depth = ((depth - min) / (max - min)) * 255
    #    depth = cv2.Sobel(depth, -1, 1, 1)

    #    # isnan = np.isnan(depth)
    #    # depth = np.nan_to_num(depth)
    #    # depth = np.where(depth > 255, 255, depth)
    #    # depth = np.where(depth < 0, 0, depth)
    #    # depth = depth.astype(np.uint8)
    #    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    #    # depth = np.where(isnan[..., None], [255, 255, 255], depth)
    #    res_depths.append(depth)
    return res_depths


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
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    depth = np.where(isnan[..., None], [255, 255, 255], depth).astype(np.uint8)
    return depth


def visualize_geometries(pcds: List[o3d.geometry.PointCloud]):
    o3d_visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    o3d_visualizer.create_window()

    geometries = combine_point_clouds(pcds)
    o3d_visualizer.add_geometry(geometries)

    o3d_visualizer.get_render_option().background_color = [0, 0, 0]
    o3d_visualizer.run()
    o3d_visualizer.destroy_window()
