from enum import unique
from itertools import count
from typing import List
import open3d as o3d
import numpy as np
import cv2


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# this format to make open3d objects pickable
zv_ci = dict(
    width=1920, height=1080,
    fx=2.76012e+03, fy=2.75978e+03,
    cx=9.51680e+02, cy=5.94779e+02,
)
rs_ci = dict(
    width=1920, height=1080,
    fx=1377.6448974609375, fy=1375.7239990234375,
    cx=936.4846801757812, cy=537.48779296875,
)


def imgs_to_pcd(bgr, depth, ci: dict, project_valid_depth_only: bool = True):
    ci = o3d.camera.PinholeCameraIntrinsic(**ci)

    rgb = to_rgb(bgr)
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth.astype(np.float32))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, ci, project_valid_depth_only=project_valid_depth_only)
    return pcd


def fill_to_shape(array, shape, fill_value, dtype):
    if len(shape) == 2 and len(array.shape) == 3: # append channel dimension if nesseccary
        shape = shape + (array.shape[-1],)

    s = (min(shape[0], array.shape[0]), min(shape[1], array.shape[1]))
    array_large = np.full(shape, fill_value=fill_value, dtype=dtype)
    array_large[:s[0], :s[1]] = array[:s[0], :s[1]]
    return array_large

def pcd_to_imgs_old(pcd, ci: dict, depth_scale: float = 1000.0):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # camera parameters
    f = np.array([ci['fx'], ci['fy']])
    c = np.array([ci['cx'], ci['cy']])

    # convert to 3d to 2d space
    depths = points[:, 2]
    pixels = points[:, :2] * f / np.expand_dims(depths, axis=1) + c
    pixels = np.round(pixels).astype(np.uint16)
    
    # create empty frames for final rgb and depth images
    ul_corner = np.min(pixels, axis=0)
    lr_corner = np.max(pixels, axis=0)
    picture_size = np.round_(lr_corner).astype(np.uint16).T
    rgb_frame = np.zeros((picture_size[1] + 1, picture_size[0] + 1, 3), dtype=np.uint8)
    # depth frame filled with NaNs
    depth_frame = np.empty((picture_size[1] + 1, picture_size[0] + 1), dtype=np.float32)
    depth_frame[:] = np.nan

    # fill respective pixels with depth and rgb info
    pixels = pixels.T
    pixels = (pixels[1], pixels[0])
    rgb_frame[pixels] = colors * 255
    depth_frame[pixels] = depths * depth_scale

    bgr_frame = to_bgr(rgb_frame)

    return bgr_frame, depth_frame, ul_corner, lr_corner


def pcd_to_imgs(pcd, ci: dict, depth_scale: float = 1000.0):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # camera parameters
    f = np.array([ci['fx'], ci['fy']])
    c = np.array([ci['cx'], ci['cy']])

    # convert to 3d to 2d space
    depths = points[:, 2]
    pixels = points[:, :2] * f / np.expand_dims(depths, axis=1) + c
    pixels = np.round(pixels).astype(np.uint16)
    
    # replace all depths of duplicate pixel values with smallest depth value among duplicates
    unique, inv, counts = np.unique(pixels, return_counts=True, return_inverse=True, axis=0)
    # print(f'# total pixels: {len(pixels)}, # unique pixels: {len(counts)}')

    # len(inv) == len(depths) and inv[i] corresponds to index of
    # unique elemenet at depths[i], 
    # e.g. depths[i] == unique[inv[i]]
    # unique_max_depths[inv] reconsturcts depths array with min depth
    # values of each unique element
    unique_max_depths = np.full(len(unique), np.inf)
    for i in range(len(inv)):
        if unique_max_depths[inv[i]] > depths[i]:
            unique_max_depths[inv[i]] = depths[i]
    depths = unique_max_depths[inv]
    
    # print(np.isinf(depths).any())

    # create empty frames for final rgb and depth images
    ul_corner = np.min(pixels, axis=0)
    lr_corner = np.max(pixels, axis=0)
    picture_size = np.round_(lr_corner).astype(np.uint16).T
    rgb_frame = np.zeros((picture_size[1] + 1, picture_size[0] + 1, 3), dtype=np.uint8)
    # depth frame filled with NaNs
    depth_frame = np.empty((picture_size[1] + 1, picture_size[0] + 1), dtype=np.float32)
    depth_frame[:] = np.nan

    # fill respective pixels with depth and rgb info
    pixels = pixels.T
    pixels = (pixels[1], pixels[0])
    rgb_frame[pixels] = colors * 255
    depth_frame[pixels] = depths * depth_scale

    bgr_frame = to_bgr(rgb_frame)

    return bgr_frame, depth_frame, ul_corner, lr_corner

def resize(bgr_frame, depth_frame, ul_corner, lr_corner, cropped: bool, resulting_shape: tuple):
    if not cropped:
        bgr_frame = fill_to_shape(bgr_frame, resulting_shape, 0, dtype=np.uint8)
        depth_frame = fill_to_shape(depth_frame, resulting_shape, np.nan, dtype=np.float32)
    else:
        bgr_frame = bgr_frame[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
        depth_frame = depth_frame[ul_corner[1]:lr_corner[1], ul_corner[0]:lr_corner[0]]
        bgr_frame = cv2.resize(bgr_frame, resulting_shape)
        depth_frame = cv2.resize(depth_frame, resulting_shape)
    return bgr_frame, depth_frame


def image_points_to_camera_points(image_points: np.array, ci: dict, depth_scale: float = 1000.0):
    assert image_points.shape[1] == 3, "image points must have xyz"

    v = image_points[:, 0]
    u = image_points[:, 1]
    d = image_points[:, 2]

    z = d / depth_scale
    x = (u - ci['cx']) * z / ci['fx']
    y = (v - ci['cy']) * z / ci['fy']

    return np.concatenate((x[:, None], y[:, None], z[:, None]), axis=1)

def normalize_depth(depth, min, max):
    depth = (depth - min) / (max - min)
    depth = np.where(depth > 1, 1, depth)
    depth = np.where(depth < 0, 0, depth)
    return depth

def unnormalize_depth(depth, min, max):
    depth = (depth * (max - min)) + min
    depth = np.where(depth > max, max, depth)
    depth = np.where(depth < min, min, depth)
    return depth


def combine_point_clouds(pcds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    all_points = [np.asarray(pcd.points) for pcd in pcds]
    all_colors = [np.asarray(pcd.colors) for pcd in pcds]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
    return cloud

