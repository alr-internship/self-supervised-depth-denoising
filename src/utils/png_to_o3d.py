import open3d as o3d
import os

def get_rgbd_images_from_paths(color_path: str, depth_path: str):
  """
  @param color_path str: path to directory where all color .png images are stored in
  @param depth_path str: path to directory where all depth .png images are stored in
  @return [RGBDImage]: list of Open3D RGBDImage
  """
  color_image_paths = [f for f in os.listdir(color_path) if '.png' in f.lower()]
  depth_image_paths = [f for f in os.listdir(depth_path) if '.png' in f.lower()]

  rgbd_images = []
  for color_image_filename, depth_image_filename in zip(color_image_paths, depth_image_paths):
    color_raw = o3d.io.read_image(color_path + color_image_filename)
    depth_raw = o3d.io.read_image(depth_path + depth_image_filename)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    rgbd_images.append(rgbd_image)

  return rgbd_images
