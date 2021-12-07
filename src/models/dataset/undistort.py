from typing import Sequence, Tuple
import numpy as np


def undistort_images(
    z_images: Sequence[np.array], 
    rgb_images: Sequence[np.array],
    transformation_matrix: np.array) -> Tuple[Sequence[np.array], Sequence[np.array]]:
    """
    Arguments:
        z_images: [np.array]
            array of depth images
        rgb_images: [np.array]
            array of rgb images

    Returns:
        undistorted_z_images: [np.array]
            array of undistorted depth images
        undistorted_rgb_images: [np.array]
            array of undistorted rgb images
    """

    assert transformation_matrix.shape == (4, 4)
    assert len(z_images) == len(rgb_images) != 0

    # undistorted np array shape
    distorted_shape = z_images.shape
    N = distorted_shape[0] # amount of images

    # convert z images to xyz images
    xy_indices = np.indices(distorted_shape[1:]).transpose((1, 2, 0))
    xy_indices = np.repeat(xy_indices[None, ...], N, axis=0)
    ones = np.ones(distorted_shape)
    distorted_xyz_images = np.concatenate((xy_indices, z_images[..., None], ones[..., None]), axis=3)

    # undistort images
    undistorted_xyz_images = distorted_xyz_images @ transformation_matrix
    undistorted_xyz_images = undistorted_xyz_images[..., :3] # remove trailing ones
    
    # convert x`y`z images to z images (update x and y coordinate)
    xy_indices = undistorted_xyz_images[..., :2].astype(np.int32)
    # max, min of both dims (x,y)
    min, max = np.min(xy_indices, axis=(0, 1, 2)), np.max(xy_indices, axis=(0, 1, 2))
    xy_indices -= min

    # new array with shape (n, x`, y`)
    image_size = max - min + [1, 1] # new image size
    undistorted_z_images = np.zeros((N, *image_size))
    undistorted_rgb_images = np.zeros((N, *image_size, 3), dtype=np.int32)

    # fill undistorted z and rgb images with rgb data of old x,y position and new z data from undistorted array
    for n in range(undistorted_xyz_images.shape[0]):
        for x_old in range(undistorted_xyz_images.shape[1]): 
            for y_old in range(undistorted_xyz_images.shape[2]):
                idx = (n, *xy_indices[n, x_old, y_old])
                undistorted_z_images[idx] = undistorted_xyz_images[n, x_old, y_old, 2] # copy new z value
                undistorted_rgb_images[idx] = rgb_images[n, x_old, y_old] # copy old rgb value

    return undistorted_z_images, undistorted_rgb_images