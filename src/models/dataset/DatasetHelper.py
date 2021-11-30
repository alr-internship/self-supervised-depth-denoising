from functools import reduce
import numpy as np
import open3d as o3d

def saveImageDataset(
    datasetPath: str,
    realsenseColor: list,
    realsenseDepth: list,
    zividColor: list,
    zividDepth: list,
):
    """
    Parameters
    ----------
    realsenseColor : list
        color images of realsense camera
    realsenseDepth : list
        depth images of realsense camera
    zividColor: list
        color images of zivid camera
    zividDepth: list
        depth images of zivid camera
    """

    assert len(realsenseColor) == len(realsenseDepth) == len(zividColor) == len(zividColor)

    np.savez_compressed(
        datasetPath,
        RealsenseColor=realsenseColor,
        RealsenseDepth=realsenseDepth,
        ZividColor=zividColor, 
        ZividDepth=zividDepth,
    )

def openImageDataset(
    datasetPath: str
):
    """
    Returns
    ----------
    realsenseColor, realsenseDepth, zividColor, zividDepth

    """
    with np.load(datasetPath) as dataset:
        realsenseColor = dataset["RealsenseColor"]
        realsenseDepth = dataset["RealsenseDepth"]
        zividColor = dataset["ZividColor"]
        zividDepth = dataset["ZividDepth"]

    return realsenseColor, realsenseDepth, zividColor, zividDepth

def open_image_dataset_o3d(
    dataset_path: str
):
    """
    Returns
    ----------
    rgbd_rs_images, rgbd_zv_images

    """
    realsense_color, realsense_depth, zivid_color, zivid_depth = openImageDataset(dataset_path)

    rgbd_rs_images = []
    rgbd_zv_images = []
    for rs_color_image, rs_depth_image, zv_color_image, zv_depth_image in zip(realsense_color, realsense_depth, zivid_color, zivid_depth):
        rgbd_rs_images.append(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rs_color_image), 
                o3d.geometry.Image(rs_depth_image)
            )
        )
        rgbd_zv_images.append(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(zv_color_image), 
                o3d.geometry.Image(zv_depth_image)
            )
        )
    return rgbd_rs_images, rgbd_zv_images

def save_coefficients(
    tmat=np.array, 
    path=str
): 
    '''Save the camera matrix and the distortion coefficients to given path/file.''' 
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('T', tmat)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(
    path=str
):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    transf_matrix = cv_file.getNode('T').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix, transf_matrix]