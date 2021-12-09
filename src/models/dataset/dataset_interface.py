from functools import reduce
import numpy as np
import cv2


def save_dataset(dataset_path: str, realsense_color: list,
                 realsense_depth: list, realsense_cm: np.array,
                 realsense_dc: np.array, zivid_color: list, zivid_depth: list,
                 zivid_cm: np.array, zivid_dc: np.array):
    """
    Parameters
    ----------
    dataset_path: str
        file path the compressed .npz file should be saved
    realsense_color : list
        color images of realsense camera
    realsense_depth : list
        depth images of realsense camera
    realsense_cm: np.array
        intrinsic camera matrix of realsense rgb lense
    realsense_dc: np.array
        distortion coefficients of realsense rgb lense
    zivid_color: list
        color images of zivid camera
    zivid_depth: list
        depth images of zivid camera
    zivid_cm: np.array
        intrinsic camera matrix of zivid rgb lense
    zivid_dc: np.array
        distortion coefficients of zivid rgb lense 
    """

    assert len(realsense_color) == len(realsense_depth) == len(
        zivid_color) == len(zivid_color)
    assert realsense_cm.shape == (3, 3) and zivid_cm.shape == (3, 3)
    assert realsense_dc.shape[0] >= 4 and zivid_dc.shape[0] >= 4

    np.savez_compressed(dataset_path,
                        realsense_color=realsense_color,
                        realsense_depth=realsense_depth,
                        realsense_cm=realsense_cm,
                        realsense_dc=realsense_dc,
                        zivid_color=zivid_color,
                        zivid_depth=zivid_depth,
                        zivid_cm=zivid_cm,
                        zivid_dc=zivid_dc)


def open_dataset(datasetPath: str):
    """
    opens dataset and returns all stored arrays

    Returns
    ----------
    realsense_color, realsense_depth, realsense_cm, realsense_dc, 
    zivid_color, zivid_depth, zivid_cm, zivid_dc

    """
    with np.load(datasetPath) as dataset:
        realsense_color = dataset["realsense_color"]
        realsense_depth = dataset["realsense_depth"]
        realsense_cm = dataset["realsense_cm"]
        realsense_dc = dataset["realsense_dc"]
        zivid_color = dataset["zivid_color"]
        zivid_depth = dataset["zivid_depth"]
        zivid_cm = dataset['zivid_cm']
        zivid_dc = dataset['zivid_dc']

    return realsense_color, realsense_depth, realsense_cm, realsense_dc, \
        zivid_color, zivid_depth, zivid_cm, zivid_dc


def open_dataset_images(datasetPath: str):
    """
    opens dataset and only returns all rgb and depth images

    Returns
    ---------
    realsense_color, realsense_depth, zivid_color, zivid_depth
    """
    realsense_color, realsense_depth, _, _, zivid_color, zivid_depth, _, _ = open_dataset(
        datasetPath)
    return realsense_color, realsense_depth, zivid_color, zivid_depth


def save_coefficients(tmat=np.array, path=str):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('T', tmat)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path=str):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    transf_matrix = cv_file.getNode('T').mat()

    cv_file.release()
    return transf_matrix