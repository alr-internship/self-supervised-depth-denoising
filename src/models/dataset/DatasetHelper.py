from functools import reduce
import numpy as np


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

    