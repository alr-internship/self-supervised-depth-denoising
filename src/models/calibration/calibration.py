from typing import Sequence

from pathlib import Path
from models.dataset.dataset_container import DatasetContainer
import numpy as np
import cv2


class StereoCalibration:
    def __init__(self) -> None:
        self.__map1x = None
        self.__map1y = None
        self.__map2x = None
        self.__map2y = None

    def configure(self, map1x: np.array, map1y: np.array, map2x: np.array,
                  map2y: np.array):
        self.__map1x = np.copy(map1x)
        self.__map1y = np.copy(map1y)
        self.__map2x = np.copy(map2x)
        self.__map2y = np.copy(map2y)

    def configure_from_dataset(
        self,
        charuco_dict,
        charuco_board,
        dataset_container: DatasetContainer,
        common_points_threshold: int = 40,
        alpha: int = -1,
        silent: bool = True,
    ):
        self.configure(
            charuco_dict=charuco_dict,
            charuco_board=charuco_board,
            common_points_threshold=common_points_threshold,
            imgs_1=dataset_container.realsense.rgb,
            imgs_1_cm=dataset_container.realsense.camera_matrix,
            imgs_1_dc=dataset_container.realsense.distortion_coefficients,
            imgs_2=dataset_container.zivid.rgb,
            imgs_2_cm=dataset_container.zivid.camera_matrix,
            imgs_2_dc=dataset_container.zivid.distortion_coefficients,
            alpha=alpha,
            silent=silent)

    def configure(self,
                  charuco_dict,
                  charuco_board,
                  imgs_1: Sequence[np.array],
                  imgs_1_cm: np.array,
                  imgs_1_dc: np.array,
                  imgs_2: Sequence[np.array],
                  imgs_2_cm: np.array,
                  imgs_2_dc: np.array,
                  common_points_threshold: int = 40,
                  alpha: int = -1,
                  silent: bool = True):
        """
        Arguments
        --------
        alpha: int
            Free scaling parameter. 
            If it is -1 or absent, the function performs the default scaling. 
            Otherwise, the parameter should be between 0 and 1. 
            alpha=0 means that the rectified images are zoomed and shifted so 
            that only valid pixels are visible (no black areas after rectification). 
            alpha=1 means that the rectified image is decimated and shifted so that 
            all the pixels from the original images from the cameras are retained 
            in the rectified images (no source image pixels are lost). 
            Any intermediate value yields an intermediate result between those two extreme cases.
        
        Results
        --------
        R: np.array (shape: 3x3)
            rotation matrix 
        T: np.array (shape: 3)
            translation vector

        """
        assert len(imgs_1) == len(imgs_2) > 0

        # Reference: https://stackoverflow.com/questions/64612924/opencv-stereocalibration-of-two-cameras-using-charuco
        imgs_1_points = []
        imgs_2_points = []
        obj_points = []

        img_1_size = tuple(reversed(imgs_1[0].shape[:2]))
        img_2_size = tuple(reversed(imgs_2[0].shape[:2]))

        for img_1, img_2 in zip(imgs_1, imgs_2):
            try:
                # convert BGR to Gray
                img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

                # Find markers corners
                img_1_corners, img_1_ids, _ = cv2.aruco.detectMarkers(
                    img_1, charuco_dict)
                img_2_corners, img_2_ids, _ = cv2.aruco.detectMarkers(
                    img_2, charuco_dict)
                if not img_1_corners or not img_2_corners:
                    raise Exception("No markers detected")

                # find charcuo corners # TODO: try with cameraMatrix/distCoeffs
                retA, img_1_corners, img_1_ids = cv2.aruco.interpolateCornersCharuco(
                    img_1_corners, img_1_ids, img_1, charuco_board)
                retB, img_2_corners, img_2_ids = cv2.aruco.interpolateCornersCharuco(
                    img_2_corners, img_2_ids, img_2, charuco_board)
                if not retA or not retB:
                    raise Exception("Can't interpolate corners")

                # Find common points in both frames (is there a nicer way?)
                obj_points_1, img_1_common_frame_points = cv2.aruco.getBoardObjectAndImagePoints(
                    charuco_board, img_1_corners, img_1_ids)
                obj_points_2, img_2_points = cv2.aruco.getBoardObjectAndImagePoints(
                    charuco_board, img_2_corners, img_2_ids)

                # Create dictionary for each frame objectPoint:imagePoint to get common markers detected
                img_1_obj_to_points = {
                    tuple(a): tuple(b)
                    for a, b in zip(obj_points_1[:, 0],
                                    img_1_common_frame_points[:, 0])
                }
                img_2_obj_to_points = {
                    tuple(a): tuple(b)
                    for a, b in zip(obj_points_2[:, 0], img_2_points[:, 0])
                }
                common = set(img_1_obj_to_points.keys()) & set(
                    img_2_obj_to_points.keys()
                )  # intersection between obj points

                if len(common) < common_points_threshold:
                    raise Exception("To few respective points found in images")

                # fill arrays where each index specifies one markers objectPoint and both
                # respective imagePoints
                img_common_obj_points = []
                img_1_common_frame_points = []
                img_2_common_frame_points = []
                for objP in common:
                    img_common_obj_points.append(np.array(objP))
                    img_1_common_frame_points.append(
                        np.array(img_1_obj_to_points[objP]))
                    img_2_common_frame_points.append(
                        np.array(img_2_obj_to_points[objP]))
                imgs_1_points.append(np.array(img_1_common_frame_points))
                imgs_2_points.append(np.array(img_2_common_frame_points))
                obj_points.append(np.array(img_common_obj_points))

            except Exception as e:
                if not silent:
                    print(f"Skipped frame: {e}")
                continue

        # calculates transformation (T) and rotation (R) between both cameras
        results = cv2.stereoCalibrate(obj_points,
                                      imgs_1_points,
                                      imgs_2_points,
                                      imgs_1_cm,
                                      imgs_1_dc,
                                      imgs_2_cm,
                                      imgs_2_dc,
                                      img_1_size,
                                      flags=cv2.CALIB_FIX_INTRINSIC)

        _, _, _, _, _, R, T, _, _ = results

        if not silent:
            print("R")
            print(R)
            print("T")
            print(T)

        # calculate rectification and projection matrizes for both cameras
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(imgs_1_cm,
                                                    imgs_1_dc,
                                                    imgs_2_cm,
                                                    imgs_2_dc,
                                                    img_1_size,
                                                    R,
                                                    T,
                                                    alpha=alpha)

        # produces maps to remap images later on
        self.__map1x, self.__map1y = cv2.initUndistortRectifyMap(
            imgs_1_cm, imgs_1_dc, R1, P1, img_1_size, cv2.CV_32FC1)
        self.__map2x, self.__map2y = cv2.initUndistortRectifyMap(
            imgs_2_cm, imgs_2_dc, R2, P2, img_2_size, cv2.CV_32FC1)

    def remap(self, img1: np.array, img2: np.array):
        """
        Arguments
        ---------
        img1: np.array
            first image of image pair to remap
        img2: np.array
            second image of image pair to remap

        Returns
        --------
        imgU1: np.array
            first image of image pair that is remapped
        imgU2: np.array
            second image of image pair that is remapped
        """
        assert self.__map1x is not None and self.__map1y is not None
        assert self.__map2x is not None and self.__map2y is not None

        # remap both images resulting in an pixel-wise correspondance
        imgU1 = cv2.remap(img1, self.__map1x, self.__map1y, cv2.INTER_LINEAR,
                          cv2.BORDER_CONSTANT)
        imgU2 = cv2.remap(img2, self.__map2x, self.__map2y, cv2.INTER_LINEAR,
                          cv2.BORDER_CONSTANT)

        return imgU1, imgU2

    def get_calibration(self):
        """
        returns all np.arrays needed to configure calibration 

        Returns:
        map1x, map1y, map2x, map2y
        """
        return map(np.copy,
                   [self.__map1x, self.__map1y, self.__map2x, self.__map2y])

    def save_calibration(self, path: Path):
        assert self.__map1x is not None and self.__map1y is not None
        assert self.__map2x is not None and self.__map2y is not None

        cv_file = cv2.FileStorage(path.as_posix(), cv2.FILE_STORAGE_WRITE)
        cv_file.write('map1x', self.__map1x)
        cv_file.write('map1y', self.__map1y)
        cv_file.write('map2x', self.__map2x)
        cv_file.write('map2y', self.__map2y)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_calibration(self, path: Path):
        cv_file = cv2.FileStorage(path.as_posix(), cv2.FILE_STORAGE_READ)
        self.__map1x = cv_file.getNode('map1x').mat()
        self.__map1y = cv_file.getNode('map1y').mat()
        self.__map2x = cv_file.getNode('map2x').mat()
        self.__map2y = cv_file.getNode('map2y').mat()
        cv_file.release()
