from typing import Sequence

import numpy as np
import cv2


def calibrate_extrinsics(charuco_dict,
                         charuco_board,
                         imgs_1: Sequence[np.array], imgs_1_cm: np.array,
                         imgs_1_dc: np.array, imgs_2: Sequence[np.array],
                         imgs_2_cm: np.array, imgs_2_dc: np.array,
                         common_points_threshold: int = 40,
                         silent: bool = True
                         ):
    """
    Arguments
    --------
    
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
                for a, b in zip(obj_points_1[:,
                                             0], img_1_common_frame_points[:,
                                                                           0])
            }
            img_2_obj_to_points = {
                tuple(a): tuple(b)
                for a, b in zip(obj_points_2[:, 0], img_2_points[:, 0])
            }
            common = set(img_1_obj_to_points.keys()) & set(
                img_2_obj_to_points.keys())  # intersection between obj points

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

    return R, T