from collections import namedtuple
import math
from typing import Tuple

import numpy as np
from image_geometry import PinholeCameraModel

BOX_DIMS = namedtuple("BOX_DIMS", ["x", "y", "z"])
SMALL_BOX_DIMS = BOX_DIMS(x=0.255, y=0.155, z=0.100)
MEDIUM_BOX_DIMS = BOX_DIMS(x=0.340, y=0.250, z=0.095)


def time_filter(
    img: np.ndarray, prev_filtered_img: np.ndarray, alpha: float = 0.5, delta: int = 20
) -> np.ndarray:
    """Apply a time filter on the given imag using EMA.

    Args:
        img (np.ndarray): Current image.
        prev_filtered_img (np.ndarray): Image from last frame with EMA applied.
        alpha (float, optional): Weightage of img. Defaults to 0.5.
        delta (int, optional): Difference between pixel values above which EMA is not applied.
        Defaults to 20.

    Returns:
        np.ndarray: Output image with EMA applied.
    """
    # Check if prev_filtered_img is valid
    if not isinstance(prev_filtered_img, np.ndarray):
        return img

    # Store data type of img and convert img and prev_filtered_img to np.float32 for accurate calculation
    dtype = img.dtype
    img = img.astype(np.float32)
    prev_filtered_img = prev_filtered_img.astype(np.float32)

    # Calculate difference between pixel values of img and prev_filtered_img
    diff = np.abs(img - prev_filtered_img)

    # Apply EMA when difference is less than delta
    mask = diff > delta
    a = np.where(mask, 1.0, alpha)
    img = a * img + (1 - a) * prev_filtered_img

    return img.astype(dtype)


def quaternion_from_euler(
    ai: int | float, aj: int | float, ak: int | float
) -> np.ndarray:
    """Calculate quaternion from euler angles.

    Args:
        ai (int | float): Euler angle around x-axis.
        aj (int | float): Euler angle around y-axis.
        ak (int | float): Euler angle around z-axis.

    Returns:
        np.ndarray: Array contained quaternion angles in format [x, y, z, w].
    """
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    q[0] = cj * sc - sj * cs
    q[1] = cj * ss + sj * cc
    q[2] = cj * cs - sj * sc
    q[3] = cj * cc + sj * ss

    return q


def get_XYZ_from_Pixels(
    camera: PinholeCameraModel, u: int | float, v: int | float, d: int | float
) -> Tuple[float, float, float]:
    """_summary_

    Args:
        camera (PinholeCameraModel): Camera model intialized with camera info.
        u (int | float): x-axis co-ordinate.
        v (int | float): y-axis co-ordinate.
        d (int | float): Depth co-ordinate.

    Returns:
        Tuple[float, float, float]: Tuple contained a unit vector pointing from camera frame to pixel (u, v).
    """
    # Get unit vector
    X, Y, Z = camera.project_pixel_to_3d_ray((u, v))

    # Find scaling factor needed to make the z-component of unit vector equal to depth
    factor = d / (1000 * Z)

    return X * factor, Y * factor, Z * factor
