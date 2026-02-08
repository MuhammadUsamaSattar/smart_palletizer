from collections import namedtuple
import math
from typing import Tuple

from image_geometry import PinholeCameraModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


BOX_DIMS = namedtuple("BOX_DIMS", ["x", "y", "z"])
SMALL_BOX_DIMS = BOX_DIMS(x=0.255, y=0.155, z=0.100)
MEDIUM_BOX_DIMS = BOX_DIMS(x=0.340, y=0.250, z=0.095)

# Mapping of object categories to IDs
CLASS_MAP = {
    "medium": 1,
    "small": 2,
}


def time_filter(
    img: np.ndarray, prev_filtered_img: np.ndarray, alpha: float = 0.5, delta: int = 20
) -> np.ndarray:
    """Apply a time filter on the given image using EMA.

    Args:
        img (np.ndarray): Current image.
        prev_filtered_img (np.ndarray): Image from last frame with EMA applied.
        alpha (float, optional): Weight of current image. Defaults to 0.5.
        delta (int, optional): Difference between pixel values above which EMA
        is not applied.
        Defaults to 20.

    Returns:
        np.ndarray: Output image with the EMA applied.
    """
    # Check if prev_filtered_img is valid
    if not isinstance(prev_filtered_img, np.ndarray):
        return img

    # Store data type of img and convert img and prev_filtered_img to np.
    # float32 for accurate calculation
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
    """Get real world XYZ coordinates from pixel coordinates and depth value.

    Args:
        camera (PinholeCameraModel): Camera model initialized with camera info.
        u (int | float): x-axis coordinate.
        v (int | float): y-axis coordinate.
        d (int | float): Depth coordinate.

    Returns:
        Tuple[float, float, float]: Tuple containing a unit vector pointing
        from camera frame to pixel (u, v).
    """
    # Get unit vector
    X, Y, Z = camera.project_pixel_to_3d_ray((u, v))

    # Find scaling factor needed to make the z-component of unit vector equal
    # to depth
    factor = d / (1000 * Z)

    return X * factor, Y * factor, Z * factor


def from_euler_to_rot6d(euler):
    return list(R.from_euler("XYZ", euler).as_matrix()[:, :2].reshape(-1))


def rot6d_to_euler(rot6d, order="xyz"):
    """
    Convert 6D rotation representation to Euler angles using NumPy.

    Args:
        rot6d: np.ndarray of shape (6,) or (N,6)
               First 3 = first column, next 3 = second column
        order: string, Euler order ('xyz' or 'zyx')

    Returns:
        euler_angles: np.ndarray of shape (3,) or (N,3), in radians
    """
    rot6d = np.atleast_2d(rot6d)  # ensure batch dimension

    # Split first and second columns
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    # Normalize first column
    r1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)

    # Make second column orthogonal to first
    dot = np.sum(r1 * a2, axis=1, keepdims=True)
    r2 = a2 - dot * r1
    r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)

    # Third column = cross product
    r3 = np.cross(r1, r2)

    # Build rotation matrix [N,3,3]
    Rmat = np.stack([r1, r2, r3], axis=2)

    # Function to convert a single 3x3 rotation matrix to Euler
    def matrix_to_euler(R, order):
        if order == "xyz":
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        elif order == "zyx":
            sy = np.sqrt(R[2, 2] ** 2 + R[2, 1] ** 2)
            singular = sy < 1e-6
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            raise NotImplementedError(f"Euler order {order} not implemented")
        return np.array([x, y, z])

    # Convert each rotation matrix to Euler
    eulers = np.array([matrix_to_euler(Rmat[i], order) for i in range(Rmat.shape[0])])

    # If single input, return 1D array
    if eulers.shape[0] == 1:
        return eulers[0]
    return eulers


def rot6d_to_quaternion(rot6d):
    """
    Convert 6D rotation representation to quaternion (x, y, z, w).

    Args:
        rot6d: np.ndarray of shape (6,) or (N,6)
               First 3 = first column of rotation matrix
               Next 3  = second column

    Returns:
        quaternions: np.ndarray of shape (4,) or (N,4) [x, y, z, w]
    """
    rot6d = np.atleast_2d(rot6d)

    # Split first two columns
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    # Normalize first column
    r1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)

    # Make second column orthogonal to first
    dot = np.sum(r1 * a2, axis=1, keepdims=True)
    r2 = a2 - dot * r1
    r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)

    # Third column = cross product
    r3 = np.cross(r1, r2)

    # Build rotation matrix
    Rmat = np.stack([r1, r2, r3], axis=1)  # shape [N,3,3]

    # Convert rotation matrix to quaternion
    quaternions = []
    for i in range(Rmat.shape[0]):
        R = Rmat[i]
        q = np.empty(4)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2, 1] - R[1, 2]) * s
            q[1] = (R[0, 2] - R[2, 0]) * s
            q[2] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s
        quaternions.append(q)

    quaternions = np.array(quaternions)
    if quaternions.shape[0] == 1:
        return quaternions[0]
    return quaternions


def visualize_box_and_pose_data(
    X, y, options={"color", "bboxes", "seg_masks", "depth"}
):
    if "color" in options:
        fig, ax = plt.subplots()

        # Show base image
        ax.imshow(X[0], cmap="gray")

        if "seg_masks" in options:
            colors = plt.cm.tab10.colors  # distinct colors
            masks = y["masks"]
            for k, mask in enumerate(masks):
                color = colors[k % len(colors)]

                # Create RGBA overlay (only where mask == 1)
                overlay = np.zeros((*mask.shape, 4))
                overlay[..., :3] = color
                mask_np = mask.detach().cpu().numpy()
                overlay[..., 3] = mask_np * 0.4

                ax.imshow(overlay)

        # Draw bounding boxes
        if "bboxes" in options:
            bboxes = y["bboxes"]
            for bbox in bboxes:
                rect = patches.Rectangle(
                    bbox[:2] * torch.tensor(X[0].shape[-1::-1]),
                    bbox[2] * X[0].shape[1],
                    bbox[3] * X[0].shape[0],
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

    # Show depth image separately
    if "depth" in options:
        plt.figure()
        plt.imshow(X[1], cmap="gray")

    plt.show()
