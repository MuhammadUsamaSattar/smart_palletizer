from copy import deepcopy

import cv2
import cv_bridge
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from smart_palletizer_py import utils


class PostProcessing(Node):
    """Remove noise and holes from depth and rgb images."""

    def __init__(self) -> None:
        super().__init__("post_processing")

        self.img_depth = None
        self.img_rgb = None
        self.camera_info = None
        self.prev_img_depth = None

        # Declare parameter
        self.declare_parameter(
            "downscaling_factor",
            2,
            ParameterDescriptor(
                description="Factor by which to downsacle the rgb and depth image"
            ),
        )

        # Publisher for processed depth image
        self.filtered_depth_publisher_ = self.create_publisher(
            Image,
            "aligned_depth_to_color/image_filtered",
            10,
        )
        # Publisher for processed rgb image
        self.filtered_rgb_publisher_ = self.create_publisher(
            Image,
            "color/image_filtered",
            10,
        )
        # Publisher for processed depth image's camera info
        self.filtered_depth_camera_info_publisher_ = self.create_publisher(
            CameraInfo,
            "aligned_depth_to_color/camera_info",
            10,
        )
        # Publisher for processed rgb image's camera info
        self.filtered_rgb_camera_info_publisher_ = self.create_publisher(
            CameraInfo,
            "color/camera_info",
            10,
        )

        # Subscriber for raw depth image
        self.img_depth_subscriber_ = self.create_subscription(
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            self.add_img_depth,
            10,
        )
        # Subscriber for raw rgb image
        self.img_rgb_subscriber_ = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.add_img_rgb,
            10,
        )
        # Subscriber for raw camera info
        self.camera_info_subscriber_ = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.add_camera_info,
            10,
        )

        self.create_timer(1 / 30, self.process_imgs)

    def add_img_depth(self, msg: Image) -> None:
        """Add depth image to instance attribute.

        Args:
            msg (Image): Image message.
        """
        self.img_depth = msg

    def add_img_rgb(self, msg: Image) -> None:
        """Add rgb image to instance attribute.

        Args:
            msg (Image): Image message.
        """
        self.img_rgb = msg

    def add_camera_info(self, msg: CameraInfo) -> None:
        """Add camera info to instance attribute.

        Args:
            msg (CameraInfo): CameraInfo message.
        """
        self.camera_info = msg

    def process_imgs(self) -> None:
        """Filter images. 
        
        Apply downscaling, hole patching, median blur and EMA time filter.
        """
        # If class attributes have no value, then skip
        if self.img_depth is None or self.img_rgb is None or self.camera_info is None:
            return

        header = self.img_rgb.header
        # Convert image message to OpenCV image
        bridge = cv_bridge.CvBridge()
        img_depth = bridge.imgmsg_to_cv2(self.img_depth, "16UC1")
        img_rgb = bridge.imgmsg_to_cv2(self.img_rgb, "rgb8")

        # Downscale depth and rgb images
        down_scaling_factor = (
            self.get_parameter("downscaling_factor").get_parameter_value().integer_value
        )
        if down_scaling_factor != 1:
            img_depth = subsample(img_depth, factor=down_scaling_factor, method="auto")
            img_rgb = subsample(img_rgb, factor=down_scaling_factor, method="auto")

        # Remove holes in depth image
        img_depth = holePatching(img_depth, 3, "max", "L-2D-Excl", max_iter=8)

        # Denoise depth image through median blur
        img_depth = cv2.medianBlur(img_depth, 5)

        # Denoise depth image through time filter
        self.prev_img_depth = img_depth = utils.time_filter(
            img_depth, self.prev_img_depth, 0.2, (5) * (2**8)
        ).copy()

        # Construct messages for filtered rgb image, depth image and camera info
        msg = bridge.cv2_to_imgmsg(img_depth, "16UC1", header)
        self.filtered_depth_publisher_.publish(msg)

        msg = bridge.cv2_to_imgmsg(img_rgb, "rgb8", header)
        self.filtered_rgb_publisher_.publish(msg)

        camera_info_msg = deepcopy(self.camera_info)
        camera_info_msg.header = header
        camera_info_msg.height = camera_info_msg.height // down_scaling_factor
        camera_info_msg.width = camera_info_msg.width // down_scaling_factor
        camera_info_msg.k = camera_info_msg.k / down_scaling_factor
        camera_info_msg.p = camera_info_msg.p / down_scaling_factor

        self.filtered_depth_camera_info_publisher_.publish(camera_info_msg)
        self.filtered_rgb_camera_info_publisher_.publish(camera_info_msg)


def subsample(img: np.ndarray, factor: int = 2, method: str = "auto") -> np.ndarray:
    """Sub-sample an image (depth map or RGB) intelligently using non-zero median or mean.

    Args:
        img (np.ndarray): 2D (depth) or 3D (H,W,C) image. np.uint16 for depth.
        np.uint8 for RGB.
        factor (int, optional): Sub-sampling factor. Defaults to 2.
        method (str, optional): 'auto' chooses median for small factor (<4) and
        mean for larger factors.

    Raises:
        ValueError: Raise error if method is neither 'median' nor 'mean'. Raise
        error if image is neither 2D nor 3D.

    Returns:
        np.ndarray: Sub-sampled image of the same dtype as input.
    """
    # Input must be either 2D (depth) or 3D (RGB)
    assert img.ndim in (2, 3)
    dtype = img.dtype

    # If method is 'auto', automatically chose method
    if method == "auto":
        method = "median" if factor < 4 else "mean"

    # Make the img 3D array if image is 2D for consistency
    if img.ndim == 2:
        H, W = img.shape
        C = 1
        img_reshaped = img[:, :, np.newaxis]
    else:
        H, W, C = img.shape
        img_reshaped = img

    new_H = (H + factor - 1) // factor
    new_W = (W + factor - 1) // factor

    # Pad to make divisible by factor
    pad_H = new_H * factor - H
    pad_W = new_W * factor - W
    padded = np.pad(
        img_reshaped,
        ((0, pad_H), (0, pad_W), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Reshape into blocks
    blocks = padded.reshape(new_H, factor, new_W, factor, C)
    blocks = blocks.transpose(0, 2, 1, 3, 4)
    blocks_flat = blocks.reshape(new_H, new_W, factor * factor, C)

    valid = blocks_flat > 0

    if method == "median":
        # Apply median filter on subarray
        blocks_float = blocks_flat.astype(np.float32)
        blocks_masked = np.where(valid, blocks_float, np.nan)
        subsampled = np.nanmedian(blocks_masked, axis=2)
        subsampled = np.nan_to_num(subsampled, nan=0).astype(dtype)
    elif method == "mean":
        # Apply mean filter on subarray
        sums = np.sum(blocks_flat * valid, axis=2, dtype=np.float32)
        counts = np.sum(valid, axis=2, dtype=np.float32)
        subsampled = np.divide(
            sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0
        )
        subsampled = np.nan_to_num(subsampled, nan=0).astype(dtype)
    else:
        raise ValueError("Method must be 'median', 'mean', or 'auto'")

    # Convert to 2D if orignal image was 2D
    if img.ndim == 2:
        subsampled = subsampled[:, :, 0]

    return subsampled


def get_mask(mask_type: str, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Get a sub-array mask depending on type around (dx, dy) pixel.

    Args:
        type (str): Type of mask. Str consists of 2 or 3 parts
        Direction-Dimension-[Inclusion].
                    - Direction: U (Up), D (Down), L (Left), R (Right)
                    - Dimension: 1D, 2D
                    - Inclusion: Incl (Include pixels perpendicular to
                    direction beside central pixel),
                    Excl (Exclude pixels perpendicular to direction beside
                    central pixel)
        dy (np.ndarray): x-coordinate of pixel.
        dx (np.ndarray): y-coordinate of pixel.

    Raises:
        ValueError: Raise error if type is invalid.

    Returns:
        np.ndarray: Output mask.
    """
    if mask_type == "U-1D":
        return dy < 0
    elif mask_type == "D-1D":
        return dy > 0
    elif mask_type == "L-1D":
        return dx < 0
    elif mask_type == "R-1D":
        return dx > 0
    elif mask_type == "U-2D-Excl":
        return (dy < 0) & (np.abs(dx) <= np.abs(dy))
    elif mask_type == "D-2D-Excl":
        return (dy > 0) & (np.abs(dx) <= np.abs(dy))
    elif mask_type == "L-2D-Excl":
        return (dx < 0) & (np.abs(dy) <= np.abs(dx))
    elif mask_type == "R-2D-Excl":
        return (dx > 0) & (np.abs(dy) <= np.abs(dx))
    elif mask_type == "U-2D-Incl":
        return dy < 0
    elif mask_type == "D-2D-Incl":
        return dy > 0
    elif mask_type == "L-2D-Incl":
        return dx < 0
    elif mask_type == "R-2D-Incl":
        return dx > 0
    else:
        raise ValueError(f"Unknown type: {mask_type}")


def holePatching(
    img: np.ndarray, size: int, mode: str, mask_type: str, max_iter: int = 10
) -> np.ndarray:
    """Efficient iterative hole-filling for image using directional masks.

    Median, min, or max are applied only to non-zero neighbors.

    Args:
        img (np.ndarray): Image to hole patch. Must be np.uint16.
        size (int): Odd integer window size within which hole patching is done.
        mode (str): Mode of hole patching algorithm. Options are 'median',
        'min' or 'max'.
        mask_type (str): Type of mask. Str consists of 2 or 3 parts
        Direction-Dimension-[Inclusion].
                    - Direction: U (Up), D (Down), L (Left), R (Right)
                    - Dimension: 1D, 2D
                    - Inclusion: Incl (Include pixels perpendicular to
                    direction beside central pixel),
                    Excl (Exclude pixels perpendicular to direction beside
                    central pixel)
        max_iter (int, optional): Maximum number of iterations to run hole
        patching. Iterations are needed to patch holes in thick regions.
        Defaults to 10.

    Raises:
        ValueError: Raise error if mode is not 'median', 'max' or 'min'. Raise
        error if image is not np.uint16 or size is not odd.

    Returns:
        np.ndarray: Hole patched image.
    """
    assert img.dtype == np.uint16, "Input must be np.uint16"
    assert size % 2 == 1, "Size must be odd"

    h, w = img.shape
    pad = size // 2
    patched = img.copy()

    # Prepare offset grid
    idx = np.arange(-pad, pad + 1)
    dy, dx = np.meshgrid(idx, idx, indexing="ij")

    # Compute only the needed mask
    mask = get_mask(mask_type, dy, dx)
    dy_offsets = dy[mask]
    dx_offsets = dx[mask]

    # Run hole patching algorithm iteratively fill in hole regions.
    for _ in range(max_iter):
        zeros = patched == 0
        if not np.any(zeros):
            break

        # Pad the image
        padded = np.pad(patched, pad_width=pad, mode="edge")

        # Extract neighbours
        neighbors = np.array(
            [
                padded[pad + dy_i : h + pad + dy_i, pad + dx_i : w + pad + dx_i]
                for dy_i, dx_i in zip(dy_offsets, dx_offsets)
            ],
            dtype=np.uint16,
        )

        # Only consider non-zero neighbors
        valid = neighbors > 0

        # Take the 'median', 'min' or 'max' of the region
        if mode == "median":
            neighbors_masked = np.where(valid, neighbors, np.nan)
            newvals = np.nanmedian(neighbors_masked, axis=0)
            newvals = np.nan_to_num(newvals, nan=0).astype(np.uint16)
        elif mode == "min":
            neighbors_masked = np.where(valid, neighbors, np.iinfo(np.uint16).max)
            newvals = np.min(neighbors_masked, axis=0)
            newvals[np.all(~valid, axis=0)] = 0
        elif mode == "max":
            neighbors_masked = np.where(valid, neighbors, 0)
            newvals = np.max(neighbors_masked, axis=0)
        else:
            raise ValueError("Invalid mode. Use 'median', 'min', or 'max'.")

        # Fill only zero pixels with at least one valid neighbor
        fill_mask = zeros & (np.any(valid, axis=0))
        if not np.any(fill_mask):
            break

        patched[fill_mask] = newvals[fill_mask]

    return patched


def main(args=None):
    rclpy.init(args=args)
    node = PostProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
