import math
from typing import List, Tuple, Dict

import cv2
from cv_bridge import CvBridge
import image_geometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

from smart_palletizer_py import utils
from smart_palletizer_interfaces.msg import DetectedBox, DetectedBoxes


class BoxDetection(Node):
    """Detect boxes in image using Canny Edge detection."""

    def __init__(self) -> None:
        super().__init__("detection")

        self.bridge = CvBridge()
        self.img_depth = None
        self.img_rgb = None
        self.camera = image_geometry.PinholeCameraModel()
        self.prev_filtered_img = {"h": None, "s": None, "v": None, "depth": None}
        self.detected_boxes = {"n": 0, "box_infos": {}}

        # Publisher for image with bounding boxes for detected boxes
        self.detected_box_img_publisher_ = self.create_publisher(
            Image, "/box_detection_img", 10
        )
        # Publisher for information of detected boxes
        self.detected_boxes_publisher_ = self.create_publisher(
            DetectedBoxes, "/detected_boxes", 10
        )

        # Subscription for filtered rgb image
        self.rgb_subscription = self.create_subscription(
            Image, "/camera_filtered/color/image_filtered", self.add_img_rgb, 10
        )
        # Subscription for filtered depth image
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/camera_filtered/aligned_depth_to_color/camera_info",
            self.add_camera_info,
            10,
        )
        # Subscription for filtered camera information
        self.depth_subscription = self.create_subscription(
            Image,
            "/camera_filtered/aligned_depth_to_color/image_filtered",
            self.add_img_depth,
            10,
        )

        self.timer = self.create_timer(1 / 30, self.detect_boxes)

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
        self.camera.from_camera_info(msg)

    def detect_boxes(self) -> None:
        """Detect boxes and publish image message with bounding boxes and information of boxes"""
        # If instance attributes have not been assigned then skip
        if (
            self.img_rgb is None
            or self.img_depth is None
            or self.camera.get_tf_frame is None
        ):
            return

        header = self.img_rgb.header

        # Convert ROS2 images to OpenCV
        img_depth = self.bridge.imgmsg_to_cv2(self.img_depth, desired_encoding="16UC1")
        img_rgb = self.bridge.imgmsg_to_cv2(self.img_rgb, desired_encoding="bgr8")

        ### --- Image pre-processing --- ###
        img_depth_processed = img_depth

        # Convert BGR image to HSV and offset the color since our target boxes are close to 0 value.
        # Equalize histograms in V space to improve contrast
        img_rgb_processed = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        img_rgb_processed[:, :, 0] = 100 + img_rgb_processed[:, :, 0]
        img_rgb_processed[:, :, 2] = cv2.equalizeHist(img_rgb_processed[:, :, 2])

        # Apply time filtering to each of HSV channel
        self.prev_filtered_img["h"] = img_rgb_processed[:, :, 0] = utils.time_filter(
            img_rgb_processed[:, :, 0], self.prev_filtered_img["h"], 0.05, 180
        ).copy()
        self.prev_filtered_img["s"] = img_rgb_processed[:, :, 1] = utils.time_filter(
            img_rgb_processed[:, :, 1], self.prev_filtered_img["s"], 0.1, 50
        ).copy()
        self.prev_filtered_img["v"] = img_rgb_processed[:, :, 2] = utils.time_filter(
            img_rgb_processed[:, :, 2], self.prev_filtered_img["v"], 0.2, 50
        ).copy()

        ### --- Combined image pre-processing --- ###
        # Create masks from depth to isolate just the region of interest
        upper_depth_lim = 1775
        lower_depth_lim = 1500
        mask1 = np.zeros_like(img_depth_processed)
        _, mask1 = cv2.threshold(
            img_depth_processed, upper_depth_lim, 255, cv2.THRESH_BINARY_INV
        )
        mask1 = mask1.astype(np.uint8)

        mask2 = np.zeros_like(img_depth_processed)
        _, mask2 = cv2.threshold(
            img_depth_processed, lower_depth_lim, 255, cv2.THRESH_BINARY
        )
        mask2 = mask2.astype(np.uint8)

        # Rescale depth image, convert to uint8 and append to the uint8 HSV image
        np.clip(
            img_depth_processed, lower_depth_lim, upper_depth_lim, img_depth_processed
        )
        img_depth_processed = cv2.convertScaleAbs(
            img_depth_processed,
            alpha=255.0 / (upper_depth_lim - lower_depth_lim),
            beta=-lower_depth_lim,
        )
        img_rgb_processed = np.dstack((img_rgb_processed, img_depth_processed))

        # Apply mask to combined image
        img_rgb_processed = cv2.bitwise_and(
            img_rgb_processed, img_rgb_processed, mask=mask1
        )
        img_rgb_processed = cv2.bitwise_and(
            img_rgb_processed, img_rgb_processed, mask=mask2
        )

        ### --- Edge detection --- ###
        # Isolate depth image
        img_depth_processed = img_rgb_processed[:, :, 3]

        # Run Canny edge detection on combined image to extract boxes that can be identified by both hsv and depth change
        contours = self.findContours(
            img_rgb_processed,
            canny_range=(110, 110),
            dilation_kernel_size=(7, 7),
            dilation_iterations=1,
        )
        # Run Canny edge detection on only depth image to extract boxes that can be identified by only depth change
        contours_depth = self.findContours(
            img_depth_processed,
            canny_range=(40, 40),
            dilation_kernel_size=(5, 5),
            dilation_iterations=2,
        )

        # Detects and draws the areas representing the boxes, bounding boxes and labels
        self.getBoxes(img_depth, contours_depth + contours)
        img_rgb = self.drawBoxes(img_rgb, alpha=0.2)

        ### --- Publish messages --- ###
        detected_boxes_data = []
        for k, (_, _, box_info, updated) in self.detected_boxes["box_infos"].items():
            if not updated:
                continue
            x, y, longest_side_coords, depth, classification = box_info

            detected_box = DetectedBox()
            detected_box.x = int(x)
            detected_box.y = int(y)
            detected_box.id = k
            detected_box.longest_side_coords[0].x = int(longest_side_coords[0][0])
            detected_box.longest_side_coords[0].y = int(longest_side_coords[0][1])
            detected_box.longest_side_coords[1].x = int(longest_side_coords[1][0])
            detected_box.longest_side_coords[1].y = int(longest_side_coords[1][1])
            detected_box.depth = int(depth)
            detected_box.classification = classification
            detected_boxes_data.append(detected_box)

        detected_boxes_msg = DetectedBoxes()
        detected_boxes_msg.header = header
        detected_boxes_msg.data = detected_boxes_data

        img_msg = self.bridge.cv2_to_imgmsg(img_rgb, "passthrough", header)
        self.detected_box_img_publisher_.publish(img_msg)
        self.detected_boxes_publisher_.publish(detected_boxes_msg)

        # Reset images
        self.img_depth = None
        self.img_rgb = None

    def findContours(
        self,
        img: np.ndarray,
        canny_range: Tuple[int | float, int | float],
        dilation_kernel_size: Tuple[int, int],
        dilation_iterations: int,
    ) -> List[List[List[int]]]:
        """Find contours in image using Canny Edge Detection.

        Args:
            img (np.ndarray): Image in which contours are detected.
            canny_range (Tuple[int  |  float, int  |  float]): Thresholds for Canny in
            the format (lower_threshold, upper_threshold).
            dilation_kernel_size (int): Kernel size for dilation of contours.
            dilation_iterations (int): Number of iterations for dilation of contours.

        Returns:
            List[List[List[int, int]]]: List containing contours.
        """
        # Detect contours
        img = cv2.Canny(img, canny_range[0], canny_range[1])

        # Dilate contours to strength edge relations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        img = cv2.morphologyEx(
            img, cv2.MORPH_DILATE, kernel, iterations=dilation_iterations
        )

        return cv2.findContours(
            img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )[0]

    def getBoxes(
        self, img_depth: np.ndarray, contours: List[List[List[int]]]
    ) -> Tuple[List[int], List[int], Tuple[int, int, Tuple[int, int], int, str]]:
        """Get the boudning box, box contour and box information from contours.

        Args:
            img_depth (np.ndarray): Depth image.
            contours (List[List[List[int, int]]]): List containing contours.

        Returns:
            List[List[int, int, int, int], List[int, int], Tuple[int, int, List[int, int], int, str]]:
            Box information in the format [Bounding Box, Box Contour, Box Information].
        """
        # Assign all previous boxes as undetected
        for k, prev_cnts in self.detected_boxes["box_infos"].items():
            prev_cnts[3] = False

        for cnt in contours:
            # Calculate the average depth for each point in the contour
            depth = np.mean([img_depth[point[0][1], point[0][0]] for point in cnt])

            # Find the box contour
            box_contour = cv2.boxPoints(cv2.minAreaRect(cnt))
            box_contour = np.int0(box_contour)

            # Get real-world co-ordinates of three ponts of the box contour and find the two side lengths
            p1 = utils.get_XYZ_from_Pixels(
                self.camera, box_contour[0][0], box_contour[0][1], depth
            )
            p2 = utils.get_XYZ_from_Pixels(
                self.camera, box_contour[1][0], box_contour[1][1], depth
            )
            p3 = utils.get_XYZ_from_Pixels(
                self.camera, box_contour[2][0], box_contour[2][1], depth
            )
            l1, l2 = math.dist(p1, p2), math.dist(p2, p3)

            # Assign l1 as the larger of the two sides and l2 as the smaller
            if l1 >= l2:
                longest_side_coords = (box_contour[0], box_contour[1])
            else:
                l1, l2 = l2, l1
                longest_side_coords = (box_contour[1], box_contour[2])

            # Calculate the sum of squared error between the l1 and l2 and second-largest dim of each box
            errors = [
                (
                    (l1 - utils.SMALL_BOX_DIMS.x) ** 2
                    + (l2 - utils.SMALL_BOX_DIMS.y) ** 2
                ),
                (
                    (l1 - utils.MEDIUM_BOX_DIMS.x) ** 2
                    + (l2 - utils.MEDIUM_BOX_DIMS.y) ** 2
                ),
            ]

            # Assign classification as the box with lower error and assign error[0] as the smaller of the two errors
            if errors[0] <= errors[1]:
                classification = "small"
            else:
                classification = "medium"
                errors[0], errors[1] = errors[1], errors[0]

            # Process contour only if error is below threshold
            if errors[0] < 0.008:
                # Calculate the centroid of the contour
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Find a the bounding box for the contour
                bounding_box = cv2.boundingRect(cnt)
                # Data for the detected box
                data = [
                    bounding_box,
                    box_contour,
                    (cx, cy, longest_side_coords, depth, classification),
                    True,
                ]
                # Difference in pixel values before two boxes are considered to be separate
                diff = 20

                # Assign box data to a close detected box in previous frame. Otherwise add the box as a new box.
                for k, prev_cnts in self.detected_boxes["box_infos"].items():
                    if (
                        abs(prev_cnts[2][0] - cx) <= diff
                        and abs(prev_cnts[2][1] - cy) <= diff
                    ):
                        self.detected_boxes["box_infos"][k] = data
                        break
                else:
                    self.detected_boxes["box_infos"][
                        "".join(("box_", str(self.detected_boxes["n"])))
                    ] = data
                    self.detected_boxes["n"] += 1

    def drawBoxes(
        self,
        img: np.ndarray,
        alpha: float = 0.2,
    ) -> np.ndarray:
        """Draw the bounding box, box contour and box label.

        Args:
            img (np.ndarray): Image on which to draw.
            detected_boxes (Tuple[List[int], List[int], Tuple[int, int, Tuple[int, int], int, str]]):
            Box information in the format [Bounding Box, Box Contour, Box Information].
            alpha (float, optional): Bleding parameter. Higher value of alpha leads to stronger
            contour fill color. Defaults to 0.2.

        Returns:
            np.ndarray: Output image with bounding box, box contour and box label.
        """
        contour_mask = np.zeros(img.shape, dtype=img.dtype)

        for bounding_box, box_contour, box_info, updated in self.detected_boxes[
            "box_infos"
        ].values():
            if not updated:
                continue

            x, y, w, h = bounding_box
            cv2.drawContours(
                contour_mask, [box_contour], -1, (0, 0, 255), thickness=cv2.FILLED
            )
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
            )
            cv2.rectangle(
                img,
                (x, y - 4),
                (x + (60 if box_info[4] == "medium" else 44), y - 24),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                img,
                box_info[4],
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
            )

        return cv2.addWeighted(img, 1 - alpha, contour_mask, alpha, 0)


def main(args=None):
    rclpy.init(args=args)
    node = BoxDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
